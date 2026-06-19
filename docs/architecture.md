# NexusRT - architecture

> Beyond the OS: A Firmware-Centric Runtime for End-to-End Acceleration of
> LLM Pipelines on NVIDIA GPUs.

This document describes the system design, data flow, firmware-equivalent
layer rationale, OS-bypass boundaries, and hardware abstraction model of
NexusRT. It is the canonical reference for contributors.

---

## 1. Design summary

NexusRT is a firmware-centric, OS-bypass runtime for end-to-end LLM
workloads (preprocessing -> training -> inference -> postprocessing). It
replaces the framework + runtime + OS stack traditionally used to drive a
GPU with a single user-space + GPU-resident layer that owns scheduling,
memory paging, DMA transfers, and pipeline orchestration directly through
hardware-aware abstractions.

The runtime targets two execution paths:

| Path                | Hardware              | Lowest API used                                  |
| ------------------- | --------------------- | ------------------------------------------------ |
| NVIDIA primary      | A100 (Ampere) / H100 (Hopper) | CUDA Driver API (`cu*`), `cuFile`, `NCCL` |
| Apple unified       | M1 Pro / M2 Pro       | Metal (`MTLDevice`, `MTLBuffer`, `MTLComputePipelineState`), MLX interop |

The two paths share a single C ABI (`src/platform/abi.h`) and a single set
of higher-level C++ classes (`firmware::*`, `memory::*`, `scheduler::*`,
`pipeline::*`, `token_opt::*`). The platform dispatch layer
(`src/platform/dispatch.hpp`) selects the active backend at boot time.

---

## 2. Stack comparison

### 2.1 Traditional stack (PyTorch / TF / JAX on CUDA)

```
┌──────────────────────────────────────────────────────────┐
│  Application                                             │
├──────────────────────────────────────────────────────────┤
│  Framework dispatcher, autograd, device allocator        │
├──────────────────────────────────────────────────────────┤
│  CUDA Runtime (cudart) - wrappers over driver            │
├──────────────────────────────────────────────────────────┤
│  CUDA Driver (libcuda) - user-space                      │
├──────────────────────────────────────────────────────────┤
│  Host OS (Linux page cache, IOMMU, IRQ routing)          │  <- page-fault latency
├──────────────────────────────────────────────────────────┤     IOMMU stalls
│  GPU firmware (closed, NVIDIA-only boot path)            │     context switches
└──────────────────────────────────────────────────────────┘
```

In this stack, CUDA Unified Memory routes page faults through the host
Linux kernel, which:

1. requires a modern kernel (6.1.24+, 6.2.11+) for full UM functionality;
2. on every fault, performs a kernel-mode switch, page-table update, and
   page migration - latency spikes of tens of microseconds;
3. uses `cudaMemAdvise` and `cudaMemPrefetchAsync` as *hints* to the OS
   memory manager, not as authoritative commands.

### 2.2 NexusRT stack

```
┌──────────────────────────────────────────────────────────┐
│  Application (calls nexusrt_* C ABI / Python bindings)   │
├──────────────────────────────────────────────────────────┤
│  NexusRT micro-kernel (user-space + GPU-resident)        │
│   • firmware-equivalent boot sequence                    │
│   • GPU-driven virtual memory (DREAM-style)              │
│   • warp-specialized task graph executor                 │
│   • GDS / GRDMA / TMA / ILC integration                  │
├──────────────────────────────────────────────────────────┤
│  CUDA Driver API (cu*) / Metal API (MTL*)                │
├──────────────────────────────────────────────────────────┤
│  GPU firmware (vendor)                                   │
└──────────────────────────────────────────────────────────┘
```

The host OS is demoted from "memory & I/O authority" to "interrupt router."
Page faults are handled by GPU-resident firmware-equivalent threads
reading from a fault buffer in HBM, exactly as described in the DREAM
project. The OS kernel is never entered for memory management.

---

## 3. Firmware boundary (important)

True vendor firmware modification is restricted on NVIDIA devices. NexusRT
therefore implements a firmware-equivalent user-space micro-kernel that
uses only the lowest publicly available CUDA Driver APIs:

| Operation                    | Vendor firmware would      | NexusRT uses                       |
| ---------------------------- | -------------------------- | ---------------------------------- |
| Boot / clock config          | Direct PLL writes          | `cuDevicePrimaryCtxCreate` + clock lock |
| HBM controller init          | Memory-controller registers| `cuMemAddressReserve` + `cuMemMap` |
| DMA engine registration      | DMA-descriptor ring setup  | `cuStreamCreateWithPriority` + `cuMemcpyAsync` (or `cuTensorMapEncodeTiled` on Hopper) |
| Page-fault handling          | Fault-handler thread on SM | `cuMemAdvise` + HBM-resident fault buffer polled by host thread |
| GRDMA fetch                  | NIC register writes        | NCCL / IBV verbs (via `cuFile` + GDS) |
| ILC tagging                  | Hardware allocator         | `CU_MEM_ALLOCATION_PROP_COMPRESSED` at `cuMemCreate` |
| TMA descriptor encoding      | Inline in device code      | `cuTensorMapEncodeTiled` (host-side) |

This boundary is *non-negotiable*. NexusRT never patches vendor firmware,
never loads a kernel module, and never requires elevated privileges beyond
what cuFile / NCCL already require. Everything else - task scheduling,
memory paging, KV-cache pruning, pipeline orchestration - is implemented
in the NexusRT layer.

### 3.1 What "firmware-equivalent" means in practice

The firmware-equivalent micro-kernel is a small C++17 static library
(`src/firmware/`) plus a GPU-resident code bundle (PTX / Metal shaders)
that together:

1. Boots by acquiring the primary CUDA context (or MTLDevice),
   reserving a virtual address range, mapping the HBM pool, registering
   DMA engines, spawning the stream pool, installing the fault buffer,
   and enabling ILC (H100 only). See `firmware::boot()` in
   `src/firmware/boot.cpp`.
2. Schedules by acquiring streams from a per-class pool (compute / DMA
   / fence) and launching kernels with explicit grid + warp specialization
   configuration. See `firmware::MicroKernel` in
   `src/firmware/microkernel.cpp`.
3. Manages memory by intercepting faults through an HBM-resident ring
   buffer and resolving them via GRDMA / GDS / blit copies. See
   `firmware::FaultHandler` in `src/firmware/fault_handler.cpp`.
4. Encodes TMA descriptors for Hopper and caches them in HBM. See
   `firmware::TmaEngine` in `src/firmware/tma_engine.cpp`.
5. Tags allocations with ILC on H100. See `firmware::IlcManager` in
   `src/firmware/ilc_manager.cpp`.

The boot sequence is logged phase-by-phase via `BootOptions::on_event`,
which is the primary diagnostic surface for the firmware-equivalent layer.

---

## 4. Module layout

```
nexusrt/
├── src/
│   ├── firmware/      GPU micro-kernel & bare-metal-equivalent abstractions
│   │   ├── boot.{hpp,cpp}            Phase-by-phase boot sequence
│   │   ├── context.{hpp,cpp}         FirmwareContext (owns every subsystem)
│   │   ├── microkernel.{hpp,cpp}     Stream pool + module table + launcher
│   │   ├── dma_engine.{hpp,cpp}      GDS / GRDMA / D2D request abstraction
│   │   ├── tma_engine.{hpp,cpp}      Hopper TMA descriptor cache (H100 only)
│   │   ├── fault_handler.{hpp,cpp}   DREAM-style HBM-resident fault buffer
│   │   ├── ilc_manager.{hpp,cpp}     Inline Compression tag manager (H100)
│   │   ├── cuda_driver_shim.cu       CUDA Driver API bridge (the only libcuda caller)
│   │   ├── tma_descriptor.cu         cuTensorMapEncodeTiled wrappers
│   │   ├── ilc_allocations.cu        CU_MEM_ALLOCATION_PROP_COMPRESSED
│   │   └── metal_shim.mm             Metal / MLX bridge
│   ├── memory/        GPU-driven virtual memory, GDS/GRDMA, coalescer
│   │   ├── page_table.{hpp,cpp}      DREAM-style page table (host mirror)
│   │   ├── manager.{hpp,cpp}         High-level allocator (nexusrt_mem_alloc)
│   │   ├── coalescer.{hpp,cpp}       Warp-aligned fragment merger
│   │   ├── gds_router.cpp            GPUDirect Storage fetch router
│   │   └── grdma_router.cpp          GPUDirect RDMA fetch router
│   ├── scheduler/     Async task graph + warp specialization
│   │   ├── graph.{hpp,cpp}           DAG executor with stage contracts
│   │   ├── warp_specialization.{hpp,cpp}  Per-arch policy (2+14 on Ampere/Hopper)
│   │   └── stream_pool.{hpp,cpp}     RAII stream acquire/release
│   ├── pipeline/      End-to-end LLM orchestrator
│   │   ├── orchestrator.{hpp,cpp}    Pipeline builder + run() entry
│   │   ├── preprocess.{hpp,cpp}      GDS streaming + GPU tokenize
│   │   ├── train.{hpp,cpp}           Fwd+bwd + NCCL gradient sync
│   │   ├── infer.{hpp,cpp}           Autoregressive decode + KV mgmt
│   │   ├── postprocess.{hpp,cpp}     Detokenize + format
│   │   └── kernels/*.cu              tokenize, transformer, decode, detokenize
│   ├── token_opt/     ICM layered context + KV-cache pruning
│   │   ├── scope.{hpp,cpp}           Layered context (L0..L4)
│   │   ├── prefetcher.{hpp,cpp}      Attention-weighted slot prefetch
│   │   └── kv_cache.{hpp,cpp}        PagedAttention-style page table
│   └── platform/      Hardware dispatch (NVIDIA CUDA vs Apple Metal)
│       ├── dispatch.{hpp,cpp}        Singleton backend selector
│       ├── abi.h                     Unified C ABI
│       ├── cuda_include.h            CUDA header guard
│       ├── cuda_driver_loader.cpp    Link shim
│       └── metal_loader.mm           Link shim
└── python/nexusrt/    Python bindings (ctypes) + CLI
    ├── _abi.py           Low-level ctypes binding
    ├── firmware.py       Boot / shutdown / device info
    ├── memory.py         alloc / free / prefetch
    ├── scheduler.py      Stage contract builder + submit
    ├── token_opt.py      ICM layers + prefetch_attention + prune
    ├── pipeline.py       End-to-end run_infer
    ├── platform.py       detect() / is_nvidia() / is_apple()
    ├── config.py         YAML config loader
    ├── cli_run.py        nexusrt-run
    └── cli_bench.py      nexusrt-bench
```

---

## 5. Data flow

### 5.1 Preprocess (GDS -> HBM -> tokenize)

```
   NVMe SSD  ──GDS zero-copy──▶  HBM (raw bytes)  ──GPU kernel──▶  HBM (token IDs)
                                       │
                                       └── Warp-specialized producer warps issue
                                           cp.async (A100) or TMA bulk (H100) reads
                                           while consumer warps tokenize in parallel.
```

### 5.2 Train (forward -> backward -> allreduce)

```
   HBM (weights) ──┐
   HBM (tokens)  ──┼──▶  Forward kernel  ──▶  Activations  ──▶  Loss
                   │                                                │
                   │                                                ▼
                   │                                           Backward kernel
                   │                                                │
                   │                                                ▼
                   │                                            Gradients
                   │                                                │
                   └──── NCCL allreduce (GRDMA-preferred) ◀────────┘
                                          │
                                          ▼
                                     Updated weights
```

### 5.3 Infer (autoregressive decode + KV management)

```
   Prompt tokens  ──▶  Prefill (chunks of 256 tok)  ──▶  KV cache
                                                            │
                                                            ▼
                              ┌── Decode step ◀─────────────┤
                              │   1. TokenOpt::step()       │
                              │   2. Attention prefetcher   │
                              │   3. Decode kernel          │
                              │   4. Append next token      │
                              │   5. KV-cache prune         │
                              └────────────────────────────┘
                                          │
                                          ▼
                                    Output token IDs
                                          │
                                          ▼
                                  Postprocess (host copy on final)
```

### 5.4 Postprocess

```
   HBM (token IDs)  ──GPU detokenize kernel──▶  HBM (UTF-8 bytes)
                                                       │
                                                       └──▶ Host (only on final answer)
```

---

## 6. Hardware abstraction model

The `platform::PlatformInterface` (in `src/platform/dispatch.hpp`) is the
single point of dispatch between NVIDIA and Apple paths. Every other layer
calls `platform::PlatformDispatch::instance().<method>()` and is unaware
of which backend is active.

### 6.1 NVIDIA path

* CUDA Driver API (libcuda.so / nvcuda.dll) - context, streams,
  modules, kernel launches. The shim lives in
  `src/firmware/cuda_driver_shim.cu`.
* cuFile (libcufile.so) - GPUDirect Storage. Loaded lazily via dlopen
  so NexusRT can build without cuFile headers. Falls back to host-pinned
  bounce buffers + `cuMemcpyHtoDAsync` when unavailable.
* NCCL (libnccl.so) - multi-GPU collectives, used as the GRDMA
  fallback when raw IBV verbs are not available.
* cuTensorMapEncodeTiled (H100 only) - TMA descriptor encoding. See
  `src/firmware/tma_descriptor.cu`.
* CU_MEM_ALLOCATION_PROP_COMPRESSED (H100 only) - ILC. See
  `src/firmware/ilc_allocations.cu`.

### 6.2 Apple path

* Metal (`MTLDevice`, `MTLCommandQueue`, `MTLComputeCommandEncoder`,
  `MTLBuffer`) - replaces CUDA streams and kernels. The shim lives in
  `src/firmware/metal_shim.mm`.
* Unified memory - `MTLBuffer` with `MTLResourceStorageModeShared`
  returns a pointer that is simultaneously CPU- and GPU-readable. The
  fault handler is a no-op on this path; the prefetcher is a no-op.
* MLX interop - MLX tensors are backed by `MTLBuffer` and can be
  passed directly to NexusRT via the `MTLBuffer` handle.

### 6.3 Conditional compilation

The following macros gate Hopper-specific code:

| Macro                       | When defined                  | Effect                                |
| --------------------------- | ----------------------------- | ------------------------------------- |
| `NEXUSRT_HAVE_CUDA`         | CMake detects CUDA 12.x       | Enables `cuda_driver_shim.cu` etc.    |
| `NEXUSRT_HAVE_METAL`        | Building on macOS with Metal  | Enables `metal_shim.mm`               |
| `NEXUSRT_HAVE_CUDA_HOPPER`  | CUDA + sm_90 target           | Enables TMA / ILC / DSM code paths    |
| `__CUDA_ARCH__ >= 900`      | Inside .cu compiled for sm_90 | Enables device-side TMA intrinsics    |

At runtime, the firmware-equivalent layer additionally probes
`DeviceDesc::features` to disable Hopper-only features on A100. This makes
the binary forward-compatible: a single build runs on both A100 and H100.

---

## 7. Memory management

### 7.1 GPU-driven virtual memory (DREAM-inspired)

Standard CUDA Unified Memory routes faults through the host OS kernel.
NexusRT instead maintains a page table in HBM (host mirror in
`memory::PageTableManager`) and routes faults through an HBM-resident fault
buffer (`firmware::FaultHandler`).

```
   Compute warp touches unmapped address
            │
            ▼
   HBM fault buffer (ring of FaultRecord)
            │
            ▼ (polled by firmware-equivalent host thread)
   Resolve via:
     • GRDMA fetch from remote GPU
     • GDS fetch from NVMe
     • cuMemcpyHtoD from host pinned
            │
            ▼
   Update page table entry -> mark HBM-resident
            │
            ▼
   Wake compute warp (write doorbell via cuStreamWriteValue32)
```

The fault buffer is allocated in HBM via the platform allocator (not the
user-facing memory pool) so the buffer itself cannot trigger faults. This
breaks the recursion that would otherwise deadlock the poller.

### 7.2 Eviction policy

The default policy is LRU-with-refcount (`memory::EvictionPolicy::LruRefcount`):
pages with `refcount > 0` are never evicted. The page table walks all
HBM-resident pages, sorts by `last_access_ns`, and evicts in order until
HBM occupancy drops below the low watermark (default 65%).

Evicted pages are spilled to NVMe via GDS or to host pinned memory. The
`fetch_count` field on each `PageEntry` tracks how many times a page has
been paged in - useful for detecting thrashing.

### 7.3 ILC (inline compression) - H100 only

ILC compresses individual memory allocations during HBM transactions. It
does *not* reduce the application's memory footprint, but it does increase
effective memory bandwidth: TMA and SMs operate on compressed data,
transferring fewer bits over the memory bus.

NexusRT exposes ILC transparently through `AllocHints::ilc=true`. The
allocation goes through `cuMemCreate` with
`CU_MEM_ALLOCATION_PROP_COMPRESSED` set; the resulting allocation is
tagged in the `IlcManager` for untagging on free.

### 7.4 Coalescer

The coalescer (`memory::Coalescer`) merges fragmented allocations and
aligns them to HBM warp boundaries (128B on both Ampere and Hopper with
128B swizzle). This ensures the TMA / async-copy engine can issue bulk
transactions without straddling pages, which would otherwise cause TLB
misses.

---

## 8. Scheduler

### 8.1 Task graph

The scheduler (`scheduler::TaskGraph`) is a DAG executor with explicit
dependency tracking. Each node is a `StageContract` declaring inputs,
outputs, memory footprint, SM footprint, token budget, grid/block
configuration, and a list of upstream stages it depends on.

The scheduler:

1. Validates the DAG (Kahn's algorithm for cycle detection).
2. Enforces stage contracts (token budget, SM footprint, memory footprint)
   before launching.
3. Picks a stream class (compute vs DMA) based on the stage's module name
   pattern.
4. Launches ready nodes in topological order.
5. Synchronizes via stream-level events (in the production build) or
   stream sync (in the smoke build).

### 8.2 Warp specialization

Each CTA is partitioned into producer / consumer / fence warps:

| Arch   | Producer | Consumer | Fence | Async-copy strategy |
| ------ | -------- | -------- | ----- | ------------------- |
| Hopper | 2        | 14       | 0     | TMA bulk (`cp.async.bulk.tensor`) |
| Ampere | 2        | 14       | 0     | `cp.async`           |
| Apple  | 0        | 16       | 0     | Blit encoder         |

Producer warps issue async copies from global to shared memory; consumer
warps compute on the data once it arrives. On Hopper, fence warps would
arrive at mbarriers to signal data readiness - currently disabled in
favor of `__syncthreads()` for portability.

### 8.3 Stream pool

The stream pool is partitioned by `StreamClass`:
- `Compute` - SM-bound work, default priority.
- `Dma` - GDS / GRDMA / TMA copies, higher (negative) priority so the GPU
  scheduler prefers them for copy work.
- `Fence` - doorbell writers (mbarrier arrievement on H100,
  `cuStreamWriteValue32` on Ampere).

`ScopedStream` (`scheduler/stream_pool.hpp`) provides RAII acquire / release.

---

## 9. Token optimization (ICM)

### 9.1 Layered context model

NexusRT implements an ICM-inspired layered context delivery model. Context
is divided into 5 layers, each with its own residency policy:

| Layer | Purpose              | Max tokens | Residency     |
| ----- | -------------------- | ---------- | ------------- |
| L0    | system prompt        | 256        | pinned HBM    |
| L1    | persona / role      | 512        | pinned HBM    |
| L2    | task instructions   | 1024       | HBM           |
| L3    | reference (RAG)     | 8192       | HBM-paged     |
| L4    | working artifacts   | 4096       | HBM-working   |

Each stage declares which layers it may access
(`config/pipeline.yaml:stages.<name>.context_routing.icm_layers`). The
scheduler enforces the routing rules: a stage that declares only `L4_working`
cannot touch `L3_reference` pages.

### 9.2 Attention-weighted prefetcher

The prefetcher (`token_opt::AttentionPrefetcher`) maintains a sliding
window of recent attention rows. At each decode step, it averages the
window, picks the top-k slots by score, and issues prefetch hints to the
fault handler. This ensures the next step's KV slots are paged into HBM
before the compute kernel touches them.

### 9.3 KV-cache pruning

The KV cache (`token_opt::KvCache`) is organized as PagedAttention-style
token pages (default 16 tokens/page). Pages are evicted in bands of 2
(default) to amortize TLB cost. The eviction policy is
attention-weighted LRU: pages with the lowest recent attention weight are
evicted first, but only if `refcount == 0`.

### 9.4 Token budgeting

Each stage enforces a per-stage context window (default 2k-8k tokens,
overridable in `config/pipeline.yaml`). When the working set exceeds the
budget, the overflow spills to L3 (HBM-paged). If L3 also overflows, it
spills to NVMe via GDS. The spill is logged via the
`nexusrt.token.spill` metric.

---

## 10. Configuration

Three YAML files control runtime behavior:

* `config/hardware.yaml` - device-specific defaults (HBM capacity, NVLink
  bandwidth, TMA / ILC availability, Metal cache limits). Auto-loaded at
  init.
* `config/runtime.yaml` - memory pool sizes, fault thresholds, eviction
  policies, token budgets per stage, stream pool sizes, ILC enable flag.
* `config/pipeline.yaml` - stage contracts, context routing rules,
  optional human-in-the-loop review gates (ICM-inspired).

See the files themselves for the full schema.

---

## 11. Failure modes & graceful degradation

| Failure                          | Detection                  | Fallback                                 |
| -------------------------------- | -------------------------- | ---------------------------------------- |
| No GPU detected                  | `probe()` returns false    | `boot()` returns `Status::DeviceNotFound`|
| cuFile not installed             | `gds_init()` fails         | Host-pinned bounce + `cuMemcpyHtoDAsync` |
| IBV / GRDMA unavailable          | `grdma_init()` fails       | NCCL collectives (`grdma.nccl_fallback`)|
| H100-only feature on A100       | `DeviceDesc::features`     | Conditional compile + runtime disable    |
| Metal unsupported                | `MTLCreateSystemDefaultDevice` returns nil | Falls through to NVIDIA, then NullPlatform |
| ILC unsupported                  | `IlcManager::enable()` returns `NotImplemented` | Allocations proceed without compression |
| Fault buffer overflow            | Poller detects `status==2` | Increments `failed_faults_` counter, logs |
| Stage contract violation         | `validate()` returns `ContractViolation` | Reject stage (configurable: log or abort) |

All fallbacks are documented in `docs/research.md` section "Failure Mode Analysis".

---

## 12. Thread safety & RAII

* Every `FirmwareContext` member is either immutable after boot or guarded
  by an internal mutex.
* Subsystems are stored in `std::shared_ptr` and torn down in
  reverse-of-declaration order, which mirrors the correct teardown
  sequence (faults -> mk -> dma -> ilc -> plat).
* `ScopedStream` provides RAII acquire / release of streams.
* `memory::Allocation` does not own its HBM pointer; the caller must call
  `memory::free()` explicitly. (In the Python binding, `Allocation.__del__`
  calls free if the user forgot.)
* The handle registry (`FirmwareContext::registry_`) is guarded by a
  single mutex; lookups are O(1).

### 12.1 Memory-leak audit

The runtime includes a leak audit (config: `diagnostics.leak_audit.enabled`)
that walks the live allocation map every 30s and logs any allocation that
has not been freed. In CI mode (`fail_on_leak: true`) the audit fails the
process exit code, which the test harness picks up.

---

## 13. Build system

See `packaging/CMakeLists.txt` for the top-level build. Highlights:

* C++17 required.
* CUDA 12.x optional (`NEXUSRT_ENABLE_CUDA`).
* Metal SDK optional, auto-detected on macOS (`NEXUSRT_ENABLE_METAL`).
* GoogleTest required for tests (`NEXUSRT_BUILD_TESTS`).
* Python 3.10+ required for bindings.
* Coverage threshold: >=85% for core modules (enforced in CI via
  `packaging/setup_hooks/post_test_coverage.sh`).

---

## 14. References

See `docs/research.md` for the full literature synthesis, citations, and
benchmark tables.
