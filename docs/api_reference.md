# NexusRT - API reference

> Full C / Python API documentation with usage examples, memory safety
> guarantees, and pipeline contract schemas.

This document covers:

1. The unified C ABI (`src/platform/abi.h`) - every function the rest of
   the codebase and external bindings call.
2. The C++ class layer (`firmware::*`, `memory::*`, `scheduler::*`,
   `pipeline::*`, `token_opt::*`) - for in-tree contributors.
3. The Python bindings (`nexusrt.*`) - for application developers.
4. The pipeline contract schema (`config/pipeline.yaml`) - declarative
   stage definitions.

---

## 1. C ABI

All C ABI functions return `int32_t`. Non-negative values are success;
negative values are `nexusrt::firmware::Status` codes (see section 1.4).

### 1.1 Version

```c
int32_t nexusrt_version(int32_t* out_major,
                        int32_t* out_minor,
                        int32_t* out_patch);
```

Returns the NexusRT library version. Always succeeds.

### 1.2 Firmware

```c
int32_t nexusrt_firmware_init(const char* profile, void out_ctx);
int32_t nexusrt_firmware_shutdown(void* ctx);
int32_t nexusrt_firmware_device_desc(void* ctx,
                                     int32_t*  out_vendor,
                                     int32_t*  out_arch,
                                     char*     out_name, size_t name_cap,
                                     uint64_t* out_hbm_bytes,
                                     uint32_t* out_sm_count,
                                     uint32_t* out_features_bits);
int32_t nexusrt_firmware_task_submit(void* ctx,
                                     const char* module,
                                     const char* function,
                                     uint32_t gx, uint32_t gy, uint32_t gz,
                                     uint32_t bx, uint32_t by, uint32_t bz,
                                     uint32_t shared_mem_bytes,
                                     void kernel_args, uint32_t n_args,
                                     int32_t stream_class,
                                     int32_t stream_priority);
int32_t nexusrt_firmware_fault_handler(void* ctx,
                                       uint64_t faulting_addr,
                                       int32_t timeout_ms);
```

#### `nexusrt_firmware_init`

Boot the firmware-equivalent layer.

| Argument  | Type          | Description                                                   |
| --------- | ------------- | ------------------------------------------------------------- |
| `profile` | `const char*` | `"auto"`, `"a100"`, `"h100"`, or `"m1pro"`. May be NULL.      |
| `out_ctx` | `void`      | Receives an opaque handle used by every other ABI function.  |

Returns 0 on success, negative `Status` on failure. On success, the
caller must call `nexusrt_firmware_shutdown` to release resources.

Memory safety: The returned handle is owned by the caller. The
firmware layer maintains an internal `shared_ptr` registry so the handle
remains valid even if the caller does not hold a C++ reference.

#### `nexusrt_firmware_shutdown`

Tear down the firmware layer. Idempotent - calling twice returns
`InvalidArgument`.

#### `nexusrt_firmware_device_desc`

Copy out the discovered `DeviceDesc`. The `features_bits` field packs
the feature flags into a single `uint32_t`:

| Bit | Feature          |
| --- | ---------------- |
| 0   | TMA              |
| 1   | ILC              |
| 2   | DSM              |
| 3   | GDS              |
| 4   | GRDMA            |
| 5   | Thread Block Clusters |
| 6   | Unified memory native (Apple) |

#### `nexusrt_firmware_task_submit`

Submit a kernel launch through the micro-kernel. Acquires a stream of
the given class, launches the kernel, synchronizes, and releases the
stream. Returns 0 on success.

`stream_class` values:
- `0` - Compute
- `1` - DMA
- `2` - Fence
- `3` - Control

`stream_priority` is in `[-2, +2]`; negative is higher priority. The
firmware layer clamps to the device's supported range.

#### `nexusrt_firmware_fault_handler`

Manually inject a fetch hint for a faulting address. Used by the
`token_opt` prefetcher to explicitly page in attention-anticipated KV
slots.

### 1.3 Memory

```c
int32_t nexusrt_mem_alloc(void* ctx, uint64_t bytes, uint32_t flags,
                          void out_ptr);
int32_t nexusrt_mem_free(void* ctx, void* ptr);
int32_t nexusrt_mem_prefetch(void* ctx, void* ptr, uint64_t bytes);
int32_t nexusrt_mem_advise_read_mostly(void* ctx, void* ptr, uint64_t bytes);
```

`flags` is a bitfield:

| Bit | Flag           | Description                                     |
| --- | -------------- | ----------------------------------------------- |
| 0   | ILC            | Request Inline Compression (H100 only, ignored elsewhere) |
| 1   | READ_MOSTLY    | Hint that the range will be read but not written |
| 2   | PINNED_HOST    | Allocate in host-pinned (DMA-able) memory       |
| 3   | GDS_READABLE   | Map for GDS reads (zero-copy NVMe)              |
| 4   | GRDMA_VISIBLE  | Expose for remote RDMA write                    |

Memory safety: Every allocation returned by `nexusrt_mem_alloc`
*must* be freed by `nexusrt_mem_free`. The firmware layer's leak audit
(config: `diagnostics.leak_audit.enabled`) will report unfreed
allocations.

### 1.4 Status codes

| Code | Name                    | Description                              |
| ---- | ----------------------- | ---------------------------------------- |
| 0    | Ok                      | Success                                  |
| -1   | InvalidArgument         | Bad argument (null pointer, zero size, ...) |
| -2   | OutOfMemory             | Host memory allocation failed            |
| -3   | OutOfHbm                | HBM allocation failed                    |
| -4   | DeviceNotFound          | No GPU detected                          |
| -5   | DriverError             | CUDA / Metal driver returned an error    |
| -6   | NotImplemented          | Feature not implemented on this platform |
| -7   | FaultBufferOverflow     | HBM fault buffer overflowed              |
| -8   | ContractViolation       | Stage contract failed validation         |
| -9   | Timeout                 | Wait timed out                           |
| -10  | IoError                 | I/O (GDS / NVMe) error                   |
| -11  | Aborted                 | Operation aborted (e.g., stage failed)   |

### 1.5 Scheduler

```c
int32_t nexusrt_submit_stage(void* ctx, const char* stage_name,
                             const char* module, const char* function,
                             void inputs,  uint32_t n_inputs,
                             void outputs, uint32_t n_outputs,
                             uint32_t token_budget,
                             uint32_t sm_footprint_mb,
                             uint32_t mem_footprint_mb,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             uint32_t bx, uint32_t by, uint32_t bz,
                             uint32_t shared_mem_bytes,
                             void kernel_args, uint32_t n_args);
int32_t nexusrt_wait_barrier(void* ctx, const char* stage_name,
                             uint32_t timeout_ms);
int32_t nexusrt_stream_overlap(void* ctx, const char* a, const char* b,
                               int32_t enable);
```

`nexusrt_submit_stage` submits a single stage synchronously (it returns
after the stage has completed). For asynchronous submission, use the C++
`scheduler::TaskGraph` API directly.

### 1.6 Token optimization

```c
int32_t nexusrt_context_scope(void* ctx, const char* stage,
                              uint32_t layer_mask,
                              uint64_t* out_token_budget);
int32_t nexusrt_prefetch_attention(void* ctx, void* kv_cache,
                                   uint64_t n_slots,
                                   const uint32_t* topk_indices);
int32_t nexusrt_token_prune(void* ctx, void* kv_cache, uint32_t max_resident);
```

`layer_mask` is a bitfield over `IcmLayer`:

| Bit | Layer             |
| --- | ----------------- |
| 0   | L0_system         |
| 1   | L1_persona        |
| 2   | L2_instructions   |
| 3   | L3_reference      |
| 4   | L4_working        |

### 1.7 Diagnostics

```c
int32_t nexusrt_metrics_dump(void* ctx, char* out_json, size_t cap);
```

Writes a JSON snapshot of all metrics (stage states, allocation counts,
fault counts, average resolve latencies) into `out_json`. Returns 0 on
success, `-1` (`InvalidArgument`) if the buffer is too small.

---

## 2. C++ class layer

For contributors building inside the NexusRT tree, the C++ class layer
provides stronger typing and RAII than the C ABI.

### 2.1 `nexusrt::firmware::FirmwareContext`

The central owner of every firmware-equivalent resource. Created by
`boot()`, destroyed by `shutdown()` or by dropping the last
`shared_ptr`.

```cpp
namespace nexusrt::firmware {

class FirmwareContext : public std::enable_shared_from_this<FirmwareContext> {
public:
    explicit FirmwareContext(DeviceDesc d);
    ~FirmwareContext();

    MicroKernel&    microkernel();
    DmaEngine&      dma();
    FaultHandler&   faults();
    TmaEngine*      tma();          // nullptr on A100
    IlcManager*     ilc();          // nullptr on A100
    memory::MemoryManager& memory();  // lazy-init

    const DeviceDesc device;
    BootOptions      options;
    uint64_t         hbm_pool_bytes = 0;
    PlatformContextHandle plat;

    // Handle registry for the C ABI.
    static void register_handle(void* k, std::shared_ptr<FirmwareContext> sp);
    static std::shared_ptr<FirmwareContext> lookup_handle(void* k);
    static std::shared_ptr<FirmwareContext> take_handle(void* k);
};

} // namespace nexusrt::firmware
```

### 2.2 `nexusrt::firmware::boot`

```cpp
BootResult boot(const BootOptions& opts);
Status     shutdown(FirmwareContext& ctx);
DeviceDesc probe_device(const std::string& profile_hint);
```

`BootOptions::on_event` is called for every boot phase transition. Use
it to wire up logging or progress bars.

### 2.3 `nexusrt::memory::MemoryManager`

```cpp
class MemoryManager {
public:
    explicit MemoryManager(FirmwareContext& ctx);

    Status alloc(uint64_t bytes, AllocHints const& hints, Allocation& out);
    Status free(Allocation const& a);
    Status prefetch(void* hbm_ptr, uint64_t bytes);
    Status advise_read_mostly(void* hbm_ptr, uint64_t bytes);
    Status coalesce(uint32_t* out_n_merged);

    PageTableManager& page_table();
    uint64_t bytes_allocated() const;
    uint64_t bytes_resident_hbm() const;
    uint64_t bytes_spilled() const;
    uint32_t fragmentation_pct() const;
};
```

### 2.4 `nexusrt::scheduler::TaskGraph`

```cpp
class TaskGraph {
public:
    explicit TaskGraph(FirmwareContext& ctx);

    Status add_stage(StageContract const& c);
    Status validate();
    Status run();
    Status wait(const std::string& stage_name, uint32_t timeout_ms);
    Status wait_all(uint32_t timeout_ms);
    Status set_overlap(const std::string& a, const std::string& b, bool enable);

    StageNode const* find(const std::string& name) const;
    std::vector<std::string> stage_names() const;
};
```

### 2.5 `nexusrt::token_opt::ContextScope`

```cpp
class ContextScope {
public:
    explicit ContextScope(FirmwareContext& ctx);

    void set_layer(IcmLayer l, LayerSpec const& s);
    Status step(uint32_t generated_so_far);
    void bind_kv_cache(void* kv_hbm, uint64_t bytes, uint32_t max_resident);
    Status prefetch_attention(const uint32_t* topk_indices, uint32_t k);
    Status prune(uint32_t max_resident);

    uint64_t total_token_budget() const;
    uint64_t total_bytes_used() const;
    LayerSpec const& layer(IcmLayer l) const;
};
```

---

## 3. Python bindings

### 3.1 `nexusrt.firmware`

```python
import nexusrt

info = nexusrt.firmware.init("auto")
# DeviceInfo(vendor="nvidia", arch="hopper", name="NVIDIA H100 SXM5",
#            hbm_bytes=8567668736, sm_count=132,
#            features={"tma": True, "ilc": True, "dsm": True, "gds": True,
#                      "grdma": True, "clusters": True, "unified_native": False})

# ... use the runtime ...

nexusrt.firmware.shutdown()
```

### 3.2 `nexusrt.memory`

```python
import nexusrt.memory as mem

# Allocate 64 MB with ILC compression (H100 only; no-op on A100).
a = mem.alloc(64 << 20, ilc=True, read_mostly=True)
print(f"ptr=0x{a.ptr:x}, bytes={a.bytes}")

# Prefetch into HBM (no-op if already resident).
mem.prefetch(a.ptr, a.bytes)

# Free.
mem.free(a)
```

### 3.3 `nexusrt.scheduler`

```python
import nexusrt.scheduler as sch

s = sch.stage("infer.transformer_block_0",
              module="nexusrt.kernels.infer",
              function="decode_step",
              inputs=[kv_cache_ptr, weights_ptr],
              outputs=[logits_ptr],
              token_budget=4096,
              sm_footprint_mb=64,
              mem_footprint_mb=16384,
              grid=(32,1,1), block=(256,1,1),
              shared_mem_bytes=48*1024,
              depends_on=["infer.embed"],
              kernel_args=[kv_cache_ptr, weights_ptr, &last_token, logits_ptr])

sch.submit_stage(s)
sch.wait_barrier("infer.transformer_block_0", timeout_ms=10000)
```

### 3.4 `nexusrt.token_opt`

```python
import nexusrt.token_opt as tk

# Total budget across L0+L1+L2+L3+L4.
budget = tk.context_scope("infer",
    [tk.IcmLayer.L0_SYSTEM, tk.IcmLayer.L1_PERSONA,
     tk.IcmLayer.L2_INSTRUCTIONS, tk.IcmLayer.L3_REFERENCE,
     tk.IcmLayer.L4_WORKING])
print(f"total budget: {budget} tokens")

# Prefetch the top-32 KV slots by attention score.
tk.prefetch_attention(kv_cache_ptr, n_slots=8192,
                      topk_indices=[42, 100, 7, 999, ...])

# Prune KV cache to 4096 resident tokens.
tk.token_prune(kv_cache_ptr, max_resident=4096)
```

### 3.5 `nexusrt.pipeline`

```python
import nexusrt.pipeline as pl

# Preprocess: stream corpus via GDS.
n = pl.run_preprocess("/data/corpus.txt", bytes_to_read=1 << 20)

# Inference.
out = pl.run_infer(prompt_tokens=[1, 2, 3, 4, 5], max_new_tokens=32)

# Postprocess.
text = pl.run_postprocess(out)
print(text)
```

### 3.6 CLI

```bash
# Run an end-to-end pipeline.
nexusrt-run --profile auto --prompt "Hello, NexusRT." --max-tokens 32

# Benchmark.
nexusrt-bench --stage all --json
```

---

## 4. Pipeline contract schema

The contract catalog lives in `config/pipeline.yaml`. Each entry is a
stage contract:

```yaml
stages:
  <stage_name>:
    description: "<human-readable>"

    inputs:
      - { name: <id>, source: gds_nvme | hbm | host, dtype: uint8 | int32 | bf16 | fp32, layout: raw | packed | row_major }

    outputs:
      - { name: <id>, dtype: ..., layout: ..., destination: hbm | host }

    memory_footprint_mb: <int>
    sm_footprint_mb: <int>
    token_budget: <int>

    context_routing:
      icm_layer: L0_system | L1_persona | L2_instructions | L3_reference | L4_working
      icm_layers: [<layer>, ...]      # alternative: multi-layer access
      prefetch_policy: attention_weighted | none
      kv_cache_prune: true | false

    kernel:
      bundle: nexusrt.kernels.<name>
      entrypoint: <function_name>

    collective:                       # optional - multi-GPU collective
      type: nccl_allreduce | nccl_allgather | grdma_send | grdma_recv
      grdma_preferred: true | false

    autoregressive:                   # optional - inference only
      max_new_tokens: <int>
      prefill_chunk_tokens: <int>
      speculative_decoding: true | false

    contract_enforced:
      - memory_footprint
      - sm_footprint
      - token_budget
      - kv_cache_residency
      - host_copy_only_on_final
      - collective

topology:
  edges:
    - { from: <stage>, to: <stage>, transfer: hbm_resident | weights_pinned }

review_gates:                        # optional - ICM-inspired HITL
  - name: <gate_name>
    stage: <stage_name>
    trigger: on_first_step | on_kv_cache_overflow | on_final
    description: "<human-readable>"
    timeout_s: <int>                 # 0 = no timeout
    auto_approve_if: "<python expression>"
```

### 4.1 Routing rules

```yaml
routing:
  default_policy: deny               # stages may only access declared layers
  allow_cross_stage_share: true      # L3_reference can be shared train↔infer
  spill_target: gds_nvme             # where to spill over-budget context
  spill_metric: nexusrt.token.spill
```

---

## 5. Memory safety guarantees

1. RAII everywhere. Every GPU resource (context, stream, module,
   allocation, TMA descriptor, ILC tag) is owned by a C++ object whose
   destructor releases it. The `FirmwareContext` destructor releases
   every subsystem even if `shutdown()` was not called.
2. Handle registry. The C ABI uses an internal `shared_ptr` registry
   so handles remain valid even if the caller does not hold a C++
   reference. `nexusrt_firmware_shutdown` removes the handle from the
   registry and releases the `shared_ptr`.
3. Leak audit. When `diagnostics.leak_audit.enabled` is true
   (default), the runtime walks the live allocation map every 30s and
   logs unfreed allocations. In CI mode (`fail_on_leak: true`) the audit
   fails the process exit code.
4. Thread safety. Every `FirmwareContext` member is either immutable
   after boot or guarded by an internal mutex. The handle registry is
   guarded by a single mutex; lookups are O(1).
5. Stream pool exhaustion. `acquire_stream` returns a zero-id handle
   if the pool is empty. Callers must check `h.id != 0` before use. The
   scheduler retries with backoff.
6. Fault buffer overflow. When the HBM fault buffer fills, the
   poller increments `failed_faults_` and logs. It does not block the
   compute kernel; the next fault will overwrite the oldest entry.
7. Stage contract violation. `validate()` returns
   `ContractViolation` if any stage exceeds its declared memory / SM /
   token budget. `runtime.yaml:scheduler.stage_contracts.reject_on_violation`
   controls whether the scheduler rejects the stage (default: false =
   log and demote priority).
