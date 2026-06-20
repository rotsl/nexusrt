# NexusRT - firmware-centric OS-bypass runtime for end-to-end LLM pipelines

[![Manual Artifact Build](https://github.com/rotsl/nexusrt/actions/workflows/manual-artifacts.yml/badge.svg)](https://github.com/rotsl/nexusrt/actions/workflows/manual-artifacts.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/rotsl/nexusrt/blob/main/packaging/pyproject.toml)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://github.com/rotsl/nexusrt)
[![Metal / MLX](https://img.shields.io/badge/Apple-Metal%20%2F%20MLX-lightgrey?logo=apple)](https://github.com/rotsl/nexusrt)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange)](https://github.com/rotsl/nexusrt)
<img src="assets/nexusrt-logo.JPG" align="right" width="160" alt="NexusRT logo">

# NexusRT

Your README intro text here...
NexusRT is a firmware-centric, OS-bypass runtime architecture for end-to-end LLM
workloads (preprocessing -> training -> inference -> postprocessing) that operates
below standard AI frameworks (PyTorch / TensorFlow / JAX). It targets
NVIDIA CUDA GPUs through the lowest publicly available CUDA Driver APIs, with
A100/H100 as the primary feature targets and T4/P100-class Kaggle GPUs used for
CUDA smoke validation. It also exposes a parallel unified-memory execution path
for Apple M1 Pro via Metal / MLX.

The runtime replaces conventional framework-managed scheduling, memory paging,
DMA transfers, and pipeline orchestration with hardware-aware abstractions that
live almost entirely in user-space + GPU-resident firmware-equivalent code.

In practical terms, NexusRT is for experiments where the interesting question
is not "which model API should I call?" but "how much latency can be removed
when the LLM pipeline owns scheduling, memory movement, token-cache residency,
and GPU work submission directly?" The Python package is a thin control plane
over a C ABI and C++ core; the C++ core owns the firmware-equivalent runtime.

---

## Why NexusRT?

Standard AI runtimes stack multiple layers of indirection on top of the GPU:

```
┌──────────────────────────────────────────────────────────┐
│  Application (PyTorch / TF / JAX)                        │
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

NexusRT collapses this stack:

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

The host OS is demoted from "memory & I/O authority" to "interrupt router".
Page faults are handled by GPU-resident firmware threads reading from a fault
buffer in HBM, exactly as described in the DREAM project.

See `BENCHMARKS.md` for the currently validated Kaggle GPU results and the
boundary between measured results, smoke comparisons, and projected A100/H100
performance.

---

## Hardware targets

| Target                 | Memory                 | TMA | ILC | NVLink        | SMEM/SM | Notes                                  |
| ---------------------- | ---------------------- | --- | --- | ------------- | ------- | -------------------------------------- |
| NVIDIA A100 SXM        | 40 / 80 GB HBM2e       | No  | No  | 600 GB/s Gen3 | 164 KB  | Async-copy + warp-specialization queue |
| NVIDIA H100 SXM        | 80 GB HBM3             | Yes | Yes | 900 GB/s Gen4 | 228 KB  | TMA + DSM via Thread Block Clusters    |
| NVIDIA T4 / P100 class | 14-16 GB GDDR / HBM2   | No  | No  | n/a           | device  | Kaggle CUDA build/runtime smoke path   |
| Apple M1 Pro           | 16-32 GB unified LPDDR | n/a | n/a | n/a           | n/a     | Metal + MLX unified-memory path        |

The CUDA paths require an NVIDIA GPU, a working NVIDIA driver, and CUDA Toolkit
12.x. A100/H100 remain the intended research targets for the advanced runtime
features; T4/P100 are useful for Kaggle availability checks and basic CUDA
runtime validation. The Metal path builds on macOS, but runtime boot still
depends on a supported Apple Silicon Metal device being visible to the process.
On unsupported hosts the build and non-device tests can pass while runtime
smoke commands return `DEVICE_NOT_FOUND`.

---

## Repository layout

```
nexusrt/
├── src/
│   ├── firmware/      GPU micro-kernel & bare-metal-equivalent abstractions
│   ├── memory/        GPU-driven virtual memory, page-fault routing, GDS/GRDMA
│   ├── scheduler/     Async task graph, warp specialization, TMA/GDS integration
│   ├── pipeline/      LLM stage orchestration (preprocess/train/infer/postprocess)
│   ├── token_opt/     ICM layered context, attention prefetcher, KV-cache pruning
│   └── platform/      Hardware dispatch (NVIDIA CUDA vs Apple Metal)
├── tests/
│   ├── kaggle/        CUDA GPU validation & benchmarking notebooks
│   ├── mac_unified/   M1 Pro unified-memory coherence & fallback tests
│   └── unit/          C++ (GoogleTest) + Python (pytest) unit tests
├── docs/
│   ├── architecture.md
│   ├── research.md
│   ├── api_reference.md
│   └── hardware_profiles/{a100,h100,m1pro}.yaml
├── config/
│   ├── hardware.yaml
│   ├── runtime.yaml
│   └── pipeline.yaml
├── packaging/
│   ├── pyproject.toml
│   ├── CMakeLists.txt
│   └── setup_hooks/
├── .gitignore
├── BENCHMARKS.md
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Create a Python environment

Use `.venv` by default; it is ignored by this repo. `nexusenv/` is also ignored
for local experiments.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install cmake ninja pytest pytest-cov pyyaml
```

### 2. Check your hardware path

CUDA builds require both commands to work:

```bash
nvcc --version
nvidia-smi
```

On a Mac without NVIDIA CUDA, configure with CUDA disabled:

```bash
cmake -S packaging -B packaging/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUSRT_ENABLE_CUDA=OFF \
  -DNEXUSRT_ENABLE_METAL=AUTO \
  -DNEXUSRT_BUILD_TESTS=ON
```

On an NVIDIA CUDA machine with CUDA Toolkit 12.x:

```bash
cmake -S packaging -B packaging/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUSRT_ENABLE_CUDA=ON \
  -DNEXUSRT_ENABLE_METAL=OFF \
  -DNEXUSRT_BUILD_TESTS=ON
```

Then build and test:

```bash
cmake --build packaging/build -j
ctest --test-dir packaging/build --output-on-failure
PYTHONPATH=python python -m pytest tests/unit/test_python_bindings.py
```

### 3. Install the Python package

```bash
python -m pip install -e "packaging[dev]"
```

Editable installs and source checkouts auto-discover the local shared library
from `packaging/build/outputs/lib/`. If you use a custom build location, set:

```bash
export NEXUSRT_LIB=/absolute/path/to/libnexusrt.so     # Linux
export NEXUSRT_LIB=/absolute/path/to/libnexusrt.dylib  # macOS
```

### 4. Run smoke commands

```bash
nexusrt-bench --stage detect
nexusrt-run  --pipeline config/pipeline.yaml
```

If these fail with `NexusRT error -4: DEVICE_NOT_FOUND`, the package and C ABI
loaded correctly but no supported runtime device was found. That is expected on
CPU-only hosts, Intel Macs without NVIDIA CUDA, and machines where Metal/CUDA is
not exposed to the process.

### Kaggle GPU notebooks

The notebooks in `tests/kaggle/` are intended for Kaggle GPU sessions. Enable a
GPU accelerator, attach the private Kaggle Dataset `rbrtsl/nexusrt` or upload
the source zip, then run the setup cell in each notebook. Kaggle datasets mount
read-only under `/kaggle/input`, so the setup cell copies the repo to
`/kaggle/working/nexusrt` before building.

The current notebooks use direct source imports plus a local CUDA CMake build,
then set `NEXUSRT_LIB` to the produced shared library. This avoids editable
install/build-isolation failures in private dataset runs. The setup cell prefers
the attached `/kaggle/input` dataset and refreshes the writable copy so an older
failed run in `/kaggle/working` cannot shadow the latest dataset version.
Kaggle Internet is only needed for optional PyTorch/Transformers baseline cells,
not for the NexusRT source-import + CMake path. The notebooks require CUDA; if
CUDA configure, build, or device initialization fails, the notebook fails rather
than counting a CPU-only run as validation. It writes the full setup log to
`/kaggle/working/nexusrt_setup.log`.

The four canonical Kaggle notebooks were last pulled back from successful
private Kaggle runs on 2026-06-19. All completed on Kaggle's `NvidiaTeslaT4`
shape and their raw logs are stored under `tests/kaggle/results/`.

### Downloadable GitHub build artifacts

The repository includes a manual GitHub Actions workflow for producing
downloadable user artifacts without publishing a package release. In GitHub,
open Actions -> Manual Artifact Build -> Run workflow.

The workflow uploads:

- source archives (`.tar.gz` and `.zip`)
- a Python wheel
- native Linux CMake install archive
- optional native macOS CMake install archive
- optional Linux CUDA 12 compile artifact
- optional HTML documentation

The macOS artifact is disabled by default because GitHub-hosted macOS runners
can sit in the queue for a long time on manual runs. Enable `build_macos` when
you specifically want a macOS archive. The CUDA artifact is also disabled by
default because it installs the CUDA Toolkit during the workflow run. Enable
`build_cuda` when you specifically want a CUDA-enabled Linux build archive.

### Minimal Python usage

```python
import nexusrt as nrt

# Initialize firmware-equivalent layer
dev = nrt.firmware.init(profile="auto")     # auto-detects supported CUDA/Metal devices

# Allocate HBM-resident tensor (GPU-driven virtual memory)
buf = nrt.memory.alloc(shape=(4096, 4096), dtype="bf16", ilc=True)

# Build a pipeline stage contract
stage = nrt.scheduler.stage(
    name="infer.transformer_block_0",
    inputs=[buf],
    outputs=[],
    token_budget=4096,
    sm_footprint_mb=64,
)

nrt.scheduler.submit(stage)
nrt.scheduler.wait_barrier()
```

---

## Firmware-equivalent boundary (important)

True vendor firmware modification is restricted on NVIDIA devices. NexusRT
therefore implements a firmware-equivalent user-space micro-kernel that
uses only the lowest publicly available CUDA Driver APIs:

* `cuDevicePrimaryCtxCreate` / `cuCtxSetCurrent` - primary context ownership
* `cuModuleLoad` / `cuModuleLoadData` - PTX / SASS kernel loading
* `cuStreamCreateWithPriority` - prioritized async streams (DMA vs compute)
* `cuMemAddressReserve` / `cuMemMap` / `cuMemSetAccess` - virtual address reservation & mapping
* `cuMemAdvise` + `CU_MEM_ATTRIBUTE_*` - prefetch / read-mostly hints
* `cuMemcpyAsync` with custom streams - GDS-style async paths
* `cuTensorMapEncodeTiled` (Hopper only) - TMA descriptor encoding
* `cuStreamWaitValue32` / `cuStreamWriteValue32` - doorbell-style sync

The boundary is documented explicitly in `docs/architecture.md` section "Firmware
Boundary".

---

## License

MIT - see `LICENSE`. Attributions to NVIDIA CUDA, Apple Metal, and the
underlying research (DREAM, KOKARYOKU, TrainMover) are documented in
`docs/research.md`.
