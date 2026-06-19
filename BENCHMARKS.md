# NexusRT benchmarks and validation results

This document summarizes the performance evidence currently stored in the
repository. It separates measured test results from architectural expectations,
because the current notebooks are validation workloads rather than a full
model-equivalent benchmark suite.

## Summary

| Area | Platform | Test source | Result |
| --- | --- | --- | --- |
| CUDA runtime boot and device detection | Kaggle `NvidiaTeslaT4` | `tests/kaggle/01_gpu_detection_bench.ipynb` | Passed; NexusRT detected `nvidia` / `turing`, `Tesla T4`, 14 GB HBM, 40 SMs. |
| Oversubscribed memory paging | Kaggle `NvidiaTeslaT4` | `tests/kaggle/02_memory_paging_test.ipynb` | Passed; 925 live 16 MB pages, `7.8 us / 16MB page` prefetch timing. |
| Token context reduction | Kaggle `NvidiaTeslaT4` | `tests/kaggle/03_token_efficiency.ipynb` | Passed; ICM budget `14080 tokens` vs monolithic `32768 tokens`, an `8.0x` L4 working-set reduction. |
| End-to-end pipeline smoke comparison | Kaggle `NvidiaTeslaT4` | `tests/kaggle/04_pipeline_end2end.ipynb` | Passed; NexusRT stub path reported 32 tokens in `68.1 us`; optional PyTorch/GPT-2 baseline reported 32 tokens in `1521067.9 us`. |
| macOS / Metal path | Local macOS build and tests | CMake + CTest + Python unit tests | Build and unit tests validated locally; no captured Mac performance run is currently checked into the repo. |

Raw Kaggle logs are stored in `tests/kaggle/results/*.logs.json`. The compact
Kaggle run summary is in `tests/kaggle/results/README.md`.

## What is validated

The Kaggle run validated that NexusRT can build and execute its CUDA backend on
a fresh hosted GPU runtime without relying on a framework stack. The setup cells
used direct source imports plus a CMake CUDA build, then loaded the local
`libnexusrt` through the Python ABI layer.

Common Kaggle environment:

- Machine shape: `NvidiaTeslaT4`
- Visible GPUs: two `Tesla T4` devices
- Driver: `580.159.04`
- CUDA reported by `nvidia-smi`: `13.0`
- CMake CUDA toolkit/compiler: `12.8.93`
- CMake architectures: `60;75;80;90`

All four Kaggle notebooks completed successfully on 2026-06-19.

## GPU results

### 1. Device detection and runtime boot

Source: `tests/kaggle/results/01_gpu_detection_bench.logs.json`

NexusRT detected the Kaggle GPU as:

| Field | Value |
| --- | --- |
| Vendor | `nvidia` |
| Architecture | `turing` |
| Device | `Tesla T4` |
| HBM / device memory | `14 GB` |
| SM count | `40` |
| Feature flags | `gds=True`, `grdma=True`, `tma=False`, `ilc=False`, `dsm=False`, `unified_native=False` |

The notebook also records prefetch timing across allocation sizes. These numbers
are useful as a regression signal for the NexusRT memory path, but they should
not be read as physical T4 HBM bandwidth; the current benchmark times prefetch
submission/handling and can exceed physical memory bandwidth.

### 2. Oversubscribed memory paging

Source: `tests/kaggle/results/02_memory_paging_test.logs.json`

The memory paging notebook intentionally requested a workload larger than the
available T4 memory:

| Metric | Value |
| --- | --- |
| Detected HBM capacity | `14 GB` |
| Target workload | `28 GB` (`2.0x HBM`) |
| Page size | `16 MB` |
| Live pages before OOM boundary | `925` |
| NexusRT prefetch timing | `7.8 us / 16MB page` |

The optional PyTorch baseline was skipped because allocating the full 28 GB
baseline buffer exceeded the 14 GB T4 capacity. That result still validates the
NexusRT path for bounded oversubscription behavior: it reached the expected
out-of-memory boundary, reported it cleanly, measured page prefetch timing, and
completed the test.

### 3. Token context efficiency

Source: `tests/kaggle/results/03_token_efficiency.logs.json`

The token efficiency notebook compares a monolithic context with NexusRT's
ICM-staged working-set model.

| Metric | Monolithic | NexusRT ICM-staged |
| --- | ---: | ---: |
| Logical token budget | `32768` | `14080` |
| Active L4 working tokens | `32768` | `4096` |
| L4 working-set reduction | baseline | `8.0x` fewer active L4 tokens |
| Measured step time in current stub test | `16.0 us/step` | `124.7 us/step` |

The validated win here is memory residency and token working-set reduction, not
raw step latency on T4. The current T4 stub path reports `0.13x` speed for the
ICM simulation, so it should not be described as faster yet. On H100/A100-class
targets, the project expectation is that smaller active working sets combine
with TMA/ILC/DSM and improved scheduling to reduce end-to-end latency, but that
feature-target benchmark is still pending.

### 4. End-to-end pipeline smoke comparison

Source: `tests/kaggle/results/04_pipeline_end2end.logs.json`

The end-to-end notebook exercises the NexusRT pipeline stubs and then runs an
optional PyTorch/Transformers GPT-2 baseline.

| Path | Output | Timing |
| --- | --- | ---: |
| NexusRT stub pipeline | 32 generated stub tokens | `68.1 us` (`469773 tok/s`) |
| PyTorch/Transformers GPT-2 baseline | 32 generated GPT-2 tokens | `1521067.9 us` (`21 tok/s`) |
| Ratio reported by notebook | PyTorch time / NexusRT stub time | `22329.90x` |

This is a useful framework-overhead smoke comparison, but it is not a
model-equivalent quality benchmark. NexusRT is running repository pipeline
stubs; PyTorch is running GPT-2 generation through `transformers`. The result
validates that the NexusRT pipeline path has very low control-plane overhead,
not that a full NexusRT LLM implementation is 22k times faster than PyTorch.

## Mac / Metal validation

The repository includes Apple unified-memory notebooks under
`tests/mac_unified/`, but they currently have no captured output cells checked
in. The Mac path has been validated locally at the build/unit-test level:

```bash
cmake -S packaging -B packaging/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUSRT_ENABLE_CUDA=OFF \
  -DNEXUSRT_ENABLE_METAL=AUTO \
  -DNEXUSRT_BUILD_TESTS=ON
cmake --build packaging/build -j
ctest --test-dir packaging/build --output-on-failure
PYTHONPATH=python python -m pytest tests/unit/test_python_bindings.py
```

Use the Mac notebooks to collect performance numbers on an Apple Silicon host:

- `tests/mac_unified/01_unified_memory_coherence.ipynb`
- `tests/mac_unified/02_token_pipeline_mps.ipynb`

Once those notebooks are executed and their outputs are committed, this file
should be updated with measured Metal coherence, token pipeline, and fallback
timings.

## Comparison against standard AI runtimes

NexusRT is designed to outperform standard AI runtimes by reducing framework
dispatch, allocator, paging, and pipeline orchestration overhead. The current
checked-in tests support narrower claims:

| Claim | Status | Evidence |
| --- | --- | --- |
| NexusRT can build and boot a CUDA runtime path without PyTorch/TensorFlow/JAX. | Validated | All four Kaggle notebooks completed through direct source import + CMake CUDA build. |
| NexusRT can handle CUDA device detection and expose low-level feature flags. | Validated | Notebook 01 detected T4/Turing, memory, SMs, and feature flags. |
| NexusRT can run an oversubscribed paging smoke workload and measure page prefetch timing. | Validated | Notebook 02 measured `7.8 us / 16MB page`. |
| NexusRT reduces active token working-set size with ICM. | Validated | Notebook 03 reduced L4 working tokens from `32768` to `4096`. |
| NexusRT's stub pipeline has far lower control-plane overhead than a PyTorch/Transformers GPT-2 generation call. | Validated as a smoke comparison | Notebook 04 reported `68.1 us` vs `1521067.9 us`, but the workloads are not model-equivalent. |
| NexusRT outperforms standard runtimes on full A100/H100 LLM inference/training. | Not yet validated | Requires real model kernels and A100/H100 benchmark runs. |
| NexusRT outperforms standard runtimes on Mac/Metal. | Not yet validated | Mac notebooks need captured output and baseline comparisons. |

## Reproducing the GPU results

1. Upload or attach the private Kaggle Dataset `rbrtsl/nexusrt`.
2. Start a Kaggle notebook with a GPU accelerator.
3. Run the notebooks in `tests/kaggle/`.
4. Pull the completed notebooks and logs with:

```bash
kaggle kernels pull rbrtsl/notebook1nexusrtworking
kaggle kernels pull rbrtsl/notebook2nexusrt-importtime
kaggle kernels pull rbrtsl/notebook3nexusrt-importtime
kaggle kernels pull rbrtsl/notebook4nexusrt-importtime
```

5. Store raw logs under `tests/kaggle/results/` and update this document.

## Next benchmark work

- Run the same Kaggle notebooks on P100 and any available A100/H100 runtime.
- Add model-equivalent PyTorch vs NexusRT benchmarks once transformer kernels
  are no longer stubs.
- Execute and commit output-bearing Mac/Metal notebooks.
- Add throughput, latency p50/p95/p99, CPU utilization, memory residency, and
  energy metrics for each benchmark.
- Track results by commit SHA and hardware profile.
