# Kaggle run results

Pulled with `kaggle kernels pull` and `kaggle kernels logs` on 2026-06-19
after all four private Kaggle notebooks completed successfully.

Common environment:

- Kaggle machine shape: `NvidiaTeslaT4`
- Visible GPUs: two `Tesla T4` devices
- Driver: `580.159.04`
- Runtime CUDA reported by `nvidia-smi`: `13.0`
- CMake CUDA toolkit/compiler: `12.8.93`
- NexusRT CMake options: `NEXUSRT_ENABLE_CUDA=ON`,
  `NEXUSRT_ENABLE_METAL=OFF`, `NEXUSRT_BUILD_TESTS=OFF`,
  `CMAKE_CUDA_ARCHITECTURES=60;75;80;90`
- NexusRT detected device: vendor `nvidia`, arch `turing`, HBM `14 GB`,
  SM count `40`

| Notebook | Kaggle kernel | Status | Raw log |
| --- | --- | --- | --- |
| `01_gpu_detection_bench.ipynb` | `rbrtsl/notebook1nexusrtworking` | Complete | `01_gpu_detection_bench.logs.json` |
| `02_memory_paging_test.ipynb` | `rbrtsl/notebook2nexusrt-importtime` | Complete | `02_memory_paging_test.logs.json` |
| `03_token_efficiency.ipynb` | `rbrtsl/notebook3nexusrt-importtime` | Complete | `03_token_efficiency.logs.json` |
| `04_pipeline_end2end.ipynb` | `rbrtsl/notebook4nexusrt-importtime` | Complete | `04_pipeline_end2end.logs.json` |

Key observed outputs:

- GPU detection completed and reported Turing/T4 capabilities. TMA, ILC, DSM,
  and native unified memory were false on T4; GDS and GRDMA were true.
- Memory paging allocated 925 pages of 16 MB each before the expected
  out-of-memory boundary, then measured NexusRT page-in latency at
  `7.8 us / 16MB page`. The PyTorch baseline was skipped because the 28 GB
  baseline allocation exceeded the 14 GB T4 HBM capacity.
- Token efficiency completed with an ICM total budget of `14080 tokens` versus
  `32768 tokens` for the monolithic context.
- End-to-end pipeline completed successfully. Optional external baseline paths
  remain best-effort and are not required for NexusRT validation.

The `*.logs.json` files are the raw Kaggle CLI log payloads. They intentionally
preserve stdout, stderr, and timing fields from Kaggle.
