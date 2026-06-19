# NexusRT Kaggle GPU notebooks

These notebooks are for Kaggle GPU sessions, especially T4/P100/A100/H100
validation when available. They require a visible CUDA device and are not
expected to pass on CPU-only or TPU-only Kaggle kernels.

## Kaggle setup

1. Start a Kaggle notebook with a GPU accelerator enabled.
2. Enable Internet in the Kaggle notebook settings only if you plan to run the
   optional PyTorch/Transformers baseline cells. The NexusRT setup cells use
   direct source imports plus a local CMake CUDA build.
3. Confirm the GPU is visible:

```bash
!nvidia-smi
!nvcc --version
```

4. Attach the private Kaggle Dataset `rbrtsl/nexusrt`, or upload/clone this
   repository into `/kaggle/working/nexusrt`.
5. Run the setup cell near the top of each notebook. If the repo is mounted
   from `/kaggle/input/nexusrt`, the cell copies it to `/kaggle/working/nexusrt`
   first because Kaggle datasets are read-only and CMake needs
   a writable checkout. The setup cell prefers `/kaggle/input` and refreshes
   `/kaggle/working/nexusrt`, so stale files from previous runs do not shadow
   the latest dataset version. It adds `python/` to `sys.path`, builds the
   shared CUDA library with CMake directly, and sets `NEXUSRT_LIB` to the
   resulting `libnexusrt` path. The setup cell also writes a full log to:

```text
/kaggle/working/nexusrt_setup.log
```

If CUDA configure, build, or device initialization fails, the notebook fails.
That is intentional: these notebooks validate CUDA behavior, not CPU fallback.

6. If the runtime returns `DEVICE_NOT_FOUND`, CUDA is not visible to NexusRT or
   the selected Kaggle accelerator is unsupported for that test. The notebooks
   set `NEXUSRT_BOOT_TRACE=1`, so the Kaggle logs include CUDA probe details.

## Notebook purpose

| Notebook | Purpose |
| --- | --- |
| `01_gpu_detection_bench.ipynb` | Detect the runtime device and run basic memory prefetch timing. |
| `02_memory_paging_test.ipynb` | Exercise oversubscribed paging and compare with a CUDA UM baseline when PyTorch is available. |
| `03_token_efficiency.ipynb` | Compare monolithic context touches with ICM-scoped token-cache behavior. |
| `04_pipeline_end2end.ipynb` | Run the NexusRT pipeline stubs and optional PyTorch baseline. |

The optional PyTorch/Transformers cells are baselines only. The NexusRT source
tree intentionally avoids importing framework stacks.

## Latest successful Kaggle runs

All four notebooks were pulled back from successful private Kaggle runs on
2026-06-19 and stored here as the current canonical notebook copies.

| Notebook | Kaggle kernel | Kaggle status | Accelerator |
| --- | --- | --- | --- |
| `01_gpu_detection_bench.ipynb` | `rbrtsl/notebook1nexusrtworking` | Complete | `NvidiaTeslaT4` |
| `02_memory_paging_test.ipynb` | `rbrtsl/notebook2nexusrt-importtime` | Complete | `NvidiaTeslaT4` |
| `03_token_efficiency.ipynb` | `rbrtsl/notebook3nexusrt-importtime` | Complete | `NvidiaTeslaT4` |
| `04_pipeline_end2end.ipynb` | `rbrtsl/notebook4nexusrt-importtime` | Complete | `NvidiaTeslaT4` |

The matching raw Kaggle CLI logs are in `tests/kaggle/results/`, with a compact
run summary in `tests/kaggle/results/README.md`.
