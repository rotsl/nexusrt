# NexusRT documentation index

| Document | Description |
| --- | --- |
| [architecture.md](architecture.md) | System design, data flow, firmware-equivalent layer, OS-bypass boundaries, hardware abstraction model. |
| [research.md](research.md) | Literature synthesis (DREAM, CUDA UM, GDS/GRDMA, Hopper TMA/ILC, M1 Pro unified memory, ICM), benchmark tables, failure mode analysis, citations. |
| [../BENCHMARKS.md](../BENCHMARKS.md) | Validated Kaggle GPU benchmark results, Mac validation status, and benchmark caveats. |
| [api_reference.md](api_reference.md) | Full C ABI + C++ class + Python binding documentation, with usage examples and memory safety guarantees. |
| [hardware_profiles/](hardware_profiles/) | Per-device tuning matrices: [a100.yaml](hardware_profiles/a100.yaml), [h100.yaml](hardware_profiles/h100.yaml), [m1pro.yaml](hardware_profiles/m1pro.yaml). |
| [../tests/kaggle/README.md](../tests/kaggle/README.md) | Kaggle GPU notebook setup and latest T4 validation runs. |

## Quick links

- Boot sequence: `architecture.md` section 3 (Firmware Boundary) and section 3.1 (What "firmware-equivalent" means).
- Memory management: `architecture.md` section 7; `api_reference.md` section 1.3.
- Scheduler: `architecture.md` section 8; `api_reference.md` section 1.5, section 2.4.
- Token optimization (ICM): `architecture.md` section 9; `research.md` section 6.
- Hopper features (TMA/ILC/DSM): `research.md` section 5; `hardware_profiles/h100.yaml`.
- Apple M1 Pro path: `research.md` section 7; `hardware_profiles/m1pro.yaml`.
- Failure modes: `architecture.md` section 11; `research.md` section 10.
- Build and smoke tests: root `README.md` section Quick Start; `packaging/README.md`.
- Kaggle validation: `tests/kaggle/README.md`.
- Benchmark results: `BENCHMARKS.md`.
- Citations: `research.md` section 13.
