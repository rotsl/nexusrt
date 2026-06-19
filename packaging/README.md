# NexusRT packaging - build setup

## Build the C++ core

From the repository root, first create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install cmake ninja pytest pytest-cov pyyaml
```

CUDA build for an NVIDIA machine with CUDA Toolkit 12.x:

```bash
cmake -S packaging -B packaging/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUSRT_ENABLE_CUDA=ON \
  -DNEXUSRT_ENABLE_METAL=OFF \
  -DNEXUSRT_BUILD_TESTS=ON

cmake --build packaging/build -j
```

Local macOS/Metal build without CUDA:

```bash
cmake -S packaging -B packaging/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNEXUSRT_ENABLE_CUDA=OFF \
  -DNEXUSRT_ENABLE_METAL=AUTO \
  -DNEXUSRT_BUILD_TESTS=ON

cmake --build packaging/build -j
```

## Install Python bindings

```bash
python -m pip install -e "packaging[dev]"
```

## Run tests

```bash
ctest --test-dir packaging/build --output-on-failure
PYTHONPATH=python python -m pytest tests/unit/test_python_bindings.py
```

Hardware-dependent C++ tests skip when no supported runtime device is visible.
Runtime CLI smoke tests may still return `DEVICE_NOT_FOUND` on unsupported
hosts even when the build and unit tests pass.

## Kaggle validation

The notebooks under `../tests/kaggle/` use source imports plus a direct CUDA
CMake build instead of editable `pip install`. This is the path validated on
Kaggle T4 runs, where the repository is usually mounted read-only from
`/kaggle/input` and copied to `/kaggle/working/nexusrt` before building.

Raw successful run logs are stored in `../tests/kaggle/results/`.

## Manual GitHub artifact builds

`.github/workflows/manual-artifacts.yml` is a manual-only workflow that uploads
downloadable artifacts from the GitHub Actions run page:

- source `.tar.gz` and `.zip` archives
- Python wheel
- native Linux CMake install archive
- optional native macOS CMake install archive
- optional Linux CUDA 12 archive
- optional HTML docs

Run it from Actions -> Manual Artifact Build -> Run workflow. Keep
`build_macos` disabled unless you specifically need a macOS archive, because
hosted macOS runners may queue for a long time. Keep `build_cuda` disabled
unless you want the workflow to install CUDA Toolkit 12 and compile the
CUDA-enabled Linux artifact.

## Build a wheel

```bash
python -m pip install build
python -m build packaging
```

## Files in this directory

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Python package metadata + scikit-build-core config |
| `CMakeLists.txt` | Top-level CMake build for the C++ core + shared library |
| `nexusrt.pc.in` | pkg-config template |
| `setup_hooks/pre-commit-config.yaml` | Pre-commit hooks (ruff, clang-format, markdownlint, no-framework-imports) |
| `setup_hooks/ci.yml` | GitHub Actions CI workflow |
| `setup_hooks/run_benchmarks.sh` | Benchmark runner |
| `setup_hooks/post_test_coverage.sh` | Post-test coverage check (>=85%) |
| `setup_hooks/run_leak_audit.sh` | Memory leak audit |
