"""Pytest unit tests for the Python bindings.

These run *without* a GPU — they validate the binding structure and the
pure-Python fallbacks. When the C++ core is built and importable, the
tests also exercise the C ABI end-to-end.
"""
from __future__ import annotations

import os
import sys
import pathlib

import pytest

# Make sure the package is importable from the repo.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))


def test_imports():
    import nexusrt
    assert nexusrt.__version__ == "0.1.0"
    # All submodules must import successfully even without a GPU.
    from nexusrt import firmware, memory, scheduler, token_opt, platform, pipeline, config
    assert firmware
    assert memory
    assert scheduler
    assert token_opt
    assert platform
    assert pipeline
    assert config


def test_status_codes_match_c_abi():
    from nexusrt import _abi
    assert _abi.Status.OK == 0
    assert _abi.Status.DEVICE_NOT_FOUND == -4
    assert _abi.Status.NOT_IMPLEMENTED == -6


def test_status_name_lookup():
    from nexusrt import _abi
    assert _abi.status_name(0) == "OK"
    assert _abi.status_name(-4) == "DEVICE_NOT_FOUND"
    assert _abi.status_name(999) == "UNKNOWN"


def test_check_raises_on_error():
    from nexusrt import _abi
    with pytest.raises(_abi.NexusRTError) as exc:
        _abi.check(-4)
    assert exc.value.code == -4


def test_check_noop_on_success():
    from nexusrt import _abi
    _abi.check(0)  # should not raise


def test_token_opt_layer_enum():
    from nexusrt import token_opt as tk
    assert int(tk.IcmLayer.L0_SYSTEM) == 0
    assert int(tk.IcmLayer.L4_WORKING) == 4


def test_memory_alloc_hints_bitfield():
    from nexusrt import memory as mem
    assert mem.ILC == 0x01
    assert mem.READ_MOSTLY == 0x02
    assert mem.GRDMA_VISIBLE == 0x10


def test_scheduler_stage_builder():
    from nexusrt import scheduler as sch
    s = sch.stage("foo", module="m", function="f",
                  token_budget=4096, grid=(32,1,1), block=(256,1,1))
    assert s.name == "foo"
    assert s.token_budget == 4096
    assert s.grid == (32, 1, 1)


def test_pipeline_run_infer_stub():
    """The pure-Python stub returns prompt + max_new_tokens values."""
    from nexusrt import pipeline as pl
    out = pl.run_infer([1, 2, 3], max_new_tokens=4)
    assert len(out) == 7
    assert out[:3] == [1, 2, 3]


def test_pipeline_postprocess_printable():
    from nexusrt import pipeline as pl
    out = pl.run_postprocess([72, 73, 33])   # 'H', 'I', '!'
    assert out == "HI!"


def test_platform_detect_without_gpu():
    """On a CPU-only CI host, detect() should not crash."""
    from nexusrt import platform
    v = platform.detect()
    assert v in ("nvidia", "apple", "unknown")


def test_config_loader_missing_file():
    from nexusrt import config
    with pytest.raises(FileNotFoundError):
        config.load("__nonexistent__")


def test_cli_run_help():
    """Smoke test that the CLI script parses args without crashing."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nexusrt_cli_run",
        str(ROOT / "python" / "nexusrt" / "cli_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_cli_bench_help():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nexusrt_cli_bench",
        str(ROOT / "python" / "nexusrt" / "cli_bench.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
