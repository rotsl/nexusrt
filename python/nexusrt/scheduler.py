"""Scheduler — task graph + stage contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import _abi, firmware as _fw


@dataclass
class StageContract:
    name: str
    module: str
    function: str
    inputs: List[int] = field(default_factory=list)
    outputs: List[int] = field(default_factory=list)
    token_budget: int = 0
    sm_footprint_mb: int = 0
    mem_footprint_mb: int = 0
    grid: tuple = (1, 1, 1)
    block: tuple = (1, 1, 1)
    shared_mem_bytes: int = 0
    depends_on: List[str] = field(default_factory=list)
    kernel_args: List[int] = field(default_factory=list)


def submit_stage(c: StageContract) -> int:
    """Submit a single stage synchronously. Returns 0 on success."""
    import ctypes
    n_in  = len(c.inputs)
    n_out = len(c.outputs)
    n_arg = len(c.kernel_args)
    in_arr  = (ctypes.c_void_p * n_in)(*c.inputs) if n_in else None
    out_arr = (ctypes.c_void_p * n_out)(*c.outputs) if n_out else None
    arg_arr = (ctypes.c_void_p * n_arg)(*c.kernel_args) if n_arg else None
    rc = _abi.nexusrt_submit_stage(
        _fw.ctx(),
        c.name.encode(), c.module.encode(), c.function.encode(),
        in_arr, n_in, out_arr, n_out,
        c.token_budget, c.sm_footprint_mb, c.mem_footprint_mb,
        c.grid[0], c.grid[1], c.grid[2],
        c.block[0], c.block[1], c.block[2],
        c.shared_mem_bytes, arg_arr, n_arg)
    _abi.check(rc)
    return rc


def wait_barrier(stage_name: str, timeout_ms: int = 0xFFFFFFFF) -> int:
    rc = _abi.nexusrt_wait_barrier(_fw.ctx(),
                                   stage_name.encode(), timeout_ms)
    _abi.check(rc)
    return rc


def stream_overlap(stage_a: str, stage_b: str, enable: bool = True) -> None:
    rc = _abi.nexusrt_stream_overlap(_fw.ctx(),
                                     stage_a.encode(), stage_b.encode(),
                                     1 if enable else 0)
    _abi.check(rc)


# Convenience: a builder for typical stage contracts.
def stage(name: str, *, module: str, function: str,
          inputs=None, outputs=None,
          token_budget: int = 0,
          sm_footprint_mb: int = 0,
          mem_footprint_mb: int = 0,
          grid=(1,1,1), block=(1,1,1),
          shared_mem_bytes: int = 0,
          depends_on=None,
          kernel_args=None) -> StageContract:
    return StageContract(
        name=name, module=module, function=function,
        inputs=list(inputs or []), outputs=list(outputs or []),
        token_budget=token_budget,
        sm_footprint_mb=sm_footprint_mb,
        mem_footprint_mb=mem_footprint_mb,
        grid=tuple(grid), block=tuple(block),
        shared_mem_bytes=shared_mem_bytes,
        depends_on=list(depends_on or []),
        kernel_args=list(kernel_args or []),
    )
