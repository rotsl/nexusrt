"""Memory manager — HBM allocation, prefetch, advice."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

from . import _abi, firmware as _fw


def _ctx() -> ctypes.c_void_p:
    """Return the active firmware context handle (or raise)."""
    return _fw.ctx()


# AllocHints bitfield (must match src/platform/abi.h).
ILC           = 0x01
READ_MOSTLY   = 0x02
PINNED_HOST   = 0x04
GDS_READABLE  = 0x08
GRDMA_VISIBLE = 0x10


@dataclass
class Allocation:
    ptr: int
    bytes: int
    hints: int


def alloc(bytes: int, *, ilc: bool = False, read_mostly: bool = False,
          pinned_host: bool = False, gds_readable: bool = False,
          grdma_visible: bool = False) -> Allocation:
    """Allocate a tensor in HBM."""
    flags = 0
    if ilc:           flags |= ILC
    if read_mostly:   flags |= READ_MOSTLY
    if pinned_host:   flags |= PINNED_HOST
    if gds_readable:  flags |= GDS_READABLE
    if grdma_visible: flags |= GRDMA_VISIBLE
    ptr = ctypes.c_void_p()
    rc = _abi.nexusrt_mem_alloc(_ctx(), bytes, flags, ctypes.byref(ptr))
    _abi.check(rc)
    return Allocation(ptr=ptr.value or 0, bytes=bytes, hints=flags)


def free(a: Allocation) -> None:
    rc = _abi.nexusrt_mem_free(_ctx(), a.ptr)
    _abi.check(rc)


def prefetch(ptr: int, bytes: int) -> None:
    rc = _abi.nexusrt_mem_prefetch(_ctx(), ptr, bytes)
    _abi.check(rc)


def advise_read_mostly(ptr: int, bytes: int) -> None:
    rc = _abi.nexusrt_mem_advise_read_mostly(_ctx(), ptr, bytes)
    _abi.check(rc)
