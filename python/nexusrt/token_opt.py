"""Token optimization — ICM layered context, KV-cache pruning."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from . import _abi, firmware as _fw


class IcmLayer(IntEnum):
    L0_SYSTEM        = 0
    L1_PERSONA       = 1
    L2_INSTRUCTIONS  = 2
    L3_REFERENCE     = 3
    L4_WORKING       = 4


def context_scope(stage: str, layers: List[IcmLayer]) -> int:
    """Return the total token budget across the selected layers."""
    import ctypes
    mask = 0
    for l in layers:
        mask |= 1 << int(l)
    budget = ctypes.c_uint64(0)
    rc = _abi.nexusrt_context_scope(_fw.ctx(), stage.encode(),
                                    mask, ctypes.byref(budget))
    _abi.check(rc)
    return int(budget.value)


def prefetch_attention(kv_cache: int, n_slots: int,
                       topk_indices: List[int]) -> None:
    import ctypes
    arr = (ctypes.c_uint32 * len(topk_indices))(*topk_indices)
    rc = _abi.nexusrt_prefetch_attention(_fw.ctx(),
                                         kv_cache, n_slots, arr)
    _abi.check(rc)


def token_prune(kv_cache: int, max_resident: int) -> None:
    rc = _abi.nexusrt_token_prune(_fw.ctx(), kv_cache, max_resident)
    _abi.check(rc)
