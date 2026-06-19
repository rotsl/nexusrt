"""Platform dispatch — NVIDIA vs Apple."""
from __future__ import annotations

from . import firmware as _fw


def detect() -> str:
    """Detect the active platform. Returns 'nvidia' | 'apple' | 'unknown'."""
    try:
        info = _fw.device_info()
        return info.vendor
    except Exception:
        return "unknown"


def is_nvidia() -> bool:
    return detect() == "nvidia"


def is_apple() -> bool:
    return detect() == "apple"
