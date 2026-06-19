"""Firmware-equivalent layer — boot, shutdown, task submit."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

from . import _abi


@dataclass
class DeviceInfo:
    vendor: str           # "nvidia" | "apple" | "unknown"
    arch: str             # "pascal" | "turing" | "ampere" | "hopper" | "apple_silicon" | "unknown"
    name: str
    hbm_bytes: int
    sm_count: int
    features: dict[str, bool]


_VENDOR_NAMES = {0: "unknown", 1: "nvidia", 2: "apple"}
_ARCH_NAMES   = {
    0: "unknown",
    1: "ampere",
    2: "hopper",
    3: "apple_silicon",
    4: "pascal",
    5: "turing",
}
_FEATURE_BITS = ["tma", "ilc", "dsm", "gds", "grdma", "clusters", "unified_native"]


class Firmware:
    """High-level wrapper around nexusrt_firmware_*."""

    def __init__(self) -> None:
        self._ctx: Optional[ctypes.c_void_p] = None

    def init(self, profile: str = "auto") -> DeviceInfo:
        """Boot the firmware-equivalent layer. Returns the discovered DeviceInfo."""
        self._ctx = ctypes.c_void_p()
        rc = _abi.nexusrt_firmware_init(profile.encode(), ctypes.byref(self._ctx))
        _abi.check(rc)
        return self.device_info()

    def device_info(self) -> DeviceInfo:
        if self._ctx is None:
            raise RuntimeError("firmware not initialized — call init() first")
        vendor = ctypes.c_int32(0)
        arch   = ctypes.c_int32(0)
        name   = ctypes.create_string_buffer(256)
        hbm    = ctypes.c_uint64(0)
        sms    = ctypes.c_uint32(0)
        bits   = ctypes.c_uint32(0)
        rc = _abi.nexusrt_firmware_device_desc(
            self._ctx,
            ctypes.byref(vendor), ctypes.byref(arch),
            name, ctypes.sizeof(name),
            ctypes.byref(hbm), ctypes.byref(sms), ctypes.byref(bits))
        _abi.check(rc)
        features = {bit: bool(bits.value & (1 << i))
                    for i, bit in enumerate(_FEATURE_BITS)}
        return DeviceInfo(
            vendor=_VENDOR_NAMES.get(vendor.value, "unknown"),
            arch=_ARCH_NAMES.get(arch.value, "unknown"),
            name=name.value.decode(errors="replace"),
            hbm_bytes=int(hbm.value),
            sm_count=int(sms.value),
            features=features,
        )

    def shutdown(self) -> None:
        if self._ctx is not None:
            rc = _abi.nexusrt_firmware_shutdown(self._ctx)
            _abi.check(rc)
            self._ctx = None

    def __enter__(self) -> "Firmware":
        self.init()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


# Module-level singleton for convenience.
_default = Firmware()


def init(profile: str = "auto") -> DeviceInfo:
    """Boot the firmware-equivalent layer using the module singleton."""
    return _default.init(profile)


def shutdown() -> None:
    """Tear down the module singleton firmware context."""
    _default.shutdown()


def device_info() -> DeviceInfo:
    """Return the DeviceInfo for the booted firmware context."""
    return _default.device_info()


def ctx() -> "ctypes.c_void_p":
    """Return the active firmware context handle, or raise if not booted.

    Used by sibling submodules (memory, scheduler, token_opt) to access
    the C ABI without exposing the singleton's private state.
    """
    if _default._ctx is None:
        raise RuntimeError(
            "firmware not initialized — call nexusrt.firmware.init() first")
    return _default._ctx
