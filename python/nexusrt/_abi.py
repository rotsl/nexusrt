"""Low-level ctypes binding for the NexusRT C ABI (src/platform/abi.h).

This module is intentionally thin. Higher-level wrappers in :mod:`nexusrt.*`
add type safety, RAII, and ergonomic defaults.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the shared library. We search a few standard locations.
# ---------------------------------------------------------------------------

def _find_library() -> ctypes.CDLL:
    candidates: list[Path] = []
    env = os.environ.get("NEXUSRT_LIB")
    if env:
        candidates.append(Path(env))
    # Relative to the package directory.
    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / "libnexusrt.so")
    candidates.append(pkg_dir / "libnexusrt.dylib")
    candidates.append(pkg_dir.parent / "lib" / "libnexusrt.so")
    candidates.append(pkg_dir.parent / "lib" / "libnexusrt.dylib")
    candidates.append(pkg_dir.parent / "build" / "libnexusrt.so")
    repo_dir = pkg_dir.parent.parent
    candidates.append(repo_dir / "packaging" / "build" / "outputs" / "lib" / "libnexusrt.so")
    candidates.append(repo_dir / "packaging" / "build" / "outputs" / "lib" / "libnexusrt.dylib")
    candidates.append(Path("/usr/lib/libnexusrt.so"))
    candidates.append(Path("/usr/local/lib/libnexusrt.so"))

    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    # If nothing is found, return a stub object that raises on call. This
    # keeps `import nexusrt` working in environments without the runtime
    # built — useful for the mac_unified path that runs the pure-Python
    # fallback.
    raise OSError(
        "NexusRT shared library not found. Set NEXUSRT_LIB or build the "
        "C++ core with `cmake --build packaging/build`."
    )


try:
    _lib = _find_library()
except OSError:
    _lib = None  # type: ignore[assignment]


# Bind signatures. We bind lazily — if _lib is None, callers get a clear
# error message at use time.

class _Binding:
    """Wrap a ctypes function with a Pythonic signature."""

    def __init__(self, name: str, restype, argtypes):
        self._name = name
        self._restype = restype
        self._argtypes = argtypes
        self._fn = None

    def _resolve(self):
        if self._fn is not None:
            return self._fn
        if _lib is None:
            raise RuntimeError(
                f"NexusRT shared library not loaded — cannot call {self._name}. "
                "Build the C++ core or set NEXUSRT_LIB."
            )
        fn = getattr(_lib, self._name)
        fn.restype = self._restype
        fn.argtypes = self._argtypes
        self._fn = fn
        return fn

    def __call__(self, *args):
        return self._resolve()(*args)


# Status codes mirror src/firmware/types.hpp.
class Status:
    OK                    = 0
    INVALID_ARGUMENT      = -1
    OUT_OF_MEMORY         = -2
    OUT_OF_HBM            = -3
    DEVICE_NOT_FOUND      = -4
    DRIVER_ERROR          = -5
    NOT_IMPLEMENTED       = -6
    FAULT_BUFFER_OVERFLOW = -7
    CONTRACT_VIOLATION    = -8
    TIMEOUT               = -9
    IO_ERROR              = -10
    ABORTED               = -11


_STATUS_NAMES = {v: k for k, v in vars(Status).items() if not k.startswith("_")}


def status_name(code: int) -> str:
    return _STATUS_NAMES.get(code, "UNKNOWN")


def check(code: int) -> None:
    """Raise :class:`NexusRTError` if ``code`` is negative."""
    if code < 0:
        raise NexusRTError(code, status_name(code))


class NexusRTError(RuntimeError):
    def __init__(self, code: int, name: str):
        super().__init__(f"NexusRT error {code}: {name}")
        self.code = code
        self.name = name


# ---- ABI bindings ----------------------------------------------------------

c_void_p = ctypes.c_void_p
c_int32  = ctypes.c_int32
c_uint32 = ctypes.c_uint32
c_uint64 = ctypes.c_uint64
c_char_p = ctypes.c_char_p
c_size_t = ctypes.c_size_t

nexusrt_version = _Binding("nexusrt_version", c_int32,
                           [ctypes.POINTER(c_int32),
                            ctypes.POINTER(c_int32),
                            ctypes.POINTER(c_int32)])

nexusrt_firmware_init = _Binding("nexusrt_firmware_init", c_int32,
                                 [c_char_p, ctypes.POINTER(c_void_p)])

nexusrt_firmware_shutdown = _Binding("nexusrt_firmware_shutdown", c_int32,
                                     [c_void_p])

nexusrt_firmware_device_desc = _Binding("nexusrt_firmware_device_desc", c_int32,
    [c_void_p, ctypes.POINTER(c_int32), ctypes.POINTER(c_int32),
     ctypes.c_char_p, c_size_t,
     ctypes.POINTER(c_uint64), ctypes.POINTER(c_uint32), ctypes.POINTER(c_uint32)])

nexusrt_firmware_task_submit = _Binding("nexusrt_firmware_task_submit", c_int32,
    [c_void_p, c_char_p, c_char_p,
     c_uint32, c_uint32, c_uint32,
     c_uint32, c_uint32, c_uint32,
     c_uint32,
     ctypes.POINTER(c_void_p), c_uint32,
     c_int32, c_int32])

nexusrt_firmware_fault_handler = _Binding("nexusrt_firmware_fault_handler", c_int32,
                                          [c_void_p, c_uint64, c_int32])

nexusrt_mem_alloc = _Binding("nexusrt_mem_alloc", c_int32,
                             [c_void_p, c_uint64, c_uint32, ctypes.POINTER(c_void_p)])

nexusrt_mem_free = _Binding("nexusrt_mem_free", c_int32,
                            [c_void_p, c_void_p])

nexusrt_mem_prefetch = _Binding("nexusrt_mem_prefetch", c_int32,
                                [c_void_p, c_void_p, c_uint64])

nexusrt_mem_advise_read_mostly = _Binding(
    "nexusrt_mem_advise_read_mostly", c_int32,
    [c_void_p, c_void_p, c_uint64])

nexusrt_submit_stage = _Binding("nexusrt_submit_stage", c_int32,
    [c_void_p, c_char_p, c_char_p, c_char_p,
     ctypes.POINTER(c_void_p), c_uint32,
     ctypes.POINTER(c_void_p), c_uint32,
     c_uint32, c_uint32, c_uint32,
     c_uint32, c_uint32, c_uint32,
     c_uint32, c_uint32, c_uint32,
     c_uint32,
     ctypes.POINTER(c_void_p), c_uint32])

nexusrt_wait_barrier = _Binding("nexusrt_wait_barrier", c_int32,
                                [c_void_p, c_char_p, c_uint32])

nexusrt_stream_overlap = _Binding("nexusrt_stream_overlap", c_int32,
                                  [c_void_p, c_char_p, c_char_p, c_int32])

nexusrt_context_scope = _Binding("nexusrt_context_scope", c_int32,
    [c_void_p, c_char_p, c_uint32, ctypes.POINTER(c_uint64)])

nexusrt_prefetch_attention = _Binding("nexusrt_prefetch_attention", c_int32,
    [c_void_p, c_void_p, c_uint64, ctypes.POINTER(c_uint32)])

nexusrt_token_prune = _Binding("nexusrt_token_prune", c_int32,
                               [c_void_p, c_void_p, c_uint32])

nexusrt_metrics_dump = _Binding("nexusrt_metrics_dump", c_int32,
                                [c_void_p, c_char_p, c_size_t])
