"""platform_check.py — auto-detect M1 Pro, disable CUDA paths, enforce Metal.

Imported by every test under tests/mac_unified/ to ensure the harness does
not attempt to initialize the CUDA backend on Apple Silicon (which would
either fail loudly or silently mis-dispatch).
"""
from __future__ import annotations

import platform
import sys


def is_apple_silicon() -> bool:
    """True on M1/M2/M3 family chips."""
    if sys.platform != "darwin":
        return False
    # `platform.machine()` returns 'arm64' on Apple Silicon.
    return platform.machine() in ("arm64", "aarch64")


def assert_apple_silicon() -> None:
    """Raise RuntimeError if not running on Apple Silicon."""
    if not is_apple_silicon():
        raise RuntimeError(
            "This test requires Apple Silicon (M1 Pro / M2 Pro / ...). "
            f"Detected: platform={sys.platform}, machine={platform.machine()}"
        )


def disable_cuda() -> None:
    """Prevent any CUDA backend from initializing.

    Sets NEXUSRT_PROFILE=m1pro and unsets CUDA_VISIBLE_DEVICES so any
    code that probes for NVIDIA devices sees nothing.
    """
    import os
    os.environ["NEXUSRT_PROFILE"] = "m1pro"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    # Also null out the CUDA driver path so a stray cuInit() fails fast.
    os.environ["CUDA_DRIVER"] = ""


def enforce_metal_dispatch() -> None:
    """Convenience wrapper — disable CUDA + assert Apple Silicon."""
    disable_cuda()
    assert_apple_silicon()


if __name__ == "__main__":
    # Run as a script: print the detection result and exit 0 on Apple.
    if is_apple_silicon():
        print("Apple Silicon detected — Metal backend will be used.")
        disable_cuda()
        sys.exit(0)
    else:
        print(f"Not Apple Silicon (platform={sys.platform}, "
              f"machine={platform.machine()}).")
        sys.exit(1)
