"""NexusRT — firmware-centric OS-bypass runtime for end-to-end LLM pipelines.

Python bindings live under :mod:`nexusrt`. The public surface is:

* :mod:`nexusrt.firmware`   — boot / shutdown / task submit
* :mod:`nexusrt.memory`     — HBM allocation + prefetch
* :mod:`nexusrt.scheduler`  — task graph + stage contracts
* :mod:`nexusrt.pipeline`   — end-to-end LLM orchestrator
* :mod:`nexusrt.token_opt`  — ICM layered context, KV-cache pruning
* :mod:`nexusrt.platform`   — device detection
* :mod:`nexusrt._abi`       — low-level C ABI (ctypes); rarely used directly
"""

from __future__ import annotations

__version__ = "0.1.0"

# Public submodules are imported lazily so that `import nexusrt` works even
# on machines without a GPU — the actual boot will report DeviceNotFound.

from . import _abi          # noqa: F401  (always available — pure ctypes)
from . import firmware      # noqa: F401
from . import memory        # noqa: F401
from . import scheduler     # noqa: F401
from . import token_opt     # noqa: F401
from . import platform      # noqa: F401
from . import pipeline      # noqa: F401
from . import config        # noqa: F401

__all__ = [
    "__version__",
    "firmware",
    "memory",
    "scheduler",
    "token_opt",
    "platform",
    "pipeline",
    "config",
]
