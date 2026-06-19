#!/usr/bin/env bash
# =============================================================================
# NexusRT — pre-commit hook: forbid AI framework imports in src/
# Enforces the strict constraint that no PyTorch / TensorFlow / JAX / Flax /
# Haiku imports appear in the core runtime source. The runtime must use only
# native CUDA / Metal primitives or custom kernels.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${ROOT}/src"

if [ ! -d "${SRC}" ]; then
    echo "✗ src/ not found at ${SRC}"
    exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
    echo "⚠ ripgrep (rg) not installed — skipping framework-import check."
    exit 0
fi

if rg -n 'import (torch|tensorflow|jax|flax|haiku)' "${SRC}/"; then
    echo "✗ ERROR: framework imports are forbidden in src/."
    echo "  The NexusRT core must use only native CUDA / Metal primitives."
    echo "  See docs/architecture.md §Firmware Boundary."
    exit 1
fi

echo "✓ No framework imports in src/"
