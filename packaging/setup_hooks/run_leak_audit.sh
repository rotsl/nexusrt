#!/usr/bin/env bash
# =============================================================================
# NexusRT — leak audit
# Runs the runtime leak audit on the test harness. Used in CI to fail the
# build if any HBM allocation is not freed.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-${ROOT}/build}"

export NEXUSRT_LEAK_AUDIT_ENABLED=1
export NEXUSRT_LEAK_AUDIT_FAIL_ON_LEAK=1

cd "${BUILD}"
ctest --output-on-failure -R ".*selftest.*"

echo "✓ Leak audit passed"
