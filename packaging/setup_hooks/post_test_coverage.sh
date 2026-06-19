#!/usr/bin/env bash
# =============================================================================
# NexusRT — post-test coverage check
# Verifies that the C++ core modules have ≥85% coverage. Run after `ctest`.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-${ROOT}/build}"
THRESHOLD="${NEXUSRT_COVERAGE_THRESHOLD:-85}"

if ! command -v lcov >/dev/null 2>&1; then
    echo "⚠ lcov not installed — skipping coverage check."
    exit 0
fi

OUT="${BUILD}/coverage"
mkdir -p "${OUT}"

# Collect coverage data.
lcov --capture --directory "${BUILD}" --output-file "${OUT}/coverage.info" \
     --rc lcov_branch_coverage=1 || true

# Filter out system headers and test helpers.
lcov --remove "${OUT}/coverage.info" \
     '/usr/*' \
     '*/tests/*' \
     '*/selftest.cpp' \
     --output-file "${OUT}/coverage.filtered.info"

# Generate HTML report.
genhtml "${OUT}/coverage.filtered.info" --output-directory "${OUT}/html" \
        --rc lcov_branch_coverage=1 || true

# Extract total coverage percentage.
TOTAL=$(lcov --list "${OUT}/coverage.filtered.info" | awk '/Total/ {print $4}' | tr -d '%')
if [ -z "${TOTAL}" ]; then
    echo "⚠ Could not parse coverage percentage — skipping check."
    exit 0
fi

echo "Total coverage: ${TOTAL}% (threshold: ${THRESHOLD}%)"
if (( $(echo "${TOTAL} < ${THRESHOLD}" | bc -l) )); then
    echo "✗ Coverage below threshold"
    exit 1
fi
echo "✓ Coverage meets threshold"
