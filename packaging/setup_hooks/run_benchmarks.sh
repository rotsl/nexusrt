#!/usr/bin/env bash
# =============================================================================
# NexusRT — benchmark runner
# Runs the nexusrt-bench CLI across all stages and writes a JSON report
# to benchmark-results/<timestamp>.json.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${ROOT}/benchmark-results"
mkdir -p "${RESULTS_DIR}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="${RESULTS_DIR}/${TS}.json"

cd "${ROOT}"

# Determine profile
PROFILE="${NEXUSRT_PROFILE:-auto}"

# Run all benchmark stages.
python -m nexusrt.cli_bench --stage all --profile "${PROFILE}" --json > "${OUT}"

echo "✓ Benchmark complete: ${OUT}"
