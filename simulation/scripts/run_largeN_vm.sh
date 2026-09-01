#!/usr/bin/env bash
# P5.4 large-N controller-dynamics run (VM edition).
#
# Purpose: measure whether the curated headline metrics and the controller
# dynamics (half-life, settle, chatter, signal noise floor) survive when the
# committee is a small fraction of the validator set, i.e. toward m << N.
# The certificate layer needs no such run (Proposition prop:ch2-bps-transfer
# holds for any m <= N); this measures the dynamics layer only.
#
# VM requirements (each validator is a separate chain process, ~200 MB RSS):
#   N=54 cell:  >= 8 vCPU, >= 16 GB RAM   (~1.5-3 h for 3 seeds)
#   N=99 cell:  >= 16 vCPU, >= 32 GB RAM  (~3-6 h for 3 seeds)
# Run the N=54 cell first; add N=99 only if the VM is large enough.
#
# Setup on a fresh VM:
#   1. git clone <repo> && cd <repo>
#   2. install Go (>=1.23) and Python3 with pyyaml, numpy, pandas, matplotlib
#   3. bash poc/cosmos/scripts/run_largeN_vm.sh            # N=54 only
#      bash poc/cosmos/scripts/run_largeN_vm.sh --with-n99 # both cells
#
# Outputs (same layout as every curated scenario):
#   poc/cosmos/artifacts/largeN/LN_A1_N54_c9_k18_b033_burst/aggregated_final_table.csv
#   poc/cosmos/artifacts/largeN/LN_A1_N99_c9_k33_b033_burst/aggregated_final_table.csv
# Compare against the curated N=18 rows in
#   poc/cosmos/dissertation_final/tables/FINAL_DISSERTATION_TABLE.csv
# (default burst row: full 5.020 / 1ep 26.925 / chernoff>=1/2 13.317).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
if [[ ! -x "$BINARY" ]]; then
  echo "[largeN] chain binary missing; building..."
  GO_BIN="${GO_BIN:-$(command -v go || echo "$HOME/.local/go/bin/go")}"
  ( cd poc/cosmos/chain53/chain-five-three && "$GO_BIN" build -o build/chain-five-threed ./cmd/chain-five-threed/ )
fi
export POC_CHAIND="$BINARY"

CFG_DIR="$REPO_ROOT/poc/cosmos/generated_configs/largeN"
RUNNER="$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py"
LOG_DIR="$REPO_ROOT/poc/cosmos/artifacts/largeN/logs"
mkdir -p "$LOG_DIR"

CELLS=(LN_A1_N54_c9_k18_b033_burst)
if [[ "${1:-}" == "--with-n99" ]]; then
  CELLS+=(LN_A1_N99_c9_k33_b033_burst)
fi

for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_DIR/${cell}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[largeN] starting ${cell} -> ${LOG}"
  T0=$(date +%s)
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    echo "[largeN] ${cell} OK ($(( $(date +%s) - T0 ))s)"
  else
    echo "[largeN] ${cell} FAILED after $(( $(date +%s) - T0 ))s — see ${LOG}" >&2
    exit 1
  fi
done

echo "[largeN] done. Aggregates:"
find "$REPO_ROOT/poc/cosmos/artifacts/largeN" -name aggregated_final_table.csv
