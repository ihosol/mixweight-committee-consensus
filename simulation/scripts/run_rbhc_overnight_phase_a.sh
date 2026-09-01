#!/usr/bin/env bash
# RBHC overnight Phase A — burst regression matrix.
# 3 controllers × 3 seeds = 9 runs.
# Target wall clock: ~2.5–3 hours (each run ~15–20 minutes at FP_A1 epoch
# counts, plus localnet bootstrap overhead).
#
# Each cell writes under poc/cosmos/artifacts/article_rbhc/burst_*/runs/<seed>/.
# Re-running with the same config overwrites latest aliases but keeps
# per-seed run directories (because preserve_run_history is true).
#
# To resume after a partial failure, delete only the failed cell's runs/<seed>
# directory and re-launch this script.
#
# Phase B (drift) is deferred until M3.3 (stake-migration primitive) ships;
# this script does not include drift cells.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
if [[ ! -x "$BINARY" ]]; then
  echo "FATAL: chain binary not found at $BINARY" >&2
  echo "Build with: cd poc/cosmos/chain53/chain-five-three && go build -o build/chain-five-threed ./cmd/chain-five-threed/" >&2
  exit 2
fi

export POC_CHAIND="$BINARY"

CFG_DIR="$REPO_ROOT/poc/cosmos/generated_configs/article_rbhc"
RUNNER="$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py"

CELLS=(
  RBHC_burst_baseline
  RBHC_burst_signal
  RBHC_burst_hybrid
)

mkdir -p "$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/logs"
LOG_ROOT="$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/logs"

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[overnight] start ${START_TS}"

for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_ROOT/${cell}_${START_TS}.log"
  echo "[overnight] running ${cell} -> ${LOG}"
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    echo "[overnight] ${cell} OK"
  else
    echo "[overnight] ${cell} FAILED — see ${LOG}" >&2
    echo "[overnight] continuing with remaining cells" >&2
  fi
done

END_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[overnight] end ${END_TS}"
echo
echo "Artifacts under:"
for cell in "${CELLS[@]}"; do
  AS="$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/${cell#RBHC_}"
  echo "  ${AS}"
done
echo
echo "Next step: generate paper figures via"
echo "  python3 papers/risk_budget_controller_2026/figures/scripts/make_paper_plots.py"
