#!/usr/bin/env bash
# RBHC overnight — full Phase A + B matrix (24 runs).
#
# Block A: drift main (3 cells x 3 seeds = 9)
#   drift_baseline, drift_signal, drift_hybrid at ρ=0.02
#
# Block B: drift-rate sensitivity (2 cells x 3 seeds = 6)
#   drift_hybrid_rate01, drift_hybrid_rate03
#   (rate02 is already in block A as drift_hybrid)
#
# Block C: burst regression (3 cells x 3 seeds = 9)
#   burst_baseline, burst_signal, burst_hybrid
#
# Wall-clock estimate: ~6 hours total. Each cell is 3 seeds and each seed is
# ~10–15 minutes of localnet runtime plus bootstrap overhead.
#
# Cells run sequentially with per-cell failure isolation (one bad cell does
# not block the rest). Per-cell logs go to artifacts/article_rbhc/logs/.
#
# Resumable: a cell is skipped on re-launch once its multi-seed aggregate
# (aggregated_final_table.csv) exists, which is written only after all of that
# cell's seeds succeed. So after a partial failure just re-launch — completed
# cells are skipped and only the failed/incomplete ones re-run. Use RBHC_FORCE=1
# to re-run everything from scratch.

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
ARTIFACTS_ROOT="$REPO_ROOT/poc/cosmos/artifacts"

# Resume by default: a cell is considered complete once its multi-seed
# aggregate (aggregated_final_table.csv) exists, which epochrun_multiseed.py
# writes only after every seed of that cell succeeds. Completed cells are
# skipped so a re-launch after a partial failure does not redo finished work.
# Set RBHC_FORCE=1 to ignore existing results and re-run every cell.
FORCE="${RBHC_FORCE:-0}"

# Order: drift main first (headline), then drift sensitivity, then burst.
# This way, if the script is killed midway, the most important results land
# first.
CELLS=(
  RBHC_drift_baseline
  RBHC_drift_signal
  RBHC_drift_hybrid
  RBHC_drift_hybrid_rate01
  RBHC_drift_hybrid_rate03
  RBHC_burst_baseline
  RBHC_burst_signal
  RBHC_burst_hybrid
)

mkdir -p "$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/logs"
LOG_ROOT="$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/logs"

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[overnight] start ${START_TS}, 24 runs across 8 cells"

CELL_OK=0
CELL_FAIL=0
CELL_SKIP=0
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_ROOT/${cell}_${START_TS}.log"
  if [[ ! -f "$CFG" ]]; then
    echo "[overnight] ${cell} CONFIG MISSING ($CFG); skipping" >&2
    CELL_FAIL=$((CELL_FAIL+1))
    continue
  fi

  # Resume guard: skip a cell whose multi-seed aggregate already exists. The
  # output subdir is read from the config's experiment.artifacts_subdir rather
  # than guessed, so it stays correct if a cell is ever renamed.
  SUBDIR="$(sed -n 's/^[[:space:]]*artifacts_subdir:[[:space:]]*//p' "$CFG" | head -1 | tr -d '"'"'"' ' | tr -d '\r')"
  DONE_MARKER="$ARTIFACTS_ROOT/${SUBDIR}/aggregated_final_table.csv"
  if [[ "$FORCE" != "1" && -n "$SUBDIR" && -f "$DONE_MARKER" ]]; then
    echo "[overnight] ${cell} already complete (found ${DONE_MARKER}); skipping (RBHC_FORCE=1 to re-run)"
    CELL_SKIP=$((CELL_SKIP+1))
    continue
  fi

  echo "[overnight] starting ${cell} -> ${LOG}"
  CELL_START=$(date +%s)
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    CELL_END=$(date +%s)
    echo "[overnight] ${cell} OK ($((CELL_END-CELL_START))s)"
    CELL_OK=$((CELL_OK+1))
  else
    CELL_END=$(date +%s)
    echo "[overnight] ${cell} FAILED after $((CELL_END-CELL_START))s — see ${LOG}" >&2
    CELL_FAIL=$((CELL_FAIL+1))
  fi
done

END_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[overnight] end ${END_TS}: ${CELL_OK} ok, ${CELL_SKIP} skipped (already complete), ${CELL_FAIL} failed"
echo
echo "Artifacts under:"
for cell in "${CELLS[@]}"; do
  AS="$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/${cell#RBHC_}"
  echo "  ${AS}"
done
echo
echo "Next step:"
echo "  python3 papers/risk_budget_controller_2026/figures/scripts/make_paper_plots.py"
