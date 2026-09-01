#!/usr/bin/env bash
# RBHC drift re-run — refresh ONLY the concentration-drift cells.
#
# Why a dedicated script: the top-k coalition seat-attribution instrumentation in
# epochrun.py (realized tracked_seats_share for drift) changed, so the drift
# cells must be re-run to populate the empirical-capture metric. The burst cells
# are unaffected (their attacker tracking already worked), so this script leaves
# them alone instead of forcing the whole 8-cell matrix.
#
# Each cell is 3 seeds; ~50 min/cell on the reference box, so ~4 h for all five.
# Cells run sequentially with per-cell failure isolation; logs under
# artifacts/article_rbhc/logs/. Re-running is safe: this script always refreshes
# the drift cells (it does not consult the resume guard).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
if [[ ! -x "$BINARY" ]]; then
  echo "FATAL: chain binary not found at $BINARY" >&2
  exit 2
fi
export POC_CHAIND="$BINARY"

CFG_DIR="$REPO_ROOT/poc/cosmos/generated_configs/article_rbhc"
RUNNER="$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py"
ARTIFACTS_ROOT="$REPO_ROOT/poc/cosmos/artifacts"
LOG_ROOT="$REPO_ROOT/poc/cosmos/artifacts/article_rbhc/logs"
mkdir -p "$LOG_ROOT"

# Drift cells only (baseline / signal / hybrid + the two drift-rate variants).
CELLS=(
  RBHC_drift_baseline
  RBHC_drift_signal
  RBHC_drift_hybrid
  RBHC_drift_hybrid_rate01
  RBHC_drift_hybrid_rate03
)

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[drift-rerun] start ${START_TS}, ${#CELLS[@]} cells x 3 seeds"

# Drop stale aggregates so the (resumable) state is clean and the new
# instrumentation's output is what lands.
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  SUBDIR="$(sed -n 's/^[[:space:]]*artifacts_subdir:[[:space:]]*//p' "$CFG" | head -1 | tr -d '"'"'"' ' | tr -d '\r')"
  [[ -n "$SUBDIR" ]] && rm -f "$ARTIFACTS_ROOT/${SUBDIR}/aggregated_final_table.csv"
done

OK=0; FAIL=0
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_ROOT/${cell}_rerun_${START_TS}.log"
  if [[ ! -f "$CFG" ]]; then
    echo "[drift-rerun] ${cell} CONFIG MISSING; skipping" >&2; FAIL=$((FAIL+1)); continue
  fi
  echo "[drift-rerun] starting ${cell} -> ${LOG}"
  CELL_START=$(date +%s)
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    echo "[drift-rerun] ${cell} OK ($(( $(date +%s) - CELL_START ))s)"; OK=$((OK+1))
  else
    echo "[drift-rerun] ${cell} FAILED after $(( $(date +%s) - CELL_START ))s — see ${LOG}" >&2; FAIL=$((FAIL+1))
  fi
done

echo "[drift-rerun] end: ${OK} ok, ${FAIL} failed"
echo
echo "Next: python3 papers/risk_budget_controller_2026/figures/scripts/make_drift_plots.py"
