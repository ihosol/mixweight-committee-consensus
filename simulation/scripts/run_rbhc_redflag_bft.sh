#!/usr/bin/env bash
# RBHC red-flag BFT demonstration (theta=1/3, eps=0.5).
#
# Emulates stake-concentration drift within a monitored top-5 coalition in a
# committee-based PoS localnet and exercises the capture-risk certificate so that
# the trajectory passes through safe -> acting -> violated/infeasible, i.e. the
# method raises a red flag (saturation) when the coalition's committee-threshold
# risk budget becomes infeasible. Writes to a separate suite so the theta=1/2
# variant-A package stays untouched:
#   poc/cosmos/artifacts/article_rbhc_redflag_bft/
#
# Minimal set for the figure: cc_drift_baseline (uncontrolled B0 + realised) and
# cc_drift_hybrid_concave (controlled B(lambda) + saturation). The signal and
# capped cells are included for completeness; comment them out to shorten the run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
if [[ ! -x "$BINARY" ]]; then
  echo "FATAL: chain binary not found at $BINARY" >&2
  echo "Build: cd poc/cosmos/chain53/chain-five-three && \\" >&2
  echo "       ~/.local/go/bin/go build -o build/chain-five-threed ./cmd/chain-five-threed/" >&2
  exit 2
fi
export POC_CHAIND="$BINARY"
export RISK_BUDGET_CONCAVE_GAMMA_PPM="${RISK_BUDGET_CONCAVE_GAMMA_PPM:-500000}"
# The risk-budget assessment uses the with-replacement draw, under which the binomial
# committee-capture certificate is the exact law of the realised committee draw.
export COMMITTEE_DRAW_MODE="${COMMITTEE_DRAW_MODE:-wr}"

SUITE_ROOT="article_rbhc_redflag_bft"
CFG_DIR="$REPO_ROOT/poc/cosmos/generated_configs/${SUITE_ROOT}"
RUNNER="$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py"
LOG_ROOT="$REPO_ROOT/poc/cosmos/artifacts/${SUITE_ROOT}/logs"
mkdir -p "$LOG_ROOT"

CELLS=(
  RBHC_cc_drift_baseline
  RBHC_cc_drift_signal
  RBHC_cc_drift_hybrid_capped
  RBHC_cc_drift_hybrid_concave
)

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[redflag] start ${START_TS}, ${#CELLS[@]} cells x 3 seeds, theta=1/3, eps=0.5, draws=16"

OK=0; FAIL=0
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_ROOT/${cell}_${START_TS}.log"
  if [[ ! -f "$CFG" ]]; then
    echo "[redflag] ${cell} CONFIG MISSING; skipping" >&2; FAIL=$((FAIL+1)); continue
  fi
  echo "[redflag] starting ${cell} -> ${LOG}"
  CELL_START=$(date +%s)
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    echo "[redflag] ${cell} OK ($(( $(date +%s) - CELL_START ))s)"; OK=$((OK+1))
  else
    echo "[redflag] ${cell} FAILED after $(( $(date +%s) - CELL_START ))s — see ${LOG}" >&2; FAIL=$((FAIL+1))
  fi
done

echo "[redflag] end: ${OK} ok, ${FAIL} failed"
echo "Next: python3 poc/cosmos/scripts/make_redflag_figure.py --suite ${SUITE_ROOT}"
