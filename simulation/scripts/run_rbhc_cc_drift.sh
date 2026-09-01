#!/usr/bin/env bash
# RBHC larger competitive-committee drift run (variant A).
#
# This is the tuned follow-up to the original cc-drift script. The old N=12,
# m=5, top-3 setup made capture observable, but it was small enough to invite
# "toy regime" criticism. The default run here scales the validator set up to
# N=24 and monitors a top-5 coalition under m=9. That keeps the committee
# competitive (5/9 > theta=0.5) without relying on an ultra-small committee, and
# it preserves the original goal: expose the drift-sensitive gap between the
# signal-only controller and the hybrid risk-budget controller.
#
# Artifacts are written to a separate suite root:
#   poc/cosmos/artifacts/article_rbhc_variant_a/
# so the legacy cc-drift package stays untouched.
#
# Cells (3 seeds each):
#   cc_drift_baseline           lambda=0 reference
#   cc_drift_signal             signal-only controller
#   cc_drift_hybrid_capped      hybrid, capped_stake baseline (robustness variant)
#   cc_drift_hybrid_concave     hybrid, concave_stake baseline (main variant)
#
# The concave baseline u_i ∝ stake_i^gamma is read by the chain keeper from
# RISK_BUDGET_CONCAVE_GAMMA_PPM (ppm). gamma=0.5 (500000) is Square Root Stake
# Weight, the literature-standard anchor; override to run a sensitivity sweep.

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

# Concave-stake exponent for the concave_stake baseline (ppm). 500000 = sqrt.
export RISK_BUDGET_CONCAVE_GAMMA_PPM="${RISK_BUDGET_CONCAVE_GAMMA_PPM:-500000}"

SUITE_ROOT="article_rbhc_variant_a"
CFG_DIR="$REPO_ROOT/poc/cosmos/generated_configs/${SUITE_ROOT}"
RUNNER="$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py"
ARTIFACTS_ROOT="$REPO_ROOT/poc/cosmos/artifacts"
LOG_ROOT="$ARTIFACTS_ROOT/${SUITE_ROOT}/logs"
mkdir -p "$LOG_ROOT"

CELLS=(
  RBHC_cc_drift_baseline
  RBHC_cc_drift_signal
  RBHC_cc_drift_hybrid_capped
  RBHC_cc_drift_hybrid_concave
)

START_TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[cc-drift] start ${START_TS}, ${#CELLS[@]} cells x 3 seeds, gamma_ppm=${RISK_BUDGET_CONCAVE_GAMMA_PPM}"

# Always refresh: drop stale aggregates for these cells.
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  SUBDIR="$(sed -n 's/^[[:space:]]*artifacts_subdir:[[:space:]]*//p' "$CFG" | head -1 | tr -d '"'"'"' ' | tr -d '\r')"
  [[ -n "$SUBDIR" ]] && rm -f "$ARTIFACTS_ROOT/${SUBDIR}/aggregated_final_table.csv"
done

OK=0; FAIL=0
for cell in "${CELLS[@]}"; do
  CFG="$CFG_DIR/${cell}.yaml"
  LOG="$LOG_ROOT/${cell}_${START_TS}.log"
  if [[ ! -f "$CFG" ]]; then
    echo "[cc-drift] ${cell} CONFIG MISSING; skipping" >&2; FAIL=$((FAIL+1)); continue
  fi
  echo "[cc-drift] starting ${cell} -> ${LOG}"
  CELL_START=$(date +%s)
  if python3 "$RUNNER" "$CFG" > "$LOG" 2>&1; then
    echo "[cc-drift] ${cell} OK ($(( $(date +%s) - CELL_START ))s)"; OK=$((OK+1))
  else
    echo "[cc-drift] ${cell} FAILED after $(( $(date +%s) - CELL_START ))s — see ${LOG}" >&2; FAIL=$((FAIL+1))
  fi
done

echo "[cc-drift] end: ${OK} ok, ${FAIL} failed"
echo
echo "Artifacts root: $ARTIFACTS_ROOT/$SUITE_ROOT"
echo "Next: python3 papers/risk_budget_controller_2026/figures/scripts/make_drift_plots.py --cells cc --suite $SUITE_ROOT"
