#!/usr/bin/env bash
# Reproduce the RBHC committee-capture red-flag results (theta=1/3, eps=0.5).
#
# Pipeline:
#   1) build the chain binary that carries the WITH-REPLACEMENT committee sampler
#      (each of the m seats is an independent draw in proportion to the mixed
#      weights, so the coalition seat count is exactly Binomial(m, p_t(lambda))
#      and the binomial certificate is the exact law of the realised draw);
#   2) run the 4-cell x 3-seed red-flag suite;
#   3) regenerate the paper figures and the cc-drift summary table.
#
# Requirements: Go 1.24+ (set GO_BIN if `go` is not on PATH), Python 3 + matplotlib.
# Resource note: each cell runs a 24-validator localnet x 31 epochs x 16 draws for
# each of 3 seeds, for 4 controller cells; budget a few cores and a few GB RAM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

# --- 0. locate Go --------------------------------------------------------------
GO_BIN="${GO_BIN:-$(command -v go || true)}"
if [[ -z "$GO_BIN" && -x "$HOME/.local/go/bin/go" ]]; then
  GO_BIN="$HOME/.local/go/bin/go"
fi
if [[ -z "$GO_BIN" ]] || ! "$GO_BIN" version >/dev/null 2>&1; then
  echo "FATAL: Go toolchain not found. Install Go 1.24+ or set GO_BIN=/path/to/go" >&2
  exit 2
fi
echo "[reproduce] using Go: $("$GO_BIN" version)"

# --- 1. build the chain binary (with-replacement sampler) ----------------------
CHAIN_DIR="$REPO_ROOT/poc/cosmos/chain53/chain-five-three"
echo "[reproduce] building chain-five-threed ..."
( cd "$CHAIN_DIR" && "$GO_BIN" build -o build/chain-five-threed ./cmd/chain-five-threed )
export POC_CHAIND="$CHAIN_DIR/build/chain-five-threed"
echo "[reproduce] binary: $POC_CHAIND"

# --- 2. run the red-flag suite (theta=1/3, eps=0.5, 4 cells x 3 seeds) ----------
echo "[reproduce] running red-flag suite ..."
bash "$REPO_ROOT/poc/cosmos/scripts/run_rbhc_redflag_bft.sh"

# --- 3. regenerate figures + summary table -------------------------------------
SUITE=article_rbhc_redflag_bft
EN_FIG="$REPO_ROOT/papers/risk_budget_controller_2026/figures"
UA_FIG="$REPO_ROOT/papers/risk_budget_controller_2026_ua/figures"
echo "[reproduce] regenerating figures ..."
python3 "$REPO_ROOT/poc/cosmos/scripts/make_redflag_figure.py" --suite "$SUITE" --theta 0.3333 --eps 0.5
python3 "$REPO_ROOT/papers/risk_budget_controller_2026/figures/scripts/make_drift_plots.py" --cells cc --suite "$SUITE" --eps 0.5
# mirror the cc-drift figures into the UA paper figures dir
cp -f "$EN_FIG/fig_cc_drift_gini_lambda.png" "$UA_FIG/" 2>/dev/null || true
cp -f "$EN_FIG/fig_cc_drift_seat_share.png"  "$UA_FIG/" 2>/dev/null || true

echo "[reproduce] done."
echo "  UA figures      : $UA_FIG/{fig_redflag_bft.png,fig_headroom_gini.png,fig_cc_drift_gini_lambda.png,fig_cc_drift_seat_share.png}"
echo "  per-epoch CSV   : $UA_FIG/fig_redflag_bft.csv"
echo "  summary table   : $EN_FIG/cc_drift_summary_table.csv"
echo
echo "Next: refresh the numbers in papers/risk_budget_controller_2026_ua/sections/05_results.tex"
echo "      (tab:rbhc-cc-summary, tab:rbhc-cc-tightness, realised-capture prose) from the CSVs above."
echo "      Under the with-replacement sampler the realised capture now MATCHES the binomial"
echo "      certificate, so the previous 'conservative bound' wording becomes 'matches'."
