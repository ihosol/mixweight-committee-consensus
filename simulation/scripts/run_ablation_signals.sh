#!/usr/bin/env bash
# run_ablation_signals.sh
# Three-signal ablation: isolates the contribution of freshness, Gini, and
# split-pressure signals to the adaptive controller. Each run pins all weight
# to a single signal (others = 0) while keeping LAM_MAX=0.65, ALPHA_DOWN=0.02
# (the overnight_v5 best-reduction regime), so results are directly comparable
# to the canonical headline run.
#
# Usage (from repo root):
#   bash poc/cosmos/scripts/run_ablation_signals.sh
#
# Output artifacts:
#   poc/cosmos/artifacts/ablation_freshness_only_burst/
#   poc/cosmos/artifacts/ablation_gini_only_burst/
#   poc/cosmos/artifacts/ablation_split_only_burst/
#
# Each artifact dir contains the canonical layout produced by epochrun.py:
#   results/epoch_final_table_latest.csv
#   results/epoch_summary_<RUN_ID>.csv
#   results/epoch_draws_<RUN_ID>.csv
#   plots/*.png

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$REPO_ROOT/poc/cosmos/scripts"
CONFIGS="$REPO_ROOT/poc/cosmos/generated_configs/final_package"
BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
LOG_DIR="/tmp/ablation_signals_runs"

# ── Common controller knobs (matches FP_A1_alphadown02_lammax065_burst) ───────
export ADAPTIVE_LAM_MAX=0.65
export ADAPTIVE_ALPHA_DOWN=0.02
export POC_CHAIND="$BINARY"

# ── Helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL:${NC} $*" >&2; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────────
log "=== Pre-flight checks ==="

[[ -x "$BINARY" ]] || fail "Binary not found: $BINARY
  Build with: cd poc/cosmos/chain53/chain-five-three && make install"

python3 -c "import yaml, csv, pathlib" 2>/dev/null || fail "Python3 missing yaml/csv/pathlib. Run: pip3 install pyyaml"

for cfg in \
    FP_ablation_freshness_only_burst.yaml \
    FP_ablation_gini_only_burst.yaml \
    FP_ablation_split_only_burst.yaml; do
    [[ -f "$CONFIGS/$cfg" ]] || fail "Config not found: $CONFIGS/$cfg"
done

# Warn if ports are already in use (leftover from a previous crashed run)
if ss -tlnp 2>/dev/null | grep -q ":26680\b"; then
    warn "Port 26680 is already in use — a previous run may still be alive."
    warn "Kill it with: pkill -9 -f chain-five-threed && pkill -f epochrun.py"
    read -rp "Continue anyway? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || exit 1
fi

mkdir -p "$LOG_DIR"
log "Binary  : $BINARY"
log "LAM_MAX : $ADAPTIVE_LAM_MAX  ALPHA_DOWN: $ADAPTIVE_ALPHA_DOWN"
log "Logs    : $LOG_DIR/"
echo

# ── Run function ───────────────────────────────────────────────────────────────
# Args: artifact_name, config_name, freshness_w, gini_w, split_w
run_sim() {
    local name="$1"
    local config="$2"
    local fw="$3"
    local gw="$4"
    local sw="$5"
    local logfile="$LOG_DIR/${name}.log"
    local artifact_dir="$REPO_ROOT/poc/cosmos/artifacts/${name}"

    log "=== Starting: $name ==="
    log "Config       : $config"
    log "Signal weights: w_F=$fw  w_G=$gw  w_S=$sw"
    log "Output       : $artifact_dir"
    log "Log          : $logfile"
    echo

    local start_ts
    start_ts=$(date +%s)

    ADAPTIVE_FRESHNESS_W="$fw" \
    ADAPTIVE_GINI_W="$gw" \
    ADAPTIVE_SPLIT_W="$sw" \
    python3 "$SCRIPTS/epochrun.py" "$config" 2>&1 | tee "$logfile"

    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - start_ts ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ $exit_code -ne 0 ]]; then
        fail "$name exited with code $exit_code after ${mins}m${secs}s. See $logfile"
    fi

    # Verify final table exists
    local final_csv="$artifact_dir/results/epoch_final_table_latest.csv"
    if [[ ! -f "$final_csv" ]]; then
        fail "Simulation completed but $final_csv is missing."
    fi

    log "=== Done: $name (${mins}m${secs}s) ==="
    echo
}

# ── Three ablation runs ───────────────────────────────────────────────────────
run_sim "ablation_freshness_only_burst" \
        "$CONFIGS/FP_ablation_freshness_only_burst.yaml" \
        "1.0" "0.0" "0.0"

run_sim "ablation_gini_only_burst" \
        "$CONFIGS/FP_ablation_gini_only_burst.yaml" \
        "0.0" "1.0" "0.0"

run_sim "ablation_split_only_burst" \
        "$CONFIGS/FP_ablation_split_only_burst.yaml" \
        "0.0" "0.0" "1.0"

# ── Comparison table (vs canonical alphadown02 baseline) ──────────────────────
log "=== Comparison: ablation runs vs canonical alphadown02 baseline ==="
echo

python3 - <<'PYEOF'
import csv
from pathlib import Path

ROOT = Path("poc/cosmos/artifacts")

# Canonical reference plus the three ablation runs.
candidates = [
    "v2_pub_alphadown02_lammax065_burst",   # full controller (w_F=0.75, w_G=0.10, w_S=0.15)
    "ablation_freshness_only_burst",         # w_F=1.0
    "ablation_gini_only_burst",              # w_G=1.0
    "ablation_split_only_burst",             # w_S=1.0
]

HDR = (f"{'artifact':<40}  {'red_full%':>9}  {'weight':>8}  {'seats':>7}  "
       f"{'lam_peak':>9}  {'half_life':>9}  {'chern12_red%':>12}")
SEP = "=" * len(HDR)
print(HDR)
print(SEP)

for name in candidates:
    d = ROOT / name
    f = d / "results/epoch_final_table_latest.csv"
    if not f.exists():
        print(f"  {name:<40}  (no final_table_latest.csv yet)")
        continue
    with f.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"  {name:<40}  (empty)")
        continue
    row = rows[0]
    red  = row.get("reduction_vs_baseline_full_pct", "—")
    wt   = row.get("post_attacker_weight_mean", "—")
    seat = row.get("post_attacker_seat_mean", "—")
    lpk  = row.get("post_lambda_peak_ppm", "—")
    hl   = row.get("post_lambda_half_life_draws", "—")
    r12  = row.get("chernoff_bound_reduction_ge_1_2_pct", "—")
    print(f"  {name:<40}  {red:>9}  {wt:>8}  {seat:>7}  "
          f"{lpk:>9}  {hl:>9}  {r12:>12}")

print(SEP)
print("  red_full%   = reduction_vs_baseline_full_pct (post-attack window)")
print("  chern12_red% = Chernoff certificate reduction at theta=1/2")
PYEOF

echo
log "All done. Logs in $LOG_DIR/"
log "Artifacts:"
log "  poc/cosmos/artifacts/ablation_freshness_only_burst/"
log "  poc/cosmos/artifacts/ablation_gini_only_burst/"
log "  poc/cosmos/artifacts/ablation_split_only_burst/"
