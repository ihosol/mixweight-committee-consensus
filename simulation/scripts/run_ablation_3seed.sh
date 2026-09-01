#!/usr/bin/env bash
# run_ablation_3seed.sh
#
# Multi-seed (3 independent runs) signal-weight ablation, mirroring the
# canonical overnight_v5 headline protocol so that the ablation rows
# become directly comparable to the 3-seed canonical headline of
# Table 4 in Chapter 4. Closes the methodological gap where the
# previous ablation rows were single-seed and therefore not formally
# comparable to the 3-seed canonical row 7.98% / 31.82% / 20.89%.
#
# Each run pins all signal weight to one component at a time, with the
# headline regime (lambda_max=0.65, alpha_down=0.02) held fixed:
#   freshness-only:  w_F=1.0, w_G=0.0, w_S=0.0
#   gini-only:       w_F=0.0, w_G=1.0, w_S=0.0
#   split-only:      w_F=0.0, w_G=0.0, w_S=1.0
#
# Usage (from repo root):
#   bash poc/cosmos/scripts/run_ablation_3seed.sh
#
# Env overrides:
#   SEEDS="1,2,3"     # default; widen to "1,2,3,4,5" for a 5-seed run
#   FORCE=1           # ignore existing aggregated_final_table.csv and rerun
#
# Walltime: each ablation single-seed is ~30-50 min on a localnet of
# eighteen validators; the full 3 scenarios x 3 seeds therefore runs in
# 4.5-7.5h. Schedule for an overnight batch.
#
# Output artifacts:
#   poc/cosmos/artifacts/overnight_v5/ablation_freshness_only_burst/
#   poc/cosmos/artifacts/overnight_v5/ablation_gini_only_burst/
#   poc/cosmos/artifacts/overnight_v5/ablation_split_only_burst/
#
# Each artifact dir contains:
#   aggregated_final_table.csv  — mean +/- std across seeds, ready to
#                                  drop into FINAL_DISSERTATION_TABLE.csv
#   seed_<N>/                   — per-seed raw outputs (epoch_final_table,
#                                  epoch_draws, plots/)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$REPO_ROOT/poc/cosmos/scripts"
CONFIGS="$REPO_ROOT/poc/cosmos/generated_configs/final_package"
BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
ARTIFACTS_ROOT="$REPO_ROOT/poc/cosmos/artifacts"
ARTIFACTS_NS="overnight_v5"
LOG_DIR="/tmp/ablation_3seed_runs"

# ── Configuration ──────────────────────────────────────────────────────────────
SEEDS="${SEEDS:-1,2,3}"
FORCE="${FORCE:-0}"

# Controller knobs match the alphadown02 headline regime in run_overnight_full.sh
export ADAPTIVE_LAM_MAX=0.65
export ADAPTIVE_ALPHA_DOWN=0.02
# Between seeds the runner waits for every node port to become bindable again,
# with a 20 s default. After a localnet of twenty or more nodes stops, the sockets
# sit in TIME_WAIT past that window and the batch aborts although nothing is stuck.
export ORPHAN_CLEANUP_TIMEOUT_S="${ORPHAN_CLEANUP_TIMEOUT_S:-180}"
export POC_CHAIND="$BINARY"

# ── Helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; GRAY='\033[0;37m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
skip() { echo -e "${GRAY}[$(date '+%H:%M:%S')] SKIP:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL:${NC} $*" >&2; exit 1; }
section() { echo; echo -e "${BLUE}═══ $* ═══${NC}"; echo; }

# Kill any leftover chain/runner processes and clear stale tmp dirs.
cleanup_poc() {
    pkill -9 -f 'chain-five-threed'    >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun.py'          >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun_multiseed'   >/dev/null 2>&1 || true
    rm -rf /tmp/poc_epoch_* /tmp/poc-* >/dev/null 2>&1 || true
    sleep 3
}

# ── Pre-flight checks ──────────────────────────────────────────────────────────
section "Pre-flight checks"

[[ -x "$BINARY" ]] || fail "Binary not found: $BINARY
  Build with:
    export PATH=\"\$HOME/.local/go/bin:\$PATH\"
    cd poc/cosmos/chain53/chain-five-three
    go build -o build/chain-five-threed ./cmd/chain-five-threed"

python3 -c "import yaml, csv, statistics, pathlib" 2>/dev/null \
    || fail "Python3 missing deps. Run: pip3 install pyyaml"

"$BINARY" tx adaptivecommittee --help >/dev/null 2>&1 \
    || fail "Binary does not expose adaptivecommittee module. Rebuild required."

# Verify every required config is present.
REQUIRED_CONFIGS=(
    "FP_ablation_freshness_only_burst.yaml"
    "FP_ablation_gini_only_burst.yaml"
    "FP_ablation_split_only_burst.yaml"
)
for cfg in "${REQUIRED_CONFIGS[@]}"; do
    [[ -f "$CONFIGS/$cfg" ]] || fail "Missing config: $CONFIGS/$cfg"
done

[[ -x "$SCRIPTS/epochrun_multiseed.py" ]] \
    || fail "Multi-seed runner not found: $SCRIPTS/epochrun_multiseed.py"

if ss -tlnp 2>/dev/null | grep -q ":26680\b"; then
    warn "Port 26680 is in use. Kill leftover runs with: pkill -9 -f chain-five-threed && pkill -f epochrun"
    read -rp "Continue anyway? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || exit 1
fi

mkdir -p "$LOG_DIR" "$ARTIFACTS_ROOT/$ARTIFACTS_NS"

log "Binary        : $BINARY ($(stat -c '%y' "$BINARY" | cut -d. -f1))"
log "Seeds         : $SEEDS"
log "LAM_MAX       : $ADAPTIVE_LAM_MAX"
log "ALPHA_DOWN    : $ADAPTIVE_ALPHA_DOWN"
log "Artifacts ns  : $ARTIFACTS_ROOT/$ARTIFACTS_NS/"
log "Logs          : $LOG_DIR"
log "Force re-run  : $FORCE"

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

# ── Run function ───────────────────────────────────────────────────────────────
# $1: scenario name (under overnight_v5/, e.g. "ablation_freshness_only_burst")
# $2: source config YAML basename under CONFIGS
# $3: freshness weight
# $4: gini weight
# $5: split weight
run_ablation() {
    local name="$1"
    local config_basename="$2"
    local fw="$3"
    local gw="$4"
    local sw="$5"

    local config_path="$CONFIGS/$config_basename"
    local subdir="$ARTIFACTS_NS/$name"
    local logfile="$LOG_DIR/${name}.log"
    local agg_marker="$ARTIFACTS_ROOT/$subdir/aggregated_final_table.csv"

    [[ -f "$config_path" ]] || fail "Config not found: $config_path"

    if [[ "$FORCE" != "1" && -f "$agg_marker" ]]; then
        skip "$name — already has aggregated_final_table.csv (FORCE=1 to override)"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi

    log "── $name"
    log "   config   : $config_basename"
    log "   weights  : w_F=$fw  w_G=$gw  w_S=$sw"
    log "   seeds    : $SEEDS"
    log "   out      : $ARTIFACTS_ROOT/$subdir/"

    cleanup_poc

    local start_ts
    start_ts=$(date +%s)

    set +e
    ADAPTIVE_FRESHNESS_W="$fw" \
    ADAPTIVE_GINI_W="$gw" \
    ADAPTIVE_SPLIT_W="$sw" \
    python3 "$SCRIPTS/epochrun_multiseed.py" "$config_path" \
        --seeds "$SEEDS" \
        --artifacts-subdir "$subdir" \
        2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local elapsed=$(( $(date +%s) - start_ts ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ $exit_code -ne 0 ]]; then
        warn "$name exited with code $exit_code after ${mins}m${secs}s. See $logfile"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        return 1
    fi

    if [[ ! -f "$agg_marker" ]]; then
        warn "$name finished but $agg_marker is missing. See $logfile"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        return 1
    fi

    log "✓ $name (${mins}m${secs}s)"
    TOTAL_RUN=$((TOTAL_RUN + 1))
    echo
}

# ── Three ablation runs ───────────────────────────────────────────────────────
section "Multi-seed signal ablation (3 scenarios x ${SEEDS} seeds)"

run_ablation "ablation_freshness_only_burst" \
    "FP_ablation_freshness_only_burst.yaml" \
    "1.0" "0.0" "0.0" || true

run_ablation "ablation_gini_only_burst" \
    "FP_ablation_gini_only_burst.yaml" \
    "0.0" "1.0" "0.0" || true

run_ablation "ablation_split_only_burst" \
    "FP_ablation_split_only_burst.yaml" \
    "0.0" "0.0" "1.0" || true

# ── Comparison table ──────────────────────────────────────────────────────────
section "Comparison: 3-seed ablation vs 3-seed canonical headline"

python3 - <<'PYEOF'
import csv
from pathlib import Path

ROOT = Path("poc/cosmos/artifacts/overnight_v5")

# Canonical 3-seed reference + three 3-seed ablation runs from this batch.
candidates = [
    ("pub_alphadown02_lammax065_burst", "canonical multi-signal"),
    ("ablation_freshness_only_burst",   "freshness-only (w_F=1)"),
    ("ablation_gini_only_burst",        "Gini-only (w_G=1)"),
    ("ablation_split_only_burst",       "split-only (w_S=1)"),
]

HDR = (f"{'scenario':<40}  {'red_full %':>12}  {'red_1ep %':>10}  "
       f"{'lam_peak ppm':>13}  {'chern>=1/2 %':>13}")
SEP = "=" * len(HDR)
print(HDR)
print(SEP)

def fmt(val_key, std_key, row):
    val = row.get(val_key, "—")
    std = row.get(std_key, "")
    if std and std not in ("0.000", "0"):
        return f"{val:>7}±{std:<4}"
    return f"{val:>12}"

for name, label in candidates:
    f = ROOT / name / "aggregated_final_table.csv"
    if not f.exists():
        print(f"  {label:<40}  (no aggregated_final_table.csv yet)")
        continue
    with f.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"  {label:<40}  (empty)")
        continue
    row = rows[0]
    red_full = fmt("reduction_vs_baseline_full_pct",
                   "reduction_vs_baseline_full_pct_std", row)
    red_1ep  = fmt("reduction_vs_baseline_1ep_pct",
                   "reduction_vs_baseline_1ep_pct_std", row)
    lpk      = row.get("post_lambda_peak_ppm", "—")
    r12      = fmt("chernoff_bound_reduction_ge_1_2_pct",
                   "chernoff_bound_reduction_ge_1_2_pct_std", row)
    print(f"  {label:<40}  {red_full}  {red_1ep}  {lpk:>13}  {r12}")

print(SEP)
print("  red_full %  = reduction_vs_baseline_full_pct  (mean +/- std across seeds)")
print("  red_1ep %   = reduction_vs_baseline_1ep_pct   (mean +/- std across seeds)")
print("  chern>=1/2 %= chernoff_bound_reduction_ge_1_2_pct (majority-capture proxy)")
PYEOF

echo
section "Summary"
log "Runs completed : $TOTAL_RUN"
log "Runs skipped   : $TOTAL_SKIP"
log "Runs failed    : $TOTAL_FAIL"
log "Artifacts root : $ARTIFACTS_ROOT/$ARTIFACTS_NS/"
log "Logs           : $LOG_DIR/"

if [[ $TOTAL_FAIL -gt 0 ]]; then
    fail "$TOTAL_FAIL scenario(s) failed. Inspect logs in $LOG_DIR/"
fi

log "Next step: re-key the ablation rows in"
log "  poc/cosmos/dissertation_final/tables/FINAL_DISSERTATION_TABLE.csv"
log "to point to $ARTIFACTS_NS/ablation_*_burst and bump n_seeds to 3 with"
log "the mean+/-std values from each aggregated_final_table.csv."
