#!/usr/bin/env bash
# run_overnight_full.sh
#
# Single comprehensive overnight runner. Executes every scenario needed to
# refresh the dissertation's empirical claims in one go, sequentially, with
# resume-safe skipping so the batch can be interrupted and restarted without
# losing progress.
#
# All artifacts are isolated under:
#   poc/cosmos/artifacts/overnight_v5/<scenario>/
# so this run does not collide with any prior v3_/v4_/pub_ artifacts.
#
# Phases (default: all enabled):
#
#   A. Security headline (6 scenarios, 3 seeds each, ~13-14h)
#      overnight_v5/pub_default_{burst,trickle}
#      overnight_v5/pub_tuned_{burst,trickle}                (λ_max=0.65)
#      overnight_v5/pub_alphadown02_lammax065_{burst,trickle} (α_down=0.02, λ_max=0.65)
#
#   B. β sweep — dose-response curve for attacker stake share (4 scenarios, 3 seeds, ~9h)
#      overnight_v5/pub_beta020_burst     (β=0.20)
#      overnight_v5/pub_beta025_burst     (β=0.25)
#      overnight_v5/pub_beta040_burst     (β=0.40)
#      overnight_v5/pub_beta050_burst     (β=0.50)
#      NB: β=0.33 is covered by overnight_v5/pub_default_burst in phase A.
#
#   C. Age-farming threat model (2 scenarios, 3 seeds, ~4-5h)
#      overnight_v5/agefarming_default_burst
#      overnight_v5/agefarming_tuned_burst                   (λ_max=0.65)
#
#   D. Fairness (honest newcomer) (4 scenarios, 3 seeds, ~9h)
#      overnight_v5/fairness_default_{burst,trickle}
#      overnight_v5/fairness_tuned_{burst,trickle}           (λ_max=0.65)
#
# Total walltime: ≈ 35-37h (plan for 3 consecutive nights, or use PHASES to slice).
#
# Resume behavior:
#   A scenario is skipped if its aggregated_final_table.csv already exists
#   under overnight_v5/<scenario>/. Set FORCE=1 to re-run everything.
#
# Usage (from repo root):
#
#   bash poc/cosmos/scripts/run_overnight_full.sh                    # all phases
#   PHASES="A"        bash poc/cosmos/scripts/run_overnight_full.sh  # security only
#   PHASES="A,B"      bash poc/cosmos/scripts/run_overnight_full.sh  # security + β-sweep
#   PHASES="B,C"      bash poc/cosmos/scripts/run_overnight_full.sh  # β-sweep + age-farming
#   SEEDS="1,2,3,4,5" bash poc/cosmos/scripts/run_overnight_full.sh  # wider seed budget
#   FORCE=1           bash poc/cosmos/scripts/run_overnight_full.sh  # ignore existing artifacts

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$REPO_ROOT/poc/cosmos/scripts"
CONFIGS="$REPO_ROOT/poc/cosmos/generated_configs/final_package"
BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
ARTIFACTS_ROOT="$REPO_ROOT/poc/cosmos/artifacts"
ARTIFACTS_NS="overnight_v5"
LOG_DIR="/tmp/${ARTIFACTS_NS}_runs"

# ── Configuration ──────────────────────────────────────────────────────────────
PHASES="${PHASES:-A,B,C,D}"
SEEDS="${SEEDS:-1,2,3}"
FORCE="${FORCE:-0}"

export POC_CHAIND="$BINARY"

# ── Helpers ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; GRAY='\033[0;37m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
skip() { echo -e "${GRAY}[$(date '+%H:%M:%S')] SKIP:${NC} $*"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL:${NC} $*" >&2; exit 1; }
section() { echo; echo -e "${BLUE}═══ $* ═══${NC}"; echo; }

has_phase() {
    [[ ",$PHASES," == *",$1,"* ]]
}

# Kill any leftover chain/epochrun processes and clear stale tmp dirs.
# Always called before each scenario to guarantee a clean slate.
cleanup_poc() {
    pkill -9 -f 'chain-five-threed'    >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun.py'           >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun_multiseed'    >/dev/null 2>&1 || true
    rm -rf /tmp/poc_epoch_* /tmp/poc-*  >/dev/null 2>&1 || true
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

if ss -tlnp 2>/dev/null | grep -q ":26680\b"; then
    warn "Port 26680 is in use. Kill leftover runs with: pkill -9 -f chain-five-threed && pkill -f epochrun"
    read -rp "Continue anyway? [y/N] " ans
    [[ "${ans,,}" == "y" ]] || exit 1
fi

# Verify every config referenced below is present.
REQUIRED_CONFIGS=(
    "FP_A1_main_12h_c9_k6_b033_burst.yaml"
    "FP_A2_companion_12h_c9_k6_b033_trickle.yaml"
    "FP_A1_alphadown02_lammax065_burst.yaml"
    "FP_A2_alphadown02_lammax065_trickle.yaml"
    "FP_A1_beta020_12h_c9_k6_b020_burst.yaml"
    "FP_A1_beta025_12h_c9_k6_b025_burst.yaml"
    "FP_A1_beta040_12h_c9_k6_b040_burst.yaml"
    "FP_A1_beta050_12h_c9_k6_b050_burst.yaml"
    "FP_A4_agefarming_12h_c9_k6_b033_burst.yaml"
    "FP_H1_honest_newcomer_12h_c9_k1_b010_burst.yaml"
    "FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml"
)
for cfg in "${REQUIRED_CONFIGS[@]}"; do
    [[ -f "$CONFIGS/$cfg" ]] || fail "Missing config: $CONFIGS/$cfg"
done

mkdir -p "$LOG_DIR" "$ARTIFACTS_ROOT/$ARTIFACTS_NS"

log "Binary        : $BINARY ($(stat -c '%y' "$BINARY" | cut -d. -f1))"
log "Phases        : $PHASES"
log "Seeds         : $SEEDS"
log "Artifacts ns  : $ARTIFACTS_ROOT/$ARTIFACTS_NS/"
log "Logs          : $LOG_DIR"
log "Force re-run  : $FORCE"

TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

# ── Run function ───────────────────────────────────────────────────────────────
# $1: scenario name (under overnight_v5/, e.g. "pub_default_burst")
# $2: source config YAML basename under CONFIGS
# $3: seeds (comma-separated)
# env: extra env overrides via caller (export VAR=val before call; unset after)
run_scenario() {
    local name="$1"
    local config_basename="$2"
    local seeds="$3"
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
    log "   config  : $config_basename"
    log "   seeds   : $seeds"
    log "   out     : $ARTIFACTS_ROOT/$subdir/"

    # Hard reset: kill any orphaned chain/runner processes from a prior
    # scenario or seed before binding ports for this one.
    cleanup_poc

    local start_ts
    start_ts=$(date +%s)

    set +e
    python3 "$SCRIPTS/epochrun_multiseed.py" "$config_path" \
        --seeds "$seeds" \
        --artifacts-subdir "$subdir" \
        2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local elapsed=$(( $(date +%s) - start_ts ))
    local mins=$(( elapsed / 60 ))

    if [[ $exit_code -ne 0 ]]; then
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        warn "$name failed (rc=$exit_code) after ${mins}m. See $logfile. Continuing with remaining scenarios."
        return 0
    fi
    TOTAL_RUN=$((TOTAL_RUN + 1))
    log "   done    : ${mins}m"
}

# ── Phase A: Security headline ─────────────────────────────────────────────────
if has_phase A; then
    section "Phase A: Security headline (seeds=$SEEDS)"

    # A.1: default knobs
    run_scenario "pub_default_burst" \
        "FP_A1_main_12h_c9_k6_b033_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_default_trickle" \
        "FP_A2_companion_12h_c9_k6_b033_trickle.yaml" \
        "$SEEDS"

    # A.2: tuned λ_max=0.65
    export ADAPTIVE_LAM_MAX=0.65
    run_scenario "pub_tuned_burst" \
        "FP_A1_main_12h_c9_k6_b033_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_tuned_trickle" \
        "FP_A2_companion_12h_c9_k6_b033_trickle.yaml" \
        "$SEEDS"
    unset ADAPTIVE_LAM_MAX

    # A.3: αdown=0.02 + λ_max=0.65
    # Both knobs are passed via env vars (the Go keeper reads ADAPTIVE_ALPHA_DOWN
    # and ADAPTIVE_LAM_MAX in defaultAdaptiveControllerKnobs). The dedicated YAML
    # files have no payload differing from FP_A1_main / FP_A2_companion — they
    # only exist to give this sub-phase a distinct name for log/artifact paths.
    export ADAPTIVE_ALPHA_DOWN=0.02
    export ADAPTIVE_LAM_MAX=0.65
    run_scenario "pub_alphadown02_lammax065_burst" \
        "FP_A1_alphadown02_lammax065_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_alphadown02_lammax065_trickle" \
        "FP_A2_alphadown02_lammax065_trickle.yaml" \
        "$SEEDS"
    unset ADAPTIVE_ALPHA_DOWN
    unset ADAPTIVE_LAM_MAX
fi

# ── Phase B: β sweep ───────────────────────────────────────────────────────────
if has_phase B; then
    section "Phase B: β sweep (seeds=$SEEDS)"

    # All four β points under default knobs for a clean dose-response curve.
    # β=0.33 is the default_burst scenario in Phase A (not repeated here).
    run_scenario "pub_beta020_burst" \
        "FP_A1_beta020_12h_c9_k6_b020_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_beta025_burst" \
        "FP_A1_beta025_12h_c9_k6_b025_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_beta040_burst" \
        "FP_A1_beta040_12h_c9_k6_b040_burst.yaml" \
        "$SEEDS"
    run_scenario "pub_beta050_burst" \
        "FP_A1_beta050_12h_c9_k6_b050_burst.yaml" \
        "$SEEDS"
fi

# ── Phase C: Age-farming ───────────────────────────────────────────────────────
if has_phase C; then
    section "Phase C: Age-farming threat model (seeds=$SEEDS)"

    run_scenario "agefarming_default_burst" \
        "FP_A4_agefarming_12h_c9_k6_b033_burst.yaml" \
        "$SEEDS"

    export ADAPTIVE_LAM_MAX=0.65
    run_scenario "agefarming_tuned_burst" \
        "FP_A4_agefarming_12h_c9_k6_b033_burst.yaml" \
        "$SEEDS"
    unset ADAPTIVE_LAM_MAX
fi

# ── Phase D: Fairness (honest newcomer) ────────────────────────────────────────
if has_phase D; then
    section "Phase D: Fairness — honest newcomer (seeds=$SEEDS)"

    run_scenario "fairness_default_burst" \
        "FP_H1_honest_newcomer_12h_c9_k1_b010_burst.yaml" \
        "$SEEDS"
    run_scenario "fairness_default_trickle" \
        "FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml" \
        "$SEEDS"

    export ADAPTIVE_LAM_MAX=0.65
    run_scenario "fairness_tuned_burst" \
        "FP_H1_honest_newcomer_12h_c9_k1_b010_burst.yaml" \
        "$SEEDS"
    run_scenario "fairness_tuned_trickle" \
        "FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml" \
        "$SEEDS"
    unset ADAPTIVE_LAM_MAX
fi

# ── Post-run summary ───────────────────────────────────────────────────────────
section "Summary"

log "Scenarios run   : $TOTAL_RUN"
log "Scenarios skip  : $TOTAL_SKIP"
log "Scenarios fail  : $TOTAL_FAIL"

python3 - <<PYEOF
import csv
from pathlib import Path

ROOT = Path("${ARTIFACTS_ROOT}/${ARTIFACTS_NS}")
if not ROOT.exists():
    print("[summary] no artifacts dir yet")
    raise SystemExit(0)

dirs = sorted(d for d in ROOT.iterdir() if d.is_dir())

HDR = f"{'scenario':<40}  {'red_full%':>14}  {'red_1ep%':>10}  {'lam_peak':>9}  {'chern12%':>10}  {'seeds':>6}"
SEP = "=" * len(HDR)
print(HDR); print(SEP)

for d in dirs:
    aggr = d / "aggregated_final_table.csv"
    final = d / "results" / "epoch_final_table_latest.csv"
    if aggr.exists():
        with aggr.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[0]
        seeds = row.get("n_seeds", "?")
        red = row.get("reduction_vs_baseline_full_pct_mean",
                      row.get("reduction_vs_baseline_full_pct", "—"))
        red_std = row.get("reduction_vs_baseline_full_pct_std", "")
        r1ep = row.get("reduction_vs_baseline_1ep_pct_mean",
                       row.get("reduction_vs_baseline_1ep_pct", "—"))
        lpk = row.get("post_lambda_peak_ppm_mean",
                      row.get("post_lambda_peak_ppm", "—"))
        r12 = row.get("chernoff_bound_reduction_ge_1_2_pct_mean",
                      row.get("chernoff_bound_reduction_ge_1_2_pct", "—"))
        try:
            red_str = f"{float(red):.3f}" + (f"±{float(red_std):.2f}" if red_std else "")
        except (ValueError, TypeError):
            red_str = str(red)
        print(f"  {d.name:<38}  {red_str:>14}  {r1ep:>10.8}  {lpk:>9.9}  {r12:>10.8}  {seeds:>6}")
    elif final.exists():
        with final.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        row = rows[0]
        red = row.get("reduction_vs_baseline_full_pct", "—")
        r1ep = row.get("reduction_vs_baseline_1ep_pct", "—")
        lpk = row.get("post_lambda_peak_ppm", "—")
        r12 = row.get("chernoff_bound_reduction_ge_1_2_pct", "—")
        print(f"  {d.name:<38}  {red:>14}  {r1ep:>10.8}  {lpk:>9.9}  {r12:>10.8}  {'1':>6}")
    else:
        print(f"  {d.name:<38}  (no final table)")

print(SEP)
PYEOF

log "All scenarios in $PHASES attempted. Detailed logs in $LOG_DIR/"
log "Next steps after the full overnight completes:"
log "  1. Review the comparison table above for sanity"
log "  2. python3 poc/cosmos/scripts/build_dissertation_summary.py"
log "  3. Update Ch4 §4.3 (β sweep) and Ch3 §3.6 (age-farming) with new numbers"

if [[ $TOTAL_FAIL -gt 0 ]]; then
    warn "$TOTAL_FAIL scenarios failed. Review logs, fix, and re-run — skipped scenarios will not repeat."
    exit 1
fi
