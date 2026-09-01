#!/usr/bin/env bash
# Audit follow-up batch (2026-07-22), sequential, resume-safe.
#
#   1. fairness trickle pair with FRESH seeds 4,5,6 (true replication)
#        -> overnight_v5/fairness_default_trickle_s456
#        -> overnight_v5/fairness_tuned_trickle_s456
#   2. third seed (seed 3) for the four cc_drift cells
#        -> article_rbhc_variant_a/<cell>/seed_3
#   3. largeN N=36 seeds 2,3
#        -> largeN/LN_A1_N36_c9_k12_b033_burst/seed_2, seed_3
#
# Existing canonical artifacts are never overwritten.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$REPO_ROOT/poc/cosmos/scripts"
CONFIGS="$REPO_ROOT/poc/cosmos/generated_configs"
BINARY="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
ARTIFACTS="$REPO_ROOT/poc/cosmos/artifacts"
LOG_DIR="$ARTIFACTS/audit_followup_2026-07-22_logs"
mkdir -p "$LOG_DIR"

export POC_CHAIND="$BINARY"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup_poc() {
    pkill -9 -f 'chain-five-threed' >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun'          >/dev/null 2>&1 || true
    rm -rf /tmp/poc_epoch_* /tmp/poc-* >/dev/null 2>&1 || true
    sleep 5
}

run_step() {
    local name="$1"; shift
    local marker="$1"; shift
    if [[ -f "$marker" ]]; then
        log "SKIP $name (є $marker)"
        return 0
    fi
    log "START $name"
    cleanup_poc
    local t0=$(date +%s)
    if "$@" > "$LOG_DIR/${name}.log" 2>&1; then
        log "OK   $name ($(( ( $(date +%s) - t0 ) / 60 )) хв)"
    else
        log "FAIL $name — див. $LOG_DIR/${name}.log (продовжую далі)"
    fi
}

log "=== 1. Fairness trickle, seeds 4,5,6 ==="

run_step fairness_default_trickle_s456 \
    "$ARTIFACTS/overnight_v5/fairness_default_trickle_s456/aggregated_final_table.csv" \
    python3 "$SCRIPTS/epochrun_multiseed.py" \
        "$CONFIGS/final_package/FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml" \
        --seeds 4,5,6 \
        --artifacts-subdir overnight_v5/fairness_default_trickle_s456

export ADAPTIVE_LAM_MAX=0.65
run_step fairness_tuned_trickle_s456 \
    "$ARTIFACTS/overnight_v5/fairness_tuned_trickle_s456/aggregated_final_table.csv" \
    python3 "$SCRIPTS/epochrun_multiseed.py" \
        "$CONFIGS/final_package/FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml" \
        --seeds 4,5,6 \
        --artifacts-subdir overnight_v5/fairness_tuned_trickle_s456
unset ADAPTIVE_LAM_MAX

log "=== 2. cc_drift, seed 3 для чотирьох комірок ==="

export RISK_BUDGET_CONCAVE_GAMMA_PPM="${RISK_BUDGET_CONCAVE_GAMMA_PPM:-500000}"
for cell in RBHC_cc_drift_baseline RBHC_cc_drift_signal RBHC_cc_drift_hybrid_capped RBHC_cc_drift_hybrid_concave; do
    subdir="$(sed -n 's/^[[:space:]]*artifacts_subdir:[[:space:]]*//p' "$CONFIGS/article_rbhc_variant_a/${cell}.yaml" | head -1 | tr -d ' ')"
    run_step "${cell}_seed3" \
        "$ARTIFACTS/${subdir}/seed_3/results/epoch_final_table_latest.csv" \
        python3 "$SCRIPTS/epochrun_multiseed.py" \
            "$CONFIGS/article_rbhc_variant_a/${cell}.yaml" \
            --seeds 3 \
            --artifacts-subdir "${subdir}/seed_3"
done

log "=== 3. largeN N=36, seeds 2,3 ==="

for s in 2 3; do
    run_step "LN_A1_N36_seed${s}" \
        "$ARTIFACTS/largeN/LN_A1_N36_c9_k12_b033_burst/seed_${s}/results/epoch_final_table_latest.csv" \
        python3 "$SCRIPTS/epochrun_multiseed.py" \
            "$CONFIGS/largeN/LN_A1_N36_seed1.yaml" \
            --seeds "$s" \
            --artifacts-subdir "largeN/LN_A1_N36_c9_k12_b033_burst/seed_${s}"
done

cleanup_poc
log "=== БАТЧ ЗАВЕРШЕНО ==="
