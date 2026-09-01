#!/usr/bin/env bash
# Resume of the audit follow-up batch (2026-07-23) after the 2026-07-22 run was stopped.
#
#   1. fairness_default_trickle_s456: only seed 6 (seeds 4,5 already on disk),
#      then manual aggregation of seeds 4+5+6 into aggregated_final_table.csv.
#   2. re-run the original orchestrator: fairness steps are skipped by markers,
#      cc_drift seed_3 cells and largeN N=36 seeds 2,3 run as planned.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$REPO_ROOT/poc/cosmos/scripts"
CONFIGS="$REPO_ROOT/poc/cosmos/generated_configs"
ARTIFACTS="$REPO_ROOT/poc/cosmos/artifacts"
LOG_DIR="$ARTIFACTS/audit_followup_2026-07-22_logs"
BASE="$ARTIFACTS/overnight_v5/fairness_default_trickle_s456"
mkdir -p "$LOG_DIR"

export POC_CHAIND="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup_poc() {
    pkill -9 -f 'chain-five-threed' >/dev/null 2>&1 || true
    pkill -9 -f 'epochrun' >/dev/null 2>&1 || true
    rm -rf /tmp/poc_epoch_* /tmp/poc-* >/dev/null 2>&1 || true
    sleep 10
}

if [[ ! -f "$BASE/seed_6/results/epoch_final_table_latest.csv" ]]; then
    log "START fairness_default_trickle_s456 seed 6"
    cleanup_poc
    python3 "$SCRIPTS/epochrun_multiseed.py" \
        "$CONFIGS/final_package/FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml" \
        --seeds 6 \
        --artifacts-subdir overnight_v5/fairness_default_trickle_s456/seed_6 \
        > "$LOG_DIR/fairness_default_trickle_s456_seed6_resume.log" 2>&1 \
        && log "OK   seed 6" || log "FAIL seed 6 — див. лог"
else
    log "SKIP seed 6 (вже є)"
fi

if [[ -f "$BASE/seed_6/results/epoch_final_table_latest.csv" && ! -f "$BASE/aggregated_final_table.csv" ]]; then
    log "Агрегація сідів 4+5+6"
    python3 - "$BASE" "$SCRIPTS" <<'EOF'
import sys
from pathlib import Path
base = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from epochrun_multiseed import _aggregate, _print_aggregate_summary
tables = [(s, base / f"seed_{s}/results/epoch_final_table_latest.csv") for s in (4, 5, 6)]
out = base / "aggregated_final_table.csv"
_aggregate(tables, out)
print(f"aggregated -> {out}")
_print_aggregate_summary(out)
EOF
fi

log "Передаю керування основному оркестратору (готові кроки пропустить за маркерами)"
bash "$SCRIPTS/run_audit_followup_2026-07-22.sh"
