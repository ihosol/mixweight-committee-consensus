#!/usr/bin/env bash
# P5.4 large-N: 1 seed, без cgroup-лімітів (запуск на повній швидкості).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"; cd "$REPO_ROOT"
export ORPHAN_CLEANUP_TIMEOUT_S=120
export POC_CHAIND="$REPO_ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
CFG="$REPO_ROOT/poc/cosmos/generated_configs/largeN/LN_A1_N36_seed1.yaml"
LOG="$REPO_ROOT/poc/cosmos/artifacts/largeN/logs/gentle_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$LOG")"
echo "[gentle] лог: $LOG"
exec python3 "$REPO_ROOT/poc/cosmos/scripts/epochrun_multiseed.py" "$CFG" > "$LOG" 2>&1
