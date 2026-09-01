#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${FAIRNESS_CFG:-$ROOT/poc/cosmos/generated_configs/final_package/FP_H2_honest_newcomer_12h_c9_k1_b010_trickle.yaml}"
RUNNER="$ROOT/poc/cosmos/scripts/epochrun.py"
ART="$ROOT/poc/cosmos/artifacts"
CHAIND_DEFAULT="$ROOT/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"
resolve_source_dir() {
  local cfg="$1"
  local base="$ART/$(basename "$cfg" .yaml)"
  local ptr="$base/results/latest_run_dir.txt"
  if [[ -f "$ptr" ]]; then
    cat "$ptr"
  else
    echo "$base"
  fi
}
run_one() {
  local name="$1"; shift
  echo "=== $name ==="
  env POC_CHAIND="${POC_CHAIND:-$CHAIND_DEFAULT}" "$@" python3 "$RUNNER" "$CFG"
  local src
  src="$(resolve_source_dir "$CFG")"
  rm -rf "$ART/$name"
  cp -r "$src" "$ART/$name"
}
run_one fairness_trickle_default
run_one fairness_trickle_tuned ADAPTIVE_LAM_MAX=0.65
