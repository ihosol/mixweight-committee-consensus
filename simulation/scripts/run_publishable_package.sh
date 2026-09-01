#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFGDIR="$ROOT/poc/cosmos/generated_configs/final_package"
RUNNER="$ROOT/poc/cosmos/scripts/epochrun.py"
ART="$ROOT/poc/cosmos/artifacts"
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
  local cfg="$1"; shift
  echo "=== $name ==="
  env "$@" python3 "$RUNNER" "$cfg"
  local src
  src="$(resolve_source_dir "$cfg")"
  rm -rf "$ART/$name"
  cp -r "$src" "$ART/$name"
}
run_one pub_default_burst "$CFGDIR/FP_A1_main_12h_c9_k6_b033_burst.yaml"
run_one pub_default_trickle "$CFGDIR/FP_A2_companion_12h_c9_k6_b033_trickle.yaml"
run_one pub_tuned_burst "$CFGDIR/FP_A1_main_12h_c9_k6_b033_burst.yaml" ADAPTIVE_LAM_MAX=0.65
run_one pub_tuned_trickle "$CFGDIR/FP_A2_companion_12h_c9_k6_b033_trickle.yaml" ADAPTIVE_LAM_MAX=0.65
