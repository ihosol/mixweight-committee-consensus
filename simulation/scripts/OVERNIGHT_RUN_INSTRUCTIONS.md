# Overnight Simulation Run — Instructions

## What this runs

Two new configurations with tuned controller parameters:
- `ADAPTIVE_LAM_MAX=0.65` + `ADAPTIVE_ALPHA_DOWN=0.02`

Expected results based on probe runs:
- **Burst**: ~8% reduction (vs current best 5.97%)
- **Trickle**: ~4.5% reduction (vs current best 3.47%)

## Prerequisites

### 1. Clone / sync the repo
```bash
git clone <your-repo-url> mixweight-committee-consensus
cd mixweight-committee-consensus
```
Or if already cloned, pull latest:
```bash
git pull origin feat/adaptive-redesign-poc
```

### 2. Build the chain binary
```bash
cd poc/cosmos/chain53/chain-five-three
make install
# OR build directly:
go build -o build/chain-five-threed ./cmd/chaind
cd -   # back to repo root
```
Confirm binary exists:
```bash
ls -la poc/cosmos/chain53/chain-five-three/build/chain-five-threed
```

### 3. Check Python deps
```bash
python3 -c "import yaml, csv, pathlib; print('ok')"
```

## Run the simulations

### Option A: Run sequentially (safest, ~2-3 hours total)

Open a terminal in the repo root and run:

```bash
cd /path/to/mixweight-committee-consensus

export POC_CHAIND="$PWD/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"

# Run 1: Burst with tuned alpha_down (30-40 min)
ADAPTIVE_LAM_MAX=0.65 ADAPTIVE_ALPHA_DOWN=0.02 \
python3 poc/cosmos/scripts/epochrun.py \
  poc/cosmos/generated_configs/final_package/FP_A1_alphadown02_lammax065_burst.yaml \
  2>&1 | tee /tmp/run_burst_alphadown02.log

echo "=== BURST DONE ==="

# Run 2: Trickle with tuned alpha_down (60-90 min)
ADAPTIVE_LAM_MAX=0.65 ADAPTIVE_ALPHA_DOWN=0.02 \
python3 poc/cosmos/scripts/epochrun.py \
  poc/cosmos/generated_configs/final_package/FP_A2_alphadown02_lammax065_trickle.yaml \
  2>&1 | tee /tmp/run_trickle_alphadown02.log

echo "=== TRICKLE DONE ==="
```

### Option B: Run with nohup (overnight, detached)

```bash
cd /path/to/mixweight-committee-consensus
export POC_CHAIND="$PWD/poc/cosmos/chain53/chain-five-three/build/chain-five-threed"

nohup bash -c '
  set -e
  echo "[burst start] $(date)"
  ADAPTIVE_LAM_MAX=0.65 ADAPTIVE_ALPHA_DOWN=0.02 \
    python3 poc/cosmos/scripts/epochrun.py \
    poc/cosmos/generated_configs/final_package/FP_A1_alphadown02_lammax065_burst.yaml
  echo "[burst done] $(date)"

  echo "[trickle start] $(date)"
  ADAPTIVE_LAM_MAX=0.65 ADAPTIVE_ALPHA_DOWN=0.02 \
    python3 poc/cosmos/scripts/epochrun.py \
    poc/cosmos/generated_configs/final_package/FP_A2_alphadown02_lammax065_trickle.yaml
  echo "[trickle done] $(date)"
' > /tmp/overnight_run.log 2>&1 &

echo "PID: $!"
echo "Monitor: tail -f /tmp/overnight_run.log"
```

## Monitor progress

```bash
# Live log
tail -f /tmp/overnight_run.log

# Draw count for burst
grep -c "post_attack" poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_burst/results/epoch_draws_latest.csv 2>/dev/null

# Draw count for trickle
grep -c "post_attack" poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_trickle/results/epoch_draws_latest.csv 2>/dev/null
```

## Verify completion

After the run, check final results exist:
```bash
ls poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_burst/results/epoch_final_table_latest.csv
ls poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_trickle/results/epoch_final_table_latest.csv
```

Quick result preview:
```bash
python3 - <<'EOF'
import csv
from pathlib import Path
ROOT = Path("poc/cosmos/artifacts")
for d in ["v2_pub_alphadown02_lammax065_burst", "v2_pub_alphadown02_lammax065_trickle"]:
    f = ROOT / d / "results/epoch_final_table_latest.csv"
    if not f.exists():
        print(f"{d}: NOT FOUND"); continue
    with f.open() as fh:
        row = list(csv.DictReader(fh))[0]
    print(f"{d}:")
    print(f"  reduction_vs_baseline_full_pct = {row.get('reduction_vs_baseline_full_pct','?')}")
    print(f"  post_attacker_weight_mean      = {row.get('post_attacker_weight_mean','?')}")
    print(f"  chernoff_ge_1_2_reduction      = {row.get('chernoff_bound_reduction_ge_1_2_pct','?')}")
EOF
```

## Artifact locations

Results will be stored in:
```
poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_burst/
poc/cosmos/artifacts/v2_pub_alphadown02_lammax065_trickle/
```

These names clearly identify:
- `alphadown02` = ADAPTIVE_ALPHA_DOWN=0.02
- `lammax065`   = ADAPTIVE_LAM_MAX=0.65
- `burst` / `trickle` = attacker profile

## Hardware notes

- The simulation runs 18 Cosmos validator nodes on localhost (ports 26680–26697, 36657–36674, 39090–39107)
- Each node uses ~5-10% CPU. Recommended: 8+ cores, 16+ GB RAM
- Burst run: ~30-40 min on the dev server; expect 10-20 min on faster hardware
- Trickle run: ~60-90 min on faster hardware (one sybil injection per epoch adds overhead)
- Do NOT run multiple simulations in parallel — they share the same port range

## If something goes wrong

Kill all chain processes:
```bash
pkill -9 -f "chain-five-threed"
pkill -f "epochrun.py"
```

Clean temp dirs:
```bash
rm -rf /tmp/epoch_* /tmp/localnet_*
```

Then re-run from the beginning.

## After the run — compare all results

Run this script to see the full comparison table:
```bash
python3 - <<'EOF'
import csv
from pathlib import Path

ROOT = Path("poc/cosmos/artifacts")
dirs = sorted(d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("v2_"))

print(f"{'artifact':<42}  {'red_full%':>9}  {'weight':>8}  {'lam_peak':>9}  {'chern12_red%':>12}")
print("="*90)
for d in dirs:
    f = d / "results/epoch_final_table_latest.csv"
    if not f.exists(): continue
    with f.open() as fh:
        row = list(csv.DictReader(fh))[0]
    mark = " ◄ NEW" if "alphadown02" in d.name else ""
    print(f"  {d.name+mark:<40}  "
          f"{row.get('reduction_vs_baseline_full_pct','—'):>9}  "
          f"{row.get('post_attacker_weight_mean','—'):>8}  "
          f"{row.get('post_lambda_peak_ppm','—'):>9}  "
          f"{row.get('chernoff_bound_reduction_ge_1_2_pct','—'):>12}")
EOF
```
