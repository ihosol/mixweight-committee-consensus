# Simulation scripts

Runners, aggregation and post-processing for the adaptive committee-selection
experiments: an on-chain controller that mixes stake-proportional weights with a
tenure-based baseline, `q_i(λ) = (1−λ)·w_i + λ·b_i`, and adjusts λ from measured
decentralization signals.

These are the scripts the experiments were run with, published so the procedure
can be read and checked. The repository root already holds the snapshot
collection and static risk evaluation (`snapshots.py`, `risk_sim.py`,
`risk_batch.py`); this directory covers the dynamic, on-chain part.

## What is here

| Path | Contents |
|---|---|
| `scripts/epochrun.py` | main epoch-mode runner: drives a local network, submits draws, writes per-draw and per-epoch CSVs |
| `scripts/epochrun_multiseed.py` | runs one scenario across several seeds and aggregates mean ± sd |
| `scripts/compute_realized_capture.py` | realized capture frequency from the per-draw layer, with Clopper–Pearson bounds |
| `scripts/run_*.sh` | scenario batches: publishable package, signal ablation, fairness pairs, tuning sweep, drift evaluation |
| `scripts/make_*.py` | figure and table generation from the aggregated results |
| `configs/` | scenario definitions: attacker profile, cohort size, committee size, epoch layout |

## What is not here

The chain itself. The scripts drive a Cosmos-SDK application exposing an
`x/adaptivecommittee` module, and expect its binary via `POC_CHAIND` or at a
pinned build path. Without that binary the scripts are readable and their logic
checkable, but a full run is not reproducible from this directory alone.

## Running

```bash
export POC_CHAIND=/path/to/chain-binary
python3 scripts/epochrun_multiseed.py configs/FP_A1_main_12h_c9_k6_b033_burst.yaml \
    --seeds 1,2,3 --artifacts-subdir pub_default_burst
```

Batches take the same environment:

```bash
bash scripts/run_publishable_package.sh
bash scripts/run_ablation_3seed.sh
```

Results are written under an `artifacts/<subdir>/` tree: per-draw CSV, per-epoch
summary, aggregated final table, and plots.

### Two operational notes

`epochrun_multiseed.py` waits between seeds until every node port is bindable
again, with a 20 s default. After a local network of twenty or more nodes stops,
listening sockets sit in `TIME_WAIT` past that window and the batch aborts even
though nothing is stuck. Raise it:

```bash
export ORPHAN_CLEANUP_TIMEOUT_S=180
```

Progress is easy to misread: the `[multi-seed]` markers go through a buffered
`print()`, while the child process writes to the same descriptor directly, so a
log can look frozen while work continues. The reliable signal is the appearance
of `seed_*/results/epoch_final_table_latest.csv`.

## Controller knobs

Set through the environment, read by the chain module:

| Variable | Meaning |
|---|---|
| `ADAPTIVE_LAM_MAX` | ceiling on the mixing parameter |
| `ADAPTIVE_ALPHA_UP` / `ADAPTIVE_ALPHA_DOWN` | asymmetric first-order filter coefficients |
| `ADAPTIVE_FRESHNESS_W` / `ADAPTIVE_GINI_W` / `ADAPTIVE_SPLIT_W` | signal weights in the composite indicator |
| `COMMITTEE_DRAW_MODE` | `wor` (default, sequential PPSWOR) or `wr` (independent, with replacement) |

The draw mode changes the statistical status of the binomial risk estimate: it
is the exact law of the draw only under `wr`, and a model estimate under `wor`,
conservative for coalitions holding the largest mixed weights and an empirical
assumption otherwise.

## Licence

Apache-2.0, matching Cosmos SDK and CometBFT, which the module builds on.
