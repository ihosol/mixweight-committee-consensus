# Simulation scripts

Runners, aggregation and post-processing for the adaptive committee-selection
experiments: an on-chain controller that mixes stake-proportional weights with a
tenure-based baseline, `q_i(λ) = (1−λ)·w_i + λ·b_i`, and adjusts λ from measured
decentralization signals.

This directory is self-contained: the chain module, the scripts that drive it,
the scenario definitions, and the summary tables the experiments produced. The
repository root holds the complementary static half — snapshot collection and
risk evaluation across live PoS networks (`snapshots.py`, `risk_sim.py`,
`risk_batch.py`).

## What is here

| Path | Contents |
|---|---|
| `chain/` | the Cosmos-SDK application, including `x/adaptivecommittee`: controller state, mixed-weight distribution, committee draw, diagnostic event |
| `build_chain.sh` | fetches dependencies and builds the chain binary |
| `results/tables/` | the three curated tables the dissertation reports |
| `results/scenarios/` | per-scenario aggregates, mean ± sd across three seeds |
| `scripts/epochrun.py` | main epoch-mode runner: drives a local network, submits draws, writes per-draw and per-epoch CSVs |
| `scripts/epochrun_multiseed.py` | runs one scenario across several seeds and aggregates mean ± sd |
| `scripts/compute_realized_capture.py` | realized capture frequency from the per-draw layer, with Clopper–Pearson bounds |
| `scripts/run_*.sh` | scenario batches: publishable package, signal ablation, fairness pairs, tuning sweep, drift evaluation |
| `scripts/make_*.py` | figure and table generation from the aggregated results |
| `configs/` | scenario definitions: attacker profile, cohort size, committee size, epoch layout |

## Building

```bash
bash build_chain.sh
```

The script locates a Go toolchain (including the common off-`PATH` install
locations), checks the version against what `go.mod` requires, downloads and
verifies the modules, builds the binary, and confirms that the resulting
executable actually exposes the `adaptivecommittee` module rather than only
compiling. The first run fetches the Cosmos SDK and takes several minutes.

## Running

```bash
export POC_CHAIND="$PWD/chain/build/chain-five-threed"
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

## Published results

`results/scenarios/` carries the aggregated table of each of the twenty-three
canonical scenarios — mean ± standard deviation across three independent runs —
covering the headline security package, the trickle companion, the signal
ablation, the attacker-share sweep, the fairness pairs and the age-farming case.

`results/tables/` carries the three curated tables: the final results table, the
realized capture frequency with its bounds, and the per-draw rows those
frequencies are computed from, so the counts can be rederived rather than taken
on trust:

```bash
python3 scripts/compute_realized_capture.py --draws-csv results/tables/REALIZED_CAPTURE_DRAWS.csv
```

The raw per-run trees are not published: they run to well over a gigabyte and
are regenerable from the configurations above.

One caveat on reading them. Three runs establish the repeatability of the
harness, not a statistical difference between configurations. For weight-share
metrics the between-run spread is at most 0.03 percentage points, so the mean is
a fair headline value. For event-counting metrics it is near 5 points, larger
than the differences between arms, so those columns describe what was observed
and do not generalize.

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
