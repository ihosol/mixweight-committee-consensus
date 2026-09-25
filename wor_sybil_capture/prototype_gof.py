#!/usr/bin/env python3
"""Goodness of fit of Theorem 1 and of the binomial model to the committee draws of the prototype.

Reconstructs the mixed weights of every post-attack draw from the logged stakes, tenure scores and
mixing level and computes the distribution pi_j of the adversary's seat count X for every draw
under Theorem 1 and under Binomial(m, p). The Pearson statistic of the pooled histogram (cells
merged to expected counts >= 5) is calibrated by simulating every draw from its own pi_j, since
the pooled histogram of non-identical draws is not multinomial.

  python3 prototype_gof.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from capture_exact import capture_exact

ROOT = Path(__file__).resolve().parents[3]
# Directory with the prototype runs <scenario>/seed_<n>/results/*.csv; override with WOR_PROTOTYPE_RUNS.
ART = Path(os.environ.get("WOR_PROTOTYPE_RUNS", ROOT / "poc/cosmos/artifacts/overnight_v5"))


def draws():
    for sc in ("pub_default_burst", "pub_tuned_burst"):
        for seed in (1, 2, 3):
            v = pd.read_csv(ART / sc / f"seed_{seed}/results/validator_metrics_latest.csv")
            d = pd.read_csv(ART / sc / f"seed_{seed}/results/epoch_draws_latest.csv").set_index("tag")
            for tag, g in v.groupby("tag", sort=False):
                if d.loc[tag, "phase"] != "post_attack":
                    continue
                lam = d.loc[tag, "lambda_auto_ppm"] / 1e6
                st = g.validator_stake.to_numpy(float)
                ag = g.validator_age_score.to_numpy(float)
                w = (1 - lam) * st / st.sum() + lam * ag / ag.sum()
                att = g.is_attacker.to_numpy(bool)
                yield w[~att], w[att], int(g.committee_size.iloc[0]), int(d.loc[tag, "attacker_seats"])


def per_draw_distributions():
    """Per-draw distributions of X under Theorem 1 and under Binomial(m, p), plus observed X.

    Cached in OUT/prototype_per_draw.npz because the exact evaluation takes about an hour."""
    from capture_exact import OUT
    cache_file = OUT / "prototype_per_draw.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        return z["exact"], z["binom"], z["obs"]
    ex, bi, ob, memo = [], [], [], {}
    for h, a, m, x in draws():
        key = (tuple(np.round(h, 12)), tuple(np.round(a, 12)), m)
        if key not in memo:
            tails = [1.0] + [capture_exact(h, a, m, q) for q in range(1, m + 1)] + [0.0]
            memo[key] = np.clip([tails[k] - tails[k + 1] for k in range(m + 1)], 0.0, 1.0)
        pe = memo[key] / memo[key].sum()
        ex.append(pe)
        bi.append(binom.pmf(np.arange(m + 1), m, a.sum()))
        ob.append(x)
    ex, bi, ob = np.array(ex), np.array(bi), np.array(ob)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, exact=ex, binom=bi, obs=ob)
    return ex, bi, ob


def pearson(counts, mean, cells):
    c = np.array([counts[list(g)].sum() for g in cells])
    e = np.array([mean[list(g)].sum() for g in cells])
    return float(((c - e) ** 2 / e).sum())


def calibrated_test(pi, obs, B=100_000, seed=20260924):
    """Pearson statistic on fixed cells, calibrated by simulating every draw from its own pi_j."""
    rng = np.random.default_rng(seed)
    mean = pi.sum(0)
    # cells fixed before looking at the data: merge from both ends until expected counts >= 5
    idx = list(range(pi.shape[1]))
    cells = [[i] for i in idx]
    while mean[cells[0]].sum() < 5:
        cells[1] = cells[0] + cells[1]; cells.pop(0)
    while mean[cells[-1]].sum() < 5:
        cells[-2] = cells[-2] + cells[-1]; cells.pop()
    counts = np.bincount(obs, minlength=pi.shape[1]).astype(float)
    t_obs = pearson(counts, mean, cells)
    cum = np.cumsum(pi, axis=1)
    exceed = 0
    for _ in range(B // 1000):
        u = rng.random((1000, pi.shape[0], 1))
        cats = (u > cum[None, :, :]).sum(2)                     # (1000, draws)
        for row in cats:
            if pearson(np.bincount(row, minlength=pi.shape[1]).astype(float), mean, cells) >= t_obs:
                exceed += 1
    return t_obs, (1 + exceed) / (B + 1), cells, counts, mean


def main():
    ex, bi, ob = per_draw_distributions()
    print(f"draws: {len(ob)}")
    for name, pi in (("Theorem 1", ex), ("binomial", bi)):
        t, pv, cells, counts, mean = calibrated_test(pi, ob)
        print(f"{name}: statistic={t:.2f} on {len(cells)} cells, simulated p-value={pv:.3g} (B=100000)")
        print("   observed:", [int(counts[list(g)].sum()) for g in cells])
        print("   expected:", [round(float(mean[list(g)].sum()), 1) for g in cells])


if __name__ == "__main__":
    main()
