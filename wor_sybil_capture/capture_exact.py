#!/usr/bin/env python3
"""Exact capture probabilities under successive weighted sampling without replacement.

Implements the formulas of the paper "Proof-of-stake committee capture risk under stake splitting
in weighted sampling without replacement":
  Theorem 1  Pr[X_A >= q] for an arbitrary coalition (capture_exact),
  Theorem 2  P*(q), the largest value over all splits of the coalition mass p (capture_dust),
  Theorem 3  the equal split into K identities, the worst split for at most K identities
             (capture_equal_split),
together with the binomial tail B^wr(q) of sampling with replacement and an auxiliary upper
bound on P*(q) (upper_bound, not used in the paper).

Scripts in this folder:
  capture_exact.py    formulas; the ten-network panel; --sensitivity; --check against Monte Carlo
  verify_exact.py     exact enumeration of successive sampling on small instances; known-answer cases
  check_precision.py  double-precision quadrature against mpmath and a Gauss-Legendre rule
  prototype_gof.py    the formula and the binomial model against the prototype's committee draws
  make_figures.py     the figure, from the panel results

Inputs (environment variables):
  WOR_SNAPSHOTS       stake snapshots, <dir>/<date>/<network>_weights.csv (as written by snapshots.py)
                      or <dir>/<network>/<date>_weights.csv; only the column "share" is used
  WOR_PROTOTYPE_RUNS  prototype runs, <scenario>/seed_<n>/results/*_latest.csv (prototype_gof.py only)
  WOR_OUT             output directory (default: build/ next to this folder)

Dependencies: numpy, scipy, pandas, matplotlib, mpmath.

Usage:
  python3 capture_exact.py                  # panel
  python3 capture_exact.py --sensitivity    # sensitivity to the coalition share and threshold
  python3 capture_exact.py --check          # formulas vs Monte Carlo
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import binom, gamma

ROOT = Path(__file__).resolve().parents[3]
# Stake snapshots: either <dir>/<network>/<date>_weights.csv or the layout written by
# snapshots.py, <dir>/<date>/<network>_weights.csv. Override with WOR_SNAPSHOTS.
SNAPSHOTS = Path(os.environ.get(
    "WOR_SNAPSHOTS", ROOT / "research/datasets/_unpacked/ch2_static_snapshots_2025-09_10nets"))
OUT = Path(os.environ.get("WOR_OUT", Path(__file__).resolve().parents[1] / "build"))


def snapshot_files() -> dict[str, list[Path]]:
    """Map network -> date-sorted list of weight files, for either snapshot layout."""
    files: dict[str, list[Path]] = {}
    for f in SNAPSHOTS.glob("*/*_weights.csv"):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.parent.name):      # <date>/<network>_weights.csv
            net, date = f.name[: -len("_weights.csv")], f.parent.name
        else:                                                      # <network>/<date>_weights.csv
            net, date = f.parent.name, f.name[:10]
        files.setdefault(net, []).append((date, f))
    return {net: [f for _, f in sorted(v)] for net, v in sorted(files.items())}


def snapshot_dates(net: str) -> list[tuple[str, Path]]:
    return [(f.parent.name if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.parent.name) else f.name[:10], f)
            for f in snapshot_files()[net]]

# Tenure transform of the prototype (Chapter 3): tilde tau = min(tau, tau_max)^zeta with
# tau_max = 2000 blocks and zeta = 0.5, so a mature validator has sqrt(2000) and a new one 1.
TAU_MATURE = math.sqrt(2000.0)


def pb_cdf(probs: np.ndarray, r: int) -> float:
    """Pr[sum of independent Bernoulli(probs) <= r], by the standard recursion truncated at r+1."""
    if r < 0:
        return 0.0
    d = np.zeros(r + 2)
    d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * (1.0 - p) + d[:-1] * p   # cells 0..r are exact; the last one is scratch
        d[0] *= 1.0 - p
    return float(d[: r + 1].sum())


def pb_pmf(probs: np.ndarray) -> np.ndarray:
    d = np.zeros(len(probs) + 1)
    d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * (1.0 - p) + d[:-1] * p
        d[0] *= 1.0 - p
    return d


TAIL_TOL = 1e-13


def _integrate(f, scale: float, remainder=None) -> float:
    """Integrate f over [0, L]. Breakpoints cover the first-seat scale (t ~ 1, the weights sum to one)
    and the scale s = q/p of the q-th coalition key; the endpoint L starts at 512 s and is extended
    until remainder(L), an upper bound on the omitted mass, is below TAIL_TOL."""
    s = scale
    pts = sorted({x for x in (1.0, 10.0, 100.0) if x < s / 4} | {0.0, s / 4, s, 4 * s, 16 * s, 64 * s, 512 * s})
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        total += quad(f, a, b, limit=400, epsabs=0.0, epsrel=1e-10)[0]
    L = pts[-1]
    while remainder is not None and remainder(L) > TAIL_TOL:
        total += quad(f, L, 8 * L, limit=400, epsabs=0.0, epsrel=1e-10)[0]
        L *= 8
    return total


def capture_dust(h: np.ndarray, p: float, m: int, q: int) -> float:
    """Theorem 2: worst case over splits of mass p against honest weights h."""
    if len(h) < m - q + 1:
        return 1.0
    f = lambda t: gamma.pdf(t, q, scale=1.0 / p) * pb_cdf(1.0 - np.exp(-h * t), m - q)
    return _integrate(f, q / p, remainder=lambda L: gamma.sf(L, q, scale=1.0 / p))


def capture_exact(h: np.ndarray, v: np.ndarray, m: int, q: int) -> float:
    """Theorem 1: coalition with identity weights v against honest weights h."""
    if len(v) < q:
        return 0.0
    if len(h) < m - q + 1:
        return 1.0

    def f(t):
        ea = np.exp(-v * t)
        pa = 1.0 - ea
        dens = sum(v[l] * ea[l] * pb_pmf(np.delete(pa, l))[q - 1] for l in range(len(v)))
        return dens * pb_cdf(1.0 - np.exp(-h * t), m - q)

    rem = lambda L: pb_cdf(-np.expm1(-v * L), q - 1) * pb_cdf(-np.expm1(-h * L), m - q)
    return _integrate(f, q / v.sum(), remainder=rem)


def capture_equal_split(h: np.ndarray, p: float, k: int, m: int, q: int) -> float:
    """Theorem 1 for k identities of equal weight p/k (closed-form density of the q-th key)."""
    if k < q:
        return 0.0
    if len(h) < m - q + 1:
        return 1.0
    v = p / k
    logc = math.lgamma(k + 1) - math.lgamma(q) - math.lgamma(k - q + 1)

    def f(t):
        a = -math.expm1(-v * t)
        if a <= 0.0:
            return 0.0
        dens = math.exp(logc + (q - 1) * math.log(a) - v * t * (k - q + 1)) * v
        return dens * pb_cdf(1.0 - np.exp(-h * t), m - q)

    rem = lambda L: binom.cdf(q - 1, k, -math.expm1(-v * L)) * pb_cdf(-np.expm1(-h * L), m - q)
    return _integrate(f, q / p, remainder=rem)


def upper_bound(h: np.ndarray, p: float, m: int, q: int) -> float:
    """Statement 2: tail of independent Bernoulli(min(1, p / (1 - S_r))), r = 0..m-1."""
    S = np.concatenate([[0.0], np.cumsum(np.sort(h)[::-1])])[:m]
    probs = np.minimum(1.0, p / np.maximum(1e-300, 1.0 - S))
    return float(pb_pmf(probs)[q:].sum())


def mixed_weights(stake: np.ndarray, alpha: float, lam: float) -> tuple[np.ndarray, float]:
    """Honest weights and coalition mass for a new coalition of stake share alpha.

    Honest validators keep the network's stake shape (scaled to 1 - alpha) and are mature
    (tilde tau = TAU_MATURE); the coalition's identities are new (tilde tau = 1). The mixed weight
    of a new identity is linear in its stake, so any split of the coalition keeps its mass p.
    """
    s = stake / stake.sum() * (1.0 - alpha)
    D = TAU_MATURE * (1.0 - alpha) + alpha          # normaliser of the tenure-weighted baseline
    h = s * ((1.0 - lam) + lam * TAU_MATURE / D)
    p = alpha * ((1.0 - lam) + lam / D)
    return h, p


def mc_capture(h, v, m, q, rng, R=200_000):
    w = np.concatenate([h, v])
    keys = rng.exponential(size=(R, len(w))) / w
    sel = np.argpartition(keys, m, axis=1)[:, :m]
    return float(np.mean((sel >= len(h)).sum(1) >= q))


def check():
    rng = np.random.default_rng(20260923)
    s = np.arange(1, 41) ** -0.8
    for p, k, m, q in [(0.165, 5, 9, 3), (0.33, 6, 9, 3), (0.20, 20, 21, 7)]:
        h = s[1:] / s[1:].sum() * (1 - p)
        v = np.full(k, p / k)
        print(f"p={p} k={k} m={m} q={q}: Theorem 1 {capture_exact(h, v, m, q):.4f}  "
              f"MC {mc_capture(h, v, m, q, rng):.4f}  |  P* {capture_dust(h, p, m, q):.4f}  "
              f"MC(k=400) {mc_capture(h, np.full(400, p / 400), m, q, rng, R=50_000):.4f}  "
              f"UB {upper_bound(h, p, m, q):.4f}  Bwr {binom.sf(q - 1, m, p):.4f}")


def panel():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for net in snapshot_files():
        for date, f in snapshot_dates(net):
            stake = pd.read_csv(f)["share"].to_numpy(float)
            stake = stake[stake > 0]
            N = len(stake)
            top1 = stake.max() / stake.sum()
            hhi = float(((stake / stake.sum()) ** 2).sum())
            for regime, m in (("m21", 21), ("rho0.4", int(0.4 * N))):
                if m < 3:
                    continue
                q = math.ceil(m / 3)
                for alpha in (0.20, 0.33):
                    for lam in (0.0, 0.3, 0.55):
                        h, p = mixed_weights(stake, alpha, lam)
                        rows.append(dict(
                            network=net, date=date, N=N, top1=top1, hhi=hhi,
                            regime=regime, m=m, q=q, alpha=alpha, lam=lam, p=p,
                            B_wr=binom.sf(q - 1, m, p),
                            P_star=capture_dust(h, p, m, q),
                            UB=upper_bound(h, p, m, q),
                        ))
            print("done", net, date, flush=True)
    df = pd.DataFrame(rows)
    df["ratio"] = df["P_star"] / df["B_wr"]
    df.to_csv(OUT / "panel_exact.csv", index=False)
    med = (df.groupby(["network", "regime", "alpha", "lam"])
             [["N", "m", "q", "p", "top1", "B_wr", "P_star", "UB", "ratio"]].median().reset_index())
    med.to_csv(OUT / "panel_exact_median.csv", index=False)
    print(med.to_string())


def sensitivity():
    """Sensitivity to the coalition share alpha and the threshold theta, m = 21, lambda = 0 (every snapshot)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for net in snapshot_files():
        for date, f in snapshot_dates(net):
            stake = pd.read_csv(f)["share"].to_numpy(float)
            stake = stake[stake > 0]
            m = 21
            for theta, tname in ((1 / 3, "1/3"), (1 / 2, "1/2")):
                q = math.ceil(theta * m)
                for alpha in (0.10, 0.20, 0.33):
                    h, p = mixed_weights(stake, alpha, 0.0)
                    rows.append(dict(network=net, date=date, m=m, theta=tname, q=q,
                                     alpha=alpha, B_wr=binom.sf(q - 1, m, p), P_star=capture_dust(h, p, m, q)))
        print("done", net, flush=True)
    df = pd.DataFrame(rows)
    df["ratio"] = df.P_star / df.B_wr
    df.to_csv(OUT / "sensitivity.csv", index=False)
    med = df.groupby(["theta", "alpha", "network"])[["q", "B_wr", "P_star", "ratio"]].median().reset_index()
    summ = med.groupby(["theta", "alpha"]).agg(q=("q", "first"), B_wr=("B_wr", "first"),
                                              P_min=("P_star", "min"), P_max=("P_star", "max"),
                                              r_min=("ratio", "min"), r_max=("ratio", "max")).reset_index()
    summ.to_csv(OUT / "sensitivity_summary.csv", index=False)
    print(summ.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.sensitivity:
        sensitivity()
    else:
        panel()
