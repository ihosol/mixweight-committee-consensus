#!/usr/bin/env python3
"""Accuracy check of the double-precision evaluation of the worst-case formula (Theorem 2).

  python3 check_precision.py

For moderate networks the integral is recomputed with mpmath at 30 significant digits
(tanh-sinh quadrature, Poisson-binomial recursion in multiprecision). For the largest
committee (Solana, m = 382) an independent double-precision Gauss-Legendre rule with
4000 nodes on the same subintervals is used instead.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
from scipy.stats import gamma

from capture_exact import snapshot_files, capture_dust, mixed_weights, pb_cdf

mp.mp.dps = 30


def dust_mp(h, p, m, q):
    h = [mp.mpf(float(x)) for x in h]
    p = mp.mpf(float(p))
    r = m - q

    def cdf(t):
        d = [mp.mpf(0)] * (r + 2)
        d[0] = mp.mpf(1)
        for w in h:
            a = -mp.expm1(-w * t)
            for j in range(r + 1, 0, -1):
                d[j] = d[j] * (1 - a) + d[j - 1] * a
            d[0] *= 1 - a
        return mp.fsum(d[: r + 1])

    dens = lambda t: p ** q * t ** (q - 1) * mp.exp(-p * t) / mp.factorial(q - 1)
    s = q / p
    return mp.quad(lambda t: dens(t) * cdf(t), [0, s / 4, s, 4 * s, 16 * s, 64 * s, 512 * s])


def dust_gl(h, p, m, q, nodes=4000):
    x, w = roots_legendre(nodes)
    s = q / p
    pts = [0.0, s / 4, s, 4 * s, 16 * s, 64 * s, 512 * s]
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        f = np.array([gamma.pdf(ti, q, scale=1.0 / p) * pb_cdf(1.0 - np.exp(-h * ti), m - q) for ti in t])
        total += 0.5 * (b - a) * float(np.dot(w, f))
    return total


def snapshot(net):
    f = snapshot_files()[net][-1]
    st = pd.read_csv(f)["share"].to_numpy(float)
    return st[st > 0]


def main():
    cases = [("cosmoshub", 21, 0.20, 0.0), ("cosmoshub", 21, 0.20, 0.55), ("cosmoshub", 80, 0.20, 0.0),
             ("celestia", 40, 0.20, 0.55), ("sei", 21, 0.10, 0.0)]
    print("double vs mpmath (30 digits):")
    for net, m, alpha, lam in cases:
        h, p = mixed_weights(snapshot(net), alpha, lam)
        q = math.ceil(m / 3)
        d = capture_dust(h, p, m, q)
        e = dust_mp(h, p, m, q)
        print(f"  {net:9s} m={m:3d} alpha={alpha} lam={lam}: double={d:.12g}  mp={mp.nstr(e, 12)}  "
              f"rel.err={abs(d - float(e)) / float(e):.1e}")
    print("adaptive QUADPACK vs 4000-node Gauss-Legendre, largest committee:")
    for net, alpha, lam in (("solana", 0.20, 0.0), ("solana", 0.20, 0.55), ("avalanche", 0.20, 0.3)):
        st = snapshot(net)
        m = int(0.4 * len(st))
        q = math.ceil(m / 3)
        h, p = mixed_weights(st, alpha, lam)
        a, b = capture_dust(h, p, m, q), dust_gl(h, p, m, q)
        print(f"  {net:9s} m={m} alpha={alpha} lam={lam}: quadpack={a:.10g}  gauss-legendre={b:.10g}  "
              f"rel.diff={abs(a - b) / max(b, 1e-300):.1e}")


if __name__ == "__main__":
    main()
