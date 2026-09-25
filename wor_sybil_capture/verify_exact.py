#!/usr/bin/env python3
"""Checks Theorems 1-3 and the refinement property against exact enumeration of successive sampling
on small instances:  python3 verify_exact.py"""
# Exact (brute-force) verification of the paper's formulas on small instances.
import sys, math, itertools, functools
import numpy as np
from scipy.stats import binom
sys.path.insert(0, __import__("os").path.dirname(__file__))
from capture_exact import capture_exact, capture_dust, upper_bound

def exact_seats(w, A, m):
    """Exact distribution of coalition seats under successive sampling, by recursion over drawn sets."""
    n = len(w); A = set(A)
    @functools.lru_cache(None)
    def rec(taken, k):            # taken: bitmask; returns dict seats->prob for remaining k draws
        if k == 0: return {0: 1.0}
        rem = [i for i in range(n) if not taken >> i & 1]
        tot = sum(w[i] for i in rem); out = {}
        for i in rem:
            sub = rec(taken | 1 << i, k - 1); a = 1 if i in A else 0
            for s, pr in sub.items(): out[s + a] = out.get(s + a, 0) + w[i] / tot * pr
        return out
    return rec(0, m)

def main():
    rng = np.random.default_rng(1)
    worst = 0.0
    print("Theorem 1 vs exact enumeration:")
    for trial in range(6):
        nh, K = rng.integers(3, 7), rng.integers(1, 5); m = int(rng.integers(2, min(nh + K, 7) + 1))
        h = rng.random(nh) ** 2 + 0.05; v = rng.random(K) + 0.05; tot = h.sum() + v.sum(); h /= tot; v /= tot
        d = exact_seats(tuple(np.concatenate([h, v])), tuple(range(nh, nh + K)), m)
        for q in range(1, m + 1):
            ex = sum(p for s, p in d.items() if s >= q); th = capture_exact(h, v, m, q)
            worst = max(worst, abs(ex - th))
        print(f"  nh={nh} K={K} m={m}: max |exact - formula| over q = {max(abs(sum(p for s,p in d.items() if s>=q)-capture_exact(h,v,m,q)) for q in range(1,m+1)):.2e}")
    print(f"  overall max error {worst:.2e}")

    print("Statement 1 (refinement never lowers any tail), exact:")
    h = np.array([0.30, 0.20, 0.12, 0.08]); m = 4
    for v in ([0.30], [0.15, 0.15], [0.20, 0.10], [0.10, 0.10, 0.10], [0.05]*6):
        v = np.array(v); w = np.concatenate([h, v]); w = w / w.sum()
        d = exact_seats(tuple(w), tuple(range(4, 4 + len(v))), m)
        print("  split", list(v), " tails q=1..4:", [round(sum(p for s, p in d.items() if s >= q), 4) for q in range(1, 5)])

    print("Theorem 2 and Statement 2 on the same honest weights (p = 0.30 of total 1.0):")
    hh = h / (h.sum() + 0.30); p = 0.30 / (h.sum() + 0.30)
    for q in range(1, 5):
        print(f"  q={q}: Bwr={binom.sf(q-1, m, p):.4f} <= P*={capture_dust(hh, p, m, q):.4f} <= UB={upper_bound(hh, p, m, q):.4f};"
              f"  6 pieces exact={sum(pr for s, pr in exact_seats(tuple(np.concatenate([hh, np.full(6, p/6)])), tuple(range(4, 10)), m).items() if s >= q):.4f}")


if __name__ == "__main__":
    main()


def structural_tests():
    """Cases with known answers: equal weights (hypergeometric), K < q, q = m, q = 1, forced capture."""
    from math import comb
    from capture_exact import capture_exact, capture_dust
    err = 0.0
    # all weights equal: X is hypergeometric
    for n, K, m in ((6, 3, 4), (8, 4, 5), (5, 5, 6)):
        w = 1.0 / (n + K)
        for q in range(1, m + 1):
            hyper = sum(comb(K, k) * comb(n, m - k) for k in range(q, min(K, m) + 1)) / comb(n + K, m)
            err = max(err, abs(capture_exact(np.full(n, w), np.full(K, w), m, q) - hyper))
    h = np.array([0.3, 0.25, 0.15]); p = 0.3
    zero = capture_exact(h, np.full(2, p / 2), 4, 3)                     # K < q
    full = capture_dust(h, p, 3, 3) - p ** 3                             # q = m in the limit
    q1 = max(abs(capture_exact(h, v, 3, 1) - capture_dust(h, p, 3, 1))  # q = 1 split-invariant
             for v in (np.array([p]), np.array([0.2, 0.1]), np.full(5, p / 5)))
    forced = capture_exact(h[:2] / h[:2].sum() * (1 - p), np.full(4, p / 4), 4, 2)  # |H| < m - q + 1
    print(f"structural: hypergeometric max error {err:.1e}; K<q gives {zero}; P*(m)-p^m = {full:.1e}; "
          f"q=1 spread {q1:.1e}; forced capture gives {forced}")


if __name__ == "__main__":
    structural_tests()
