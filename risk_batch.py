#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_batch.py — daily batch runner over *_weights.csv for multiple alphas/lambdas.

Reads per-validator weights from a snapshot folder (data/YYYY-MM-DD),
chooses per-chain committee size m<=n by a strategy, scans alphas & lambdas,
runs Monte Carlo + Chernoff bound, and writes a consolidated CSV.

Config via env (with sensible defaults):
  RB_PATH=./data/$(date -u +%F)
  RB_ALPHAS=0.20,0.25,0.30,0.33
  RB_LAMBDAS=0,0.1,0.2,0.3
  RB_TRIALS=20000
  RB_M_STRATEGY=fraction:0.4   # options: fixed:60  OR fraction:0.4
  DEBUG=1 to see progress logs

Output:
  <path>/risk_results.csv (appends if exists)
"""

import os, csv, math, glob, sys, datetime as dt
from typing import List, Dict, Tuple
import random

DEBUG = os.environ.get("DEBUG","0") == "1"

def log(level: str, msg: str):
    ts = dt.datetime.utcnow().isoformat(timespec="seconds")+"Z"
    print(f"[{ts}] {level} {msg}")

def dbg(msg: str):
    if DEBUG: log("DEBUG", msg)

def parse_env_list(name: str, default: str) -> List[float]:
    raw = os.environ.get(name, default)
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out

def parse_m_strategy(raw: str, n: int) -> int:
    """
    raw like 'fixed:60' or 'fraction:0.4'
    returns m <= n and >= 1
    """
    if ":" not in raw:
        raise ValueError("RB_M_STRATEGY must be fixed:<int> or fraction:<float>")
    kind, val = raw.split(":",1)
    kind = kind.strip().lower()
    val = val.strip()
    if kind == "fixed":
        m = int(val)
    elif kind == "fraction":
        frac = float(val)
        m = max(1, int(math.floor(frac * n)))
    else:
        raise ValueError("unknown m strategy")
    return min(m, n)

def load_shares(path: str) -> List[float]:
    import pandas as pd
    df = pd.read_csv(path)
    if "share" not in df.columns:
        raise RuntimeError(f"'share' not found in {path}")
    shares = [float(s) for s in df["share"].tolist() if float(s) > 0.0]
    s = sum(shares) or 1.0
    shares = [x/s for x in shares]
    return shares

def worst_adversary_mask(shares_desc: List[float], alpha: float) -> List[int]:
    mask = [0]*len(shares_desc); s=0.0
    for i,v in enumerate(shares_desc):
        if s >= alpha: break
        mask[i]=1; s+=v
    return mask

def mix_weights(shares: List[float], lam: float) -> List[float]:
    n = len(shares)
    base = (1.0 - lam); add = lam / n
    w = [base*s + add for s in shares]
    Z = sum(w) or 1.0
    return [x/Z for x in w]

def mu_expected(m: int, w_desc: List[float], A_desc: List[int]) -> float:
    piA = sum(w for w,a in zip(w_desc, A_desc) if a)
    return m * piA

def chernoff_upper_bound(m: int, mu: float, theta: float = 1/3) -> float:
    t = m*theta
    if mu >= t: return 1.0
    if mu <= 0: return 0.0
    num = (t - mu)**2; den = 3.0 * mu
    return math.exp(- num/den)

def sample_ppswor(weights: List[float], k: int) -> List[int]:
    keys=[]
    for i,w in enumerate(weights):
        if w<=0: continue
        u=random.random()
        keys.append((-math.log(u)/w, i))
    keys.sort(key=lambda x:x[0])
    return [i for _,i in keys[:k]]

def empirical_risk(weights: List[float], A_unsorted: List[int], trials: int, m: int, theta: float=1/3) -> float:
    if m > len(weights):
        return 0.0  # degenerate; sampling picks all
    cutoff = math.ceil(theta*m)
    bad=0
    for t in range(trials):
        if DEBUG and t % max(1, trials//10)==0:
            dbg(f"MC {t}/{trials}")
        idx = sample_ppswor(weights, m)
        y = sum(A_unsorted[i] for i in idx)
        if y >= cutoff: bad+=1
    return bad/trials if trials>0 else 0.0

def main():
    path = os.environ.get("RB_PATH", f"./data/{dt.datetime.utcnow().date().isoformat()}")
    alphas = parse_env_list("RB_ALPHAS", "0.20,0.25,0.30,0.33")
    lambdas = parse_env_list("RB_LAMBDAS", "0,0.1,0.2,0.3")
    trials = int(os.environ.get("RB_TRIALS","20000"))
    m_strategy = os.environ.get("RB_M_STRATEGY","fraction:0.4")

    log("INFO", f"Batch path={path}, alphas={alphas}, lambdas={lambdas}, trials={trials}, m_strategy={m_strategy}")

    files = sorted(glob.glob(os.path.join(path, "*_weights.csv")))
    if not files:
        log("ERROR", f"No *_weights.csv in {path}")
        sys.exit(1)
    log("INFO", f"Found {len(files)} weight files.")

    out_csv = os.path.join(path, "risk_results.csv")
    header = ["file","n","lambda","m","alpha","mu","chernoff_bound","empirical_risk"]
    # overwrite per day (idempotent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for fi in files:
            import pandas as pd
            df = pd.read_csv(fi)
            n = int(len(df))
            m = parse_m_strategy(m_strategy, n)
            shares = load_shares(fi)

            # sort once
            perm = sorted(range(len(shares)), key=lambda i: shares[i], reverse=True)
            shares_desc = [shares[i] for i in perm]

            for alpha in alphas:
                # worst-case attacker set on this day
                A_desc = worst_adversary_mask(shares_desc, alpha)
                A_unsorted = [0]*len(shares)
                for j,i in enumerate(perm):
                    A_unsorted[i] = A_desc[j]

                for lam in lambdas:
                    wgt = mix_weights(shares, lam)
                    w_desc = [wgt[i] for i in perm]
                    mu = mu_expected(m, w_desc, A_desc)
                    bound = chernoff_upper_bound(m, mu, theta=1/3)

                    # adaptive trials: more when bound is not tiny
                    t = trials
                    if bound > 1e-2:
                        t = max(trials, 100000)
                    elif bound > 1e-3:
                        t = max(trials, 50000)

                    erisk = empirical_risk(wgt, A_unsorted, t, m, theta=1/3)
                    row = {
                        "file": os.path.basename(fi),
                        "n": str(n),
                        "lambda": f"{lam:.3f}",
                        "m": str(m),
                        "alpha": f"{alpha:.3f}",
                        "mu": f"{mu:.6f}",
                        "chernoff_bound": f"{bound:.6e}",
                        "empirical_risk": f"{erisk:.6f}",
                    }
                    w.writerow(row)
                    log("INFO", f"{os.path.basename(fi)} alpha={alpha:.2f} λ={lam:.2f} m={m}: mu={mu:.3f} "
                                f"bound≤{bound:.3e} emp={erisk:.6f}")

    log("INFO", f"Wrote consolidated results: {out_csv}")

if __name__ == "__main__":
    random.seed(42)
    main()
