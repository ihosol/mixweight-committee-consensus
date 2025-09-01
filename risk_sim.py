#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk simulator for "unsafe committee" events:
- Baseline (stake-proportional) vs our mixed weight with parameter lambda.

Reads *_weights.csv exported by snapshots.py, computes:
  - Worst-case adversary (greedy top shares to alpha)
  - Theoretical Chernoff-style upper bound on risk
  - Empirical risk via PPSWOR Monte Carlo
Prints results as CSV to stdout.

Logging:
  - INFO: overall progress
  - DEBUG: detailed steps (enable with env DEBUG=1)
"""

import os, csv, math, random, argparse, glob, sys
from typing import List, Tuple, Dict

DEBUG = os.environ.get("DEBUG", "0") == "1"

def log(level: str, msg: str):
    import datetime as dt
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[{ts}] {level} {msg}", file=sys.stdout)

def dbg(msg: str):
    if DEBUG:
        log("DEBUG", msg)

def load_shares_from_csv(path: str) -> List[float]:
    dbg(f"Loading shares from {path}")
    shares = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "share" not in r.fieldnames:
            raise RuntimeError(f"'share' column not found in {path}")
        for row in r:
            try:
                s = float(row["share"])
                if s > 0:
                    shares.append(s)
            except Exception:
                continue
    total = sum(shares) or 1.0
    shares = [s/total for s in shares]
    dbg(f"Loaded {len(shares)} shares; sum(normalized)={sum(shares):.6f}")
    return shares

def worst_adversary_mask(shares_desc: List[float], alpha: float) -> List[int]:
    mask = [0]*len(shares_desc)
    s = 0.0
    for i, v in enumerate(shares_desc):
        if s >= alpha:
            break
        mask[i] = 1
        s += v
    dbg(f"Worst-case adversary: alpha_target={alpha:.3f}, achieved≈{s:.6f}, attackers={sum(mask)}")
    return mask

def mix_weights(shares: List[float], lam: float) -> List[float]:
    n = len(shares)
    base = (1.0 - lam)
    add  = lam / n
    w = [base*s + add for s in shares]
    Z = sum(w) or 1.0
    w = [x/Z for x in w]
    return w

def mu_expected_malicious(m: int, w_desc: List[float], adv_mask: List[int]) -> float:
    piA = sum(w for w, a in zip(w_desc, adv_mask) if a)
    return m * piA

def chernoff_upper_bound(m: int, mu: float, theta: float = 1/3) -> float:
    t = m*theta
    if mu >= t:
        return 1.0
    if mu <= 0:
        return 0.0
    num = (t - mu)**2
    den = 3.0 * mu
    return math.exp(- num/den)

def sample_ppswor(weights: List[float], k: int) -> List[int]:
    # Efraimidis & Spirakis (2006): key_i = -ln(U)/w_i; take k smallest keys
    keys = []
    for i, w in enumerate(weights):
        if w <= 0:  # skip zero-weight
            continue
        u = random.random()
        key = -math.log(u) / w
        keys.append((key, i))
    keys.sort(key=lambda x: x[0])
    return [i for _, i in keys[:k]]

def empirical_risk(weights: List[float], adv_mask_unsorted: List[int], trials: int, m: int, theta: float = 1/3) -> float:
    cutoff = math.ceil(theta*m)
    bad = 0
    for t in range(trials):
        if t % max(1, trials//10) == 0:
            dbg(f"Monte Carlo progress: {t}/{trials} trials …")
        idx = sample_ppswor(weights, m)
        y = sum(adv_mask_unsorted[i] for i in idx)
        if y >= cutoff:
            bad += 1
    return bad / trials if trials > 0 else 0.0

def analyze_file(csv_path: str, m: int, alpha: float, lambdas: List[float], trials: int) -> List[Dict[str, str]]:
    log("INFO", f"Analyzing file: {os.path.basename(csv_path)} (m={m}, alpha={alpha}, trials={trials})")
    shares = load_shares_from_csv(csv_path)
    if not shares:
        raise RuntimeError(f"No nonzero shares in {csv_path}")

    # Sort once, build worst-case adversary on descending shares
    perm = sorted(range(len(shares)), key=lambda i: shares[i], reverse=True)
    shares_desc = [shares[i] for i in perm]
    A_desc = worst_adversary_mask(shares_desc, alpha)

    # Map mask back to original order
    adv_mask_unsorted = [0]*len(shares)
    for j, i in enumerate(perm):
        adv_mask_unsorted[i] = A_desc[j]

    out = []
    for lam in lambdas:
        log("INFO", f"λ={lam:.3f}: mixing weights and computing risks …")
        w = mix_weights(shares, lam)
        w_desc = [w[i] for i in perm]
        mu = mu_expected_malicious(m, w_desc, A_desc)
        bound = chernoff_upper_bound(m, mu, theta=1/3)
        erisk = empirical_risk(w, adv_mask_unsorted, trials, m, theta=1/3)
        out.append({
            "file": os.path.basename(csv_path),
            "n": str(len(shares)),
            "lambda": f"{lam:.3f}",
            "m": str(m),
            "alpha": f"{alpha:.3f}",
            "mu": f"{mu:.6f}",
            "chernoff_bound": f"{bound:.6e}",
            "empirical_risk": f"{erisk:.6f}",
        })
        log("INFO", f"λ={lam:.3f}: mu={mu:.4f}, Chernoff≤{bound:.3e}, empirical={erisk:.6f}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Folder with *_weights.csv OR a single *_weights.csv file")
    ap.add_argument("--m", type=int, default=60, help="Committee size")
    ap.add_argument("--alpha", type=float, default=0.20, help="Total adversarial stake fraction")
    ap.add_argument("--lambdas", type=str, default="0,0.1,0.2,0.3", help="Comma-separated list of λ")
    ap.add_argument("--trials", type=int, default=5000, help="Monte Carlo trials")
    ap.add_argument("--out", type=str, default="", help="Optional output CSV path (also prints to stdout)")
    args = ap.parse_args()

    # Gather files
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*_weights.csv")))
        if not files:
            log("ERROR", f"No *_weights.csv files found in {args.path}")
            sys.exit(1)
        log("INFO", f"Found {len(files)} weight files in folder.")
    else:
        files = [args.path]
        log("INFO", f"Single file mode: {files[0]}")

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    random.seed(42)

    # Run analysis
    all_rows = []
    for idx, f in enumerate(files, 1):
        log("INFO", f"Progress: {idx}/{len(files)} → {os.path.basename(f)}")
        rows = analyze_file(f, args.m, args.alpha, lambdas, args.trials)
        all_rows.extend(rows)

    # Print CSV to stdout
    header = ["file","n","lambda","m","alpha","mu","chernoff_bound","empirical_risk"]
    print(",".join(header))
    for r in all_rows:
        print(",".join(r[h] for h in header))

    # Optional write to file
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        log("INFO", f"Wrote results CSV: {args.out}")

    log("INFO", "Risk simulation finished.")

if __name__ == "__main__":
    main()
