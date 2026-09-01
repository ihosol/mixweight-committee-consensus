#!/usr/bin/env python3
"""Saturated-regime summary table for the RBHC red-flag suite (theta=1/3).

Why this exists: cc_drift_summary_table.csv (from make_drift_plots.py) reports the
realised seat share at the single FINAL epoch and the mixing lambda averaged over
the WHOLE run. Those two windows differ, which makes the table's "mean lambda"
look small (the controller ramps before saturating) and the single-epoch seat
share noisy (~48 draws). This script instead aggregates every realised quantity
over ONE window -- the saturation plateau (the last `WINDOW` epochs, where the
coalition stake mass alpha_t is at its maximum) -- pooling all draws across seeds
(~288 draws/cell), and adds the realised capture probabilities P(>=ceil(theta*m))
that cc_drift_summary_table.csv does not carry.

Output columns (per controller cell):
  seat_share         mean realised top-k seat share over the plateau
  P_ge_third         realised P(coalition seats >= ceil(m/3))   [Byzantine]
  P_ge_half          realised P(coalition seats >= ceil(m/2))   [majority]
  lambda_bar         mean realised mixing over the plateau (== lambda_max for RB)
  beta               coalition baseline mass (from cc_drift_summary_table.csv)
  binom_cert_pct     binomial certificate at the controller's lambda, per cent
  sat_frac           budget-satisfied fraction over the full run
  n_draws            pooled draws over the plateau

Usage:
  python3 poc/cosmos/scripts/make_cc_drift_saturated_table.py \
      [--suite article_rbhc_redflag_bft] [--window 6] [--m 9] [--theta 0.3333]
"""
import argparse
import csv
import glob
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CELLS = ["cc_drift_baseline", "cc_drift_signal", "cc_drift_hybrid_capped", "cc_drift_hybrid_concave"]


def _summary_csv_lookup():
    """Read cc_drift_summary_table.csv for beta, certificate and sat-frac."""
    p = REPO / "papers" / "risk_budget_controller_2026" / "figures" / "cc_drift_summary_table.csv"
    out = {}
    if not p.exists():
        return out
    for r in csv.DictReader(open(p)):
        out[r["cell"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="article_rbhc_redflag_bft")
    ap.add_argument("--window", type=int, default=6, help="number of trailing epochs forming the saturation plateau")
    ap.add_argument("--m", type=int, default=9)
    ap.add_argument("--theta", type=float, default=1.0 / 3.0)
    ap.add_argument("--out", default=str(REPO / "papers" / "risk_budget_controller_2026_ua"
                                         / "figures" / "cc_drift_saturated_table.csv"))
    args = ap.parse_args()

    root = REPO / "poc" / "cosmos" / "artifacts" / args.suite
    m = args.m
    q3 = math.ceil(args.theta * m)
    q5 = math.ceil(0.5 * m)
    summ = _summary_csv_lookup()

    rows = []
    for c in CELLS:
        seats, lams = [], []
        ge3 = ge5 = n = 0
        epochs = set()
        files = sorted(glob.glob(str(root / c / "seed_*" / "results" / "epoch_draws_latest.csv")))
        # Determine the plateau: the trailing `window` distinct epoch indices.
        all_e = set()
        for f in files:
            for r in csv.DictReader(open(f)):
                try:
                    all_e.add(int(r["epoch_idx"]))
                except (KeyError, ValueError):
                    pass
        plateau = set(sorted(all_e)[-args.window:]) if all_e else set()
        for f in files:
            for r in csv.DictReader(open(f)):
                try:
                    e = int(r["epoch_idx"])
                except (KeyError, ValueError):
                    continue
                if e not in plateau:
                    continue
                ts = int(float(r["tracked_seats"]))
                seats.append(float(r["tracked_seats_share"]))
                lams.append(float(r.get("lambda_auto_ppm") or 0) / 1e6)
                n += 1
                epochs.add(e)
                if ts >= q3:
                    ge3 += 1
                if ts >= q5:
                    ge5 += 1
        if n == 0:
            continue
        s = summ.get(c, {})
        beta = s.get("final_beta", "")
        try:
            bl = float(s.get("final_log10_Blambda", "nan"))
            cert = round(100.0 * (10 ** bl), 1)
        except ValueError:
            cert = ""
        rows.append({
            "cell": c,
            "seat_share": round(sum(seats) / n, 4),
            "P_ge_third": round(100.0 * ge3 / n, 1),
            "P_ge_half": round(100.0 * ge5 / n, 1),
            "lambda_bar": round(sum(lams) / len(lams), 4),
            "beta": beta,
            "binom_cert_pct": cert,
            "sat_frac": s.get("frac_epochs_satisfied", ""),
            "plateau_epochs": "-".join(str(x) for x in sorted(epochs)),
            "n_draws": n,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hdr = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  (q[1/3]={q3}, q[1/2]={q5}, window={args.window})")
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(str(r[k]) for k in hdr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
