#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
from pathlib import Path
from collections import defaultdict


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout


def parse_attrs_from_tx(tx_json):
    # Prefer top-level events; fallback to logs/events
    attrs = {}
    events = tx_json.get("events", [])
    if events:
        for ev in events:
            if ev.get("type") == "committee_drawn":
                for a in ev.get("attributes", []):
                    attrs[a.get("key")] = a.get("value")
                return attrs

    for lg in tx_json.get("logs", []):
        for ev in lg.get("events", []):
            if ev.get("type") == "committee_drawn":
                for a in ev.get("attributes", []):
                    attrs[a.get("key")] = a.get("value")
                return attrs
    return attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="poc/cosmos/artifacts/results/sybil_seats_vs_lambda.csv")
    ap.add_argument("--node", default="tcp://127.0.0.1:36657")
    ap.add_argument("--out", default="poc/cosmos/artifacts/results/sybil_dynamics_from_tx.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    if args.limit > 0:
        rows = rows[: args.limit]

    has_debug_cols = all(c in rows[0] for c in [
        "attacker_stake_ppm", "attacker_age_ppm", "attacker_weight_ppm", "attacker_validators"
    ])

    out_rows = []
    for r in rows:
        txh = r["txhash"]
        if has_debug_cols and (r.get("attacker_stake_ppm") or r.get("attacker_age_ppm") or r.get("attacker_weight_ppm")):
            attrs = {
                "attacker_stake_ppm": r.get("attacker_stake_ppm", ""),
                "attacker_age_ppm": r.get("attacker_age_ppm", ""),
                "attacker_weight_ppm": r.get("attacker_weight_ppm", ""),
                "attacker_validators": r.get("attacker_validators", ""),
                "persistence_tau_max_blocks": r.get("persistence_tau_max_blocks", ""),
                "persistence_zeta_ppm": r.get("persistence_zeta_ppm", ""),
                "tag": r.get("tag", ""),
            }
        else:
            try:
                raw = run(["chaind", "query", "tx", txh, "--node", args.node, "-o", "json"])
                txj = json.loads(raw)
                attrs = parse_attrs_from_tx(txj)
            except Exception:
                # If tx is unavailable (fresh-chain runs), keep row but empty attrs.
                attrs = {}

        out_rows.append(
            {
                "k": r["k"],
                "lambda_ppm": r["lambda_ppm"],
                "draw_i": r["draw_i"],
                "txhash": txh,
                "attacker_seats_share": r["attacker_seats_share"],
                "attacker_stake_ppm": attrs.get("attacker_stake_ppm", ""),
                "attacker_age_ppm": attrs.get("attacker_age_ppm", ""),
                "attacker_weight_ppm": attrs.get("attacker_weight_ppm", ""),
                "attacker_validators": attrs.get("attacker_validators", ""),
                "persistence_tau_max_blocks": attrs.get("persistence_tau_max_blocks", ""),
                "persistence_zeta_ppm": attrs.get("persistence_zeta_ppm", ""),
                "tag": attrs.get("tag", r.get("tag", "")),
            }
        )

    if not out_rows:
        raise SystemExit("No rows to process")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # quick aggregate
    agg = defaultdict(lambda: {"n": 0, "seat": 0.0, "stake": 0.0, "age": 0.0, "w": 0.0})
    for r in out_rows:
        key = (r["k"], r["lambda_ppm"])
        agg[key]["n"] += 1
        agg[key]["seat"] += float(r["attacker_seats_share"])
        agg[key]["stake"] += float(r["attacker_stake_ppm"] or 0) / 1e6
        agg[key]["age"] += float(r["attacker_age_ppm"] or 0) / 1e6
        agg[key]["w"] += float(r["attacker_weight_ppm"] or 0) / 1e6

    print(f"Wrote: {out_path}")
    print("k,lambda,n,mean_seat_share,mean_stake_share,mean_age_share,mean_weight_share")
    for (k, lam), v in sorted(agg.items(), key=lambda x: (int(x[0][0]), int(x[0][1]))):
        n = v["n"]
        print(
            f"{k},{lam},{n},{v['seat']/n:.6f},{v['stake']/n:.6f},{v['age']/n:.6f},{v['w']/n:.6f}"
        )


if __name__ == "__main__":
    main()
