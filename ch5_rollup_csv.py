#!/usr/bin/env python3
"""
ch5_rollup_csv.py — Chapter 5 roll-up from CSV-only daily artifacts.

INPUT per day (data/YYYY-MM-DD/):
  - risk_results.csv
  - viz/summary_alpha_0.20.csv, viz/summary_alpha_0.25.csv, viz/summary_alpha_0.30.csv, viz/summary_alpha_0.33.csv

OUTPUTS under ch5_out/<START>_to_<END>/:
  tables/
    rollup_daily.csv
    table_ch5_alpha025.csv
    daily_alpha025_lambda0_to_03.csv
    viz_summaries_merged.csv
  figures/
    # per-chain (as before)
    {chain}_mu_vs_lambda_alpha{a}.png
    {chain}_log10bound_vs_lambda_alpha{a}.png
    {chain}_timeseries_mu_reduction_alpha025.png
    # NEW combined, all networks
    combined_log10bound_vs_lambda_alpha0.25.png
    combined_mu_vs_lambda_alpha0.20.png
    combined_mu_vs_lambda_alpha0.25.png
    combined_mu_vs_lambda_alpha0.30.png
    combined_mu_vs_lambda_alpha0.33.png
    combined_daily_mu_reduction_alpha0.25.png
"""

import argparse, re, sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def log(msg): print(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}", flush=True)

def date_range(start, end):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)

def infer_chain(file_name: str) -> str:
    n = Path(file_name).name
    if n.endswith("_weights.csv"):
        return n[:-len("_weights.csv")].lower()
    return Path(n).stem.lower()

def find_day_dirs(root: Path):
    if not root.exists(): return []
    days = [p for p in root.iterdir() if p.is_dir() and DATE_DIR_RE.match(p.name)]
    days.sort()
    return days

def read_risk_csv(day_dir: Path) -> pd.DataFrame | None:
    for candidate in ["risk_results.csv", "risk_result.csv"]:
        p = day_dir / candidate
        if p.exists():
            try:
                df = pd.read_csv(p)
                needed = ["file","n","lambda","m","alpha","mu","chernoff_bound","empirical_risk"]
                miss = [c for c in needed if c not in df.columns]
                if miss:
                    log(f"WARN {p} missing {miss}; skipping")
                    return None
                df["chain"] = df["file"].map(infer_chain)
                df["date"] = day_dir.name
                for c in ["n","lambda","m","alpha","mu","chernoff_bound","empirical_risk"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                if "baseline" not in df.columns:
                    df["baseline"] = "uniform"
                return df
            except Exception as e:
                log(f"WARN failed reading {p}: {e}")
                return None
    log(f"WARN no risk CSV in {day_dir}")
    return None

def read_viz_summaries(day_dir: Path, alphas=(0.20,0.25,0.30,0.33)) -> list[pd.DataFrame]:
    out = []
    vdir = day_dir / "viz"
    if not vdir.exists(): return out
    for a in alphas:
        p = vdir / f"summary_alpha_{a}.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                cols = {c.lower(): c for c in df.columns}
                req = ["chain", "bound_reduction_%", "mu_reduction_%"]
                missing = [x for x in req if x not in cols]
                alt = {"bound_reduction_%": ["bound_reduction", "bound_red_%", "bound_red"],
                       "mu_reduction_%": ["mu_reduction", "mu_red_%", "mu_red"]}
                if missing:
                    for need in missing[:]:
                        for altname in alt.get(need, []):
                            if altname in cols:
                                cols[need] = cols[altname]
                                missing.remove(need)
                                break
                if any(x not in cols for x in req):
                    log(f"WARN {p} lacks required columns; skipping")
                    continue
                df = df.rename(columns={
                    cols["chain"]: "chain",
                    cols["bound_reduction_%"]: "bound_reduction_%",
                    cols["mu_reduction_%"]: "mu_reduction_%"
                })
                df["alpha"] = float(a)
                df["date"] = day_dir.name
                df["chain"] = df["chain"].astype(str).str.strip().str.lower()
                out.append(df[["date","chain","alpha","bound_reduction_%","mu_reduction_%"]].copy())
            except Exception as e:
                log(f"WARN failed reading {p}: {e}")
                continue
    return out

def safe_log10(x):
    x = float(x)
    return -300.0 if x <= 0 else float(np.log10(x))

def build_daily_mu_reduction_from_risk(risk_df: pd.DataFrame, alpha_target=0.25):
    if risk_df.empty: return pd.DataFrame()
    cut = risk_df[np.isclose(risk_df["alpha"], alpha_target)]
    rows = []
    for (d, ch, base), g in cut.groupby(["date","chain","baseline"]):
        g0 = g[np.isclose(g["lambda"], 0.0)]
        g3 = g[np.isclose(g["lambda"], 0.3)]
        if g0.empty or g3.empty: 
            continue
        mu0 = float(g0["mu"].mean()); mu3 = float(g3["mu"].mean())
        red = 100.0 * (1.0 - mu3 / max(mu0, 1e-12))
        b0 = g0["chernoff_bound"].apply(safe_log10).median()
        b3 = g3["chernoff_bound"].apply(safe_log10).median()
        d_orders = b0 - b3
        rows.append({
            "date": d, "chain": ch, "baseline": base, "alpha": alpha_target,
            "mu0": mu0, "mu3": mu3, "mu_reduction_%": red,
            "delta_orders_log10_bound": d_orders
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="CSV-only Chapter 5 roll-up (with combined figures)")
    ap.add_argument("--root", default="data", help="Root with YYYY-MM-DD subfolders (default: data)")
    ap.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD). If unset, use --days window")
    ap.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), inclusive (default: today UTC)")
    ap.add_argument("--days", type=int, default=60, help="If --start unset, window size (default: 60)")
    ap.add_argument("--alphas", default="0.20,0.25,0.30,0.33", help="Comma-separated α list expected in viz/")
    ap.add_argument("--outdir", default="ch5_out", help="Output dir (default: ch5_out)")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    end = args.end or datetime.utcnow().strftime("%Y-%m-%d")
    if args.start:
        start = args.start
    else:
        d1 = datetime.strptime(end, "%Y-%m-%d").date()
        d0 = d1 - timedelta(days=args.days-1)
        start = d0.strftime("%Y-%m-%d")

    root = Path(args.root)
    day_dirs = [d for d in find_day_dirs(root) if start <= d.name <= end]
    if not day_dirs:
        log(f"ERR no dated folders in range {start}..{end} under {root}")
        sys.exit(2)

    out_root = Path(args.outdir) / f"{start}_to_{end}"
    tables = out_root / "tables"
    figs = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    log(f"Window: {start} → {end}  (days={len(day_dirs)})")

    # ingest
    risk_frames, viz_frames = [], []
    for d in day_dirs:
        r = read_risk_csv(d)
        if r is not None: risk_frames.append(r)
        viz_frames.extend(read_viz_summaries(d, alphas))

    if not risk_frames:
        log("ERR no risk_results.csv found across the window")
        sys.exit(2)

    risk = pd.concat(risk_frames, ignore_index=True)
    risk.to_csv(tables / "rollup_daily.csv", index=False)
    log(f"[OK] wrote {tables/'rollup_daily.csv'} rows={len(risk)}")

    viz = pd.concat(viz_frames, ignore_index=True) if viz_frames else pd.DataFrame()
    if not viz.empty:
        viz.to_csv(tables / "viz_summaries_merged.csv", index=False)
        log(f"[OK] wrote {tables/'viz_summaries_merged.csv'} rows={len(viz)}")
    else:
        log("INFO no viz summaries found; reductions will be derived from risk CSV")

    # main table α=0.25
    a_sel = 0.25
    cut = risk[np.isclose(risk["alpha"], a_sel)]
    main_rows = []
    for (ch, base), g in cut.groupby(["chain","baseline"]):
        g0 = g[np.isclose(g["lambda"], 0.0)]
        g3 = g[np.isclose(g["lambda"], 0.3)]
        if g0.empty or g3.empty: 
            continue
        mu0 = g0["mu"].median()
        mu3 = g3["mu"].median()
        pct = 100.0 * (1.0 - mu3 / max(mu0, 1e-12))
        d_orders = g0["chernoff_bound"].apply(safe_log10).median() - \
                   g3["chernoff_bound"].apply(safe_log10).median()
        main_rows.append({
            "chain": ch, "baseline": base, "alpha": a_sel,
            "mu(0)_median": mu0, "mu(0.3)_median": mu3,
            "mu_reduction_%_median": pct,
            "delta_orders_log10_bound_median": d_orders
        })
    ch5 = pd.DataFrame(main_rows).sort_values(["baseline","chain"])
    ch5_out = tables / "table_ch5_alpha025.csv"
    ch5.to_csv(ch5_out, index=False)
    log(f"[OK] wrote {ch5_out} rows={len(ch5)}")

    # daily μ-reduction time series α=0.25
    if not viz.empty and "mu_reduction_%" in viz.columns:
        daily_ts = viz[viz["alpha"].round(6) == round(a_sel,6)][["date","chain","alpha","mu_reduction_%"]].copy()
        daily_ts["baseline"] = "uniform"
    else:
        daily_ts = build_daily_mu_reduction_from_risk(risk, alpha_target=a_sel)
    ts_out = tables / "daily_alpha025_lambda0_to_03.csv"
    daily_ts.to_csv(ts_out, index=False)
    log(f"[OK] wrote {ts_out} rows={len(daily_ts)}")

    # ---------- per-chain figures (kept) ----------
    chains = sorted(risk["chain"].unique())
    alphas_present = sorted(risk["alpha"].dropna().unique().tolist())

    def plot_mu_vs_lambda_IQR(chain, alpha):
        sub = risk[(risk["chain"]==chain) & (np.isclose(risk["alpha"], alpha))]
        if sub.empty: return
        xs = sorted(sub["lambda"].unique())
        meds, q25, q75 = [], [], []
        for lam in xs:
            vals = sub[np.isclose(sub["lambda"], lam)]["mu"].dropna()
            if len(vals)==0: continue
            meds.append(np.median(vals))
            q25.append(np.quantile(vals, 0.25))
            q75.append(np.quantile(vals, 0.75))
        if not meds: return
        plt.figure()
        plt.plot(xs, meds, marker='o')
        plt.fill_between(xs, q25, q75, alpha=0.2)
        plt.xlabel("λ"); plt.ylabel("Mean adversarial seats μ")
        plt.title(f"{chain}: μ vs λ (α={alpha})")
        plt.tight_layout()
        plt.savefig(figs / f"{chain}_mu_vs_lambda_alpha{alpha}.png")
        plt.close()

    def plot_log10bound_vs_lambda_IQR(chain, alpha):
        sub = risk[(risk["chain"]==chain) & (np.isclose(risk["alpha"], alpha))]
        if sub.empty: return
        xs = sorted(sub["lambda"].unique())
        meds, q25, q75 = [], [], []
        for lam in xs:
            vals = sub[np.isclose(sub["lambda"], lam)]["chernoff_bound"].dropna().clip(lower=1e-300)
            if len(vals)==0: continue
            v = np.log10(vals)
            meds.append(np.median(v))
            q25.append(np.quantile(v, 0.25))
            q75.append(np.quantile(v, 0.75))
        if not meds: return
        plt.figure()
        plt.plot(xs, meds, marker='o')
        plt.fill_between(xs, q25, q75, alpha=0.2)
        plt.xlabel("λ"); plt.ylabel("log10 Chernoff bound")
        plt.title(f"{chain}: log10 bound vs λ (α={alpha})")
        plt.tight_layout()
        plt.savefig(figs / f"{chain}_log10bound_vs_lambda_alpha{alpha}.png")
        plt.close()

    def plot_timeseries_mu_reduction_alpha025(chain):
        sub = daily_ts[daily_ts["chain"]==chain]
        if sub.empty: return
        sub = sub.sort_values("date")
        x = np.arange(len(sub))
        plt.figure(figsize=(10,3.5))
        plt.plot(x, sub["mu_reduction_%"], marker='o')
        step = max(1, len(x)//10)
        plt.xticks(x[::step], sub["date"].iloc[::step], rotation=45, ha='right')
        plt.ylabel("% μ reduction (λ: 0→0.3)"); plt.xlabel("date")
        plt.title(f"{chain}: daily μ reduction at α=0.25")
        plt.tight_layout()
        plt.savefig(figs / f"{chain}_timeseries_mu_reduction_alpha025.png")
        plt.close()

    for ch in chains:
        for a in alphas_present:
            plot_mu_vs_lambda_IQR(ch, a)
            plot_log10bound_vs_lambda_IQR(ch, a)
        plot_timeseries_mu_reduction_alpha025(ch)

    # ---------- NEW: combined, all networks figures ----------

    def combined_mu_vs_lambda(alpha):
        sub = risk[np.isclose(risk["alpha"], alpha)]
        if sub.empty: return
        plt.figure()
        for ch, g in sub.groupby("chain"):
            med = g.groupby("lambda")["mu"].median()
            if len(med)==0: continue
            plt.plot(med.index, med.values, marker='o', label=ch)
        plt.xlabel("λ"); plt.ylabel("Median μ across days")
        plt.title(f"All networks — μ vs λ (α={alpha})")
        plt.grid(True, linewidth=0.4, alpha=0.4)
        # place legend outside if many series
        n = sub["chain"].nunique()
        if n > 10:
            plt.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=8)
            plt.tight_layout(rect=(0,0,0.8,1))
        else:
            plt.legend()
            plt.tight_layout()
        plt.savefig(figs / f"combined_mu_vs_lambda_alpha{alpha}.png", dpi=160)
        plt.close()

    def combined_log10bound_vs_lambda(alpha):
        sub = risk[np.isclose(risk["alpha"], alpha)]
        if sub.empty: return
        plt.figure()
        for ch, g in sub.groupby("chain"):
            med = g.groupby("lambda")["chernoff_bound"].median().clip(lower=1e-300)
            if len(med)==0: continue
            plt.plot(med.index, np.log10(med.values), marker='o', label=ch)
        plt.xlabel("λ"); plt.ylabel("Median log10 Chernoff bound")
        plt.title(f"All networks — log10 bound vs λ (α={alpha})")
        plt.grid(True, linewidth=0.4, alpha=0.4)
        n = sub["chain"].nunique()
        if n > 10:
            plt.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=8)
            plt.tight_layout(rect=(0,0,0.8,1))
        else:
            plt.legend()
            plt.tight_layout()
        plt.savefig(figs / f"combined_log10bound_vs_lambda_alpha{alpha}.png", dpi=160)
        plt.close()

    def combined_daily_reduction_alpha025():
        sub = daily_ts.copy()
        if sub.empty: return
        sub = sub.sort_values(["chain","date"])
        # to keep x-axis consistent across chains:
        dates = sorted(sub["date"].unique())
        x = np.arange(len(dates))
        plt.figure(figsize=(11,4))
        for ch, g in sub.groupby("chain"):
            g = g.set_index("date").reindex(dates)
            plt.plot(x, g["mu_reduction_%"], marker='o', linewidth=1.0, label=ch)
        step = max(1, len(x)//10)
        plt.xticks(x[::step], [dates[i] for i in x[::step]], rotation=45, ha='right')
        plt.ylabel("% μ reduction (λ: 0→0.3) at α=0.25"); plt.xlabel("date")
        plt.title("All networks — daily μ reduction time series (α=0.25)")
        plt.grid(True, linewidth=0.4, alpha=0.4)
        n = sub["chain"].nunique()
        if n > 10:
            plt.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=8, ncols=1)
            plt.tight_layout(rect=(0,0,0.8,1))
        else:
            plt.legend()
            plt.tight_layout()
        plt.savefig(figs / f"combined_daily_mu_reduction_alpha0.25.png", dpi=160)
        plt.close()

    # generate combined plots
    for a in sorted(alphas_present):
        combined_mu_vs_lambda(a)
    combined_log10bound_vs_lambda(0.25)  # specifically requested α=0.25
    combined_daily_reduction_alpha025()

    log(f"[DONE] Outputs under: {out_root}")
    log(f" - Tables: {tables}")
    log(f" - Figures: {figs}")

if __name__ == "__main__":
    main()



