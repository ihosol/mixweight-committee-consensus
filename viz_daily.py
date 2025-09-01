#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_results(path):
    fp = os.path.join(path, "risk_results.csv")
    if not os.path.exists(fp):
        print(f"[WARN] risk_results.csv not found in {path}", file=sys.stderr)
        sys.exit(0)
    df = pd.read_csv(fp)
    df["chain"] = df["file"].str.replace("_weights.csv","", regex=False)
    for col in ["lambda","alpha","mu","chernoff_bound","empirical_risk"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def plot_line(sel, ycol, title, ylabel, out_png):
    if sel.empty: return
    plt.figure()
    for chain, g in sel.groupby("chain"):
        gg = g.sort_values("lambda")
        plt.plot(gg["lambda"], gg[ycol], marker="o", label=chain)
    plt.xlabel("λ (mixing weight)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def build_reduction_table(df, alpha):
    rows = []
    sel = df[df["alpha"].round(6) == round(alpha,6)]
    for chain, g in sel.groupby("chain"):
        g = g.sort_values("lambda")
        base = g[g["lambda"] == g["lambda"].min()]
        last = g[g["lambda"] == g["lambda"].max()]
        if base.empty or last.empty: continue
        b0, b1 = float(base["chernoff_bound"].iloc[0]), float(last["chernoff_bound"].iloc[0])
        mu0, mu1 = float(base["mu"].iloc[0]), float(last["mu"].iloc[0])
        red_b = 100.0*(b0 - b1)/b0 if b0>0 else float("nan")
        red_mu = 100.0*(mu0 - mu1)/mu0 if mu0>0 else float("nan")
        rows.append({"chain": chain, "bound_reduction_%": round(red_b,2), "mu_reduction_%": round(red_mu,2)})
    return pd.DataFrame(rows).sort_values("chain").reset_index(drop=True)

def plot_bound_reduction_bar(summary, alpha, out_png):
    if summary.empty: return
    plt.figure()
    plt.bar(summary["chain"], summary["bound_reduction_%"])
    plt.xlabel("Chain")
    plt.ylabel("Percent reduction in Chernoff bound")
    plt.title(f"Reduction at λ_max vs λ_min (alpha={alpha})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def write_index_html(root, alphas, artifacts):
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'><title>Daily risk viz</title>")
    html.append("<style>body{font-family:system-ui,Arial,Helvetica,sans-serif;margin:24px;}h2{margin-top:28px}img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;padding:4px;margin:8px 0}</style>")
    html.append("</head><body><h1>Daily risk visualizations</h1>")
    for a in alphas:
        png1 = artifacts.get((a,"chernoff"))
        png2 = artifacts.get((a,"mu"))
        png3 = artifacts.get((a,"bar"))
        summ = artifacts.get((a,"csv"))
        html.append(f"<h2>alpha = {a}</h2>")
        if png1: html.append(f"<h3>Chernoff bound vs λ</h3><img src='{os.path.basename(png1)}'/>")
        if png2: html.append(f"<h3>μ vs λ</h3><img src='{os.path.basename(png2)}'/>")
        if png3: html.append(f"<h3>Percent reduction</h3><img src='{os.path.basename(png3)}'/>")
        if summ: html.append(f"<p><a href='{os.path.basename(summ)}'>Download CSV summary</a></p>")
    html.append("</body></html>")
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="data/YYYY-MM-DD folder with risk_results.csv")
    args = ap.parse_args()

    df = load_results(args.path)
    out_dir = os.path.join(args.path, "viz")
    ensure_dir(out_dir)

    alphas = sorted(df["alpha"].dropna().unique().tolist())
    artifacts = {}
    for a in alphas:
        sel = df[df["alpha"].round(6) == round(a,6)]
        chern_png = os.path.join(out_dir, f"chernoff_vs_lambda_alpha_{a}.png")
        mu_png    = os.path.join(out_dir, f"mu_vs_lambda_alpha_{a}.png")
        plot_line(sel, "chernoff_bound", f"Risk upper bound vs λ (alpha={a})", "Chernoff upper bound", chern_png)
        plot_line(sel, "mu",            f"Expected malicious seats μ vs λ (alpha={a})", "μ", mu_png)
        summ = build_reduction_table(df, a)
        summ_csv = os.path.join(out_dir, f"summary_alpha_{a}.csv")
        summ.to_csv(summ_csv, index=False)
        bar_png = os.path.join(out_dir, f"bound_reduction_alpha_{a}.png")
        plot_bound_reduction_bar(summ, a, bar_png)
        artifacts[(a,"chernoff")] = chern_png
        artifacts[(a,"mu")] = mu_png
        artifacts[(a,"bar")] = bar_png
        artifacts[(a,"csv")] = summ_csv

    write_index_html(out_dir, alphas, artifacts)
    print(f"[OK] Visualizations written to: {out_dir}")

if __name__ == "__main__":
    main()

