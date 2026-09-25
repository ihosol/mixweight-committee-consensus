#!/usr/bin/env python3
"""Figures of the paper. Reads panel_exact*.csv from WOR_OUT (written by capture_exact.py) and the snapshots.

  python3 make_figures.py
Writes figures/fig_underestimation.png next to the code folder
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binom

from capture_exact import OUT, snapshot_files, capture_dust, capture_equal_split, mixed_weights

HERE = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"avalanche": "Avalanche", "celestia": "Celestia", "cosmoshub": "Cosmos Hub", "injective": "Injective",
         "near": "NEAR", "osmosis": "Osmosis", "sei": "Sei", "solana": "Solana", "sui": "Sui", "tezos": "Tezos"}


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7.5)


def main():
    med = pd.read_csv(OUT / "panel_exact_median.csv")
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "mathtext.fontset": "cm"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) underestimation factor against the committee size relative to the effective number
    #     of validators N_eff = 1/HHI (median over daily snapshots), m = 21, alpha = 0.20
    full = pd.read_csv(OUT / "panel_exact.csv")
    full["x"] = full.m * full.hhi
    s = (full[(full.regime == "m21") & (full.alpha == 0.20)]
         .groupby(["network", "lam"])[["x", "ratio"]].median().reset_index())
    for lam, color, marker in ((0.0, BLUE, "o"), (0.55, ORANGE, "s")):
        g = s[s.lam == lam]
        a.plot(g.x, g.ratio, linestyle="none", marker=marker, markersize=5.5, color=color,
               markeredgecolor="white", markeredgewidth=1.0, label=fr"$\lambda={lam:g}$")
    # NEAR/Osmosis and Avalanche/Solana nearly coincide, so each pair shares one label
    offsets = {"avalanche": (-14, 18), "sui": (5, -3), "near": (-8, 8),
               "celestia": (5, -4), "injective": (-58, 2), "sei": (-8, 7),
               "cosmoshub": (6, -10), "tezos": (-30, 7)}
    labels = dict(NAMES, near="NEAR, Osmosis", avalanche="Avalanche, Solana")
    for _, r in s[s.lam == 0.0].iterrows():
        if r.network in offsets:
            a.annotate(labels[r.network], (r.x, r.ratio), textcoords="offset points",
                       xytext=offsets[r.network], fontsize=7, color=MUTED)
    a.set_ylim(0.75, 6.3)
    a.axhline(1.0, color=MUTED, linewidth=1.0, linestyle="--")
    a.set_xlabel(r"committee size relative to effective validators, $m/N_{\mathrm{eff}}$", color=INK)
    a.set_ylabel(r"$P^{\ast}(q)\,/\,B^{\mathrm{wr}}(q)$", color=INK)
    a.set_title(r"(a) $m=21$, $\alpha=0.20$", fontsize=8.5, color=INK, loc="left")
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(a)

    # (b) capture probability of k equal identities, Cosmos Hub, alpha = 0.20, lambda = 0
    f = snapshot_files()["cosmoshub"][-1]
    st = pd.read_csv(f)["share"].to_numpy(float)
    st = st[st > 0]
    for m, color, marker in ((21, BLUE, "o"), (int(0.4 * len(st)), ORANGE, "s")):
        q = math.ceil(m / 3)
        h, p = mixed_weights(st, 0.20, 0.0)
        ks = [q, int(1.5 * q), 2 * q, 3 * q, 4 * q, 6 * q, 10 * q]
        ys = [capture_equal_split(h, p, k, m, q) for k in ks]
        x = [k / q for k in ks]
        b.plot(x, ys, color=color, linewidth=2, marker=marker, markersize=4.5,
               markeredgecolor="white", markeredgewidth=1.0, label=f"$m={m}$, $q={q}$")
        b.axhline(capture_dust(h, p, m, q), color=color, linewidth=1.0, linestyle=":")
        bw = binom.sf(q - 1, m, p)
        b.axhline(bw, color=color, linewidth=1.0, linestyle="--")
        b.annotate(fr"$B^{{\mathrm{{wr}}}}={bw:.2g}$", (10.0, bw), textcoords="offset points",
                   xytext=(0, 3), fontsize=7, color=color, ha="right")
    b.set_xlabel(r"number of coalition identities $K$, in units of $q$", color=INK)
    b.set_ylabel(r"$\Pr[X_A\geq q]$", color=INK)
    b.set_title(r"(b) Cosmos Hub, $\alpha=0.20$, $\lambda=0$", fontsize=8.5, color=INK, loc="left")
    b.text(10.0, 0.37, r"dotted: worst case $P^{\ast}(q)$" + "\n" + r"dashed: binomial $B^{\mathrm{wr}}(q)$",
           fontsize=7, color=MUTED, ha="right", va="top")
    b.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.28, 0.70))
    b.set_ylim(0, 0.95)
    style(b)

    fig.tight_layout()
    out = HERE / "figures/fig_underestimation.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
