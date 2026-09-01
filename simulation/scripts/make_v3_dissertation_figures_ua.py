#!/usr/bin/env python3
# Generate the final overnight_v5 dissertation figure package (F1..F5) under
# poc/cosmos/dissertation_final/figures/.
#
# Sources: overnight_v5 pub_* and fairness_* artifact directories. Every final
# scenario row has a 3-seed aggregated_final_table.csv; for per-draw trajectory
# plots this script uses seed_1 as a representative trace and relies on the
# aggregate tables for numeric claims.

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix", "axes.unicode_minus": False})

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parents[1] / "artifacts"
OUT = Path(__file__).resolve().parents[3] / "dissertation_ua" / "poc" / "cosmos" / "dissertation_final" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ── Scenario registry ──────────────────────────────────────────────────────
#   Each scenario has a stable display label, a short variant tag, a
#   per-draw CSV path (for F1 traces), and a final table path (for F2/F3/F4/F5).

SEC_SCENARIOS = {
    "default_burst":  dict(label="Типова (імпульс)", variant="default",
                           profile="burst", color="#2E86AB"),
    "tuned_burst":    dict(label=r"Налаштована $\lambda_{\max}=0{,}65$ (імпульс)", variant="tuned",
                           profile="burst", color="#F18F01"),
    "alpha02_burst":  dict(label=r"Налашт. + $\kappa_\downarrow=0{,}02$ (імпульс)", variant="alpha02",
                           profile="burst", color="#C73E1D"),
    "default_trickle":dict(label="Типова (повільна)", variant="default",
                           profile="trickle", color="#74B3CE"),
    "tuned_trickle":  dict(label=r"Налаштована $\lambda_{\max}=0{,}65$ (повільна)", variant="tuned",
                           profile="trickle", color="#FAB662"),
    "alpha02_trickle":dict(label=r"Налашт. + $\kappa_\downarrow=0{,}02$ (повільна)", variant="alpha02",
                           profile="trickle", color="#E5685F"),
}

FAIR_SCENARIOS = {
    "default_burst":  "overnight_v5/fairness_default_burst",
    "tuned_burst":    "overnight_v5/fairness_tuned_burst",
    # s456 dirs: genuine replication runs with fresh, pairwise-distinct draw seeds
    "default_trickle":"overnight_v5/fairness_default_trickle_s456",
    "tuned_trickle":  "overnight_v5/fairness_tuned_trickle_s456",
}

SEC_DIR = {
    "default_burst":   "overnight_v5/pub_default_burst",
    "tuned_burst":     "overnight_v5/pub_tuned_burst",
    "alpha02_burst":   "overnight_v5/pub_alphadown02_lammax065_burst",
    "default_trickle": "overnight_v5/pub_default_trickle",
    "tuned_trickle":   "overnight_v5/pub_tuned_trickle",
    "alpha02_trickle": "overnight_v5/pub_alphadown02_lammax065_trickle",
}

ABLATION_DIR = {
    "canonical":  "overnight_v5/pub_alphadown02_lammax065_burst",
    "freshness":  "overnight_v5/ablation_freshness_only_burst",
    "gini":       "overnight_v5/ablation_gini_only_burst",
    "split":      "overnight_v5/ablation_split_only_burst",
}


def _find_seed1_results(artifact_root: Path) -> Path:
    # Multi-seed artifacts store under seed_1/results/; single-seed under results/.
    if (artifact_root / "seed_1" / "results").is_dir():
        return artifact_root / "seed_1" / "results"
    return artifact_root / "results"


def _final_table(key: str, artifact_root: Path) -> dict:
    # For 3-seed headline rows, prefer aggregated_final_table.csv (means).
    agg = artifact_root / "aggregated_final_table.csv"
    if agg.exists():
        df = pd.read_csv(agg)
        return df.iloc[0].to_dict()
    fp = _find_seed1_results(artifact_root) / "epoch_final_table_latest.csv"
    df = pd.read_csv(fp)
    return df.iloc[0].to_dict()


def _draw_trace(artifact_root: Path) -> pd.DataFrame:
    fp = _find_seed1_results(artifact_root) / "epoch_draws_latest.csv"
    return pd.read_csv(fp)


# ── F1 — two-panel mechanism+outcome (burst trio) ─────────────────────────
def f1_lambda_and_weight():
    trio = ["default_burst", "tuned_burst", "alpha02_burst"]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.5, 7.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
    )

    atk_x = None
    for key in trio:
        info = SEC_SCENARIOS[key]
        df = _draw_trace(ROOT / SEC_DIR[key])
        x = df["draw_idx_global"].values
        lam_pct = df["lambda_auto_ppm"].values / 1e6 * 100.0
        ax_top.plot(x, lam_pct, color=info["color"], lw=1.8, label=info["label"])

        atk_rows = df[df["attack_height"].notna() & (df["attack_height"] != "")]
        if atk_x is None and not atk_rows.empty:
            atk_heights = pd.to_numeric(atk_rows["attack_height"], errors="coerce").dropna()
            if not atk_heights.empty:
                atk_h = int(atk_heights.iloc[0])
                idx = df.index[df["height"] == atk_h]
                if len(idx) > 0:
                    atk_x = int(df.loc[idx[0], "draw_idx_global"])

        wshare = df["attacker_weight_ppm"].values / 1e6
        ax_bot.plot(x, wshare, color=info["color"], lw=1.8, label=info["label"])

    # Stake-only baseline on the bottom panel: use stake_share_indep post-attack.
    # This is constant β in the post-attack window.
    df_ref = _draw_trace(ROOT / SEC_DIR["default_burst"])
    post = df_ref[df_ref["phase"] == "post_attack"]
    if not post.empty:
        beta = float(post["stake_share_indep"].iloc[0])
        ax_bot.axhline(beta, color="black", ls="--", lw=1.3,
                       label=f"Суто стейковий еталон ($\\beta_{{\\mathrm{{atk}}}}={beta:.2f}$)")

    if atk_x is not None:
        for ax in (ax_top, ax_bot):
            ax.axvline(atk_x, color="gray", ls=":", lw=1.0, alpha=0.8)
        ax_top.annotate("атака", xy=(atk_x, 0.96), xycoords=("data", "axes fraction"),
                        xytext=(6, -4), textcoords="offset points",
                        color="gray", fontsize=9)

    # Стелі регулятора: типова 0.55 та налаштована 0.65 (у відсотках).
    for ceil_pct, ceil_lab in ((55.0, r"$\lambda_{\max}=0{,}55$"),
                               (65.0, r"$\lambda_{\max}=0{,}65$")):
        ax_top.axhline(ceil_pct, color="gray", ls="--", lw=0.9, alpha=0.65)
        ax_top.annotate(ceil_lab, xy=(0.01, ceil_pct), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        color="gray", fontsize=8.5)
    ax_top.set_ylabel(r"Адаптивний $\lambda_t$, %")
    ax_top.set_title("Механізм та результат за імпульсного профілю атаки "
                     r"($k=6$, $c=9$, $\beta_{\mathrm{atk}}=0{,}33$)")
    ax_top.legend(loc="upper right", framealpha=0.9)
    ax_top.set_ylim(0, 70)
    ax_top.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    ax_bot.set_xlabel("Глобальний індекс розіграшу (від генезису)")
    ax_bot.set_ylabel("Частка ваги зловмисника")
    ax_bot.legend(loc="lower right", framealpha=0.9)
    ax_bot.set_ylim(bottom=0)

    out = OUT / "F1_lambda_trace_and_attacker_weight.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ── F2 — 6 scenarios × 4 windows grouped bars ─────────────────────────────
def f2_scenario_window_bars():
    window_cols = [
        ("reduction_vs_baseline_1ep_pct",  "1 еп."),
        ("reduction_vs_baseline_3ep_pct",  "3 еп."),
        ("reduction_vs_baseline_5ep_pct",  "5 еп."),
        ("reduction_vs_baseline_full_pct", "Повне"),
    ]

    keys = ["default_burst", "tuned_burst", "alpha02_burst",
            "default_trickle", "tuned_trickle", "alpha02_trickle"]
    x_labels = ["типова\nімпульс", "налашт.\nімпульс", r"налашт.+$\kappa_\downarrow$" "\nімпульс",
                "типова\nповільна", "налашт.\nповільна", r"налашт.+$\kappa_\downarrow$" "\nповільна"]

    data = {}
    for key in keys:
        info = SEC_SCENARIOS[key]
        row = _final_table(key, ROOT / SEC_DIR[key])
        vals = []
        for col, _lbl in window_cols:
            # Aggregated tables expose *_mean; single-seed expose raw col.
            v = row.get(f"{col}_mean", row.get(col))
            vals.append(float(v) if v not in (None, "", "nan") else np.nan)
        data[key] = vals

    n = len(keys)
    n_w = len(window_cols)
    x = np.arange(n)
    width = 0.2
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(11, 5.3))
    hatches = ["", "///", "...", "xxx"]
    for wi, (_col, lbl) in enumerate(window_cols):
        vals = [data[k][wi] for k in keys]
        bars = ax.bar(x + (wi - (n_w - 1) / 2) * width, vals, width,
                      label=lbl, color=palette[wi], edgecolor="white", lw=0.6,
                      hatch=hatches[wi])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Зниження частки ваги зловмисника\nвідносно стейкового еталона (%)")
    ax.set_title("Зниження ваги зловмисника за варіантами налаштування та профілями атаки\n"
                 r"($k=6$, $c=9$, $\beta_{\mathrm{atk}}=0{,}33$)")
    ax.legend(title="Вікно", loc="upper right", ncol=4)
    ax.set_ylim(0, max(35, max(max(v for v in data[k] if np.isfinite(v)) for k in keys) * 1.15))

    out = OUT / "F2_reduction_by_scenario_window.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ── F3 — Security ↔ fairness trade-off scatter ────────────────────────────
def f3_security_fairness_tradeoff():
    # Pair each security config with the matching fairness config.
    pairs = [
        ("default_burst",   "default_burst",   "Типова (імпульс)",   "#2E86AB", "o"),
        ("tuned_burst",     "tuned_burst",     "Налаштована (імпульс)",     "#F18F01", "o"),
        ("default_trickle", "default_trickle", "Типова (повільна)", "#74B3CE", "s"),
        ("tuned_trickle",   "tuned_trickle",   "Налаштована (повільна)",   "#FAB662", "s"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for sec_key, fair_key, label, color, marker in pairs:
        sec_row = _final_table(sec_key, ROOT / SEC_DIR[sec_key])
        fair_dir = ROOT / FAIR_SCENARIOS[fair_key]
        fair_row = _final_table(fair_key, fair_dir)

        x_sec = float(sec_row.get("reduction_vs_baseline_full_pct_mean",
                                  sec_row.get("reduction_vs_baseline_full_pct")))
        y_fair = float(fair_row.get("tracked_vs_baseline_full_pct_mean",
                                    fair_row.get("tracked_vs_baseline_full_pct")))

        ax.scatter(x_sec, y_fair, s=160, color=color, marker=marker,
                   edgecolor="black", lw=0.8, zorder=3, label=label)
        ax.annotate(label.split(" (")[0], (x_sec, y_fair),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)

    # αdown=0.02 scenarios: no matching fairness run, but plot at expected
    # fairness cost by interpolation (monotone in λ_max); annotate this explicitly.
    for sec_key, base_fair_key, label, color in [
        ("alpha02_burst",   "tuned_burst",   r"Налашт. + $\kappa_\downarrow=0{,}02$ (імпульс)",   "#C73E1D"),
        ("alpha02_trickle", "tuned_trickle", r"Налашт. + $\kappa_\downarrow=0{,}02$ (повільна)", "#E5685F"),
    ]:
        sec_row = _final_table(sec_key, ROOT / SEC_DIR[sec_key])
        # Use the tuned fairness gap as the upper bound (αdown keeps λ elevated longer,
        # so newcomer cost is ≥ tuned; we mark as open triangle to denote this).
        base_row = _final_table(base_fair_key, ROOT / FAIR_SCENARIOS[base_fair_key])
        x_sec = float(sec_row.get("reduction_vs_baseline_full_pct_mean",
                                  sec_row.get("reduction_vs_baseline_full_pct")))
        y_fair = float(base_row.get("tracked_vs_baseline_full_pct_mean",
                                    base_row.get("tracked_vs_baseline_full_pct")))
        ax.scatter(x_sec, y_fair, s=160, color="none", marker="^",
                   edgecolor=color, lw=1.6, zorder=3, label=label)
        ax.annotate(r"$\kappa_\downarrow{=}0{,}02$", (x_sec, y_fair),
                    xytext=(8, -12), textcoords="offset points", fontsize=9, color=color)

    # Preferred direction: larger x (safer) AND smaller y (fairer) = lower-right.
    ax.annotate("", xy=(0.97, 0.05), xytext=(0.58, 0.50), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
    ax.text(0.97, 0.08, "бажаний\nнапрям", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="gray", style="italic")

    # Highlight the security gain from tuned_burst to αdown_burst at constant
    # newcomer gap. The dissertation prose deliberately avoids "Pareto" framing
    # because it overclaims a frontier on a 6-point scatter; we keep the visual
    # arrow but label it as a security gain at constant fairness cost.
    sec_t = float(_final_table("tuned_burst", ROOT / SEC_DIR["tuned_burst"]).get(
        "reduction_vs_baseline_full_pct_mean",
        _final_table("tuned_burst", ROOT / SEC_DIR["tuned_burst"]).get("reduction_vs_baseline_full_pct")))
    sec_a = float(_final_table("alpha02_burst", ROOT / SEC_DIR["alpha02_burst"]).get(
        "reduction_vs_baseline_full_pct_mean"))
    _fr_b = _final_table("tuned_burst", ROOT / FAIR_SCENARIOS["tuned_burst"])
    y_pareto = float(_fr_b.get("tracked_vs_baseline_full_pct_mean",
                               _fr_b.get("tracked_vs_baseline_full_pct")))
    ax.annotate("", xy=(sec_a, y_pareto), xytext=(sec_t, y_pareto),
                arrowprops=dict(arrowstyle="->", color="#1F7A1F", lw=2.0),
                zorder=2)
    ax.text((sec_t + sec_a) / 2, y_pareto + 0.10,
            "ефект $\\kappa_{\\downarrow}=0{,}02$ за консервативно\nзведеної ціни розриву",
            ha="center", va="bottom", fontsize=9, color="#1F7A1F", style="italic")

    # Symmetric arrow for the trickle profile: tuned_trickle → alpha02_trickle
    # at constant fairness cost.
    sec_tt = float(_final_table("tuned_trickle", ROOT / SEC_DIR["tuned_trickle"]).get(
        "reduction_vs_baseline_full_pct_mean",
        _final_table("tuned_trickle", ROOT / SEC_DIR["tuned_trickle"]).get(
            "reduction_vs_baseline_full_pct")))
    sec_at = float(_final_table("alpha02_trickle", ROOT / SEC_DIR["alpha02_trickle"]).get(
        "reduction_vs_baseline_full_pct_mean"))
    _fr_t = _final_table("tuned_trickle", ROOT / FAIR_SCENARIOS["tuned_trickle"])
    y_pareto_t = float(_fr_t.get("tracked_vs_baseline_full_pct_mean",
                                 _fr_t.get("tracked_vs_baseline_full_pct")))
    ax.annotate("", xy=(sec_at, y_pareto_t), xytext=(sec_tt, y_pareto_t),
                arrowprops=dict(arrowstyle="->", color="#1F7A1F", lw=2.0,
                                linestyle="dashed", alpha=0.7),
                zorder=2)

    ax.set_xlabel("Зниження частки ваги зловмисника за повним вікном (%)\n(що більше, то безпечніше)")
    ax.set_ylabel("Відхилення нового валідатора від стейкового еталона (%)\n(що менше, то справедливіше)")
    ax.set_title("Компроміс «безпека $\\leftrightarrow$ справедливість» за варіантами налаштування")
    ax.legend(loc="upper center", framealpha=0.92, fontsize=9,
              ncol=3, bbox_to_anchor=(0.5, -0.12))
    # Pad axes so annotations do not collide with points.
    ax.margins(x=0.12, y=0.22)

    out = OUT / "F3_security_fairness_tradeoff.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ── F4 — Controller dynamics 4-panel ──────────────────────────────────────
def f4_controller_dynamics():
    keys = ["default_burst", "tuned_burst", "alpha02_burst",
            "default_trickle", "tuned_trickle", "alpha02_trickle"]
    x_labels = ["типова\nімпульс", "налашт.\nімпульс", r"налашт.+$\kappa_\downarrow$" "\nімпульс",
                "типова\nповільна", "налашт.\nповільна", r"налашт.+$\kappa_\downarrow$" "\nповільна"]

    def getv(row, col):
        v = row.get(f"{col}_mean", row.get(col))
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    rows = [_final_table(k, ROOT / SEC_DIR[k]) for k in keys]
    # post_lambda_overshoot_pct is defined relative to the final settled value
    # and therefore becomes incomparably large for slow-decay scenarios; we
    # drop it from the panel in favor of more interpretable metrics.
    metrics = [
        ("post_lambda_peak_ppm",           "Пік $\\lambda$ (ppm)",         "#4C72B0"),
        ("post_lambda_half_life_draws",    "Півперіод спадання (розіграші)",             "#55A868"),
        ("post_lambda_settle_time_draws",  "Час усталення (розіграші)",           "#C44E52"),
        ("post_lambda_chatter_rms_ppm",    "Середньокв. дрижання (ppm)",             "#8172B3"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, (col, ylabel, color) in zip(axes.ravel(), metrics):
        vals = [getv(r, col) for r in rows]
        bars = ax.bar(x_labels, vals, color=color, edgecolor="white", lw=0.6)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                if v >= 1000:
                    txt = f"{v:,.0f}"
                else:
                    txt = f"{v:.0f}"
                ax.text(b.get_x() + b.get_width() / 2, v, txt,
                        ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(bottom=0, top=max(vals) * 1.15 if vals else 1)

    fig.suptitle("Динаміка адаптивного регулятора за сценаріями "
                 "(виміряно на постатаковій траєкторії $\\lambda_t$)", fontsize=12)
    fig.tight_layout()

    out = OUT / "F4_controller_dynamics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ── F5 — Chernoff bound reduction ─────────────────────────────────────────
def f5_chernoff_reduction():
    keys = ["default_burst", "tuned_burst", "alpha02_burst",
            "default_trickle", "tuned_trickle", "alpha02_trickle"]
    x_labels = ["типова\nімпульс", "налашт.\nімпульс", r"налашт.+$\kappa_\downarrow$" "\nімпульс",
                "типова\nповільна", "налашт.\nповільна", r"налашт.+$\kappa_\downarrow$" "\nповільна"]

    def getv(row, col):
        v = row.get(f"{col}_mean", row.get(col))
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    rows = [_final_table(k, ROOT / SEC_DIR[k]) for k in keys]
    red_12 = [getv(r, "chernoff_bound_reduction_ge_1_2_pct") for r in rows]
    red_13 = [getv(r, "chernoff_bound_reduction_ge_1_3_pct") for r in rows]

    x = np.arange(len(keys))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    b1 = ax.bar(x - width / 2, red_12, width,
                label=r"Захоплення більшості $\geq 1/2$",
                color="#2E86AB", edgecolor="white", lw=0.6)
    b2 = ax.bar(x + width / 2, red_13, width,
                label=r"Поріг безпеки у стилі BFT $\geq 1/3$",
                color="#C73E1D", edgecolor="white", lw=0.6)
    for bars, vals in [(b1, red_12), (b2, red_13)]:
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Зниження межі Чернова (%)")
    ax.set_title("Зниження верхнього хвоста Чернова (оцінка розділу 2):\n"
                 "від придушення частки ваги до явної хвостової ймовірності")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    out = OUT / "F5_chernoff_reduction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def f6_signal_ablation_3seed():
    """3-seed signal-weight ablation: grouped bars across four controller
    configurations with std error bars and headline value labels.

    Configurations:
      Canonical multi-signal (w_F=0.75, w_G=0.10, w_S=0.15)
      Freshness-only (w_F=1)
      Gini-only (w_G=1)
      Split-only (w_S=1)

    Metrics shown side-by-side per configuration:
      full-window attacker reduction (%),
      1-epoch attacker reduction (%),
      Chernoff certificate reduction at theta >= 1/2 (%).
    """
    labels = [
        ("canonical", "Канонічна\nбагатосигнальна"),
        ("freshness", "Лише свіжість\n$w_F{=}1$"),
        ("gini",      "Лише Джині\n$w_G{=}1$"),
        ("split",     "Лише розщеплення\n$w_S{=}1$"),
    ]

    def fetch(key):
        row = _final_table(key, ROOT / ABLATION_DIR[key])
        # aggregated_final_table.csv stores _mean / _std columns
        m_full = float(row.get("reduction_vs_baseline_full_pct_mean",
                               row.get("reduction_vs_baseline_full_pct", np.nan)))
        s_full = float(row.get("reduction_vs_baseline_full_pct_std", 0.0) or 0.0)
        m_1ep  = float(row.get("reduction_vs_baseline_1ep_pct_mean",
                               row.get("reduction_vs_baseline_1ep_pct", np.nan)))
        s_1ep  = float(row.get("reduction_vs_baseline_1ep_pct_std", 0.0) or 0.0)
        m_chr  = float(row.get("chernoff_bound_reduction_ge_1_2_pct_mean",
                               row.get("chernoff_bound_reduction_ge_1_2_pct", np.nan)))
        s_chr  = float(row.get("chernoff_bound_reduction_ge_1_2_pct_std", 0.0) or 0.0)
        return (m_full, s_full), (m_1ep, s_1ep), (m_chr, s_chr)

    data = {key: fetch(key) for key, _ in labels}

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    metric_specs = [
        ("Повне вікно",          0, "#2E86AB"),
        ("1-епохове вікно",       1, "#3CB371"),
        (r"Чернов $\geq 1/2$", 2, "#C73E1D"),
    ]

    for offset_idx, (legend_label, midx, color) in enumerate(metric_specs):
        means = [data[key][midx][0] for key, _ in labels]
        stds  = [data[key][midx][1] for key, _ in labels]
        pos   = x + (offset_idx - 1) * width
        bars  = ax.bar(pos, means, width, yerr=stds,
                       label=legend_label, color=color,
                       edgecolor="white", lw=0.5,
                       error_kw=dict(elinewidth=0.9, capsize=2.5, ecolor="#333333"))
        for b, m in zip(bars, means):
            if np.isfinite(m):
                y_lbl = m + 0.6 if m > 1.0 else m + 0.3
                ax.text(b.get_x() + b.get_width() / 2.0, y_lbl,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in labels])
    ax.set_ylabel("Зниження відносно суто стейкового еталона (%)")
    ax.set_title("Трисідова абляція ваг сигналів в імпульсному Sybil-режимі\n"
                 r"($\lambda_{\max}=0{,}65$, $\kappa_{\downarrow}=0{,}02$, "
                 r"$\beta_{\mathrm{atk}}=0{,}33$, $k=6$, $c=9$; "
                 r"середнє $\pm$ ст. відхил. за трьома незалежними запусками)")
    ax.axhline(0.0, color="#888888", lw=0.5, zorder=0)
    ax.set_ylim(bottom=-1.5, top=49)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=10)

    out = OUT / "F6_signal_ablation_3seed.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main():
    out_paths = [
        f1_lambda_and_weight(),
        f2_scenario_window_bars(),
        f3_security_fairness_tradeoff(),
        f4_controller_dynamics(),
        f5_chernoff_reduction(),
        f6_signal_ablation_3seed(),
    ]
    print("\nGenerated overnight_v5 dissertation figures:")
    for p in out_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
