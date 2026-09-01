#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "article_rbhc"
OUT = ART / "regenerated_plots"
OUT.mkdir(parents=True, exist_ok=True)

POLICIES = {
    "baseline": ART / "burst_baseline",
    "signal": ART / "burst_signal",
    "hybrid": ART / "burst_hybrid",
}
COLORS = {
    "baseline": "#666666",
    "signal": "#1f77b4",
    "hybrid": "#d62728",
}
LABELS = {
    "baseline": "baseline",
    "signal": "signal",
    "hybrid": "hybrid",
}


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_epoch_metric(metric: str):
    data = {}
    for policy, base in POLICIES.items():
        seed_map = {}
        for seed_dir in sorted(base.glob("seed_*")):
            rows = read_csv(seed_dir / "results" / "epoch_summary_latest.csv")
            xs, ys = [], []
            for r in rows:
                xs.append(int(r["epoch_idx"]))
                ys.append(float(r[metric]))
            seed_map[seed_dir.name] = (xs, ys)
        data[policy] = seed_map
    return data


def aggregate_series(seed_map):
    epoch_values = defaultdict(list)
    for _, (xs, ys) in seed_map.items():
        for x, y in zip(xs, ys):
            epoch_values[x].append(y)
    xs = sorted(epoch_values)
    med = [median(epoch_values[x]) for x in xs]
    lo = [min(epoch_values[x]) for x in xs]
    hi = [max(epoch_values[x]) for x in xs]
    return xs, med, lo, hi


def load_window_summary():
    windows = {
        "early\n(1-5 ep)": range(1, 6),
        "middle\n(6-15 ep)": range(6, 16),
        "late\n(16-30 ep)": range(16, 31),
        "full\n(1-30 ep)": range(1, 31),
    }
    summary = {}
    epoch_metric = load_epoch_metric("mean_attacker_share")
    for policy, seed_map in epoch_metric.items():
        seed_window_vals = defaultdict(list)
        for _, (xs, ys) in seed_map.items():
            per_epoch = dict(zip(xs, ys))
            for wlabel, wrange in windows.items():
                vals = [per_epoch[e] for e in wrange if e in per_epoch]
                if vals:
                    seed_window_vals[wlabel].append(mean(vals))
        summary[policy] = {
            w: {
                "mean": mean(vals),
                "min": min(vals),
                "max": max(vals),
            }
            for w, vals in seed_window_vals.items()
        }
    return windows, summary


def plot_attacker_share():
    plt.figure(figsize=(9, 5))
    data = load_epoch_metric("mean_attacker_share")
    for policy in ["baseline", "signal", "hybrid"]:
        xs, med, lo, hi = aggregate_series(data[policy])
        plt.fill_between(xs, lo, hi, color=COLORS[policy], alpha=0.15)
        plt.plot(xs, med, color=COLORS[policy], linewidth=2.2, label=LABELS[policy])
    plt.axhline(0.33, color="black", linestyle="--", linewidth=1, alpha=0.7, label="stake baseline")
    plt.xlabel("post-attack epoch")
    plt.ylabel("attacker seat share")
    plt.title("Burst scenario: attacker seat share (median with seed range)")
    plt.legend(frameon=False, ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "burst_attacker_seat_share_median_range.png", dpi=220)
    plt.close()


def plot_lambda_trace():
    plt.figure(figsize=(9, 5))
    data = load_epoch_metric("mean_lambda_auto_ppm")
    for policy in ["signal", "hybrid"]:
        xs, med, lo, hi = aggregate_series(data[policy])
        med = [v / 1_000_000 for v in med]
        lo = [v / 1_000_000 for v in lo]
        hi = [v / 1_000_000 for v in hi]
        plt.fill_between(xs, lo, hi, color=COLORS[policy], alpha=0.15)
        plt.plot(xs, med, color=COLORS[policy], linewidth=2.2, label=LABELS[policy])
    plt.xlabel("post-attack epoch")
    plt.ylabel("lambda")
    plt.title("Burst scenario: controller response (median with seed range)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "burst_lambda_median_range.png", dpi=220)
    plt.close()


def plot_window_summary():
    windows, summary = load_window_summary()
    order = list(windows.keys())
    x = list(range(len(order)))
    width = 0.24
    plt.figure(figsize=(9, 5))
    offsets = {"baseline": -width, "signal": 0, "hybrid": width}
    for policy in ["baseline", "signal", "hybrid"]:
        means = [summary[policy][w]["mean"] for w in order]
        mins = [summary[policy][w]["min"] for w in order]
        maxs = [summary[policy][w]["max"] for w in order]
        yerr = [
            [m - lo for m, lo in zip(means, mins)],
            [hi - m for m, hi in zip(means, maxs)],
        ]
        xpos = [v + offsets[policy] for v in x]
        plt.bar(xpos, means, width=width, color=COLORS[policy], alpha=0.85, label=LABELS[policy])
        plt.errorbar(xpos, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=3)
    plt.axhline(0.33, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.xticks(x, order)
    plt.ylabel("mean attacker seat share")
    plt.title("Burst scenario: windowed summary across seeds")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "burst_windowed_attacker_share.png", dpi=220)
    plt.close()


def write_summary_csv():
    windows, summary = load_window_summary()
    out = OUT / "burst_window_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "window", "mean_attacker_share", "min_seed_mean", "max_seed_mean"])
        for policy in ["baseline", "signal", "hybrid"]:
            for window in windows.keys():
                row = summary[policy][window]
                w.writerow([policy, window, f"{row['mean']:.6f}", f"{row['min']:.6f}", f"{row['max']:.6f}"])


def main():
    plot_attacker_share()
    plot_lambda_trace()
    plot_window_summary()
    write_summary_csv()
    print(OUT)


if __name__ == "__main__":
    main()
