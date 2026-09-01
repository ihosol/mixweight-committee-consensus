#!/usr/bin/env python3
"""Realized committee-capture frequency from the per-draw artifact layer.

Chapter 2 defines the risk event as the tail event X_A >= q with q = ceil(theta*m),
but the headline table of Chapter 4 reports only reductions of the attacker's
weight share and of the analytical bounds. The event itself is never counted in
the burst and trickle scenarios, even though every draw already records the
attacker's seat count.

This script closes that gap directly from `epoch_draws_latest.csv`: it counts the
post-attack draws in which the tracked coalition reached q seats.

IMPORTANT — unit of analysis. Draws inside one run are NOT independent: they share
a chain, a validator panel that evolves across epochs, and one controller
trajectory. Pooling every draw of every seed into one binomial sample and putting
a Clopper-Pearson interval on it understates the uncertainty by roughly the
within-run correlation, and a two-proportion test on the pooled counts is invalid.
The independent replicate here is the RUN, not the draw.

The script therefore reports the per-seed frequency and the mean +/- sd ACROSS
seeds, and compares arms with a paired t-test on the per-seed differences. With
three runs per arm that test has two degrees of freedom and almost no power; the
output says so rather than implying significance. Pooled Clopper-Pearson bounds
are still printed, explicitly labelled as a within-run repeatability range, never
as a confidence interval for the mechanism.

Usage:
    python3 poc/cosmos/scripts/compute_realized_capture.py
    python3 poc/cosmos/scripts/compute_realized_capture.py SCENARIO_DIR [...]
    python3 poc/cosmos/scripts/compute_realized_capture.py --csv out.csv

A scenario directory is either a leaf run (containing `results/`) or a multi-seed
directory (containing `seed_*/`), in which case seeds are pooled and also listed
individually.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO_ROOT / "poc" / "cosmos" / "artifacts"

# Per-draw rows committed to the repository, so the table can be rebuilt from a
# clean checkout where the artifact tree is absent.
SHIPPED_DRAWS = (REPO_ROOT / "poc" / "cosmos" / "dissertation_final" / "tables"
                 / "REALIZED_CAPTURE_DRAWS.csv")
SHIPPED_SUMMARY = SHIPPED_DRAWS.with_name("REALIZED_CAPTURE.csv")

DEFAULT_SCENARIOS = [
    ("імпульсний, еталон lambda=0", "overnight_v5/ablation_gini_only_burst"),
    ("імпульсний, типовий", "overnight_v5/pub_default_burst"),
    ("імпульсний, налаштований", "overnight_v5/pub_tuned_burst"),
    ("імпульсний, налаштований + a_down", "overnight_v5/pub_alphadown02_lammax065_burst"),
    ("повільний, типовий", "overnight_v5/pub_default_trickle"),
    ("повільний, налаштований", "overnight_v5/pub_tuned_trickle"),
    ("повільний, налаштований + a_down", "overnight_v5/pub_alphadown02_lammax065_trickle"),
]

THRESHOLDS = {"ge_1_3": 1.0 / 3.0, "ge_1_2": 1.0 / 2.0}

# Fixed column order for the emitted table. The "model" columns are pipeline
# outputs, not per-draw quantities: when rebuilding from --draws-csv they are
# carried over from the sibling summary, which is why they are listed here
# explicitly rather than derived from whatever keys a row happens to carry.
CSV_COLUMNS = [
    "scenario", "artifacts_subdir", "row_kind", "seed", "n_seeds", "n_post_draws",
    "committee_size",
    "q_ge_1_3", "hits_ge_1_3", "realized_ge_1_3_pct",
    "repeatability_lo_ge_1_3_pct", "repeatability_hi_ge_1_3_pct",
    "model_ge_1_3_pct",
    "q_ge_1_2", "hits_ge_1_2", "realized_ge_1_2_pct",
    "repeatability_lo_ge_1_2_pct", "repeatability_hi_ge_1_2_pct",
    "model_ge_1_2_pct",
]


def _betaincinv(a: float, b: float, y: float) -> float:
    """Inverse regularized incomplete beta, by bisection on the CDF.

    Avoids a scipy dependency: the pipeline only needs pyyaml/csv elsewhere and
    this script must run in the same bare environment.
    """
    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via its continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    result = front * (f - 1.0)
    if a > 1.0 and b > 1.0 and x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    return min(max(result, 0.0), 1.0)


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval. Degenerate counts collapse correctly."""
    if trials == 0:
        return (0.0, 1.0)
    lo = 0.0 if successes == 0 else _betaincinv(successes, trials - successes + 1, alpha / 2.0)
    hi = 1.0 if successes == trials else _betaincinv(successes + 1, trials - successes, 1.0 - alpha / 2.0)
    return (lo, hi)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    """Sample mean and sample sd across runs. sd is undefined for a single run."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float("nan")
    return m, (sum((v - m) ** 2 for v in values) / (n - 1)) ** 0.5


def _seat_column(row: dict) -> str:
    for name in ("tracked_seats", "attacker_seats"):
        if name in row and row[name] != "":
            return name
    raise KeyError("no seat column in draws CSV")


def count_run(results_dir: Path) -> dict | None:
    """Count post-attack capture events in one leaf run."""
    draws = results_dir / "epoch_draws_latest.csv"
    if not draws.exists():
        return None
    with draws.open() as fh:
        rows = list(csv.DictReader(fh))
    post = [r for r in rows if r.get("phase") != "pre_attack"]
    if not post:
        return None

    seat_col = _seat_column(post[0])
    m = int(post[0]["committee_size"])
    out = {"n_post": len(post), "committee_size": m}
    for name, theta in THRESHOLDS.items():
        q = math.ceil(theta * m)
        hits = sum(1 for r in post if int(r[seat_col]) >= q)
        lo, hi = clopper_pearson(hits, len(post))
        out[name] = {"q": q, "hits": hits, "pct": 100.0 * hits / len(post),
                     "lo": 100.0 * lo, "hi": 100.0 * hi}

    # Model value the pipeline already computed, for a like-for-like comparison.
    policy = results_dir / "epoch_final_policy_table_latest.csv"
    if policy.exists():
        with policy.open() as fh:
            prows = list(csv.DictReader(fh))
        for pr in prows:
            if pr.get("policy") == "adaptive":
                for name in THRESHOLDS:
                    key = f"post_capture_{name}_model_pct"
                    if key in pr and pr[key] != "":
                        out[name]["model_pct"] = float(pr[key])
            if pr.get("policy") == "baseline_stake":
                for name in THRESHOLDS:
                    key = f"post_capture_{name}_model_pct"
                    if key in pr and pr[key] != "":
                        out[name]["model_base_pct"] = float(pr[key])
    return out


def collect(scenario_dir: Path) -> tuple[dict | None, list[tuple[int, dict]]]:
    """Return (pooled, per_seed). Pooled sums counts across seeds before the CI."""
    seed_dirs = sorted(scenario_dir.glob("seed_*"))
    per_seed: list[tuple[int, dict]] = []
    if seed_dirs:
        for d in seed_dirs:
            res = count_run(d / "results")
            if res:
                per_seed.append((int(d.name.split("_")[1]), res))
    else:
        res = count_run(scenario_dir / "results")
        if res:
            per_seed.append((0, res))

    if not per_seed:
        return None, []

    pooled = {"n_post": sum(r["n_post"] for _, r in per_seed),
              "committee_size": per_seed[0][1]["committee_size"],
              "n_seeds": len(per_seed)}
    for name in THRESHOLDS:
        hits = sum(r[name]["hits"] for _, r in per_seed)
        lo, hi = clopper_pearson(hits, pooled["n_post"])
        entry = {"q": per_seed[0][1][name]["q"], "hits": hits,
                 "pct": 100.0 * hits / pooled["n_post"],
                 "lo": 100.0 * lo, "hi": 100.0 * hi}
        models = [r[name].get("model_pct") for _, r in per_seed if r[name].get("model_pct") is not None]
        if models:
            entry["model_pct"] = sum(models) / len(models)
        bases = [r[name].get("model_base_pct") for _, r in per_seed if r[name].get("model_base_pct") is not None]
        if bases:
            entry["model_base_pct"] = sum(bases) / len(bases)
        pooled[name] = entry
    return pooled, per_seed


def _summarize(rows: list[dict]) -> dict:
    """Count capture events in one run's post-attack rows."""
    m = int(rows[0]["committee_size"])
    out = {"n_post": len(rows), "committee_size": m}
    for name, theta in THRESHOLDS.items():
        q = math.ceil(theta * m)
        hits = sum(1 for r in rows if int(r["attacker_seats"]) >= q)
        lo, hi = clopper_pearson(hits, len(rows))
        out[name] = {"q": q, "hits": hits, "pct": 100.0 * hits / len(rows),
                     "lo": 100.0 * lo, "hi": 100.0 * hi}
    return out


def collect_from_draws(path: Path) -> tuple[dict[str, tuple[dict, list[tuple[int, dict]]]], dict[str, str]]:
    """Rebuild every scenario from the committed per-draw CSV.

    The model columns are pipeline outputs rather than per-draw quantities, so
    they are carried over from the sibling summary CSV when it is present and
    left blank otherwise. Everything else is recomputed from the raw rows.
    """
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    models: dict[str, dict[str, float]] = {}
    if SHIPPED_SUMMARY.exists():
        with SHIPPED_SUMMARY.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("row_kind") != "pooled":
                    continue
                for name in THRESHOLDS:
                    v = r.get(f"model_{name}_pct", "")
                    if v not in ("", None):
                        models.setdefault(r["scenario"], {})[name] = float(v)

    by_scenario: dict[str, dict[int, list[dict]]] = {}
    order: list[str] = []
    subdirs: dict[str, str] = {}
    for r in rows:
        sc = r["scenario"]
        if sc not in by_scenario:
            by_scenario[sc] = {}
            order.append(sc)
            subdirs[sc] = r.get("artifacts_subdir", "")
        by_scenario[sc].setdefault(int(r["seed"]), []).append(r)

    out: dict[str, tuple[dict, list[tuple[int, dict]]]] = {}
    for sc in order:
        per_seed = [(seed, _summarize(rs)) for seed, rs in sorted(by_scenario[sc].items())]
        pooled = {"n_post": sum(r["n_post"] for _, r in per_seed),
                  "committee_size": per_seed[0][1]["committee_size"],
                  "n_seeds": len(per_seed)}
        for name in THRESHOLDS:
            hits = sum(r[name]["hits"] for _, r in per_seed)
            lo, hi = clopper_pearson(hits, pooled["n_post"])
            entry = {"q": per_seed[0][1][name]["q"], "hits": hits,
                     "pct": 100.0 * hits / pooled["n_post"],
                     "lo": 100.0 * lo, "hi": 100.0 * hi}
            if sc in models and name in models[sc]:
                entry["model_pct"] = models[sc][name]
            pooled[name] = entry
        out[sc] = (pooled, per_seed)
    return out, subdirs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", help="scenario dirs under poc/cosmos/artifacts")
    ap.add_argument("--csv", help="also write a machine-readable table here")
    ap.add_argument("--per-seed", action="store_true", help="print each seed separately")
    ap.add_argument("--draws-csv", nargs="?", const=str(SHIPPED_DRAWS), default=None,
                    help="rebuild from a committed per-draw CSV instead of the artifact "
                         "tree; with no value, uses the shipped REALIZED_CAPTURE_DRAWS.csv")
    args = ap.parse_args()

    # A clean checkout has no artifact tree. Fall back to the shipped per-draw rows
    # so the table is reproducible with one command and no simulation run.
    source = args.draws_csv
    if source is None and not args.scenarios and not ARTIFACTS.exists() and SHIPPED_DRAWS.exists():
        source = str(SHIPPED_DRAWS)
        print(f"[дерево артефактів відсутнє — читаю {SHIPPED_DRAWS.name}]")

    if source:
        src = Path(source)
        if not src.exists():
            print(f"немає файлу: {src}", file=sys.stderr)
            return 1
        rebuilt, subdirs = collect_from_draws(src)
        targets = [(sc, sc) for sc in rebuilt]
    else:
        rebuilt = subdirs = None
        targets = ([(s, s) for s in args.scenarios] if args.scenarios else DEFAULT_SCENARIOS)

    hdr = (f"{'сценарій':<36}{'сідів':>6}{'розіг.':>8}"
           f"{'>=1/3 сер.±sd':>18}{'внутр.розкид':>18}{'модель':>9}"
           f"{'>=1/2 сер.±sd':>18}{'внутр.розкид':>18}{'модель':>9}")
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for label, rel in targets:
        if rebuilt is not None:
            pooled, per_seed = rebuilt[rel]
            rel = subdirs[rel]
        else:
            d = ARTIFACTS / rel
            if not d.exists():
                print(f"{label:<36}  (немає: {rel})")
                continue
            pooled, per_seed = collect(d)
        if pooled is None:
            print(f"{label:<36}  (немає даних розіграшів)")
            continue

        a, b = pooled["ge_1_3"], pooled["ge_1_2"]
        # Headline is the across-run mean +/- sd: the run is the independent replicate.
        m3, s3 = _mean_sd([r["ge_1_3"]["pct"] for _, r in per_seed])
        m5, s5 = _mean_sd([r["ge_1_2"]["pct"] for _, r in per_seed])
        agg_a = "{:.1f} ± {:.1f}".format(m3, s3)
        agg_b = "{:.1f} ± {:.1f}".format(m5, s5)
        # Pooled Clopper-Pearson: a within-run repeatability band, NOT a confidence
        # interval for the mechanism. Draws inside a run are not independent.
        band_a = "[{:.1f}; {:.1f}]".format(a["lo"], a["hi"])
        band_b = "[{:.1f}; {:.1f}]".format(b["lo"], b["hi"])
        ma = a.get("model_pct", float("nan"))
        mb = b.get("model_pct", float("nan"))
        print(f"{label:<36}{pooled.get('n_seeds', 1):>6}{pooled['n_post']:>8}"
              f"{agg_a:>18}{band_a:>18}{ma:>8.1f}%"
              f"{agg_b:>18}{band_b:>18}{mb:>8.1f}%")

        if args.per_seed:
            for seed, r in per_seed:
                print(f"    seed {seed:<30}{'':>6}{r['n_post']:>8}"
                      f"{r['ge_1_3']['pct']:>15.1f}%{'':>18}{'':>9}"
                      f"{r['ge_1_2']['pct']:>15.1f}%")

        for seed, r in per_seed:
            out_rows.append({
                "scenario": label, "artifacts_subdir": rel, "row_kind": "seed",
                "seed": seed, "n_seeds": "", "n_post_draws": r["n_post"],
                "committee_size": r["committee_size"],
                "q_ge_1_3": r["ge_1_3"]["q"], "hits_ge_1_3": r["ge_1_3"]["hits"],
                "realized_ge_1_3_pct": round(r["ge_1_3"]["pct"], 3),
                "repeatability_lo_ge_1_3_pct": "", "repeatability_hi_ge_1_3_pct": "",
                "model_ge_1_3_pct": "",
                "q_ge_1_2": r["ge_1_2"]["q"], "hits_ge_1_2": r["ge_1_2"]["hits"],
                "realized_ge_1_2_pct": round(r["ge_1_2"]["pct"], 3),
                "repeatability_lo_ge_1_2_pct": "", "repeatability_hi_ge_1_2_pct": "",
                "model_ge_1_2_pct": "",
            })
        out_rows.append({
            "scenario": label, "artifacts_subdir": rel, "row_kind": "pooled", "seed": "",
            "n_seeds": pooled.get("n_seeds", 1), "n_post_draws": pooled["n_post"],
            "committee_size": pooled["committee_size"],
            "q_ge_1_3": a["q"], "hits_ge_1_3": a["hits"],
            "realized_ge_1_3_pct": round(a["pct"], 3),
            "repeatability_lo_ge_1_3_pct": round(a["lo"], 3), "repeatability_hi_ge_1_3_pct": round(a["hi"], 3),
            "model_ge_1_3_pct": round(a["model_pct"], 3) if "model_pct" in a else "",
            "model_baseline_ge_1_3_pct": round(a["model_base_pct"], 3) if "model_base_pct" in a else "",
            "q_ge_1_2": b["q"], "hits_ge_1_2": b["hits"],
            "realized_ge_1_2_pct": round(b["pct"], 3),
            "repeatability_lo_ge_1_2_pct": round(b["lo"], 3), "repeatability_hi_ge_1_2_pct": round(b["hi"], 3),
            "model_ge_1_2_pct": round(b["model_pct"], 3) if "model_pct" in b else "",
            "model_baseline_ge_1_2_pct": round(b["model_base_pct"], 3) if "model_base_pct" in b else "",
        })

    print()
    print("  сер.±sd     — середнє та вибіркове sd ЗА ЗАПУСКАМИ; запуск є незалежною реплікою")
    print("  внутр.розкид — Клоппер–Пірсон за об'єднаними розіграшами: діапазон повторюваності")
    print("                 стенда, НЕ довірчий інтервал (розіграші в межах запуску залежні)")

    if args.csv and out_rows:
        # Deterministic serialization: fixed column order, LF endings, no trailing
        # whitespace. Rebuilding from --draws-csv must reproduce the committed file
        # byte for byte, so nothing here may depend on the source or on dict order.
        with io.open(args.csv, "w", encoding="utf-8", newline="\n") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            for row in out_rows:
                w.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
        print(f"\nЗаписано: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
