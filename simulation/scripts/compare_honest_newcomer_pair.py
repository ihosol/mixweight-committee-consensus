#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def single_row(path: Path) -> Dict[str, str]:
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[0]


def to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def to_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def post_rows(draw_rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = [r for r in draw_rows if (r.get("phase", "") or "") == "post_attack"]
    out.sort(key=lambda r: to_int(r.get("draw_idx_post_attack", "0"), 0))
    return out


def early_metrics(draw_rows: List[Dict[str, str]], draws_per_epoch: int) -> Dict[str, object]:
    post = post_rows(draw_rows)
    first_inclusion = None
    for r in post:
        share = to_float(r.get("tracked_seats_share", r.get("attacker_seats_share", "0")))
        if share > 0:
            first_inclusion = to_int(r.get("draw_idx_post_attack", "0"), 0)
            break

    def window(n_epochs: int) -> List[Dict[str, str]]:
        limit = n_epochs * max(1, draws_per_epoch)
        return post[:limit]

    def hits(rows: List[Dict[str, str]]) -> int:
        return sum(1 for r in rows if to_float(r.get("tracked_seats_share", r.get("attacker_seats_share", "0"))) > 0)

    def mean_share(rows: List[Dict[str, str]]) -> float:
        vals = [to_float(r.get("tracked_seats_share", r.get("attacker_seats_share", "0"))) for r in rows]
        return (sum(vals) / len(vals)) if vals else 0.0

    w3 = window(3)
    w5 = window(5)
    return {
        "first_inclusion": first_inclusion,
        "hits_3": hits(w3),
        "hits_5": hits(w5),
        "mean_3": mean_share(w3),
        "mean_5": mean_share(w5),
        "mean_full": mean_share(post),
    }


def compare_pair(default_dir: Path, tuned_dir: Path, title: str) -> str:
    d_final = single_row(default_dir / "results" / "epoch_final_table_latest.csv")
    t_final = single_row(tuned_dir / "results" / "epoch_final_table_latest.csv")
    d_draws = read_rows(default_dir / "results" / "epoch_draws_latest.csv")
    t_draws = read_rows(tuned_dir / "results" / "epoch_draws_latest.csv")
    draws_per_epoch = to_int(d_final.get("draws_per_epoch_cfg", "0"), 0) or to_int(t_final.get("draws_per_epoch_cfg", "0"), 0) or 1

    d_early = early_metrics(d_draws, draws_per_epoch)
    t_early = early_metrics(t_draws, draws_per_epoch)

    tracked_label = d_final.get("tracked_entity_label", "honest newcomer")
    baseline_mode = d_final.get("baseline_comparison_mode", "gap")

    lines = [
        f"# Honest newcomer fairness summary ({title})",
        "",
        "Scenario:",
        f"- tracked entity: `{tracked_label}`",
        f"- comparison semantics: `{baseline_mode}`",
        f"- draws per epoch: `{draws_per_epoch}`",
        f"- comparison: `{default_dir.name}` vs `{tuned_dir.name}` (`ADAPTIVE_LAM_MAX=0.65`)",
        "",
        "## Early onboarding",
        f"- First committee inclusion: `draw {d_early['first_inclusion']} -> draw {t_early['first_inclusion']}`",
        f"- {tracked_label.capitalize()} committee hits, first 3 post-attack epochs: `{d_early['hits_3']} -> {t_early['hits_3']}`",
        f"- {tracked_label.capitalize()} committee hits, first 5 post-attack epochs: `{d_early['hits_5']} -> {t_early['hits_5']}`",
        f"- Mean {tracked_label} seat share, first 3 epochs: `{d_early['mean_3']:.4f} -> {t_early['mean_3']:.4f}`",
        f"- Mean {tracked_label} seat share, first 5 epochs: `{d_early['mean_5']:.4f} -> {t_early['mean_5']:.4f}`",
        "",
        "## Longer-window fairness",
        f"- Full post-attack mean seat share: `{d_early['mean_full']:.5f} -> {t_early['mean_full']:.5f}`",
        f"- Time to 95% of stake baseline (draws): `{d_final.get('time_to_95pct_tracked_baseline_draws', d_final.get('time_to_95pct_baseline_draws', ''))} -> {t_final.get('time_to_95pct_tracked_baseline_draws', t_final.get('time_to_95pct_baseline_draws', ''))}`",
        f"- 1-epoch baseline gap: `{d_final.get('tracked_vs_baseline_1ep_pct', d_final.get('reduction_vs_baseline_1ep_pct', ''))}% -> {t_final.get('tracked_vs_baseline_1ep_pct', t_final.get('reduction_vs_baseline_1ep_pct', ''))}%`",
        f"- 3-epoch baseline gap: `{d_final.get('tracked_vs_baseline_3ep_pct', d_final.get('reduction_vs_baseline_3ep_pct', ''))}% -> {t_final.get('tracked_vs_baseline_3ep_pct', t_final.get('reduction_vs_baseline_3ep_pct', ''))}%`",
        f"- 5-epoch baseline gap: `{d_final.get('tracked_vs_baseline_5ep_pct', d_final.get('reduction_vs_baseline_5ep_pct', ''))}% -> {t_final.get('tracked_vs_baseline_5ep_pct', t_final.get('reduction_vs_baseline_5ep_pct', ''))}%`",
        f"- Full-window baseline gap: `{d_final.get('tracked_vs_baseline_full_pct', d_final.get('reduction_vs_baseline_full_pct', ''))}% -> {t_final.get('tracked_vs_baseline_full_pct', t_final.get('reduction_vs_baseline_full_pct', ''))}%`",
        "",
        "## Control response",
        f"- Peak lambda ppm: `{d_final.get('post_lambda_peak_ppm', '')} -> {t_final.get('post_lambda_peak_ppm', '')}`",
        "",
        "## Artifact paths",
        f"- `{default_dir}`",
        f"- `{tuned_dir}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default-dir", required=True)
    ap.add_argument("--tuned-dir", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text = compare_pair(Path(args.default_dir), Path(args.tuned_dir), args.title)
    Path(args.out).write_text(text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
