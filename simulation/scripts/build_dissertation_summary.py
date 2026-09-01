#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


SCENARIOS = [
    # ── Headline tuned result (ALPHA_DOWN=0.02, LAM_MAX=0.65) ─────────────────
    {
        "alias": "v2_pub_alphadown02_lammax065_burst",
        "role": "headline tuned burst",
        "note": "Best result: ALPHA_DOWN=0.02 + LAM_MAX=0.65; λ half-life 73 draws vs 47 default.",
    },
    # ── Main realistic scenarios ───────────────────────────────────────────────
    {
        "alias": "v2_pub_default_burst",
        "role": "main realistic",
        "note": "Primary realistic security case, default controller bounds.",
    },
    {
        "alias": "v2_pub_tuned_burst",
        "role": "main realistic tuned",
        "note": "Primary realistic case with stronger early response via LAM_MAX=0.65.",
    },
    # ── Trickle robustness ────────────────────────────────────────────────────
    {
        "alias": "v2_pub_default_trickle",
        "role": "companion robustness",
        "note": "Companion robustness case under slower sybil injection.",
    },
    {
        "alias": "v2_pub_tuned_trickle",
        "role": "companion robustness tuned",
        "note": "Companion robustness case with LAM_MAX=0.65 tuning.",
    },
    # ── Sensitivity ───────────────────────────────────────────────────────────
    {
        "alias": "v2_pub_beta040_burst",
        "role": "sensitivity β=0.40",
        "note": "Beta sensitivity: higher attacker stake share, useful as secondary evidence.",
    },
    # ── Fairness ──────────────────────────────────────────────────────────────
    {
        "alias": "v2_fairness_default",
        "role": "fairness default",
        "note": "Honest-newcomer baseline: fairness trade-off evidence, not a security headline.",
    },
    {
        "alias": "v2_fairness_tuned",
        "role": "fairness tuned",
        "note": "Honest-newcomer tuned case: earlier onboarding with LAM_MAX=0.65.",
    },
    {
        "alias": "fairness_trickle_default",
        "role": "fairness robustness default",
        "note": "Optional: newcomer fairness under trickle injection (v2 run pending task #12).",
    },
    {
        "alias": "fairness_trickle_tuned",
        "role": "fairness robustness tuned",
        "note": "Optional: newcomer fairness under trickle injection, LAM_MAX=0.65 (v2 pending).",
    },
]


FIGURES = [
    {
        "figure": "F1",
        "title": "Headline tuned burst: λ trajectory (ALPHA_DOWN=0.02, LAM_MAX=0.65)",
        "plot": "artifacts/v2_pub_alphadown02_lammax065_burst/plots/draw_lambda_auto_trace_latest.png",
        "source_csv": "artifacts/v2_pub_alphadown02_lammax065_burst/results/epoch_summary_latest.csv",
        "why": "Shows sustained λ elevation (half-life 73 draws) — core dissertation evidence for tuned controller.",
    },
    {
        "figure": "F2",
        "title": "Main realistic case, attacker weight trajectory",
        "plot": "artifacts/v2_pub_default_burst/plots/epoch_attacker_weight_share_latest.png",
        "source_csv": "artifacts/v2_pub_default_burst/results/epoch_summary_latest.csv",
        "why": "Best single trajectory figure for the default realistic scenario.",
    },
    {
        "figure": "F3",
        "title": "Scenario comparison across burst configurations",
        "plot": "artifacts/v2_pub_tuned_burst/plots/scenario_reduction_vs_baseline_bars_latest.png",
        "source_csv": "artifacts/v2_pub_tuned_burst/results/epoch_final_table_latest.csv",
        "why": "Compact headline comparison across default/tuned/alphadown02 scenario family.",
    },
    {
        "figure": "F4",
        "title": "Newcomer fairness comparison",
        "plot": "artifacts/fairness_compare_2026-04-06/entrant_baseline_gap_compare.png",
        "source_csv": "artifacts/v2_fairness_tuned/results/epoch_final_table_latest.csv",
        "why": "Shows the mixed fairness story without overclaiming.",
    },
]


def tracked_meta_from_entry_kind(entry_kind: str) -> Dict[str, str]:
    kind = str(entry_kind or "").strip().lower()
    if kind in ("entrant", "honest_newcomer", "newcomer"):
        label = "entrant" if kind == "entrant" else "honest newcomer"
        return {
            "tracked_entity_mode": "entrant",
            "tracked_entity_label": label,
            "baseline_comparison_mode": "gap",
        }
    return {
        "tracked_entity_mode": "attacker",
        "tracked_entity_label": "attacker",
        "baseline_comparison_mode": "reduction",
    }


def read_csv_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[0]


def latest_manifest(results_dir: Path) -> Path | None:
    manifests = sorted(results_dir.glob("run_manifest_*.json"))
    return manifests[-1] if manifests else None


def load_manifest(results_dir: Path) -> Dict[str, object]:
    path = latest_manifest(results_dir)
    if not path:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    cosmos_root = repo_root / "cosmos"
    out_dir = cosmos_root / "eval_package_2026-04-06"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_csv = out_dir / "FINAL_DISSERTATION_TABLE.csv"
    table_md = out_dir / "FINAL_DISSERTATION_TABLE.md"
    figures_md = out_dir / "FINAL_DISSERTATION_FIGURES.md"

    fieldnames = [
        "alias",
        "role",
        "tracked_entity_mode",
        "tracked_entity_label",
        "baseline_comparison_mode",
        "attack_profile",
        "entry_kind",
        "k",
        "committee_size",
        "lambda_init_ppm",
        "tracked_vs_baseline_full_pct",
        "tracked_vs_baseline_1ep_pct",
        "tracked_vs_baseline_3ep_pct",
        "tracked_vs_baseline_5ep_pct",
        "post_lambda_peak_ppm",
        "time_to_95pct_tracked_baseline_draws",
        "note",
        "artifact_dir",
        "final_table_csv",
        "policy_table_csv",
        "epoch_table_csv",
        "manifest_json",
    ]

    rows: List[Dict[str, str]] = []
    for spec in SCENARIOS:
        art_dir = cosmos_root / "artifacts" / spec["alias"]
        results_dir = art_dir / "results"
        final_table = results_dir / "epoch_final_table_latest.csv"
        policy_table = results_dir / "epoch_final_policy_table_latest.csv"
        epoch_table = results_dir / "epoch_final_epoch_table_latest.csv"
        if not final_table.exists():
            continue
        manifest = load_manifest(results_dir)
        manifest_path = latest_manifest(results_dir)
        row = read_csv_row(final_table)
        entry_kind = str(manifest.get("entry_kind", row.get("tracked_entity_mode", "")))
        tracked_meta = tracked_meta_from_entry_kind(entry_kind)

        tracked_vs_full = row.get("tracked_vs_baseline_full_pct", row.get("reduction_vs_baseline_full_pct", ""))
        tracked_vs_1ep = row.get("tracked_vs_baseline_1ep_pct", row.get("reduction_vs_baseline_1ep_pct", ""))
        tracked_vs_3ep = row.get("tracked_vs_baseline_3ep_pct", row.get("reduction_vs_baseline_3ep_pct", ""))
        tracked_vs_5ep = row.get("tracked_vs_baseline_5ep_pct", row.get("reduction_vs_baseline_5ep_pct", ""))
        tracked_t95 = row.get("time_to_95pct_tracked_baseline_draws", row.get("time_to_95pct_baseline_draws", ""))

        rows.append(
            {
                "alias": spec["alias"],
                "role": spec["role"],
                "tracked_entity_mode": row.get("tracked_entity_mode", tracked_meta["tracked_entity_mode"]) if row.get("tracked_entity_mode") not in ("", "attacker") or tracked_meta["tracked_entity_mode"] == "attacker" else tracked_meta["tracked_entity_mode"],
                "tracked_entity_label": row.get("tracked_entity_label", tracked_meta["tracked_entity_label"]) if row.get("tracked_entity_label") not in ("", "attacker") or tracked_meta["tracked_entity_mode"] == "attacker" else tracked_meta["tracked_entity_label"],
                "baseline_comparison_mode": row.get("baseline_comparison_mode", tracked_meta["baseline_comparison_mode"]) if row.get("baseline_comparison_mode") not in ("", "reduction") or tracked_meta["baseline_comparison_mode"] == "reduction" else tracked_meta["baseline_comparison_mode"],
                "attack_profile": str(manifest.get("attacker_profile", "")),
                "entry_kind": entry_kind,
                "k": row.get("k", ""),
                "committee_size": row.get("committee_size", ""),
                "lambda_init_ppm": row.get("lambda_init_ppm", ""),
                "tracked_vs_baseline_full_pct": tracked_vs_full,
                "tracked_vs_baseline_1ep_pct": tracked_vs_1ep,
                "tracked_vs_baseline_3ep_pct": tracked_vs_3ep,
                "tracked_vs_baseline_5ep_pct": tracked_vs_5ep,
                "post_lambda_peak_ppm": row.get("post_lambda_peak_ppm", ""),
                "time_to_95pct_tracked_baseline_draws": tracked_t95,
                "note": spec["note"],
                "artifact_dir": rel(art_dir, repo_root),
                "final_table_csv": rel(final_table, repo_root),
                "policy_table_csv": rel(policy_table, repo_root),
                "epoch_table_csv": rel(epoch_table, repo_root),
                "manifest_json": rel(manifest_path, repo_root) if manifest_path else "",
            }
        )

    with table_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# Final dissertation table",
        "",
        "Canonical summary built from the stable alias artifact directories.",
        "",
        "| Alias | Role | Tracked entity | Baseline semantics | Full % | 1 ep % | 3 ep % | 5 ep % | Peak λ ppm | Note |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {alias} | {role} | {tracked_entity_label} | {baseline_comparison_mode} | {tracked_vs_baseline_full_pct} | {tracked_vs_baseline_1ep_pct} | {tracked_vs_baseline_3ep_pct} | {tracked_vs_baseline_5ep_pct} | {post_lambda_peak_ppm} | {note} |".format(
                **row
            )
        )
    table_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    fig_lines = [
        "# Final dissertation figures",
        "",
        "Each figure is mapped to one exact plot file and one exact source CSV.",
        "",
    ]
    for fig in FIGURES:
        fig_lines.extend(
            [
                f"## {fig['figure']} — {fig['title']}",
                f"- Plot: `{fig['plot']}`",
                f"- Source CSV: `{fig['source_csv']}`",
                f"- Why keep it: {fig['why']}",
                "",
            ]
        )
    figures_md.write_text("\n".join(fig_lines), encoding="utf-8")

    print(table_csv)
    print(table_md)
    print(figures_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
