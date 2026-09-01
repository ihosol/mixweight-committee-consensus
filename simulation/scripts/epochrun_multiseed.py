#!/usr/bin/env python3
# Multi-seed wrapper around epochrun.py.
#
# Runs the same scenario config N times with distinct chain_ids (= distinct WOR seeds
# deep inside the Go keeper), then aggregates per-seed epoch_final_table_latest.csv
# rows into mean±std and writes aggregated_final_table.csv under the base artifacts dir.
#
# Usage:
#   python3 epochrun_multiseed.py <config.yaml> [--seeds 1,2,3]
#
# The config YAML may include:
#   experiment:
#     seeds: [1, 2, 3]
# If both the --seeds flag and experiment.seeds are absent, runs once with seed=1
# (equivalent to invoking epochrun.py directly).

import argparse
import csv
import errno
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import yaml


NUMERIC_FIELDS = [
    "reduction_vs_baseline_full_pct",
    "reduction_vs_baseline_1ep_pct",
    "reduction_vs_baseline_2ep_pct",
    "reduction_vs_baseline_3ep_pct",
    "reduction_vs_baseline_5ep_pct",
    "tracked_vs_baseline_full_pct",
    "tracked_vs_baseline_1ep_pct",
    "tracked_vs_baseline_3ep_pct",
    "tracked_vs_baseline_5ep_pct",
    "post_attacker_weight_mean",
    "post_attacker_seat_mean",
    "post_tracked_weight_mean",
    "post_tracked_seat_mean",
    "post_lambda_peak_ppm",
    "post_lambda_half_life_draws",
    "post_lambda_rise_time_draws",
    "post_lambda_settle_time_draws",
    "post_lambda_overshoot_pct",
    "post_lambda_chatter_rms_ppm",
    "post_lambda_control_effort",
    "chernoff_bound_reduction_ge_1_2_pct",
    "chernoff_bound_reduction_ge_1_3_pct",
]


ROW_KEY_FIELDS = [
    "tracked_entity_mode",
    "baseline_comparison_mode",
    "attack_mode",
    "whale_share",
    "k",
    "committee_size",
    "lambda_init_ppm",
]


def _cleanup_orphans() -> None:
    """SIGKILL any chain/runner processes left over from a prior seed.

    epochrun.py spawns chain-five-threed daemons via subprocess.Popen and
    relies on its own teardown to stop them. If it exits via an uncaught
    exception (e.g. a node fails to bind to a port) those daemons are not
    reaped and keep their TCP ports bound, causing the next seed/scenario
    to die with "address already in use". Reap defensively here.
    """
    for pat in ("chain-five-threed", "epochrun.py"):
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pat],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            return  # pkill not available — skip silently


def _matching_pids(cmdline_needles: tuple[str, ...]) -> list[int]:
    """Best-effort scan of live PIDs whose cmdline contains any needle."""
    matches: list[int] = []
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if not cmdline:
            continue
        if any(needle in cmdline for needle in cmdline_needles):
            matches.append(pid)
    return sorted(matches)


def _tcp_port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, int(port)))
        return True
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, errno.EACCES):
            return False
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _ports_for_cfg(cfg: dict) -> list[int]:
    localnet = cfg.get("localnet", {}) if isinstance(cfg, dict) else {}
    experiment = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    attack = cfg.get("attack", {}) if isinstance(cfg, dict) else {}

    honest_nodes = int(experiment.get("honest_nodes", localnet.get("nodes", 8)))
    sybil_k_values = [int(x) for x in attack.get("sybil_k_values", [0])]
    attack_mode = str(attack.get("mode", "replacement")).strip().lower()
    additive_extra = 1 if attack_mode == "additive" else 0
    max_nodes = honest_nodes + additive_extra + max([0] + sybil_k_values)

    p2p_base = int(localnet.get("p2p_port_base", 26680))
    rpc_base = int(localnet.get("rpc_port_base", 36657))
    api_base = int(localnet.get("api_port_base", 31317))
    grpc_base = int(localnet.get("grpc_port_base", 39090))

    ports: list[int] = []
    for base in (p2p_base, rpc_base, api_base, grpc_base):
        ports.extend(base + i for i in range(max_nodes))
    return sorted(set(ports))


def _wait_cleanup_complete(cfg: dict | None, timeout_s: float | None = None) -> None:
    if timeout_s is None:
        timeout_s = float(os.getenv("ORPHAN_CLEANUP_TIMEOUT_S", "20"))
    """Wait until orphaned runners are gone and expected ports are bindable."""
    deadline = time.time() + timeout_s
    needles = ("chain-five-threed", "epochrun.py")
    ports = _ports_for_cfg(cfg or {}) if cfg is not None else []

    last_pids: list[int] = []
    last_busy: list[int] = []
    while time.time() < deadline:
        last_pids = _matching_pids(needles)
        last_busy = [p for p in ports if not _tcp_port_bindable(p)]
        if not last_pids and not last_busy:
            return

        # If a stubborn process survived pkill, try one direct SIGKILL before the next poll.
        for pid in last_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.25)

    detail = []
    if last_pids:
        detail.append(f"live_pids={last_pids[:8]}")
    if last_busy:
        detail.append(f"busy_ports={last_busy[:12]}")
    suffix = f" ({', '.join(detail)})" if detail else ""
    raise RuntimeError(f"orphan cleanup did not complete within {timeout_s:.1f}s{suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="YAML scenario config")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated list of integer seeds; overrides experiment.seeds")
    ap.add_argument("--skip-aggregation", action="store_true",
                    help="Run seeds only, do not aggregate")
    ap.add_argument("--artifacts-subdir", default=None,
                    help="Override experiment.artifacts_subdir from config")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(s) for s in cfg.get("experiment", {}).get("seeds", [1])]

    if not seeds:
        seeds = [1]

    if args.artifacts_subdir:
        base_subdir = args.artifacts_subdir.strip().strip("/")
    else:
        base_subdir = str(cfg.get("experiment", {}).get("artifacts_subdir", "")).strip().strip("/")
    if not base_subdir:
        print("[multi-seed] experiment.artifacts_subdir is required", file=sys.stderr)
        return 1

    base_chain_id = cfg["chain"]["chain_id"]

    script_dir = Path(__file__).resolve().parent
    epochrun = script_dir / "epochrun.py"
    # epochrun.py uses: repo = poc_root = parents[2]; artifacts_root = repo / "cosmos" / "artifacts"
    # So artifacts land in .../poc/cosmos/artifacts/<subdir>/ — mirror that here.
    poc_root = script_dir.parents[1]  # .../poc/cosmos/scripts → .../poc
    artifacts_root = poc_root / "cosmos" / "artifacts"

    print(f"[multi-seed] running {len(seeds)} seed(s): {seeds}")
    print(f"[multi-seed] base chain_id={base_chain_id}, base artifacts_subdir={base_subdir}")

    # Defensive cleanup: ensure no orphaned chain processes from a prior run
    # are still bound to the ports we're about to use.
    _cleanup_orphans()
    _wait_cleanup_complete(cfg)

    per_seed_final_tables = []
    single_seed = (len(seeds) == 1)
    for idx, seed in enumerate(seeds):
        seed_chain_id = f"{base_chain_id}-s{seed}"
        seed_subdir = base_subdir if single_seed else f"{base_subdir}/seed_{seed}"

        seed_cfg = dict(cfg)
        seed_cfg["chain"] = dict(cfg["chain"])
        seed_cfg["chain"]["chain_id"] = seed_chain_id
        seed_cfg["experiment"] = dict(cfg.get("experiment", {}))
        seed_cfg["experiment"]["artifacts_subdir"] = seed_subdir
        seed_cfg["experiment"].pop("seeds", None)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_seed{seed}.yaml", delete=False
        ) as tf:
            yaml.safe_dump(seed_cfg, tf)
            tmp_path = tf.name

        print(f"\n[multi-seed] === seed {idx + 1}/{len(seeds)} (seed={seed}, chain_id={seed_chain_id}) ===")
        try:
            rc = subprocess.call(["python3", str(epochrun), tmp_path], env=os.environ.copy())
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if rc != 0:
            print(f"[multi-seed] seed {seed} failed (rc={rc})", file=sys.stderr)
            _cleanup_orphans()
            _wait_cleanup_complete(cfg)
            return rc

        # Even on success, epochrun.py may leave background daemons if it
        # exited via an uncaught exception. Reap before the next seed binds
        # the same port range.
        _cleanup_orphans()
        _wait_cleanup_complete(cfg)

        final_csv = artifacts_root / seed_subdir / "results" / "epoch_final_table_latest.csv"
        if not final_csv.exists():
            print(f"[multi-seed] missing final table: {final_csv}", file=sys.stderr)
            return 1
        per_seed_final_tables.append((seed, final_csv))
        print(f"[multi-seed] seed {seed} OK: {final_csv}")

    if args.skip_aggregation or len(per_seed_final_tables) < 2:
        print(f"[multi-seed] skipping aggregation (seeds={len(per_seed_final_tables)})")
        return 0

    aggregated_dir = artifacts_root / base_subdir
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregated_dir / "aggregated_final_table.csv"
    _aggregate(per_seed_final_tables, aggregate_path)
    print(f"\n[multi-seed] Aggregated ({len(per_seed_final_tables)} seeds): {aggregate_path}")
    _print_aggregate_summary(aggregate_path)
    return 0


def _aggregate(per_seed_final_tables, out_path: Path) -> None:
    rows_by_seed = []
    ordered_keys = None
    first_fieldnames = None
    for seed, csv_path in per_seed_final_tables:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            seed_rows = list(reader)
            if first_fieldnames is None and reader.fieldnames is not None:
                first_fieldnames = list(reader.fieldnames)
        seed_map = {}
        seed_keys = []
        for row in seed_rows:
            key = _row_key(row)
            if key in seed_map:
                raise RuntimeError(f"duplicate aggregate key for seed {seed}: {key}")
            seed_map[key] = row
            seed_keys.append(key)
        if ordered_keys is None:
            ordered_keys = seed_keys
        else:
            ordered_key_set = set(ordered_keys)
            seed_key_set = set(seed_keys)
            missing = [key for key in ordered_keys if key not in seed_key_set]
            extra = [key for key in seed_keys if key not in ordered_key_set]
            if not missing and not extra:
                rows_by_seed.append((seed, seed_map))
                continue
            raise RuntimeError(
                "seed rows do not align by scenario key: "
                f"seed={seed}, missing={missing[:3]}, extra={extra[:3]}"
            )
        rows_by_seed.append((seed, seed_map))

    if not rows_by_seed or not ordered_keys:
        return

    out_rows = []
    seed_ids = [str(seed) for seed, _ in rows_by_seed]
    for key in ordered_keys:
        cells = [seed_map[key] for _, seed_map in rows_by_seed]
        row = dict(cells[0])
        row["n_seeds"] = str(len(cells))
        row["seeds_used"] = ",".join(seed_ids)
        for fld in NUMERIC_FIELDS:
            values = []
            for c in cells:
                raw = c.get(fld, "")
                if raw in ("", None):
                    continue
                try:
                    values.append(float(raw))
                except (ValueError, TypeError):
                    continue
            if values:
                mean = statistics.mean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row[f"{fld}_mean"] = f"{mean:.6f}"
                row[f"{fld}_std"] = f"{std:.6f}"
        out_rows.append(row)

    fieldnames = list(out_rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in ROW_KEY_FIELDS)


def _print_aggregate_summary(path: Path) -> None:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    row = rows[0]
    print("\n  Headline metrics (mean ± std across seeds):")
    for fld in [
        "reduction_vs_baseline_full_pct",
        "reduction_vs_baseline_1ep_pct",
        "chernoff_bound_reduction_ge_1_2_pct",
        "post_lambda_peak_ppm",
        "post_lambda_half_life_draws",
    ]:
        m = row.get(f"{fld}_mean", "—")
        s = row.get(f"{fld}_std", "—")
        if m != "—":
            try:
                m_f = float(m); s_f = float(s)
                print(f"    {fld:<42}  {m_f:>10.4f}  ± {s_f:>8.4f}")
            except (ValueError, TypeError):
                print(f"    {fld:<42}  {m}  ± {s}")


if __name__ == "__main__":
    sys.exit(main())
