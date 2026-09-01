#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Set

# Reuse proven helpers from epochrun.py
import epochrun as er


def sybil_valopers(env: Dict[str, str], home0: Path, node_rpc: str) -> Set[str]:
    vals = json.loads(er.sh([
        "chaind", "query", "staking", "validators", "-o", "json",
        "--home", str(home0), "--node", node_rpc,
    ], env=env, capture=True).stdout)
    arr = vals.get("validators", vals) if isinstance(vals, dict) else vals
    out: Set[str] = set()
    for v in arr:
        moniker = ((v.get("description") or {}).get("moniker") or "")
        if moniker.startswith("sybil"):
            op = v.get("operator_address") or ""
            if op:
                out.add(op)
    return out


def draw_once(env: Dict[str, str], home0: Path, node_rpc: str, chain_id: str, payer_key: str, fees: str,
             committee_size: int, tag: str) -> List[str]:
    r = er.sh([
        "chaind", "tx", "adaptivecommittee", "draw-committee", str(committee_size), tag,
        "--from", payer_key,
        "--fees", fees,
        "--broadcast-mode", "sync",
        "--chain-id", chain_id,
        "-y", "-o", "json",
        "--home", str(home0),
        "--node", node_rpc,
    ], env=env, capture=True, check=False)
    if r.returncode != 0 or not (r.stdout or "").strip():
        raise RuntimeError(f"draw-committee failed\nstdout={r.stdout}\nstderr={r.stderr}")
    txj = json.loads(r.stdout)
    if int(txj.get("code", 0)) != 0:
        raise RuntimeError(f"draw-committee code!=0: {txj}")
    txh = txj.get("txhash", "")
    if txh:
        txq = er.wait_tx_inclusion(env, home0, ["--node", node_rpc], txh, timeout_s=30.0)
        if not txq or int(txq.get("code", 0)) != 0:
            raise RuntimeError(f"draw tx not included/failed: tx={txh} q={txq}")

    qr = er.sh([
        "chaind", "query", "adaptivecommittee", "last-draw", tag,
        "-o", "json",
        "--home", str(home0),
        "--node", node_rpc,
    ], env=env, capture=True, check=False)
    if qr.returncode != 0 or not (qr.stdout or "").strip():
        return []
    try:
        j = json.loads(qr.stdout)
    except Exception:
        return []
    members_csv = j.get("membersCsv") or j.get("members_csv") or ""
    return [m.strip() for m in members_csv.split(",") if m.strip()]


def run_scenario(*, env: Dict[str, str], cfg: dict, lam_i: int, run_tag: str) -> dict:
    chain_id = cfg["chain"]["chain_id"]
    denom = cfg["chain"]["denom"]

    honest_nodes = int(cfg.get("experiment", {}).get("honest_nodes", 8))
    beta = float(cfg.get("attack", {}).get("beta", 0.33))
    sybil_k = int(cfg.get("attack", {}).get("sybil_k_values", [1])[0])

    p2p_base = int(cfg["localnet"]["p2p_port_base"])
    rpc_base = int(cfg["localnet"]["rpc_port_base"])
    api_base = int(cfg["localnet"]["api_port_base"])
    grpc_base = int(cfg["localnet"]["grpc_port_base"])

    committee_size = int(cfg["workload"]["committee_size"])
    draws = int(cfg["workload"].get("draws_per_setting", 12))

    time_cfg = cfg.get("epoch", {})
    epoch_blocks = int(time_cfg.get("epoch_blocks", 5))
    pre_attack_epochs = int(time_cfg.get("pre_attack_epochs", 5))

    from_acct_base = str(cfg["tx"]["from"])
    payer_key = f"{from_acct_base}0"
    keyring = str(cfg["tx"]["keyring_backend"])
    fees = str(cfg["tx"]["fees"])

    target_pre = pre_attack_epochs * epoch_blocks
    tmp_root = Path(f"/tmp/poc_attackcmp_{run_tag}_lam{lam_i}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    homes = []
    procs = []
    try:
        homes, procs = er.build_localnet(
            env=env,
            tmp_root=tmp_root,
            chain_id=chain_id,
            denom=denom,
            honest_nodes=honest_nodes,
            sybil_k=sybil_k,
            beta=beta,
            p2p_base=p2p_base,
            rpc_base=rpc_base,
            api_base=api_base,
            grpc_base=grpc_base,
            from_acct_base=from_acct_base,
            keyring=keyring,
            sybil_active_at_genesis=False,
        )

        er.wait_height(rpc_base + 0, target_pre, timeout_s=1200.0)
        node_rpc = f"tcp://127.0.0.1:{rpc_base}"

        ok_inject = er.inject_sybils(env, homes, honest_nodes, sybil_k, beta, chain_id, denom, from_acct_base, keyring, fees, node_rpc)
        if ok_inject != sybil_k:
            raise RuntimeError(f"partial injection: {ok_inject}/{sybil_k}")

        # Ensure sybils are visible in validator set before draws.
        er.wait_height(rpc_base + 0, target_pre + 2, timeout_s=180.0)
        syb_ops = sybil_valopers(env, homes[0], node_rpc)
        if len(syb_ops) == 0:
            raise RuntimeError("sybil validators not visible after injection")

        # Set lambda for this run.
        r = er.sh([
            "chaind", "tx", "adaptivecommittee", "set-lambda", str(lam_i),
            "--from", payer_key,
            "--fees", fees,
            "--broadcast-mode", "sync",
            "--chain-id", chain_id,
            "-y", "-o", "json",
            "--home", str(homes[0]),
            "--node", node_rpc,
        ], env=env, capture=True, check=False)
        if r.returncode != 0 or not (r.stdout or "").strip():
            raise RuntimeError(f"set-lambda failed\nstdout={r.stdout}\nstderr={r.stderr}")
        txj = json.loads(r.stdout)
        if int(txj.get("code", 0)) != 0:
            raise RuntimeError(f"set-lambda code!=0: {txj}")

        shares: List[float] = []
        for i in range(draws):
            tag = f"cmp_lam{lam_i}_{i}"
            members = draw_once(env, homes[0], node_rpc, chain_id, payer_key, fees, committee_size, tag)
            if not members:
                continue
            seats = sum(1 for m in members if m in syb_ops)
            shares.append(seats / max(1, len(members)))

        return {
            "lambda_ppm": lam_i,
            "injected": ok_inject,
            "sybil_validators": len(syb_ops),
            "draws_ok": len(shares),
            "mean_sybil_seat_share": statistics.mean(shares) if shares else 0.0,
            "min_sybil_seat_share": min(shares) if shares else 0.0,
            "max_sybil_seat_share": max(shares) if shares else 0.0,
        }
    finally:
        er.stop_all(procs)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    cfg = er.load_yaml_minimal(repo / "cosmos" / "poc_config.yaml")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/go/bin:{Path.home()}/.local/go/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")

    # Honor caller-provided POC_CHAIND.
    if not env.get("POC_CHAIND"):
        pinned = (repo / "cosmos" / "chain53" / "chain-five-three" / "build" / "chain-five-threed").resolve()
        if pinned.exists():
            env["POC_CHAIND"] = str(pinned)

    ver = er.sh(["chaind", "version"], env=env, capture=True, check=False)
    print(f"[cmp] chaind={env.get('POC_CHAIND')} rc={ver.returncode} out={ver.stdout.strip()}")

    # Minimal compare: baseline vs model lambda from config.
    model_lam = int(cfg["workload"]["lambda_ppm_values"][1] if len(cfg["workload"]["lambda_ppm_values"]) > 1 else cfg["workload"]["lambda_ppm_values"][0])
    lambda_set = [0, model_lam]

    results = []
    for lam in lambda_set:
        print(f"[cmp] running scenario lambda={lam}...")
        res = run_scenario(env=env, cfg=cfg, lam_i=lam, run_tag="lateentry")
        results.append(res)
        print(f"[cmp] result: {res}")

    print("\n=== Comparison ===")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))

    if len(results) == 2:
        d = results[1]["mean_sybil_seat_share"] - results[0]["mean_sybil_seat_share"]
        print(f"delta(model-baseline) mean_sybil_seat_share = {d:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
