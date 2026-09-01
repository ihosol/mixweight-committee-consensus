#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sh(cmd: List[str], *, env=None, cwd=None, capture: bool = False, check: bool = True, text: bool = True):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=capture, check=check, text=text)


def load_yaml_minimal(path: Path) -> dict:
    """Tiny YAML loader (subset) to avoid non-stdlib deps.

    Supports:
      - nested mappings via indentation
      - inline lists: [1, 2, 3]
      - ints, bools, strings

    This is intentionally shared with minirun.py style config.
    """
    data = {}
    stack = [(0, data)]

    def parse_value(v: str):
        v = v.strip()
        if v.startswith('[') and v.endswith(']'):
            inner = v[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(',')]
            out = []
            for p in parts:
                if p.startswith('"') and p.endswith('"'):
                    out.append(p[1:-1])
                elif p.isdigit() or (p.startswith('-') and p[1:].isdigit()):
                    out.append(int(p))
                else:
                    out.append(p)
            return out
        if v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
            return int(v)
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        line = raw.strip()
        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        key = key.strip()
        rest = rest.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]

        if rest == "":
            cur[key] = {}
            stack.append((indent + 2, cur[key]))
        else:
            cur[key] = parse_value(rest)
    return data


def wait_height(rpc_port: int, min_h: int = 1, timeout_s: float = 45.0) -> int:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{rpc_port}/status", timeout=1.0) as r:
                s = json.load(r)
            h = int(s["result"]["sync_info"]["latest_block_height"])
            if h >= min_h:
                return h
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("chain did not reach height>=min_h")


def toml_set(src: str, key: str, val: str) -> str:
    import re

    pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    if pat.search(src):
        return pat.sub(lambda m: m.group(1) + val, src, count=1)
    return src + f"\n{key} = {val}\n"


@dataclass
class NodeProc:
    i: int
    home: Path
    p: subprocess.Popen


def get_sequence(env, home0: Path, node_args: List[str], addr: str) -> int:
    r = sh([
        "chaind", "query", "auth", "account", addr,
        "-o", "json",
        "--home", str(home0),
    ] + node_args, env=env, capture=True, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return -1
    j = json.loads(r.stdout)
    acc = j.get("account", j)
    seq = acc.get("sequence", "0")
    try:
        return int(seq)
    except Exception:
        return -1


def wait_sequence_increase(env, home0: Path, node_args: List[str], addr: str, prev: int, timeout_s: float = 30.0) -> int:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = get_sequence(env, home0, node_args, addr)
        if s >= 0 and s > prev:
            return s
        time.sleep(0.5)
    raise RuntimeError(f"account sequence did not increase (prev={prev})")


def build_localnet(*,
                  env,
                  tmp_root: Path,
                  chain_id: str,
                  denom: str,
                  honest_nodes: int,
                  sybil_k: int,
                  beta: float,
                  p2p_base: int,
                  rpc_base: int,
                  api_base: int,
                  grpc_base: int,
                  from_acct: str,
                  keyring: str,
                  sybil_active_at_genesis: bool = True,
                  ) -> Tuple[List[Path], List[NodeProc]]:
    """Create fresh localnet under tmp_root with N=honest_nodes+sybil_k validators.

    First honest_nodes are honest, last sybil_k are attacker.
    """
    nodes = honest_nodes + sybil_k
    if nodes <= 0:
        raise ValueError("need at least one validator")
    if honest_nodes <= 0:
        raise ValueError("honest_nodes must be >=1")
    if sybil_k < 0:
        raise ValueError("sybil_k must be >=0")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("attack.beta must be in [0,1]")
    if sybil_k == 0 and beta != 0.0:
        # Nothing to split; treat as no attacker.
        beta = 0.0

    # Clean tmp_root
    if tmp_root.exists():
        sh(["bash", "-lc", f"rm -rf {tmp_root}"])
    tmp_root.mkdir(parents=True, exist_ok=True)

    homes: List[Path] = []
    for i in range(nodes):
        h = tmp_root / f"node{i}"
        h.mkdir(parents=True, exist_ok=True)
        moniker = (f"honest{i}" if i < honest_nodes else f"sybil{i-honest_nodes}")
        sh(["chaind", "init", moniker, "--chain-id", chain_id, "--home", str(h)], env=env)
        homes.append(h)

    # Create a local key in each home.
    addrs: List[str] = []
    for h in homes:
        # NOTE: `chaind keys add` prints a mnemonic to stdout; capture+discard to avoid leaking secrets into logs.
        sh(["chaind", "keys", "add", from_acct, "--keyring-backend", keyring, "--home", str(h)], env=env, capture=True)
        addr = sh([
            "chaind", "keys", "show", from_acct, "-a",
            "--keyring-backend", keyring,
            "--home", str(h),
        ], env=env, capture=True).stdout.strip()
        addrs.append(addr)

    # Fund all accounts in genesis (node0 canonical).
    # Keep huge so any gentx amounts are covered.
    for addr in addrs:
        sh(["chaind", "add-genesis-account", addr, f"2000000000000{denom}", "--home", str(homes[0])], env=env)

    # Copy genesis from node0 to others.
    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

    # Compute bond amounts matching stake shares.
    # Use a fixed total bond to avoid rounding surprises.
    total_bond = 1_000_000_000  # in denom units
    honest_total = 1.0 - beta
    if sybil_k > 0:
        attacker_each = beta / sybil_k
    else:
        attacker_each = 0.0

    # Integer rounding: allocate honest first, then attacker, fix residual on node0.
    bond_ints: List[int] = []
    for i in range(nodes):
        if i < honest_nodes:
            share = honest_total / honest_nodes
        else:
            share = attacker_each
        bond_ints.append(int(round(share * total_bond)))

    # Fix any rounding drift to ensure sum == total_bond.
    drift = total_bond - sum(bond_ints)
    bond_ints[0] += drift
    if bond_ints[0] <= 0:
        raise RuntimeError("invalid rounding produced non-positive bond for node0")

    gentx_dir = homes[0] / "config" / "gentx"
    gentx_dir.mkdir(parents=True, exist_ok=True)

    for i, h in enumerate(homes):
        # If Late Entry mode (sybil_active_at_genesis=False), skip gentx for Sybil nodes (i >= honest_nodes).
        # They will join later via create-validator.
        if (not sybil_active_at_genesis) and (i >= honest_nodes):
            continue

        out = gentx_dir / f"gentx-node{i}.json"
        p2p_port = p2p_base + i
        sh([
            "chaind", "gentx", from_acct, f"{bond_ints[i]}{denom}",
            "--chain-id", chain_id,
            "--keyring-backend", keyring,
            "--home", str(h),
            "--ip", "127.0.0.1",
            "--p2p-port", str(p2p_port),
            "--output-document", str(out),
        ], env=env)

    sh(["chaind", "collect-gentxs", "--home", str(homes[0])], env=env)

    # Sync final genesis to all nodes.
    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

    # Derive node IDs for persistent peers.
    node_ids: List[str] = []
    for h in homes:
        nid = sh(["chaind", "tendermint", "show-node-id", "--home", str(h)], env=env, capture=True).stdout.strip()
        node_ids.append(nid)

    peers_all = [f"{node_ids[i]}@127.0.0.1:{p2p_base+i}" for i in range(nodes)]

    procs: List[NodeProc] = []
    for i, h in enumerate(homes):
        p2p = p2p_base + i
        rpc = rpc_base + i
        api = api_base + i
        grpc = grpc_base + i

        # Localhost-only networking, explicit peers, no PEX.
        conf = h / "config" / "config.toml"
        txt = conf.read_text(encoding="utf-8")
        txt = toml_set(txt, "seeds", '""')
        peers_other = [p for p in peers_all if not p.startswith(node_ids[i] + "@")]
        txt = toml_set(txt, "persistent_peers", '"' + ",".join(peers_other) + '"')
        txt = toml_set(txt, "pex", "false")
        txt = toml_set(txt, "addr_book_strict", "false")
        # Localnet runs multiple validators behind the same IP; allow duplicates.
        txt = toml_set(txt, "allow_duplicate_ip", "true")
        txt = toml_set(txt, "external_address", '""')

        # Fast-localnet consensus timeouts (tuned for single-host experiments)
        txt = toml_set(txt, "timeout_propose", '"1500ms"')
        txt = toml_set(txt, "timeout_prevote", '"1200ms"')
        txt = toml_set(txt, "timeout_precommit", '"1200ms"')
        txt = toml_set(txt, "timeout_commit", '"3000ms"')

        if 'indexer = "null"' in txt:
            txt = txt.replace('indexer = "null"', 'indexer = "kv"')
        conf.write_text(txt, encoding="utf-8")

        ab = h / "config" / "addrbook.json"
        if ab.exists():
            ab.unlink()

        log = tmp_root / f"node{i}.log"
        cmd = [
            "chaind", "start",
            "--home", str(h),
            "--p2p.laddr", f"tcp://127.0.0.1:{p2p}",
            "--rpc.laddr", f"tcp://127.0.0.1:{rpc}",
            "--rpc.pprof_laddr", "127.0.0.1:0",
            "--grpc.address", f"127.0.0.1:{grpc}",
            "--grpc-web.enable=false",
            "--api.enable",
            "--api.address", f"tcp://127.0.0.1:{api}",
        ]
        p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, text=True)
        procs.append(NodeProc(i=i, home=h, p=p))

    return homes, procs


def stop_all(procs: List[NodeProc]):
    for np in procs:
        if np.p.poll() is None:
            np.p.send_signal(signal.SIGTERM)
    time.sleep(1.0)
    for np in procs:
        if np.p.poll() is None:
            np.p.kill()


def inject_sybils(
    env,
    homes: List[Path],
    honest_nodes: int,
    sybil_k: int,
    beta: float,
    chain_id: str,
    denom: str,
    from_acct: str,
    keyring: str,
    fees: str,
    node_rpc: str,
):
    """Dynamically create validators for the late-entry Sybils."""
    if sybil_k <= 0 or beta <= 0:
        return

    print(f"--- Injecting {sybil_k} Sybil validators (Late Entry) ---")

    # Recalculate bond amounts (logic matching build_localnet)
    total_bond = 1_000_000_000
    honest_total = 1.0 - beta
    attacker_each = beta / sybil_k
    nodes = honest_nodes + sybil_k

    bond_ints = []
    for i in range(nodes):
        if i < honest_nodes:
            share = honest_total / honest_nodes
        else:
            share = attacker_each
        bond_ints.append(int(round(share * total_bond)))
    
    # Drift fix (mostly for node0, but good to match exact logic)
    drift = total_bond - sum(bond_ints)
    bond_ints[0] += drift

    # Loop over Sybils
    for i in range(honest_nodes, nodes):
        h = homes[i]
        moniker = f"sybil{i-honest_nodes}"
        bond_amount = bond_ints[i]
        
        # Get Validator PubKey
        pubkey_out = sh(
            ["chaind", "tendermint", "show-validator", "--home", str(h)],
            env=env, capture=True
        ).stdout.strip()

        # Build create-validator command
        # Note: We send this transaction via the RPC of a running node (node0), 
        # but sign it with the key in the Sybil's home (using --home matches the keyring dir).
        # Wait: The account that pays fees and stakes is 'from_acct' in 'h'.
        
        cmd = [
            "chaind", "tx", "staking", "create-validator",
            "--amount", f"{bond_amount}{denom}",
            "--pubkey", pubkey_out,
            "--moniker", moniker,
            "--chain-id", chain_id,
            "--from", from_acct,
            "--keyring-backend", keyring,
            "--fees", fees,
            "--gas", "auto",
            "--gas-adjustment", "1.5",
            "--commission-rate", "0.10",
            "--commission-max-rate", "0.20",
            "--commission-max-change-rate", "0.01",
            "--min-self-delegation", "1",
            "--broadcast-mode", "sync", # Use sync, we will wait for inclusion manually
            "--home", str(h), # Key location
            "--node", node_rpc, # Submit to running chain
            "-y"
        ]
        
        print(f"Injecting {moniker} with {bond_amount}{denom}...")
        r = sh(cmd, env=env, capture=True, check=False)
        if r.returncode != 0:
            print(f"Error injecting {moniker}:\n{r.stdout}\n{r.stderr}")
            # Don't crash, try next?
        else:
            # Check tx code
            try:
                txj = json.loads(r.stdout)
                if int(txj.get("code", 0)) != 0:
                     print(f"Tx failed for {moniker}: {txj.get('raw_log')}")
            except Exception:
                pass


def main() -> int:
    repo = Path(__file__).resolve().parents[2]  # .../poc
    cfg_path = repo / "cosmos" / "poc_config.yaml"
    cfg = load_yaml_minimal(cfg_path)

    chain_id = cfg["chain"]["chain_id"]
    denom = cfg["chain"]["denom"]

    honest_nodes = int(cfg.get("experiment", {}).get("honest_nodes", 4))
    aging_blocks = int(cfg.get("experiment", {}).get("aging_blocks", 50))
    beta_raw = cfg.get("attack", {}).get("beta", 0)
    beta = float(beta_raw)
    sybil_k_values = list(cfg.get("attack", {}).get("sybil_k_values", [0, 1, 2, 3, 5, 8, 13, 21]))

    # Persistence baseline metadata (aligned with module defaults for now).
    persistence_tau_max_blocks = int(cfg.get("persistence", {}).get("tau_max_blocks", 2000))
    persistence_zeta_ppm = int(cfg.get("persistence", {}).get("zeta_ppm", 500000))

    p2p_base = int(cfg["localnet"]["p2p_port_base"])
    rpc_base = int(cfg["localnet"]["rpc_port_base"])
    api_base = int(cfg["localnet"]["api_port_base"])
    grpc_base = int(cfg["localnet"]["grpc_port_base"])

    committee_size = int(cfg["workload"]["committee_size"])
    draws_per_setting = int(cfg["workload"].get("draws_per_setting", 3))
    lambda_vals = list(cfg["workload"]["lambda_ppm_values"])

    from_acct = str(cfg["tx"]["from"])
    keyring = str(cfg["tx"]["keyring_backend"])
    fees = str(cfg["tx"]["fees"])
    broadcast_mode = str(cfg["tx"]["broadcast_mode"])
    if broadcast_mode == "block":
        broadcast_mode = "sync"

    topk_vals = list(cfg.get("coalition", {}).get("topk_values", [1, 3]))

    art_dir = repo / "cosmos" / "artifacts"
    (art_dir / "results").mkdir(parents=True, exist_ok=True)
    (art_dir / "plots").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/go/bin:{Path.home()}/.local/go/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")

    try:
        sh(["chaind", "version"], env=env)
    except Exception:
        print("chaind not found; run 'ignite chain build' in poc/cosmos/chain first", file=sys.stderr)
        return 2

    # Current PoC does not expose persistence params via tx; config values are recorded
    # into CSV metadata and should match module defaults unless code-level params are added.
    if (persistence_tau_max_blocks, persistence_zeta_ppm) != (2000, 500000):
        print(
            "[warn] persistence params in config differ from current module defaults "
            "(tau_max_blocks=2000, zeta_ppm=500000). "
            "They are recorded as metadata but not applied on-chain via tx yet.",
            file=sys.stderr,
        )

    # Avoid port conflicts
    subprocess.run(["bash", "-lc", "pkill -f 'chaind start --home /tmp/poc_sybil_' 2>/dev/null || true"], check=False)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_path = art_dir / "results" / f"sybil_seats_vs_lambda_{run_id}.csv"
    latest_csv_path = art_dir / "results" / "sybil_seats_vs_lambda.csv"
    diag_path = art_dir / "results" / f"sybil_diagnostics_{run_id}.csv"
    latest_diag_path = art_dir / "results" / "sybil_diagnostics_latest.csv"

    # CSV schema
    cols = [
        "lambda_ppm",
        "k",
        "total_validators",
        "beta",
        "honest_nodes",
        "committee_size",
        "draws_per_setting",
        "persistence_tau_max_blocks",
        "persistence_zeta_ppm",
        "draw_i",
        "tag",
        "txhash",
        "attacker_seats",
        "attacker_seats_share",
        "attacker_stake_ppm",
        "attacker_age_ppm",
        "attacker_weight_ppm",
        "attacker_validators",
    ] + [f"top{int(k)}_seats" for k in sorted(set(int(x) for x in topk_vals))]

    diag_cols = [
        "k",
        "lambda_ppm",
        "total_validators",
        "attacker_validators",
        "total_tokens",
        "attacker_tokens",
        "attacker_token_share",
        "draws",
        "mean_attacker_share",
        "min_attacker_share",
        "max_attacker_share",
    ]

    rows_out: List[Dict[str, str]] = []

    # Fresh-run only: no resume to avoid mixed-profile contamination.
    csv_f = csv_path.open("w", encoding="utf-8", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=cols)
    csv_w.writeheader()
    csv_f.flush()

    diag_f = diag_path.open("w", encoding="utf-8", newline="")
    diag_w = csv.DictWriter(diag_f, fieldnames=diag_cols)
    diag_w.writeheader()
    diag_f.flush()

    # IMPORTANT: evaluate each (k, lambda) on a fresh chain to avoid bias from
    # Sybil aging across lambda sweep.
    for k in sybil_k_values:
        k = int(k)
        for lam in lambda_vals:
            lam_i = int(lam)

            pending_draws = list(range(draws_per_setting))

            tmp_root = Path(f"/tmp/poc_sybil_k{k}_lam{lam_i}")
            homes: List[Path] = []
            procs: List[NodeProc] = []
            try:
                homes, procs = build_localnet(
                    env=env,
                    tmp_root=tmp_root,
                    chain_id=chain_id,
                    denom=denom,
                    honest_nodes=honest_nodes,
                    sybil_k=k,
                    beta=beta,
                    p2p_base=p2p_base,
                    rpc_base=rpc_base,
                    api_base=api_base,
                    grpc_base=grpc_base,
                    from_acct=from_acct,
                    keyring=keyring,
                    sybil_active_at_genesis=False,
                )

                # 1. Wait for aging period (Honest nodes run alone)
                print(f"[k={k}, lam={lam_i}] Waiting for aging period (height={aging_blocks})...")
                h = wait_height(rpc_base + 0, aging_blocks, timeout_s=1000.0)
                print(f"[k={k}, lam={lam_i}] [ok] aging complete at height={h}")

                node_rpc = f"tcp://127.0.0.1:{rpc_base+0}"

                # 2. Inject Sybils
                inject_sybils(
                    env=env,
                    homes=homes,
                    honest_nodes=honest_nodes,
                    sybil_k=k,
                    beta=beta,
                    chain_id=chain_id,
                    denom=denom,
                    from_acct=from_acct,
                    keyring=keyring,
                    fees=fees,
                    node_rpc=node_rpc,
                )

                # Wait for Sybils to be included in validator set (2 blocks safety)
                h2 = wait_height(rpc_base + 0, h + 2, timeout_s=60.0)
                print(f"[k={k}, lam={lam_i}] [ok] Sybils injected, height={h2}")
                sh(["chaind", "config", "chain-id", chain_id, "--home", str(homes[0])], env=env)
                sh(["chaind", "config", "keyring-backend", keyring, "--home", str(homes[0])], env=env)
                sh(["chaind", "config", "node", node_rpc, "--home", str(homes[0])], env=env)
                node_args = ["--node", node_rpc]

                from_addr = sh([
                    "chaind", "keys", "show", from_acct, "-a",
                    "--keyring-backend", keyring,
                    "--home", str(homes[0]),
                ], env=env, capture=True).stdout.strip()

                # Determine which validator operator addrs are attacker sybils.
                vals = json.loads(sh([
                    "chaind", "query", "staking", "validators",
                    "-o", "json",
                    "--home", str(homes[0]),
                ] + node_args, env=env, capture=True).stdout)
                arr = vals.get("validators", vals) if isinstance(vals, dict) else vals

                attacker_ops: Set[str] = set()
                all_ops: List[Tuple[str, int]] = []  # (op_addr, tokens_int)
                for v in arr:
                    op = v.get("operator_address")
                    mon = (v.get("description", {}) or {}).get("moniker", "")
                    tok_s = v.get("tokens", "0")
                    try:
                        tok = int(tok_s)
                    except Exception:
                        tok = 0
                    if mon.startswith("sybil"):
                        attacker_ops.add(op)
                    all_ops.append((op, tok))

                # Rank by voting power (tokens) desc for top-k coalition metrics.
                all_ops_sorted = [op for (op, tok) in sorted(all_ops, key=lambda x: x[1], reverse=True)]
                total_validators = honest_nodes + k

                token_by_op = {op: tok for (op, tok) in all_ops}
                total_tokens = sum(token_by_op.values())
                attacker_tokens = sum(token_by_op.get(op, 0) for op in attacker_ops)
                attacker_token_share = (attacker_tokens / total_tokens) if total_tokens > 0 else 0.0

                # Set lambda once per fresh (k, lam) chain.
                pair_shares: List[float] = []

                r = sh([
                    "chaind", "tx", "adaptivecommittee", "set-lambda", str(lam_i),
                    "--from", from_acct,
                    "--fees", fees,
                    "--broadcast-mode", broadcast_mode,
                    "--chain-id", chain_id,
                    "-y",
                    "-o", "json",
                    "--home", str(homes[0]),
                ] + node_args, env=env, capture=True, check=False)
                if r.returncode != 0:
                    raise RuntimeError(f"set-lambda failed (k={k}, lam={lam_i}):\n{r.stdout}\n{r.stderr}")
                txj = json.loads(r.stdout)
                if int(txj.get("code", 0)) != 0:
                    raise RuntimeError(f"set-lambda tx error: {txj.get('raw_log')}")

                seq0 = get_sequence(env, homes[0], node_args, from_addr)
                if seq0 < 0:
                    raise RuntimeError("could not read sender sequence")
                wait_sequence_increase(env, homes[0], node_args, from_addr, seq0, timeout_s=120.0)

                def query_committee_attrs(txh: str) -> Dict[str, str]:
                    for _ in range(40):
                        qr = sh([
                            "chaind", "query", "tx", txh,
                            "-o", "json",
                            "--home", str(homes[0]),
                        ] + node_args, env=env, capture=True, check=False)
                        if qr.returncode == 0 and qr.stdout.strip():
                            try:
                                qj = json.loads(qr.stdout)
                            except Exception:
                                time.sleep(0.3)
                                continue

                            # Prefer top-level events, fallback to logs/events.
                            events = qj.get("events", [])
                            for ev in events:
                                if ev.get("type") == "committee_drawn":
                                    return {a.get("key", ""): a.get("value", "") for a in ev.get("attributes", [])}

                            for lg in qj.get("logs", []):
                                for ev in lg.get("events", []):
                                    if ev.get("type") == "committee_drawn":
                                        return {a.get("key", ""): a.get("value", "") for a in ev.get("attributes", [])}
                        time.sleep(0.3)
                    return {}

                for draw_i in pending_draws:
                    tag = f"k{k}_lam{lam_i}_i{draw_i}"

                    r = sh([
                        "chaind", "tx", "adaptivecommittee", "draw-committee", str(committee_size), tag,
                        "--from", from_acct,
                        "--fees", fees,
                        "--broadcast-mode", broadcast_mode,
                        "--chain-id", chain_id,
                        "-y",
                        "-o", "json",
                        "--home", str(homes[0]),
                    ] + node_args, env=env, capture=True, check=False)
                    if r.returncode != 0:
                        raise RuntimeError(f"draw-committee failed (k={k}, lam={lam_i}):\n{r.stdout}\n{r.stderr}")
                    txj = json.loads(r.stdout)
                    if int(txj.get("code", 0)) != 0:
                        raise RuntimeError(f"draw-committee tx error: {txj.get('raw_log')}")
                    txh = txj.get("txhash", "")

                    seq1 = get_sequence(env, homes[0], node_args, from_addr)
                    if seq1 < 0:
                        raise RuntimeError("could not read sender sequence")
                    wait_sequence_increase(env, homes[0], node_args, from_addr, seq1, timeout_s=120.0)

                    def query_last_draw() -> str:
                        qr = sh([
                            "chaind", "query", "adaptivecommittee", "last-draw", tag,
                            "-o", "json",
                            "--home", str(homes[0]),
                        ] + node_args, env=env, capture=True, check=False)
                        if qr.returncode != 0 or not qr.stdout.strip():
                            return ""
                        qj = json.loads(qr.stdout)
                        return qj.get("membersCsv") or qj.get("members_csv") or ""

                    members_csv = query_last_draw()
                    if members_csv == "":
                        t0 = time.time()
                        while time.time() - t0 < 20.0:
                            time.sleep(0.5)
                            members_csv = query_last_draw()
                            if members_csv:
                                break
                        if members_csv == "":
                            raise RuntimeError("draw tx not committed (last-draw still empty)")

                    members = members_csv.split(",") if members_csv else []
                    attacker_seats = sum(1 for m in members if m in attacker_ops)
                    attacker_seats_share = attacker_seats / max(1, len(members))

                    committee_attrs = query_committee_attrs(txh)

                    counts: Dict[str, int] = {}
                    for m in members:
                        counts[m] = counts.get(m, 0) + 1

                    topk_seats: Dict[int, int] = {}
                    for kk in topk_vals:
                        topk_ops = all_ops_sorted[: int(kk)]
                        topk_seats[int(kk)] = sum(counts.get(op, 0) for op in topk_ops)

                    row: Dict[str, str] = {
                        "lambda_ppm": str(lam_i),
                        "k": str(k),
                        "total_validators": str(total_validators),
                        "beta": str(beta),
                        "honest_nodes": str(honest_nodes),
                        "committee_size": str(committee_size),
                        "draws_per_setting": str(draws_per_setting),
                        "persistence_tau_max_blocks": str(persistence_tau_max_blocks),
                        "persistence_zeta_ppm": str(persistence_zeta_ppm),
                        "draw_i": str(draw_i),
                        "tag": tag,
                        "txhash": txh,
                        "attacker_seats": str(attacker_seats),
                        "attacker_seats_share": f"{attacker_seats_share:.6f}",
                        "attacker_stake_ppm": committee_attrs.get("attacker_stake_ppm", ""),
                        "attacker_age_ppm": committee_attrs.get("attacker_age_ppm", ""),
                        "attacker_weight_ppm": committee_attrs.get("attacker_weight_ppm", ""),
                        "attacker_validators": committee_attrs.get("attacker_validators", ""),
                    }
                    for kk in sorted(topk_seats.keys()):
                        row[f"top{kk}_seats"] = str(topk_seats[kk])

                    rows_out.append(row)
                    pair_shares.append(attacker_seats_share)
                    csv_w.writerow(row)
                    csv_f.flush()

                    if (draw_i + 1) % max(1, draws_per_setting // 10) == 0 or (draw_i + 1) == draws_per_setting:
                        cur_mean = sum(pair_shares) / len(pair_shares)
                        print(
                            f"[k={k}, lam={lam_i}] progress {draw_i+1}/{draws_per_setting} "
                            f"mean_attacker_share={cur_mean:.4f}"
                        )

                if pair_shares:
                    diag_row = {
                        "k": str(k),
                        "lambda_ppm": str(lam_i),
                        "total_validators": str(total_validators),
                        "attacker_validators": str(len(attacker_ops)),
                        "total_tokens": str(total_tokens),
                        "attacker_tokens": str(attacker_tokens),
                        "attacker_token_share": f"{attacker_token_share:.6f}",
                        "draws": str(len(pair_shares)),
                        "mean_attacker_share": f"{(sum(pair_shares)/len(pair_shares)):.6f}",
                        "min_attacker_share": f"{min(pair_shares):.6f}",
                        "max_attacker_share": f"{max(pair_shares):.6f}",
                    }
                    diag_w.writerow(diag_row)
                    diag_f.flush()

            finally:
                stop_all(procs)

    # Close CSV/diagnostics file handles.
    try:
        csv_f.close()
    except Exception:
        pass
    try:
        diag_f.close()
    except Exception:
        pass

    # Update stable aliases for convenience.
    try:
        latest_csv_path.write_bytes(csv_path.read_bytes())
        latest_diag_path.write_bytes(diag_path.read_bytes())
    except Exception:
        pass

    # Plot: average attacker seat share vs lambda, one line per k.
    # Aggregate over draws.
    def avg_att_share(kv: int, lamv: int) -> float:
        xs = [float(r["attacker_seats_share"]) for r in rows_out if int(r["k"]) == kv and int(r["lambda_ppm"]) == lamv]
        return sum(xs) / max(1, len(xs))

    plt.figure(figsize=(7.2, 4.2))
    x_lams = [float(lam) / 1_000_000.0 for lam in lambda_vals]
    for k in sorted(set(int(x) for x in sybil_k_values)):
        ys = [avg_att_share(k, int(lam)) for lam in lambda_vals]
        plt.plot(x_lams, ys, marker="o", label=f"k={k}")

    plt.xlabel("lambda")
    plt.ylabel("avg attacker seats share")
    plt.title(f"Sybil stake-splitting: attacker share vs λ (β={beta}, draws={draws_per_setting}, m={committee_size})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    out_png = art_dir / "plots" / "sybil_attacker_seats_vs_lambda.png"
    plt.savefig(out_png, dpi=170, bbox_inches="tight")

    print(str(csv_path))
    print(str(diag_path))
    print(str(out_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
