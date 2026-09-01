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
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sh(cmd: List[str], *, env=None, cwd=None, capture: bool = False, check: bool = True, text: bool = True):
    # Default to not capturing because some Cosmos CLI commands print large JSON.
    # Capture only when we explicitly need stdout.
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=capture, check=check, text=text)


def load_yaml_minimal(path: Path) -> dict:
    # Minimal YAML loader to avoid extra deps. Supports the subset we write.
    # If this grows, we can add PyYAML.
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
                elif p.isdigit():
                    out.append(int(p))
                else:
                    out.append(p)
            return out
        if v.isdigit():
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


def wait_height(rpc_port: int, min_h: int = 1, timeout_s: float = 30.0) -> int:
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
    raise RuntimeError("chain did not reach height>=1")


@dataclass
class NodeProc:
    i: int
    home: Path
    p: subprocess.Popen


def main() -> int:
    repo = Path(__file__).resolve().parents[2]  # .../poc
    cfg_path = repo / "cosmos" / "poc_config.yaml"
    cfg = load_yaml_minimal(cfg_path)

    chain_id = cfg["chain"]["chain_id"]
    denom = cfg["chain"]["denom"]

    nodes = int(cfg["localnet"]["nodes"])
    p2p_base = int(cfg["localnet"]["p2p_port_base"])
    rpc_base = int(cfg["localnet"]["rpc_port_base"])
    api_base = int(cfg["localnet"]["api_port_base"])
    grpc_base = int(cfg["localnet"]["grpc_port_base"])

    committee_size = int(cfg["workload"]["committee_size"])
    draws_per_setting = int(cfg["workload"]["draws_per_setting"])
    lambda_vals = list(cfg["workload"]["lambda_ppm_values"])
    tags = list(cfg["workload"]["tags"])

    from_acct = str(cfg["tx"]["from"])
    keyring = str(cfg["tx"]["keyring_backend"])
    fees = str(cfg["tx"]["fees"])
    broadcast_mode = str(cfg["tx"]["broadcast_mode"])
    if broadcast_mode == "block":
        # some Cosmos CLIs use 'block' wording; this chaind accepts only sync|async
        broadcast_mode = "sync"

    topk_vals = list(cfg["coalition"]["topk_values"])

    chain_dir = repo / "cosmos" / "chain"
    art_dir = repo / "cosmos" / "artifacts"
    (art_dir / "results").mkdir(parents=True, exist_ok=True)
    (art_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Ensure chaind exists
    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/go/bin:{Path.home()}/.local/go/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")

    try:
        sh(["chaind", "version"], env=env)
    except Exception as e:
        print("chaind not found; run 'ignite chain build' in poc/cosmos/chain first", file=sys.stderr)
        return 2

    # Kill any leftover localnet from previous runs to avoid port conflicts.
    subprocess.run(["bash", "-lc", "pkill -f 'chaind start --home /tmp/poc_' 2>/dev/null || true"], check=False)
    subprocess.run(["bash", "-lc", "pkill -f 'chaind start --home /tmp/poc_multi' 2>/dev/null || true"], check=False)

    tmp_root = Path("/tmp/poc_multi")
    if tmp_root.exists():
        sh(["bash", "-lc", f"rm -rf {tmp_root}"])
    tmp_root.mkdir(parents=True, exist_ok=True)

    # init homes + keys
    homes: List[Path] = []
    for i in range(nodes):
        h = tmp_root / f"node{i}"
        h.mkdir(parents=True, exist_ok=True)
        moniker = f"node{i}"
        sh(["chaind", "init", moniker, "--chain-id", chain_id, "--home", str(h)], env=env)
        homes.append(h)

    # create alice key in each home (same name, different key material OK for PoC)
    addrs: List[str] = []
    for i, h in enumerate(homes):
        sh(["chaind", "keys", "add", from_acct, "--keyring-backend", keyring, "--home", str(h)], env=env)
        addr = sh(["chaind", "keys", "show", from_acct, "-a", "--keyring-backend", keyring, "--home", str(h)], env=env, capture=True).stdout.strip()
        addrs.append(addr)

    # fund all accounts in genesis (use node0 genesis as canonical)
    for addr in addrs:
        sh(["chaind", "add-genesis-account", addr, f"2000000000{denom}", "--home", str(homes[0])], env=env)

    # copy genesis from node0 to others
    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

    # gentx from each node into node0 gentx dir
    gentx_dir = homes[0] / "config" / "gentx"
    gentx_dir.mkdir(parents=True, exist_ok=True)
    bond_amounts = ["1500000000stake"] + ["150000000stake"] * (nodes - 1)
    for i, h in enumerate(homes):
        out = gentx_dir / f"gentx-node{i}.json"
        p2p_port = p2p_base + i
        sh([
            "chaind", "gentx", from_acct, bond_amounts[i],
            "--chain-id", chain_id,
            "--keyring-backend", keyring,
            "--home", str(h),
            "--ip", "127.0.0.1",
            "--p2p-port", str(p2p_port),
            "--output-document", str(out),
        ], env=env)

    sh(["chaind", "collect-gentxs", "--home", str(homes[0])], env=env)

    # sync final genesis to all nodes
    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

    # derive local peer IDs (needed for deterministic localhost-only persistent_peers)
    node_ids: List[str] = []
    for h in homes:
        nid = sh(["chaind", "tendermint", "show-node-id", "--home", str(h)], env=env, capture=True).stdout.strip()
        node_ids.append(nid)

    def toml_set(src: str, key: str, val: str) -> str:
        # very small toml patcher: replace the first occurrence of `key = ...`.
        # Use a function replacement to avoid accidental backslash escapes in regex replacement strings.
        import re
        pat = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
        if pat.search(src):
            return pat.sub(lambda m: m.group(1) + val, src, count=1)
        # fallback: append at end (rare)
        return src + f"\n{key} = {val}\n"

    # start all nodes (localhost-only, explicit peers, no PEX)
    procs: List[NodeProc] = []
    peers_all = [f"{node_ids[i]}@127.0.0.1:{p2p_base+i}" for i in range(nodes)]

    for i, h in enumerate(homes):
        p2p = p2p_base + i
        rpc = rpc_base + i
        api = api_base + i
        grpc = grpc_base + i

        # hard lock to localhost-only networking
        conf = (h / "config" / "config.toml")
        txt = conf.read_text(encoding="utf-8")
        txt = toml_set(txt, "seeds", '""')
        peers_other = [p for p in peers_all if not p.startswith(node_ids[i] + "@")]
        txt = toml_set(txt, "persistent_peers", '"' + ",".join(peers_other) + '"')
        txt = toml_set(txt, "pex", "false")
        txt = toml_set(txt, "addr_book_strict", "false")
        txt = toml_set(txt, "external_address", '""')
        # ensure tx indexing enabled so `chaind query tx <hash>` works
        if 'indexer = "null"' in txt:
            txt = txt.replace('indexer = "null"', 'indexer = "kv"')
        conf.write_text(txt, encoding="utf-8")

        # remove any cached peers (prevents dialing public IPs from stale addrbook)
        ab = h / "config" / "addrbook.json"
        if ab.exists():
            ab.unlink()

        log = tmp_root / f"node{i}.log"
        cmd = [
            "chaind", "start",
            "--home", str(h),
            "--p2p.laddr", f"tcp://127.0.0.1:{p2p}",
            "--rpc.laddr", f"tcp://127.0.0.1:{rpc}",
            "--rpc.pprof_laddr", "127.0.0.1:0",  # avoid pprof port conflicts
            "--grpc.address", f"127.0.0.1:{grpc}",
            "--grpc-web.enable=false",
            "--api.enable",
            "--api.address", f"tcp://127.0.0.1:{api}",
        ]
        p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, text=True)
        procs.append(NodeProc(i=i, home=h, p=p))

    def stop_all():
        for np in procs:
            if np.p.poll() is None:
                np.p.send_signal(signal.SIGTERM)
        time.sleep(1.0)
        for np in procs:
            if np.p.poll() is None:
                np.p.kill()

    try:
        h = wait_height(rpc_base + 0, 1, timeout_s=60.0)
        print(f"[ok] chain height={h}")

        # configure cli on node0
        node_rpc = f"tcp://127.0.0.1:{rpc_base+0}"
        sh(["chaind", "config", "chain-id", chain_id, "--home", str(homes[0])], env=env)
        sh(["chaind", "config", "keyring-backend", keyring, "--home", str(homes[0])], env=env)
        sh(["chaind", "config", "node", node_rpc, "--home", str(homes[0])], env=env)
        node_args = ["--node", node_rpc]

        def get_height() -> int:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{rpc_base}/status", timeout=1.0) as r:
                s = json.load(r)
            return int(s["result"]["sync_info"]["latest_block_height"])

        def wait_height_at_least(h: int, timeout_s: float = 10.0):
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                try:
                    if get_height() >= h:
                        return
                except Exception:
                    pass
                time.sleep(0.3)
            raise RuntimeError(f"chain did not reach height>={h}")

        def get_sequence(addr: str) -> int:
            r = sh([
                "chaind", "query", "auth", "account", addr,
                "-o", "json",
                "--home", str(homes[0]),
            ] + node_args, env=env, capture=True, check=False)
            if r.returncode != 0 or not r.stdout.strip():
                return -1
            j = json.loads(r.stdout)
            acc = j.get("account", j)
            # BaseAccount sequence is a string in JSON
            seq = acc.get("sequence", "0")
            try:
                return int(seq)
            except Exception:
                return -1

        def wait_sequence_increase(addr: str, prev: int, timeout_s: float = 20.0):
            t0 = time.time()
            while time.time() - t0 < timeout_s:
                s = get_sequence(addr)
                if s >= 0 and s > prev:
                    return s
                time.sleep(0.5)
            raise RuntimeError(f"account sequence did not increase (prev={prev})")

        def query_tx_json(txh: str, *, tries: int = 60, sleep_s: float = 0.7) -> dict:
            for _ in range(tries):
                r = sh(["chaind", "query", "tx", txh, "-o", "json", "--home", str(homes[0])] + node_args,
                       env=env, capture=True, check=False)
                if r.returncode == 0 and r.stdout.strip():
                    try:
                        return json.loads(r.stdout)
                    except Exception:
                        pass
                time.sleep(sleep_s)
            raise RuntimeError(f"query tx failed (not found): {txh}")

        # (cli config already set above)
        # resolve sender address (needed for sequence polling)
        from_addr = sh([
            "chaind", "keys", "show", from_acct, "-a",
            "--keyring-backend", keyring,
            "--home", str(homes[0]),
        ], env=env, capture=True).stdout.strip()

        # compute topk coalition from staking validators query
        vals = json.loads(sh(["chaind", "query", "staking", "validators", "-o", "json", "--home", str(homes[0])] + node_args, env=env, capture=True).stdout)
        arr = vals.get("validators", vals) if isinstance(vals, dict) else vals
        ops = [v["operator_address"] for v in arr]

        csv_path = art_dir / "results" / "minirun_onchain_draws.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["lambda_ppm", "tag", "txhash", "members_csv"] + [f"top{k}_seats" for k in topk_vals])

            for lam, tag in zip(lambda_vals, tags):
                # set lambda
                r = sh([
                    "chaind", "tx", "adaptivecommittee", "set-lambda", str(lam),
                    "--from", from_acct,
                    "--fees", fees,
                    "--broadcast-mode", broadcast_mode,
                    "--chain-id", chain_id,
                    "-y",
                    "-o", "json",
                    "--home", str(homes[0]),
                ] + node_args, env=env, capture=True, check=False)
                if r.returncode != 0:
                    print("[tx error] set-lambda failed", file=sys.stderr)
                    print(r.stdout, file=sys.stderr)
                    print(r.stderr, file=sys.stderr)
                    raise RuntimeError("set-lambda tx failed")
                txj = json.loads(r.stdout)
                if int(txj.get("code", 0)) != 0:
                    raise RuntimeError(
                        "set-lambda tx failed: "
                        + json.dumps({
                            "code": txj.get("code"),
                            "codespace": txj.get("codespace"),
                            "raw_log": txj.get("raw_log"),
                            "logs": txj.get("logs"),
                        })
                    )
                txh = txj.get("txhash", "")
                seq0 = get_sequence(from_addr)
                if seq0 < 0:
                    raise RuntimeError("could not read sender sequence")
                # set-lambda should consume one sequence
                wait_sequence_increase(from_addr, seq0, timeout_s=30.0)

                for j in range(draws_per_setting):
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
                        print("[tx error] draw-committee failed", file=sys.stderr)
                        print(r.stdout, file=sys.stderr)
                        print(r.stderr, file=sys.stderr)
                        raise RuntimeError("draw-committee tx failed")
                    txj = json.loads(r.stdout)
                    if int(txj.get("code", 0)) != 0:
                        raise RuntimeError(
                            "draw-committee tx failed: "
                            + json.dumps({
                                "code": txj.get("code"),
                                "codespace": txj.get("codespace"),
                                "raw_log": txj.get("raw_log"),
                                "logs": txj.get("logs"),
                            })
                        )
                    txh = txj.get("txhash", "")
                    seq1 = get_sequence(from_addr)
                    if seq1 < 0:
                        raise RuntimeError("could not read sender sequence")
                    wait_sequence_increase(from_addr, seq1, timeout_s=30.0)
                    # Read committee from module state to avoid tx-indexing issues.
                    qr = sh([
                        "chaind", "query", "adaptivecommittee", "last-draw", tag,
                        "-o", "json",
                        "--home", str(homes[0]),
                    ] + node_args, env=env, capture=True, check=False)
                    if qr.returncode != 0:
                        print("[query error] last-draw failed", file=sys.stderr)
                        print(qr.stdout, file=sys.stderr)
                        print(qr.stderr, file=sys.stderr)
                        raise RuntimeError("last-draw query failed")
                    qj = json.loads(qr.stdout)
                    members_csv = qj.get("membersCsv") or qj.get("members_csv") or ""

                    # Wait until the draw tx is actually committed (sequence increments only on deliver).
                    # We use module state as the source of truth.
                    if members_csv == "":
                        t0 = time.time()
                        while time.time() - t0 < 15.0:
                            time.sleep(0.5)
                            qr2 = sh([
                                "chaind", "query", "adaptivecommittee", "last-draw", tag,
                                "-o", "json",
                                "--home", str(homes[0]),
                            ] + node_args, env=env, capture=True, check=False)
                            if qr2.returncode == 0 and qr2.stdout.strip():
                                qj2 = json.loads(qr2.stdout)
                                members_csv = qj2.get("membersCsv") or qj2.get("members_csv") or ""
                                if members_csv:
                                    break
                        if members_csv == "":
                            raise RuntimeError("draw tx not committed (last-draw still empty)")
                    members = members_csv.split(",") if members_csv else []
                    counts = {op: 0 for op in ops}
                    for m in members:
                        counts[m] = counts.get(m, 0) + 1

                    seats = []
                    for k in topk_vals:
                        seats.append(sum(counts.get(op, 0) for op in ops[:k]))

                    w.writerow([lam, tag, txh, members_csv] + seats)

        # plot simple summary: avg seats for top1/top3 by lambda
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
        def avg(col: str, lam: str) -> float:
            xs = [float(r[col]) for r in rows if r["lambda_ppm"] == lam]
            return sum(xs) / max(1, len(xs))

        lams_s = [str(x) for x in lambda_vals]
        x_lams = [float(lam) / 1_000_000.0 for lam in lambda_vals]
        plt.figure(figsize=(6.3, 3.8))
        for k in topk_vals:
            ys = [avg(f"top{k}_seats", lam) for lam in lams_s]
            plt.plot(x_lams, ys, marker="o", label=f"top-{k} avg seats")
        plt.xlabel("lambda")
        plt.ylabel(f"avg seats in committee (m={committee_size})")
        plt.title(f"On-chain sanity run: seats vs lambda (draws={draws_per_setting})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = art_dir / "plots" / "minirun_onchain_seats_vs_lambda.png"
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        print(str(out_png))

    finally:
        stop_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
