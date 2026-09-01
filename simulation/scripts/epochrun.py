#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import errno
import socket

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sh(cmd: List[str], *, env=None, cwd=None, capture: bool = False, check: bool = True, text: bool = True):
    # Allow pinning chaind path to avoid PATH mismatch across environments.
    if cmd and cmd[0] == "chaind":
        pinned = None
        if env is not None:
            pinned = env.get("POC_CHAIND")
        pinned = pinned or os.environ.get("POC_CHAIND")
        if pinned:
            cmd = [pinned] + cmd[1:]

    r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=capture, check=False, text=text)
    if check and r.returncode != 0:
        out = (r.stdout or "") if capture else (r.stdout or "")
        err = (r.stderr or "")
        raise RuntimeError(
            f"command failed (rc={r.returncode}): {cmd}\n"
            f"stdout:\n{out}\n"
            f"stderr:\n{err}\n"
        )
    return r


def parse_json_output(r: subprocess.CompletedProcess, what: str) -> dict:
    raw = (r.stdout or "").strip()
    if not raw:
        raise RuntimeError(
            f"{what}: empty stdout (rc={r.returncode})\n"
            f"stderr:\n{(r.stderr or '').strip()}"
        )
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(
            f"{what}: invalid JSON output (rc={r.returncode}): {e}\n"
            f"stdout:\n{raw[:1000]}\n"
            f"stderr:\n{(r.stderr or '').strip()}"
        )


def load_yaml_minimal(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env_ms(name: str, default_ms: int) -> str:
    try:
        v = int(float(os.environ.get(name, str(default_ms))))
    except Exception:
        v = int(default_ms)
    v = max(50, v)
    return f'"{v}ms"'


def tracked_entity_context(entry_kind: str) -> Dict[str, str]:
    kind = str(entry_kind or "sybil").strip().lower()
    if kind in ("entrant", "honest_newcomer", "newcomer"):
        label = "entrant" if kind == "entrant" else "honest newcomer"
        return {
            "tracked_prefix": "entrant",
            "tracked_entity_mode": "entrant",
            "tracked_entity_label": label,
            "tracked_entity_label_title": label.capitalize(),
            "tracked_baseline_label": f"{label} stake baseline",
            "baseline_comparison_mode": "gap",
        }
    return {
        "tracked_prefix": "sybil",
        "tracked_entity_mode": "attacker",
        "tracked_entity_label": "attacker",
        "tracked_entity_label_title": "Attacker",
        "tracked_baseline_label": "stake baseline",
        "baseline_comparison_mode": "reduction",
    }


TRACKED_ALIAS_MAP = {
    "attacker_seats": "tracked_seats",
    "attacker_seats_share": "tracked_seats_share",
    "attacker_stake_ppm": "tracked_stake_ppm",
    "attacker_age_ppm": "tracked_age_ppm",
    "attacker_weight_ppm": "tracked_weight_ppm",
    "attacker_validators": "tracked_validators",
    "attacker_unique_members_len": "tracked_unique_members_len",
    "attacker_ops_len": "tracked_ops_len",
    "attacker_stakes_csv": "tracked_stakes_csv",
    "attacker_tokens": "tracked_tokens",
    "seat_minus_weight": "tracked_minus_weight",
    "seat_minus_stake_indep": "tracked_minus_stake_indep",
    "mean_attacker_share": "mean_tracked_share",
    "min_attacker_share": "min_tracked_share",
    "max_attacker_share": "max_tracked_share",
    "mean_attacker_weight_share": "mean_tracked_weight_share",
    "mean_attacker_age_share": "mean_tracked_age_share",
    "mean_attacker_stake_share": "mean_tracked_stake_share",
    "mean_attacker_weight_ppm": "mean_tracked_weight_ppm",
    "mean_attacker_age_ppm": "mean_tracked_age_ppm",
    "mean_attacker_stake_ppm": "mean_tracked_stake_ppm",
    "mean_seat_minus_weight": "mean_tracked_minus_weight",
    "mean_seat_minus_stake_indep": "mean_tracked_minus_stake_indep",
    "post_attacker_weight_mean": "post_tracked_weight_mean",
    "post_attacker_seat_mean": "post_tracked_seat_mean",
    "post_stake_baseline_mean": "post_tracked_stake_baseline_mean",
    "post_attacker_seat_mean_model": "post_tracked_seat_mean_model",
    "auc_gain_vs_stake": "auc_tracked_gain_vs_stake",
    "reduction_vs_baseline_full_pct": "tracked_vs_baseline_full_pct",
    "reduction_vs_baseline_1ep_pct": "tracked_vs_baseline_1ep_pct",
    "reduction_vs_baseline_2ep_pct": "tracked_vs_baseline_2ep_pct",
    "reduction_vs_baseline_3ep_pct": "tracked_vs_baseline_3ep_pct",
    "reduction_vs_baseline_5ep_pct": "tracked_vs_baseline_5ep_pct",
    "time_to_95pct_baseline_draws": "time_to_95pct_tracked_baseline_draws",
}


def enrich_tracked_row(row: Dict[str, str], tracked_meta: Dict[str, str]) -> Dict[str, str]:
    out = dict(row)
    out["tracked_entity_mode"] = tracked_meta["tracked_entity_mode"]
    out["tracked_entity_label"] = tracked_meta["tracked_entity_label"]
    out["tracked_baseline_label"] = tracked_meta["tracked_baseline_label"]
    out["baseline_comparison_mode"] = tracked_meta["baseline_comparison_mode"]
    for old_key, new_key in TRACKED_ALIAS_MAP.items():
        if old_key in out:
            out[new_key] = out.get(old_key, "")
    return out


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
    raise RuntimeError("chain did not reach requested height")


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

    # SDK v0.53 responses may wrap sequence under `value` / `base_account`.
    # If sequence field is absent, return -1 (unknown) instead of assuming 0.
    seq = None
    if isinstance(acc, dict):
        if "sequence" in acc:
            seq = acc.get("sequence")
        elif isinstance(acc.get("value"), dict) and "sequence" in acc.get("value", {}):
            seq = acc.get("value", {}).get("sequence")
        elif isinstance(acc.get("base_account"), dict) and "sequence" in acc.get("base_account", {}):
            seq = acc.get("base_account", {}).get("sequence")

    if seq is None:
        return -1

    try:
        return int(seq)
    except Exception:
        return -1


def wait_sequence_increase(env, home0: Path, node_args: List[str], addr: str, prev: int, timeout_s: float = 30.0) -> int:
    # If sequence is unavailable in auth query response for this chain build,
    # skip sequence-based confirmation and let tx-inclusion checks drive correctness.
    if prev < 0:
        return -1

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = get_sequence(env, home0, node_args, addr)
        if s >= 0 and s > prev:
            return s
        time.sleep(0.5)
    raise RuntimeError(f"account sequence did not increase (prev={prev})")


def wait_account_exists(env, home0: Path, node_rpc: str, addr: str, timeout_s: float = 30.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        q = sh([
            "chaind", "query", "auth", "account", addr,
            "-o", "json",
            "--home", str(home0),
            "--node", node_rpc,
        ], env=env, capture=True, check=False)
        if q.returncode == 0 and q.stdout.strip():
            try:
                return json.loads(q.stdout)
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError(f"account not found on-chain in time: {addr}")


def extract_account_meta(account_json: dict) -> Tuple[int, int]:
    acc = account_json.get("account", account_json)
    # Cosmos SDK v0.53 often returns BaseAccount wrapped under `value`.
    # Some paths can also return `base_account`.
    value = acc.get("value", {}) if isinstance(acc, dict) else {}
    base = acc.get("base_account", {}) if isinstance(acc, dict) else {}

    anum_raw = (
        (value.get("account_number") if isinstance(value, dict) else None)
        or (base.get("account_number") if isinstance(base, dict) else None)
        or (acc.get("account_number") if isinstance(acc, dict) else None)
        or "0"
    )
    seq_raw = (
        (value.get("sequence") if isinstance(value, dict) else None)
        or (base.get("sequence") if isinstance(base, dict) else None)
        or (acc.get("sequence") if isinstance(acc, dict) else None)
        or "0"
    )
    return int(anum_raw), int(seq_raw)


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
                  from_acct_base: str,
                  keyring: str,
                  sybil_active_at_genesis: bool = True,
                  attack_mode: str = "replacement",
                  whale_share: float = 0.0,
                  drift_pool_key: Optional[str] = None,
                  drift_pool_balance: Optional[int] = None,
                  ) -> Tuple[List[Path], List[NodeProc]]:
    is_additive = (str(attack_mode).strip().lower() == "additive")
    whale_offset = 1 if is_additive else 0
    nodes = honest_nodes + whale_offset + sybil_k
    if tmp_root.exists():
        sh(["bash", "-lc", f"rm -rf {tmp_root}"])
    tmp_root.mkdir(parents=True, exist_ok=True)

    homes: List[Path] = []
    for i in range(nodes):
        h = tmp_root / f"node{i}"
        h.mkdir(parents=True, exist_ok=True)
        if i < honest_nodes:
            moniker = f"honest{i}"
        elif is_additive and i == honest_nodes:
            moniker = "sybil_whale"
        else:
            moniker = f"sybil{i - honest_nodes - whale_offset}"
        sh(["chaind", "init", moniker, "--chain-id", chain_id, "--home", str(h)], env=env)
        homes.append(h)

    h0 = homes[0]

    # Create per-node keys (unique names) in each node home.
    # Then copy keyring files into node0 keyring so we can sign any node's tx from node0.
    key_names: List[str] = []
    addrs: List[str] = []
    for i, h in enumerate(homes):
        kn = f"{from_acct_base}{i}"
        key_names.append(kn)
        sh(["chaind", "keys", "add", kn, "--keyring-backend", keyring, "--home", str(h)], env=env, capture=True)
        addr = sh([
            "chaind", "keys", "show", kn, "-a",
            "--keyring-backend", keyring,
            "--home", str(h),
        ], env=env, capture=True).stdout.strip()
        addrs.append(addr)

    # Copy all keys to node0 keyring (test backend) so we can sign with any key from node0.
    # NOTE: this is a pragmatic hack to avoid CLI signing bugs; for PoC only.
    h0_keyring = homes[0] / "keyring-test"
    h0_keyring.mkdir(parents=True, exist_ok=True)
    for i in range(1, nodes):
        hi_keyring = homes[i] / "keyring-test"
        if hi_keyring.exists():
            for key_file in hi_keyring.glob("*"):
                shutil.copy(str(key_file), str(h0_keyring))

    for addr in addrs:
        sh(["chaind", "genesis", "add-genesis-account", addr, f"2000000000000{denom}", "--home", str(homes[0])], env=env)

    # Optional drift-pool account: a non-validator key funded at genesis. It is
    # used by the concentration-drift scenario to redelegate stake between
    # validators at epoch boundaries (see _drift_initial_delegations and
    # _drift_migrate_step). Has no effect when drift_pool_key is None.
    if drift_pool_key and drift_pool_balance:
        sh(["chaind", "keys", "add", drift_pool_key, "--keyring-backend", keyring, "--home", str(h0)], env=env, capture=True)
        drift_pool_addr_local = sh([
            "chaind", "keys", "show", drift_pool_key, "-a",
            "--keyring-backend", keyring, "--home", str(h0),
        ], env=env, capture=True).stdout.strip()
        sh([
            "chaind", "genesis", "add-genesis-account", drift_pool_addr_local,
            f"{drift_pool_balance}{denom}", "--home", str(h0),
        ], env=env)

    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

    total_bond = 1_000_000_000
    honest_total = 1.0 - beta
    if is_additive:
        whale_stake_frac = beta * float(whale_share)
        fresh_each = ((beta * (1.0 - float(whale_share))) / sybil_k) if sybil_k > 0 else 0.0
    else:
        whale_stake_frac = 0.0
        fresh_each = (beta / sybil_k) if sybil_k > 0 else 0.0

    bond_ints: List[int] = []
    for i in range(nodes):
        if i < honest_nodes:
            share = honest_total / honest_nodes
        elif is_additive and i == honest_nodes:
            share = whale_stake_frac
        else:
            share = fresh_each
        bond_ints.append(int(round(share * total_bond)))
    drift = total_bond - sum(bond_ints)
    bond_ints[0] += drift

    gentx_dir = homes[0] / "config" / "gentx"
    gentx_dir.mkdir(parents=True, exist_ok=True)

    for i, h in enumerate(homes):
        if is_additive:
            if i >= honest_nodes + whale_offset:
                continue
        else:
            if (not sybil_active_at_genesis) and (i >= honest_nodes):
                continue
        out = gentx_dir / f"gentx-node{i}.json"
        p2p_port = p2p_base + i
        gentx_from = key_names[i]
        sh([
            "chaind", "genesis", "gentx", gentx_from, f"{bond_ints[i]}{denom}",
            "--chain-id", chain_id,
            "--keyring-backend", keyring,
            "--home", str(h),
            "--ip", "127.0.0.1",
            "--p2p-port", str(p2p_port),
            "--output-document", str(out),
        ], env=env)

    sh(["chaind", "genesis", "collect-gentxs", "--home", str(homes[0])], env=env)
    g0 = homes[0] / "config" / "genesis.json"
    for h in homes[1:]:
        (h / "config" / "genesis.json").write_bytes(g0.read_bytes())

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

        conf = h / "config" / "config.toml"
        txt = conf.read_text(encoding="utf-8")
        txt = toml_set(txt, "seeds", '""')
        peers_other = [p for p in peers_all if not p.startswith(node_ids[i] + "@")]
        txt = toml_set(txt, "persistent_peers", '"' + ",".join(peers_other) + '"')
        txt = toml_set(txt, "pex", "false")
        txt = toml_set(txt, "addr_book_strict", "false")
        txt = toml_set(txt, "allow_duplicate_ip", "true")
        txt = toml_set(txt, "external_address", '""')
        txt = toml_set(txt, "timeout_propose", _env_ms("POC_TIMEOUT_PROPOSE_MS", 800))
        txt = toml_set(txt, "timeout_prevote", _env_ms("POC_TIMEOUT_PREVOTE_MS", 500))
        txt = toml_set(txt, "timeout_precommit", _env_ms("POC_TIMEOUT_PRECOMMIT_MS", 500))
        txt = toml_set(txt, "timeout_commit", _env_ms("POC_TIMEOUT_COMMIT_MS", 1200))
        if 'indexer = "null"' in txt:
            txt = txt.replace('indexer = "null"', 'indexer = "kv"')
        conf.write_text(txt, encoding="utf-8")

        # SDK v0.53 requires minimum gas prices to be set (app.toml or flag).
        # Set it directly in app.toml for robustness.
        app_toml = h / "config" / "app.toml"
        atxt = app_toml.read_text(encoding="utf-8")
        atxt = atxt.replace('minimum-gas-prices = ""', f'minimum-gas-prices = "0.000000001{denom}"')
        app_toml.write_text(atxt, encoding="utf-8")

        ab = h / "config" / "addrbook.json"
        if ab.exists():
            ab.unlink()

        log = tmp_root / f"node{i}.log"
        binp = (env.get("POC_CHAIND") or os.environ.get("POC_CHAIND") or "chaind")
        cmd = [
            binp, "start",
            "--home", str(h),
            "--p2p.laddr", f"tcp://127.0.0.1:{p2p}",
            "--rpc.laddr", f"tcp://127.0.0.1:{rpc}",
            "--rpc.pprof_laddr", "127.0.0.1:0",
            "--grpc.address", f"127.0.0.1:{grpc}",
            "--grpc-web.enable=false",
            "--api.enable",
            "--api.address", f"tcp://127.0.0.1:{api}",
            # SDK v0.53 requires explicit minimum gas prices (or app.toml setting).
            # Use a tiny non-zero value; some versions treat 0<denom> as unset.
            "--minimum-gas-prices", f"0.000000001{denom}",
        ]
        p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, text=True)
        procs.append(NodeProc(i=i, home=h, p=p))

    return homes, procs


def stop_all(procs: List[NodeProc]):
    for np in procs:
        if np.p.poll() is None:
            np.p.send_signal(signal.SIGTERM)

    # Give nodes a chance to shutdown cleanly.
    deadline = time.time() + 4.0
    while time.time() < deadline:
        alive = [np for np in procs if np.p.poll() is None]
        if not alive:
            break
        time.sleep(0.2)

    for np in procs:
        if np.p.poll() is None:
            np.p.kill()

    # Reap all children so their exit is fully observed before restart.
    for np in procs:
        try:
            np.p.wait(timeout=5.0)
        except Exception:
            pass

    # Let OS fully release sockets before next start.
    time.sleep(0.5)


def _tcp_port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    """True iff a fresh TCP listener can bind to host:port right now.

    This is more reliable than parsing `ss` output across distro versions and
    catches exactly the failure mode that later startup hits (`bind: address
    already in use`).
    """
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


def _pids_listening_on_proc(port: int) -> List[int]:
    """PID discovery for a LISTEN socket on `port` using only /proc (no lsof/ss).

    Robust on minimal hosts where lsof is absent and `ss` does not expose pids
    without privileges. Maps port -> listening socket inode via /proc/net/tcp{,6},
    then resolves inode -> pid by scanning /proc/<pid>/fd symlinks (our own
    leaked chain nodes are owned by this user, so the readlinks succeed)."""
    inodes: set = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                f.readline()  # header
                for line in f:
                    parts = line.split()
                    if len(parts) < 10 or parts[3] != "0A":  # 0A == TCP_LISTEN
                        continue
                    try:
                        lport = int(parts[1].rsplit(":", 1)[1], 16)
                    except (ValueError, IndexError):
                        continue
                    if lport == int(port):
                        inodes.add(parts[9])
        except Exception:
            pass
    if not inodes:
        return []
    pids: set = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.add(int(entry))
                    break
        except OSError:
            continue
    return sorted(pids)


def _pids_listening_on(port: int) -> List[int]:
    """Best-effort list of PIDs holding a LISTEN socket on host:port."""
    pids: set = set()
    try:
        r = subprocess.run(
            ["lsof", "-ti", f"tcp:{int(port)}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        for tok in (r.stdout or "").split():
            tok = tok.strip()
            if tok.isdigit():
                pids.add(int(tok))
    except Exception:
        pass
    if not pids:
        # Fallback to `ss`; parse `pid=NNN` out of its process column.
        try:
            r = subprocess.run(
                ["ss", "-ltnpH", f"( sport = :{int(port)} )"],
                capture_output=True, text=True, timeout=5,
            )
            for chunk in (r.stdout or "").split("pid=")[1:]:
                num = ""
                for ch in chunk:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    pids.add(int(num))
        except Exception:
            pass
    if not pids:
        # lsof/ss unavailable or privilege-blocked: fall back to pure /proc.
        try:
            pids.update(_pids_listening_on_proc(int(port)))
        except Exception:
            pass
    return sorted(pids)


def _force_free_port(port: int) -> bool:
    """SIGKILL any *chain-binary* process holding a LISTEN socket on `port`.

    Restricted to our chain process (cmdline match) so we never kill an
    unrelated service that happens to sit on the port. Stubborn busy ports in
    this harness are essentially always orphaned chain nodes left behind by a
    previous run that died before its teardown ran."""
    killed = False
    for pid in _pids_listening_on(port):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore")
        except Exception:
            cmdline = ""
        if ("chain-five-three" in cmdline or "chaind" in cmdline
                or "/build/chain" in cmdline):
            try:
                os.kill(pid, signal.SIGKILL)
                killed = True
                print(f"[ports] killed orphan chain pid={pid} holding :{port}",
                      file=sys.stderr)
            except Exception:
                pass
    return killed


def wait_ports_free(ports: List[int], timeout_s: float = 20.0):
    """Wait until all given host-local TCP ports are actually bindable.

    Self-heals: a port still held after a short grace is reaped if (and only
    if) a chain-binary orphan owns it, then re-checked. This keeps a crashed
    prior run (whose nodes leaked) from permanently blocking the next start."""
    deadline = time.time() + timeout_s
    uniq = sorted(set(int(p) for p in ports if int(p) > 0))
    busy: List[int] = []
    t0 = time.time()
    while time.time() < deadline:
        busy = [p for p in uniq if not _tcp_port_bindable(p)]
        if not busy:
            return
        # Give a clean SIGTERM teardown ~3s to release sockets on its own; after
        # that, actively reap chain orphans squatting on the ports we need.
        if time.time() - t0 > 3.0:
            for p in busy:
                _force_free_port(p)
        time.sleep(0.6)
    busy = [p for p in uniq if not _tcp_port_bindable(p)]
    if busy:
        # Final reap attempt before giving up.
        for p in busy:
            _force_free_port(p)
        time.sleep(1.0)
        busy = [p for p in uniq if not _tcp_port_bindable(p)]
    if busy:
        raise RuntimeError(f"ports still busy after timeout: {busy[:10]}")


def start_existing_network(*, env, tmp_root: Path, homes: List[Path], p2p_base: int, rpc_base: int, api_base: int, grpc_base: int, denom: str) -> List[NodeProc]:
    # Ensure previous node sockets are fully released before restart.
    n = len(homes)
    ports = [p2p_base + i for i in range(n)] + [rpc_base + i for i in range(n)] + [api_base + i for i in range(n)] + [grpc_base + i for i in range(n)]
    # Ownerless FIN_WAIT/LAST_ACK sockets from a simultaneous teardown drain
    # only after tcp_fin_timeout (60s on Linux) and cannot be reaped, so the
    # grace must exceed that; also scale with the validator-set size.
    wait_ports_free(ports, timeout_s=max(90.0, 2.0 * n))

    # Recompute node IDs and persistent peers for the copied checkpoint homes.
    node_ids: List[str] = []
    for h in homes:
        nid = sh(["chaind", "tendermint", "show-node-id", "--home", str(h)], env=env, capture=True).stdout.strip()
        node_ids.append(nid)
    peers_all = [f"{node_ids[i]}@127.0.0.1:{p2p_base+i}" for i in range(len(homes))]

    procs: List[NodeProc] = []
    for i, h in enumerate(homes):
        p2p = p2p_base + i
        rpc = rpc_base + i
        api = api_base + i
        grpc = grpc_base + i
        log = tmp_root / f"node{i}.log"

        conf = h / "config" / "config.toml"
        txt = conf.read_text(encoding="utf-8")
        txt = toml_set(txt, "seeds", '""')
        peers_other = [p for p in peers_all if not p.startswith(node_ids[i] + "@")]
        txt = toml_set(txt, "persistent_peers", '"' + ",".join(peers_other) + '"')
        txt = toml_set(txt, "pex", "false")
        txt = toml_set(txt, "addr_book_strict", "false")
        txt = toml_set(txt, "allow_duplicate_ip", "true")
        txt = toml_set(txt, "external_address", '""')
        txt = toml_set(txt, "timeout_propose", _env_ms("POC_TIMEOUT_PROPOSE_MS", 800))
        txt = toml_set(txt, "timeout_prevote", _env_ms("POC_TIMEOUT_PREVOTE_MS", 500))
        txt = toml_set(txt, "timeout_precommit", _env_ms("POC_TIMEOUT_PRECOMMIT_MS", 500))
        txt = toml_set(txt, "timeout_commit", _env_ms("POC_TIMEOUT_COMMIT_MS", 1200))
        if 'indexer = "null"' in txt:
            txt = txt.replace('indexer = "null"', 'indexer = "kv"')
        conf.write_text(txt, encoding="utf-8")

        # Keep app.toml minimum gas prices non-empty for SDK v0.53.
        app_toml = h / "config" / "app.toml"
        atxt = app_toml.read_text(encoding="utf-8")
        atxt = atxt.replace('minimum-gas-prices = ""', f'minimum-gas-prices = "0.000000001{denom}"')
        app_toml.write_text(atxt, encoding="utf-8")

        # Reset addrbook to avoid stale peer identity mappings from checkpoint copy.
        ab = h / "config" / "addrbook.json"
        if ab.exists():
            ab.unlink()

        binp = (env.get("POC_CHAIND") or os.environ.get("POC_CHAIND") or "chaind")
        cmd = [
            binp, "start",
            "--home", str(h),
            "--p2p.laddr", f"tcp://127.0.0.1:{p2p}",
            "--rpc.laddr", f"tcp://127.0.0.1:{rpc}",
            "--rpc.pprof_laddr", "127.0.0.1:0",
            "--grpc.address", f"127.0.0.1:{grpc}",
            "--grpc-web.enable=false",
            "--api.enable",
            "--api.address", f"tcp://127.0.0.1:{api}",
            # SDK v0.53 requires explicit minimum gas prices (or app.toml setting).
            # Use a tiny non-zero value; some versions treat 0<denom> as unset.
            "--minimum-gas-prices", f"0.000000001{denom}",
        ]
        started = False
        last_err = ""
        for attempt in range(1, 4):
            p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, text=True)
            time.sleep(1.2 * attempt)
            if p.poll() is None:
                procs.append(NodeProc(i=i, home=h, p=p))
                started = True
                break

            # Read a short log tail for diagnostics / retry decision.
            try:
                tail = "\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:])
            except Exception:
                tail = ""
            last_err = tail

            # Typical transient after restart/copy: socket still in TIME_WAIT / address in use.
            if "address already in use" in tail.lower() or "bind:" in tail.lower():
                time.sleep(1.0 * attempt)
                continue
            # Non-transient startup error: stop retry loop.
            break

        if not started:
            # Prevent orphan nodes when one later node fails during startup.
            stop_all(procs)
            raise RuntimeError(f"node{i} died immediately; log={log}\n{last_err}")
    return procs


def inject_sybils(env, homes, honest_nodes, sybil_k, beta, chain_id, denom, from_acct_base, keyring, fees, node_rpc, start_sybil_idx: int = 0, inject_count: int = None, moniker_prefix: str = "sybil", attack_mode: str = "replacement", whale_share: float = 0.0) -> int:
    """Inject late-entry validators via direct create-validator path.

    Args:
      start_sybil_idx: 0-based index within injected cohort.
      inject_count: number of validators to inject from start index (None => until end).
      moniker_prefix: prefix used to label the injected cohort (e.g. sybil, entrant).
      attack_mode: "replacement" | "additive". In additive mode, the whale occupies
        homes[honest_nodes]; fresh cohort starts at homes[honest_nodes+1].
      whale_share: fraction of beta kept by the whale (additive mode only).
    """
    if sybil_k <= 0 or beta <= 0:
        return 0

    is_additive = (str(attack_mode).strip().lower() == "additive")
    whale_offset = 1 if is_additive else 0

    total_bond = 1_000_000_000
    honest_total = 1.0 - beta
    if is_additive:
        whale_stake_frac = beta * float(whale_share)
        fresh_each = (beta * (1.0 - float(whale_share))) / sybil_k
    else:
        whale_stake_frac = 0.0
        fresh_each = beta / sybil_k
    nodes = honest_nodes + whale_offset + sybil_k

    bond_ints = []
    for i in range(nodes):
        if i < honest_nodes:
            share = honest_total / honest_nodes
        elif is_additive and i == honest_nodes:
            share = whale_stake_frac
        else:
            share = fresh_each
        bond_ints.append(int(round(share * total_bond)))
    drift = total_bond - sum(bond_ints)
    bond_ints[0] += drift

    start = max(0, int(start_sybil_idx or 0))
    if start >= sybil_k:
        return 0
    cnt = (sybil_k - start) if (inject_count is None) else max(0, int(inject_count))
    end = min(sybil_k, start + cnt)

    fresh_base = honest_nodes + whale_offset
    ok = 0
    for i in range(fresh_base + start, fresh_base + end):
        h = homes[i]
        moniker = f"{moniker_prefix}{i - fresh_base}"
        bond_amount = bond_ints[i]
        sybil_key = f"{from_acct_base}{i}"

        sybil_addr = sh([
            "chaind", "keys", "show", sybil_key, "-a",
            "--keyring-backend", keyring,
            "--home", str(h),
        ], env=env, capture=True).stdout.strip()
        sybil_valoper = sh([
            "chaind", "keys", "show", sybil_key, "--bech", "val", "-a",
            "--keyring-backend", keyring,
            "--home", str(h),
        ], env=env, capture=True).stdout.strip()

        sybil_q = wait_account_exists(env, homes[0], node_rpc, sybil_addr, timeout_s=60.0)
        sybil_anum, sybil_seq = extract_account_meta(sybil_q)

        pubkey_out = sh([
            "chaind", "tendermint", "show-validator", "--home", str(h)
        ], env=env, capture=True, check=False)
        if pubkey_out.returncode != 0 or not pubkey_out.stdout.strip():
            print(f"inject show-validator failed for {moniker}: {pubkey_out.stdout}\n{pubkey_out.stderr}")
            continue
        try:
            pubkey_obj = json.loads(pubkey_out.stdout.strip())
        except Exception:
            print(f"inject invalid validator pubkey JSON for {moniker}: {pubkey_out.stdout.strip()}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"epoch_inject_{moniker}_") as td:
            valf = Path(td) / "validator.json"
            valdoc = {
                "pubkey": pubkey_obj,
                "amount": f"{bond_amount}{denom}",
                "moniker": moniker,
                "identity": "",
                "website": "",
                "security": "",
                "details": "",
                "commission-rate": "0.10",
                "commission-max-rate": "0.20",
                "commission-max-change-rate": "0.01",
                "min-self-delegation": "1",
            }
            valf.write_text(json.dumps(valdoc), encoding="utf-8")

            r = sh([
                "chaind", "tx", "staking", "create-validator", str(valf),
                "--from", sybil_key,
                "--keyring-backend", keyring,
                "--home", str(h),
                "--node", node_rpc,
                "--chain-id", chain_id,
                "--account-number", str(sybil_anum),
                "--sequence", str(sybil_seq),
                "--sign-mode", "direct",
                "--fees", fees,
                "--gas", "2000000",
                "--broadcast-mode", "sync",
                "-y", "-o", "json",
            ], env=env, capture=True, check=False)

            # Fallback path: let SDK resolve account/sequence when explicit signer metadata is rejected.
            if r.returncode == 0 and (r.stdout or "").strip():
                try:
                    _j0 = json.loads((r.stdout or "").strip())
                except Exception:
                    _j0 = {}
                _raw = str(_j0.get("raw_log", "") or "")
                if int(_j0.get("code", 0)) != 0 and "signature verification failed" in _raw.lower():
                    r = sh([
                        "chaind", "tx", "staking", "create-validator", str(valf),
                        "--from", sybil_key,
                        "--keyring-backend", keyring,
                        "--home", str(h),
                        "--node", node_rpc,
                        "--chain-id", chain_id,
                        "--fees", fees,
                        "--gas", "2000000",
                        "--broadcast-mode", "sync",
                        "-y", "-o", "json",
                    ], env=env, capture=True, check=False)

        if r.returncode != 0:
            print(f"inject create-validator command failed for {moniker}: {r.stdout}\n{r.stderr}")
            continue

        try:
            j = json.loads((r.stdout or "").strip()) if (r.stdout or "").strip() else {}
        except Exception:
            print(f"inject create-validator non-JSON for {moniker}: {r.stdout}\n{r.stderr}")
            continue

        if int(j.get("code", 0)) != 0:
            print(f"inject create-validator code!=0 for {moniker}: {r.stdout}\n{r.stderr}")
            continue

        txh = j.get("txhash", "")
        if txh:
            txq = wait_tx_inclusion(env, homes[0], ["--node", node_rpc], txh, timeout_s=45.0)
            if not txq or int(txq.get("code", 0)) != 0:
                print(f"inject create-validator tx not included or failed for {moniker}: tx={txh} q={txq}")
                continue

        try:
            wait_sequence_increase(env, homes[0], ["--node", node_rpc], sybil_addr, sybil_seq, timeout_s=60.0)
        except Exception:
            pass

        print(f"[inject-ok] {moniker} addr={sybil_addr} valoper={sybil_valoper} tx={txh}")
        ok += 1

    return ok

def count_validators_by_prefix(env, home0: Path, node_args: List[str], prefix: str) -> int:
    vals = json.loads(sh([
        "chaind", "query", "staking", "validators", "-o", "json", "--home", str(home0)
    ] + node_args, env=env, capture=True).stdout)
    arr = vals.get("validators", vals) if isinstance(vals, dict) else vals
    return sum(1 for v in arr if (v.get("description", {}) or {}).get("moniker", "").startswith(prefix))


def build_attack_injection_schedule(profile: str, sybil_k: int, post_attack_epochs: int) -> List[int]:
    """Return per-post-attack-epoch injection counts.

    Profiles:
      - burst: inject all at epoch 1
      - gradual4: inject over first 4 epochs
      - trickle: inject one per epoch
      - pulse: 50% at epoch 1, 50% at epoch 4
    """
    prof = (profile or "burst").strip().lower()
    sched = [0] * max(1, post_attack_epochs)
    if sybil_k <= 0:
        return sched

    if prof == "burst":
        sched[0] = sybil_k
        return sched

    if prof == "gradual4":
        span = min(4, len(sched))
        base = sybil_k // span
        rem = sybil_k - base * span
        for i in range(span):
            sched[i] = base + (1 if i < rem else 0)
        return sched

    if prof == "trickle":
        i = 0
        left = sybil_k
        while left > 0 and i < len(sched):
            sched[i] = 1
            left -= 1
            i += 1
        return sched

    if prof == "pulse":
        first = max(1, sybil_k // 2)
        second = sybil_k - first
        sched[0] = first
        j = min(len(sched)-1, 3)
        sched[j] += second
        return sched

    # Unknown profile -> safe fallback
    sched[0] = sybil_k
    return sched


# ---------------------------------------------------------------------------
# Concentration-drift scenario primitives (RBHC paper PoC, M3.3).
#
# Stake-migration mechanism: a non-validator `drift_pool` account funded at
# genesis holds delegations to every validator. At each drift-active epoch the
# pool issues a `staking redelegate` from a donor validator (rank > donor_top_h)
# to a receiver validator (top_r), shifting bonded power without unbonding
# delays. Initial delegations are equal across validators so the pre-drift
# concentration profile is preserved.
#
# Why redelegate and not unbond+delegate: unbonding has a 21-day lockup that
# would freeze tokens for the entire experiment. Redelegate is instant for
# power purposes (delegation entries lock for the unbonding period but the
# effective validator power moves immediately).
# ---------------------------------------------------------------------------


def _gini_from_stakes(stakes: List[int]) -> float:
    """Population Gini coefficient of a stake vector (runner-side mirror of the
    chain's giniFromStakes). Used to verify the drift bootstrap preserves the
    start profile."""
    xs = sorted(float(s) for s in stakes if s is not None)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(xs))
    g = (2.0 * weighted) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, g))


def _query_validators_sorted(env, home: Path, node_args: List[str]) -> List[Tuple[str, int]]:
    """Return [(operator_address, tokens), ...] sorted by tokens descending.

    Excludes validators with zero or non-positive tokens (e.g., jailed).
    """
    raw = sh([
        "chaind", "query", "staking", "validators", "-o", "json", "--home", str(home),
    ] + node_args, env=env, capture=True)
    if raw.returncode != 0:
        return []
    try:
        data = json.loads(raw.stdout)
    except Exception:
        return []
    arr = data.get("validators", data) if isinstance(data, dict) else data
    out: List[Tuple[str, int]] = []
    for v in arr:
        op = v.get("operator_address", "") if isinstance(v, dict) else ""
        try:
            tok = int(v.get("tokens", "0") or 0)
        except (ValueError, TypeError):
            tok = 0
        if op and tok > 0:
            out.append((op, tok))
    out.sort(key=lambda t: -t[1])
    return out


def _drift_initial_delegations(*, env, home: Path, node_args: List[str],
                               drift_pool_key: str, validator_ops: List[str],
                               amounts: List[int], denom: str,
                               chain_id: str, keyring: str, fees: str,
                               gas: str, broadcast: str) -> int:
    """Delegate a per-validator amount (parallel to validator_ops) from
    drift_pool to each validator, one tx each.

    Fix 1: amounts are computed PROPORTIONAL to each validator's current bonded
    stake by the caller, so the bootstrap preserves the starting concentration
    profile instead of flattening it. A flat (equal) delegation would reduce
    relative dispersion (lower the Gini), perturbing the very distribution the
    drift experiment is supposed to start from.

    Returns the count of successful delegations.

    Execution note: use `sync` broadcast, then explicitly wait for sequence
    increase, tx inclusion, and at least one more block before sending the next
    same-account tx. This avoids relying on CLI `block` support differences
    while still serializing the fragile drift_pool bootstrap path.
    """
    drift_broadcast = "sync"
    addr_q = sh([
        "chaind", "keys", "show", drift_pool_key, "-a",
        "--keyring-backend", keyring,
        "--home", str(home),
    ], env=env, capture=True, check=False)
    drift_pool_addr = addr_q.stdout.strip() if addr_q.returncode == 0 else ""

    # Fail fast with an actionable message rather than emitting 12 identical
    # rc=1 warnings: if the key is absent from the keyring the restored
    # checkpoint did not include the drift_pool account (typically a stale or
    # foreign checkpoint reused across scenarios/seeds — see the checkpoint
    # signature guard in main()).
    if not drift_pool_addr:
        raise RuntimeError(
            f"drift bootstrap: key '{drift_pool_key}' not found in keyring at "
            f"{home}; the restored checkpoint has no drift_pool account. Force a "
            f"rebuild (time.rebuild_checkpoint: true, or delete "
            f"/tmp/poc_epoch_checkpoint_k*) and rerun."
        )

    # Pin the signer metadata once. Letting the CLI auto-resolve account number
    # and sequence on every tx is the source of the intermittent
    #   code 4: signature verification failed ... account number (0) ... unauthorized
    # failures seen under the rapid 12-in-a-row bootstrap: a flaky auth query
    # mid-burst returns account_number=0 (or a stale sequence), which makes a
    # subset of signatures permanently invalid. Instead we fetch the account
    # number + starting sequence once via wait_account_exists, sign every tx with
    # explicit --account-number/--sequence (--sign-mode direct), advance the
    # sequence locally only after a tx is confirmed in a block, and re-sync the
    # sequence from chain before retrying a rejected tx. This is the same robust
    # pattern used by the sybil create-validator path.
    node_rpc = ""
    if "--node" in node_args:
        node_rpc = node_args[node_args.index("--node") + 1]
    acct_json = wait_account_exists(env, home, node_rpc, drift_pool_addr, timeout_s=30.0)
    anum, seq = extract_account_meta(acct_json)

    ok = 0
    first_failure_dumped = False
    for op, per_validator_amount in zip(validator_ops, amounts):
        if per_validator_amount <= 0:
            continue
        success = False
        for attempt in range(4):
            prev_h = _rpc_latest_height(node_args)
            # rebuild cmd each attempt so the (possibly re-synced) sequence is used
            cmd = [
                "chaind", "tx", "staking", "delegate", op,
                f"{per_validator_amount}{denom}",
                "--from", drift_pool_key,
                "--keyring-backend", keyring,
                "--chain-id", chain_id,
                "--account-number", str(anum),
                "--sequence", str(seq),
                "--sign-mode", "direct",
                "--fees", fees, "--gas", gas,
                "--broadcast-mode", drift_broadcast,
                "-y", "-o", "json",
                "--home", str(home),
            ] + node_args
            r = sh(cmd, env=env, capture=True, check=False)
            j = {}
            if r.returncode == 0 and (r.stdout or "").strip():
                try:
                    j = json.loads(r.stdout)
                except Exception:
                    j = {}
            code = int(j.get("code", 0)) if j else -1
            if r.returncode == 0 and code == 0:
                txh = str(j.get("txhash", "") or "")
                txq = wait_tx_inclusion(env, home, node_args, txh, timeout_s=20.0) if txh else None
                if txq and int(txq.get("code", 0)) == 0:
                    seq += 1
                    ok += 1
                    try:
                        wait_next_block(node_args, prev_h, timeout_s=20.0)
                    except Exception:
                        pass
                    success = True
                    break
                # Broadcast accepted but the tx did not land cleanly: the
                # authoritative on-chain sequence tells us whether it actually
                # committed. Re-sync and retry.
            # Failure / ambiguous path: re-sync sequence from chain (the tx was
            # rejected at CheckTx so the sequence was not consumed, or it
            # committed and the live value already reflects it).
            live = get_sequence(env, home, node_args, drift_pool_addr)
            if live >= 0:
                seq = live
            if not first_failure_dumped:
                first_failure_dumped = True
                print("[drift] --- first-failure detail ---", file=sys.stderr)
                print(f"[drift] cmd: {' '.join(cmd)}", file=sys.stderr)
                print(f"[drift] full stdout: {r.stdout!r}", file=sys.stderr)
                print(f"[drift] full stderr: {r.stderr!r}", file=sys.stderr)
                print(f"[drift] resynced sequence -> {seq} (anum={anum})", file=sys.stderr)
                print("[drift] --- end first-failure detail ---", file=sys.stderr)
            time.sleep(0.6)
        if not success:
            print(f"[drift] WARN delegate {op} amount={per_validator_amount} "
                  f"failed after retries (anum={anum} seq={seq})", file=sys.stderr)
    return ok


def _query_pool_delegation(env, home: Path, node_args: List[str],
                           delegator_addr: str, validator_op: str) -> int:
    """Return the delegator's current delegated balance (in base denom) to the
    given validator, or 0 if none / query failure. Used to clamp redelegation
    to the donor's actually available delegation (Fix 2)."""
    if not delegator_addr or not validator_op:
        return 0
    r = sh([
        "chaind", "query", "staking", "delegation", delegator_addr, validator_op,
        "-o", "json", "--home", str(home),
    ] + node_args, env=env, capture=True, check=False)
    if r.returncode != 0:
        return 0
    try:
        j = json.loads(r.stdout)
        # SDK v0.53: {"delegation_response": {"balance": {"amount": "...", "denom": "..."}}}
        bal = (j.get("delegation_response", {}) or {}).get("balance", {}) or {}
        return int(bal.get("amount", "0") or 0)
    except Exception:
        return 0


def _drift_migrate_step(*, env, home: Path, node_args: List[str],
                        drift_pool_key: str, donor_op: str, receiver_op: str,
                        amount: int, denom: str, chain_id: str, keyring: str,
                        fees: str, gas: str, broadcast: str,
                        drift_pool_addr: str = "") -> bool:
    """Redelegate `amount` from donor to receiver via drift_pool's delegations.

    Fix 2: the requested amount is clamped to the drift_pool's actual delegation
    to the donor. Requesting more shares than the pool holds at the donor fails
    the tx (insufficient delegation shares), which would silently break the
    drift trajectory on later epochs once a donor is drained.

    As with bootstrap, redelegations use `sync` plus explicit confirmation
    barriers instead of relying on CLI `block` mode.
    """
    drift_broadcast = "sync"
    if donor_op == receiver_op or amount <= 0:
        return False
    if not drift_pool_addr:
        addr_q = sh([
            "chaind", "keys", "show", drift_pool_key, "-a",
            "--keyring-backend", keyring,
            "--home", str(home),
        ], env=env, capture=True, check=False)
        drift_pool_addr = addr_q.stdout.strip() if addr_q.returncode == 0 else ""

    # Clamp to available donor delegation, leaving a 1-token floor so the
    # delegation entry is not fully removed (keeps the donor a valid future
    # source and avoids edge cases in the staking module).
    available = _query_pool_delegation(env, home, node_args, drift_pool_addr, donor_op)
    if available <= 1:
        print(f"[drift] donor {donor_op[:20]}... exhausted (available={available}); skip migrate", file=sys.stderr)
        return False
    if amount > available - 1:
        amount = available - 1
    if amount <= 0:
        return False

    # Pin signer metadata and retry on signature/sequence rejection, mirroring
    # the bootstrap path: per-tx auto-resolution intermittently signs with
    # account_number=0 and fails CheckTx (code 4, unauthorized).
    node_rpc = ""
    if "--node" in node_args:
        node_rpc = node_args[node_args.index("--node") + 1]
    try:
        acct_json = wait_account_exists(env, home, node_rpc, drift_pool_addr, timeout_s=30.0)
        anum, seq = extract_account_meta(acct_json)
    except Exception:
        anum, seq = 0, max(get_sequence(env, home, node_args, drift_pool_addr), 0)

    last = None
    for attempt in range(4):
        prev_h = _rpc_latest_height(node_args)
        r = sh([
            "chaind", "tx", "staking", "redelegate", donor_op, receiver_op,
            f"{amount}{denom}",
            "--from", drift_pool_key,
            "--keyring-backend", keyring,
            "--chain-id", chain_id,
            "--account-number", str(anum),
            "--sequence", str(seq),
            "--sign-mode", "direct",
            "--fees", fees, "--gas", gas,
            "--broadcast-mode", drift_broadcast,
            "-y", "-o", "json",
            "--home", str(home),
        ] + node_args, env=env, capture=True, check=False)
        last = r
        j = {}
        if r.returncode == 0 and (r.stdout or "").strip():
            try:
                j = json.loads(r.stdout)
            except Exception:
                j = {}
        if r.returncode == 0 and j and int(j.get("code", 0)) == 0:
            txh = str(j.get("txhash", "") or "")
            txq = wait_tx_inclusion(env, home, node_args, txh, timeout_s=20.0) if txh else None
            if txq and int(txq.get("code", 0)) == 0:
                try:
                    wait_next_block(node_args, prev_h, timeout_s=20.0)
                except Exception:
                    pass
                return True
        # rejected / ambiguous: re-sync sequence from chain and retry
        live = get_sequence(env, home, node_args, drift_pool_addr)
        if live >= 0:
            seq = live
        time.sleep(0.6)
    print(f"[drift] WARN redelegate {donor_op}->{receiver_op} amount={amount} "
          f"failed after retries rc={last.returncode if last else '?'} "
          f"out={((last.stdout if last else '') or '')[:300]} "
          f"err={((last.stderr if last else '') or '')[:200]}",
          file=sys.stderr)
    return False


def _drift_pick_donor_receiver(validator_powers: List[Tuple[str, int]],
                               donor_top_h: int, receiver_top_r: int,
                               epoch_idx: int, scenario_seed: int) -> Tuple[Optional[str], Optional[str]]:
    """Return one (donor_op, receiver_op) pair for this epoch.

    Donor: cycles through validators ranked outside top-`donor_top_h` (in stake
    descending order) so all donor candidates get drained over time.
    Receiver: cycles through top-`receiver_top_r` validators.

    The cycling indices are deterministic per epoch + scenario_seed, which
    keeps the scenario reproducible.
    """
    if not validator_powers or donor_top_h < 0 or receiver_top_r <= 0:
        return None, None
    donors = validator_powers[donor_top_h:]
    receivers = validator_powers[:max(1, receiver_top_r)]
    if not donors or not receivers:
        return None, None
    donor_idx = (epoch_idx + scenario_seed) % len(donors)
    receiver_idx = (epoch_idx + scenario_seed) % len(receivers)
    return donors[donor_idx][0], receivers[receiver_idx][0]


def parse_last_draw_payload(payload: str) -> Tuple[str, Dict[str, str]]:
    """Parse last-draw payload stored by module.

    Format:
      members_csv
      or
      members_csv|a_stake=<ppm>|a_age=<ppm>|a_weight=<ppm>|a_vals=<n>
    """
    if not payload:
        return "", {}
    parts = payload.split("|")
    members_csv = parts[0].strip()
    meta: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip(); v = v.strip()
        if k == "a_stake":
            meta["attacker_stake_ppm"] = v
        elif k == "a_age":
            meta["attacker_age_ppm"] = v
        elif k == "a_weight":
            meta["attacker_weight_ppm"] = v
        elif k == "a_vals":
            meta["attacker_validators"] = v
        elif k == "v_metrics":
            meta["validator_metrics"] = v
        elif k == "l_auto":
            meta["lambda_auto_ppm"] = v
        elif k == "l_manual":
            meta["lambda_manual_ppm"] = v
        elif k == "gini":
            meta["gini_ppm"] = v
        elif k == "fresh":
            meta["fresh_pressure_ppm"] = v
        elif k == "l_signal":
            meta["lambda_signal_target_ppm"] = v
        elif k == "l_risk":
            meta["lambda_risk_target_ppm"] = v
        elif k == "l_target":
            meta["lambda_target_ppm"] = v
        elif k == "risk_alpha":
            meta["risk_alpha_ppm"] = v
        elif k == "risk_beta":
            meta["risk_beta_ppm"] = v
        elif k == "risk_sat":
            meta["risk_budget_satisfied"] = v
        elif k == "risk_ck":
            meta["risk_coalition_size"] = v
        elif k == "risk_b0_log10e6":
            meta["risk_bound0_log10e6"] = v
        elif k == "risk_ba_log10e6":
            meta["risk_bound_auto_log10e6"] = v
        elif k == "rb_mode":
            meta["risk_controller_mode"] = v
    return members_csv, meta


def decode_event_attrs(attrs: List[Dict[str, str]]) -> Dict[str, str]:
    # Cosmos sometimes returns event attr keys/values base64-encoded (depending on query path).
    # Try to decode b64 -> utf-8 when it looks plausible.
    def maybe_b64(s: str) -> str:
        if not isinstance(s, str) or not s:
            return ""
        # Heuristic: only b64 charset and length multiple of 4.
        import re, base64
        if len(s) % 4 != 0:
            return s
        if not re.fullmatch(r"[A-Za-z0-9+/=]+", s):
            return s
        try:
            raw = base64.b64decode(s, validate=True)
            txt = raw.decode("utf-8")
            # must be mostly printable
            if sum(32 <= ord(c) < 127 for c in txt) / max(1, len(txt)) < 0.9:
                return s
            return txt
        except Exception:
            return s

    out: Dict[str, str] = {}
    for a in attrs or []:
        k = maybe_b64(a.get("key", ""))
        v = maybe_b64(a.get("value", ""))
        out[k] = v
    return out


def _stable_seed_from_text(text: str) -> int:
    raw = hashlib.sha256((text or "").encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big", signed=False)


def _estimate_ppswor_coalition_baseline(stakes: List[int], coalition_flags: List[bool], committee_size: int,
                                        *, trials: int = 2000, seed_text: str = "") -> Tuple[float, float, float]:
    """Monte-Carlo stake-only committee baseline under PPS without replacement.

    Returns:
      (mean coalition seat share, P(seats >= ceil(m/3)), P(seats >= ceil(m/2)))

    This is used for concentration-drift runs where comparing committee seat share
    directly against raw stake share is misleading: the monitored coalition is a
    fixed set of validator identities, while the actual committee sampler draws
    distinct validators without replacement.
    """
    weights = [max(0, int(s or 0)) for s in stakes]
    if committee_size <= 0 or not weights or len(weights) != len(coalition_flags):
        return 0.0, 0.0, 0.0
    positive = [(i, w) for i, w in enumerate(weights) if w > 0]
    if not positive:
        return 0.0, 0.0, 0.0

    m = min(int(committee_size), len(positive))
    q13 = max(1, int(math.ceil(m / 3.0)))
    q12 = max(1, int(math.ceil(m / 2.0)))
    rng = random.Random(_stable_seed_from_text(seed_text))

    seat_total = 0.0
    cap13 = 0
    cap12 = 0
    for _ in range(max(1, int(trials))):
        keyed = []
        for idx, w in positive:
            # Efraimidis-Spirakis PPSWOR key: sample smallest -log(U)/w keys.
            u = max(rng.random(), 1e-12)
            keyed.append(((-math.log(u)) / float(w), idx))
        keyed.sort(key=lambda t: t[0])
        chosen = [idx for _, idx in keyed[:m]]
        seats = sum(1 for idx in chosen if coalition_flags[idx])
        seat_total += seats / float(m)
        if seats >= q13:
            cap13 += 1
        if seats >= q12:
            cap12 += 1

    denom = float(max(1, int(trials)))
    return seat_total / denom, (100.0 * cap13) / denom, (100.0 * cap12) / denom


_printed_committee_attrs_once = False

def _node_rpc_url(node_args: List[str]) -> str:
    node = "http://127.0.0.1:26657"
    for i, a in enumerate(node_args):
        if a == "--node" and i + 1 < len(node_args):
            node = node_args[i + 1]
            break
    if node.startswith("tcp://"):
        node = "http://" + node[len("tcp://"):]
    return node.rstrip("/")


def _rpc_json_get(url: str, timeout_s: float = 3.0) -> Dict[str, object]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _rpc_latest_height(node_args: List[str]) -> int:
    try:
        j = _rpc_json_get(f"{_node_rpc_url(node_args)}/status", timeout_s=2.5)
        res = j.get("result", {}) if isinstance(j, dict) else {}
        si = res.get("sync_info", {}) if isinstance(res, dict) else {}
        return int(si.get("latest_block_height", 0) or 0)
    except Exception:
        return 0


def wait_tx_inclusion(env, home0: Path, node_args: List[str], txh: str, *, timeout_s: float = 15.0) -> Dict[str, object]:
    """Poll Comet RPC `/tx` until commit (sync broadcast returns before DeliverTx/commit)."""
    del env, home0
    deadline = time.time() + timeout_s
    rpc = _node_rpc_url(node_args)
    last_h = _rpc_latest_height(node_args)
    last_h_change_at = time.time()

    while time.time() < deadline:
        try:
            q = urllib.parse.urlencode({"hash": f"0x{txh}", "prove": "false"})
            j = _rpc_json_get(f"{rpc}/tx?{q}", timeout_s=2.5)
            if isinstance(j, dict):
                res = j.get("result", {})
                if isinstance(res, dict) and res.get("tx_result") is not None:
                    txr = res.get("tx_result", {}) or {}
                    return {
                        "height": str(res.get("height", "0")),
                        "code": int(txr.get("code", 0) or 0),
                        "codespace": str(txr.get("codespace", "") or ""),
                        "log": str(txr.get("log", "") or ""),
                        "info": str(txr.get("info", "") or ""),
                    }
        except Exception:
            pass

        h = _rpc_latest_height(node_args)
        if h > last_h:
            last_h = h
            last_h_change_at = time.time()
        elif time.time() - last_h_change_at > 8.0:
            print(f"[warn] no new blocks while waiting tx commit (height={h}, tx={txh[:12]}...)")
            last_h_change_at = time.time()

        time.sleep(0.30)

    return {}


def wait_next_block(node_args: List[str], prev_h: int, timeout_s: float = 20.0) -> int:
    """Wait until RPC height exceeds prev_h."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        h = _rpc_latest_height(node_args)
        if h > prev_h:
            return h
        time.sleep(0.30)
    raise RuntimeError(f"no new block after height={prev_h}")


def query_committee_attrs(env, homes0: Path, node_args: List[str], txh: str) -> Dict[str, str]:
    global _printed_committee_attrs_once
    # Keep this short to avoid long stalls when tx indexer lags.
    for _ in range(8):
        qr = sh([
            "chaind", "query", "tx", txh,
            "-o", "json",
            "--home", str(homes0),
        ] + node_args, env=env, capture=True, check=False)
        if qr.returncode == 0 and qr.stdout.strip():
            try:
                qj = json.loads(qr.stdout)
            except Exception:
                time.sleep(0.3)
                continue

            def is_committee_drawn(t: str) -> bool:
                return t == "committee_drawn" or t.endswith(".committee_drawn") or t.endswith("/committee_drawn")

            events = qj.get("events", [])
            for ev in events:
                if is_committee_drawn(ev.get("type", "")):
                    d = decode_event_attrs(ev.get("attributes", []))
                    if not _printed_committee_attrs_once:
                        _printed_committee_attrs_once = True
                        print(f"[debug] committee_drawn attrs (sample): {d}")
                    return d

            for lg in qj.get("logs", []):
                for ev in lg.get("events", []):
                    if is_committee_drawn(ev.get("type", "")):
                        d = decode_event_attrs(ev.get("attributes", []))
                        if not _printed_committee_attrs_once:
                            _printed_committee_attrs_once = True
                            print(f"[debug] committee_drawn attrs (sample): {d}")
                        return d
        time.sleep(0.3)
    return {}



def query_tx_by_hash(env, home0: Path, node_args: List[str], txh: str):
    """Tx query helper for diagnostics via Comet RPC (CLI query tx is inconsistent across builds)."""
    del env, home0
    try:
        q = urllib.parse.urlencode({"hash": f"0x{txh}", "prove": "false"})
        j = _rpc_json_get(f"{_node_rpc_url(node_args)}/tx?{q}", timeout_s=3.0)
        return subprocess.CompletedProcess(args=["rpc-tx"], returncode=0, stdout=json.dumps(j), stderr="")
    except Exception as e:
        return subprocess.CompletedProcess(args=["rpc-tx"], returncode=1, stdout="", stderr=str(e))

def resolve_cfg_path(raw_path: str, poc_root: Path) -> Path:
    p = Path(raw_path.strip()).expanduser()
    if p.is_absolute():
        return p

    repo_root = poc_root.parent
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / p,
        script_dir / p,
        poc_root / p,
        poc_root / "cosmos" / p,
        repo_root / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main() -> int:
    poc_root = Path(__file__).resolve().parents[2]
    repo = poc_root
    cfg_path = poc_root / "cosmos" / "poc_config.yaml"
    if len(sys.argv) > 1 and sys.argv[1].strip():
        cfg_path = resolve_cfg_path(sys.argv[1], poc_root)
    cfg = load_yaml_minimal(cfg_path)

    chain_id = cfg["chain"]["chain_id"]
    denom = cfg["chain"]["denom"]

    honest_nodes = int(cfg.get("experiment", {}).get("honest_nodes", 8))
    beta = float(cfg.get("attack", {}).get("beta", 0.33))
    sybil_k_values = [int(x) for x in cfg.get("attack", {}).get("sybil_k_values", [8])]
    attacker_profile = str(cfg.get("attack", {}).get("attacker_profile", "burst")).strip().lower()
    entry_kind = str(cfg.get("attack", {}).get("entry_kind", "sybil")).strip().lower()
    attack_mode = str(cfg.get("attack", {}).get("mode", "replacement")).strip().lower()
    if attack_mode not in ("replacement", "additive"):
        raise RuntimeError(f"attack.mode must be replacement|additive, got: {attack_mode}")
    whale_share = float(cfg.get("attack", {}).get("whale_share", 0.5 if attack_mode == "additive" else 0.0))
    if attack_mode == "additive" and not (0.0 < whale_share < 1.0):
        raise RuntimeError(f"attack.whale_share must be in (0,1) for additive mode, got: {whale_share}")
    is_additive = (attack_mode == "additive")
    tracked_meta = tracked_entity_context(entry_kind)
    tracked_prefix = tracked_meta["tracked_prefix"]
    tracked_entity_mode = tracked_meta["tracked_entity_mode"]
    tracked_entity_label = tracked_meta["tracked_entity_label"]
    tracked_entity_label_title = tracked_meta["tracked_entity_label_title"]
    tracked_baseline_label = tracked_meta["tracked_baseline_label"]
    baseline_comparison_mode = tracked_meta["baseline_comparison_mode"]
    tracked_vs_baseline_axis_label = "reduction vs baseline (%)" if baseline_comparison_mode == "reduction" else "gap vs baseline (%)"

    p2p_base = int(cfg["localnet"]["p2p_port_base"])
    rpc_base = int(cfg["localnet"]["rpc_port_base"])
    api_base = int(cfg["localnet"]["api_port_base"])
    grpc_base = int(cfg["localnet"]["grpc_port_base"])

    committee_mode = str(cfg["workload"].get("committee_mode", "fixed")).strip().lower()
    if "committee_size_values" in cfg["workload"]:
        committee_size_values = [int(x) for x in cfg["workload"]["committee_size_values"]]
    else:
        committee_size_values = [int(cfg["workload"].get("committee_size", 6))]
    lambda_vals = [int(x) for x in cfg["workload"]["lambda_ppm_values"]]
    topk_vals = [int(x) for x in cfg.get("coalition", {}).get("topk_values", [1, 3, 5])]

    time_cfg = cfg.get("epoch", {})
    epoch_blocks = int(time_cfg.get("epoch_blocks", 50))
    pre_attack_epochs = int(time_cfg.get("pre_attack_epochs", 10))
    post_attack_epochs = int(time_cfg.get("post_attack_epochs", 20))
    draws_per_epoch = int(time_cfg.get("draws_per_epoch", 10))
    post_attack_draw_limit = int(time_cfg.get("post_attack_draw_limit", 0))
    reuse_checkpoint = bool(time_cfg.get("reuse_checkpoint", True))
    rebuild_checkpoint = bool(time_cfg.get("rebuild_checkpoint", False))
    sybil_at_genesis = bool(time_cfg.get("sybil_at_genesis", False))
    if is_additive and sybil_at_genesis:
        print("[warn] attack.mode=additive forces sybil_at_genesis=False (whale stays at genesis, fresh cohort injects post-attack)")
        sybil_at_genesis = False

    from_acct_base = str(cfg["tx"]["from"])
    payer_key = f"{from_acct_base}0"
    keyring = str(cfg["tx"]["keyring_backend"])
    fees = str(cfg["tx"]["fees"])
    tx_gas = str(cfg.get("tx", {}).get("gas", "2000000"))
    skip_set_lambda = bool(cfg.get("tx", {}).get("skip_set_lambda", False))
    policy_mode = str(cfg.get("tx", {}).get("policy_mode", "adaptive")).strip().lower()
    if policy_mode not in ("adaptive", "manual"):
        raise RuntimeError(f"tx.policy_mode must be adaptive|manual, got: {policy_mode}")
    if policy_mode == "manual" and skip_set_lambda:
        print("[warn] tx.policy_mode=manual with skip_set_lambda=true is inconsistent; forcing skip_set_lambda=false")
        skip_set_lambda = False
    broadcast_mode = str(cfg["tx"]["broadcast_mode"])
    # Reporting knobs.
    report_cfg = cfg.get("report", {}) if isinstance(cfg.get("report", {}), dict) else {}
    final_table_single_row = bool(report_cfg.get("final_table_single_row", False))
    static_lambda_ppm = int(report_cfg.get("static_lambda_ppm", 300000) or 300000)
    if static_lambda_ppm < 0:
        static_lambda_ppm = 0
    if static_lambda_ppm > 1_000_000:
        static_lambda_ppm = 1_000_000
    # allow: sync|async|block (do not rewrite)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifacts_root = repo / "cosmos" / "artifacts"
    artifacts_subdir = str(cfg.get("experiment", {}).get("artifacts_subdir", "")).strip().strip("/")
    preserve_run_history = bool(report_cfg.get("preserve_run_history", bool(artifacts_subdir)))
    if artifacts_subdir:
        art_base_dir = artifacts_root / artifacts_subdir
    else:
        art_base_dir = artifacts_root
    use_run_history_dir = bool(artifacts_subdir and preserve_run_history)
    if use_run_history_dir:
        art_dir = art_base_dir / "runs" / run_id
        art_latest_dir = art_base_dir
    else:
        art_dir = art_base_dir
        art_latest_dir = art_dir
    (art_dir / "results").mkdir(parents=True, exist_ok=True)
    (art_dir / "plots").mkdir(parents=True, exist_ok=True)
    (art_latest_dir / "results").mkdir(parents=True, exist_ok=True)
    (art_latest_dir / "plots").mkdir(parents=True, exist_ok=True)

    manifest_json = art_dir / "results" / f"run_manifest_{run_id}.json"
    draws_csv = art_dir / "results" / f"epoch_draws_{run_id}.csv"
    draws_debug_csv = art_dir / "results" / f"epoch_draws_debug_{run_id}.csv"
    summary_csv = art_dir / "results" / f"epoch_summary_{run_id}.csv"
    compare_csv = art_dir / "results" / f"epoch_lambda_comparison_{run_id}.csv"
    draw_summary_csv = art_dir / "results" / f"early_draws_summary_{run_id}.csv"
    validator_metrics_csv = art_dir / "results" / f"validator_metrics_{run_id}.csv"
    final_table_csv = art_dir / "results" / f"epoch_final_table_{run_id}.csv"
    final_table_single_csv = art_dir / "results" / f"epoch_final_table_single_row_{run_id}.csv"
    final_policy_csv = art_dir / "results" / f"epoch_final_policy_table_{run_id}.csv"
    final_epoch_csv = art_dir / "results" / f"epoch_final_epoch_table_{run_id}.csv"

    # Primary draw-level CSV (compact): keep only informative per-tx fields.
    draws_core_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "phase", "epoch_idx", "draw_i", "draw_idx_global", "draw_idx_post_attack", "height", "attack_height",
        "k", "committee_size", "lambda_ppm", "lambda_auto_ppm", "lambda_prev_auto_ppm", "lambda_auto_delta_ppm", "policy_mode",
        "gini_ppm", "fresh_pressure_ppm",
        "attacker_seats", "attacker_seats_share", "attacker_weight_ppm", "stake_share_indep", "seat_minus_weight", "seat_minus_stake_indep",
        "tracked_seats", "tracked_seats_share", "tracked_weight_ppm", "tracked_stake_ppm", "tracked_minus_weight", "tracked_minus_stake_indep",
        "tag",
    ]

    draws_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "k", "committee_size", "lambda_ppm", "lambda_auto_ppm", "lambda_prev_auto_ppm", "lambda_auto_delta_ppm", "lambda_manual_ppm", "policy_mode", "gini_ppm", "fresh_pressure_ppm", "phase", "epoch_idx", "draw_i", "draw_idx_global", "draw_idx_post_attack", "height", "attack_height", "tag",
        # stale/non-updating detection
        "tag_preexists", "tag_preexists_members_len",
        # tx inclusion details
        # validator-set context
        "vset_n",
        # seat metrics
        "attacker_seats", "attacker_seats_share",
        "tracked_seats", "tracked_seats_share",
        # module-emitted diagnostics (ppm)
        "attacker_stake_ppm", "attacker_age_ppm", "attacker_weight_ppm", "attacker_validators",
        "tracked_stake_ppm", "tracked_age_ppm", "tracked_weight_ppm", "tracked_validators",
        # runner-side diagnostics for debugging seat-share mismatches
        "members_len", "unique_members_len", "has_duplicate_members",
        "attacker_unique_members_len",
        "tracked_unique_members_len",
        "members_prefix",
        "attacker_ops_len",
        "tracked_ops_len",
        "attacker_stakes_csv",
        "tracked_stakes_csv",
        "stake_share_indep", "attacker_tokens", "total_tokens",
        "tracked_tokens",
        "seat_minus_weight", "seat_minus_stake_indep",
        "tracked_minus_weight", "tracked_minus_stake_indep",
        # RBHC paper diagnostics (parsed from chain payload; absent in pre-RBHC builds).
        "lambda_signal_target_ppm", "lambda_risk_target_ppm", "lambda_target_ppm",
        "risk_alpha_ppm", "risk_beta_ppm", "risk_budget_satisfied", "risk_coalition_size",
        "risk_bound0_log10e6", "risk_bound_auto_log10e6", "risk_controller_mode",
    ] + [f"top{int(k)}_seats" for k in sorted(set(int(x) for x in topk_vals))]

    summary_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "k", "committee_size", "lambda_ppm", "epoch_idx", "height",
        "draws", "mean_attacker_share", "min_attacker_share", "max_attacker_share",
        "mean_attacker_weight_share", "mean_attacker_age_share", "mean_attacker_stake_share",
        "mean_attacker_weight_ppm", "mean_attacker_age_ppm", "mean_attacker_stake_ppm",
        "mean_tracked_share", "min_tracked_share", "max_tracked_share",
        "mean_tracked_weight_share", "mean_tracked_age_share", "mean_tracked_stake_share",
        "mean_tracked_weight_ppm", "mean_tracked_age_ppm", "mean_tracked_stake_ppm",
        "mean_lambda_auto_ppm", "mean_gini_ppm", "mean_fresh_pressure_ppm",
        # runner-side aggregates
        "mean_stake_share_indep",
        "mean_seat_minus_weight", "mean_seat_minus_stake_indep",
        "mean_tracked_minus_weight", "mean_tracked_minus_stake_indep",
        "members_prefix_mode",
        # RBHC paper diagnostics (mean across draws within epoch).
        "mean_lambda_signal_target_ppm", "mean_lambda_risk_target_ppm", "mean_lambda_target_ppm",
        "mean_risk_alpha_ppm", "mean_risk_beta_ppm",
        "mean_risk_bound0_log10e6", "mean_risk_bound_auto_log10e6",
        "risk_budget_satisfied_frac",
        "risk_controller_mode",
    ]

    manifest = {
        "run_id": run_id,
        "config_path": str(cfg_path),
        "artifacts_dir": str(art_dir),
        "artifacts_base_dir": str(art_base_dir),
        "artifacts_latest_dir": str(art_latest_dir),
        "preserve_run_history": use_run_history_dir,
        "chain_id": chain_id,
        "denom": denom,
        "honest_nodes": honest_nodes,
        "sybil_k_values": sybil_k_values,
        "committee_mode": committee_mode,
        "committee_size_values": committee_size_values,
        "lambda_ppm_values": lambda_vals,
        "epoch_blocks": epoch_blocks,
        "pre_attack_epochs": pre_attack_epochs,
        "post_attack_epochs": post_attack_epochs,
        "draws_per_epoch": draws_per_epoch,
        "post_attack_draw_limit": post_attack_draw_limit,
        "beta": beta,
        "attacker_profile": attacker_profile,
        "entry_kind": entry_kind,
        "tracked_prefix": tracked_prefix,
        "tracked_entity_mode": tracked_entity_mode,
        "tracked_entity_label": tracked_entity_label,
        "tracked_baseline_label": tracked_baseline_label,
        "baseline_comparison_mode": baseline_comparison_mode,
        "tx_gas": tx_gas,
        "broadcast_mode": broadcast_mode,
        "policy_mode": policy_mode,
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validator_metric_cols = [
        "k", "committee_size", "lambda_ppm", "epoch_idx", "draw_i", "draw_idx_post_attack", "tag",
        "validator_address", "validator_stake", "validator_age_score", "validator_weight_score", "is_attacker",
    ]

    # Controller mode + risk-budget knobs (RBHC paper PoC).
    # YAML schema:
    #   controller:
    #     mode: signal | risk | hybrid          # default: signal (backwards-compatible)
    #   risk_budget:
    #     enabled: true|false
    #     epsilon: 1.0e-4
    #     theta_ppm: 333333
    #     coalition_share_ppm: 333333
    #     grid_step_ppm: 10000
    # These map to ADAPTIVE_CONTROLLER_MODE and RISK_BUDGET_* env vars consumed
    # by the chain's adaptivecommittee module (see keeper/risk_budget.go).
    controller_cfg = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
    controller_mode = str(controller_cfg.get("mode", "signal")).strip().lower()
    if controller_mode not in ("signal", "risk", "hybrid"):
        raise RuntimeError(f"controller.mode must be signal|risk|hybrid, got: {controller_mode}")
    # Dual-path baseline selection (RBHC paper). The active baseline drives both
    # the mixed-weight rule and the risk certificate's beta term.
    #   service_age  -> persistence path (anti-Sybil; default, Article 2 lineage)
    #   capped_stake -> concentration path (anti-whale; for drift scenarios)
    #   uniform      -> Article 1 degenerate baseline
    baseline_mode = str(controller_cfg.get("baseline_mode", "service_age")).strip().lower()
    if baseline_mode not in ("service_age", "capped_stake", "concave_stake", "uniform"):
        raise RuntimeError(f"controller.baseline_mode must be service_age|capped_stake|concave_stake|uniform, got: {baseline_mode}")
    concentration_cap_ppm = int(controller_cfg.get("concentration_cap_ppm", 125000))
    risk_cfg = cfg.get("risk_budget", {}) if isinstance(cfg.get("risk_budget", {}), dict) else {}
    risk_enabled = bool(risk_cfg.get("enabled", controller_mode in ("risk", "hybrid")))
    risk_epsilon = float(risk_cfg.get("epsilon", 1.0e-4))
    risk_theta_ppm = int(risk_cfg.get("theta_ppm", 333333))
    risk_coalition_mode = str(risk_cfg.get("coalition_mode", "min_share")).strip().lower()
    if risk_coalition_mode not in ("min_share", "top_k"):
        raise RuntimeError(f"risk_budget.coalition_mode must be min_share|top_k, got: {risk_coalition_mode}")
    risk_coalition_ppm = int(risk_cfg.get("coalition_share_ppm", 333333))
    risk_top_k = int(risk_cfg.get("coalition_top_k", 3))
    risk_grid_step_ppm = int(risk_cfg.get("grid_step_ppm", 10000))

    # Scenario kind (RBHC paper PoC, M3.2 + M3.3).
    # YAML schema:
    #   scenario:
    #     kind: late_entry_sybil | concentration_drift     # default: late_entry_sybil
    #   concentration_drift:
    #     migration_rate_ppm: 20000              # 2% of total bonded per epoch
    #     start_epoch: 1                         # absolute epoch index (1-based)
    #     end_epoch: 30                          # inclusive
    #     donor_top_h: 3                         # donors are validators ranked > top_h
    #     receiver_top_r: 3                      # receivers are top_r validators
    #     scenario_seed: 1                       # for deterministic donor/receiver cycling
    #     drift_pool_funding: 2000000000         # in denom units (no denom suffix here)
    scenario_cfg = cfg.get("scenario", {}) if isinstance(cfg.get("scenario", {}), dict) else {}
    scenario_kind = str(scenario_cfg.get("kind", "late_entry_sybil")).strip().lower()
    if scenario_kind not in ("late_entry_sybil", "concentration_drift"):
        raise RuntimeError(f"scenario.kind must be late_entry_sybil|concentration_drift, got: {scenario_kind}")
    drift_cfg = cfg.get("concentration_drift", {}) if isinstance(cfg.get("concentration_drift", {}), dict) else {}
    drift_active = (scenario_kind == "concentration_drift")
    if drift_active:
        # The tracked entity in drift is the monitored top-k stake coalition, not
        # an injected attacker. Relabel the metadata so every emitted column,
        # plot title and summary reads honestly ("coalition", not "attacker").
        # The natural comparison here is the coalition's committee seat share
        # against its stake share, i.e. a gap rather than a Sybil "reduction".
        tracked_entity_mode = "coalition"
        tracked_entity_label = "top-k coalition"
        tracked_entity_label_title = "Top-k coalition"
        tracked_baseline_label = "stake-only committee baseline"
        baseline_comparison_mode = "gap"
        tracked_vs_baseline_axis_label = "gap vs baseline (%)"
        tracked_meta = dict(tracked_meta)
        tracked_meta.update({
            "tracked_entity_mode": tracked_entity_mode,
            "tracked_entity_label": tracked_entity_label,
            "tracked_entity_label_title": tracked_entity_label_title,
            "tracked_baseline_label": tracked_baseline_label,
            "baseline_comparison_mode": baseline_comparison_mode,
        })
        manifest["tracked_entity_mode"] = tracked_entity_mode
        manifest["tracked_entity_label"] = tracked_entity_label
        manifest["tracked_baseline_label"] = tracked_baseline_label
        manifest["baseline_comparison_mode"] = baseline_comparison_mode
    drift_migration_rate_ppm = int(drift_cfg.get("migration_rate_ppm", 20000))
    drift_start_epoch = int(drift_cfg.get("start_epoch", 1))
    drift_end_epoch = int(drift_cfg.get("end_epoch", 30))
    drift_donor_top_h = int(drift_cfg.get("donor_top_h", 3))
    drift_receiver_top_r = int(drift_cfg.get("receiver_top_r", 3))
    drift_scenario_seed = int(drift_cfg.get("scenario_seed", 1))
    drift_pool_funding = int(drift_cfg.get("drift_pool_funding", 2_000_000_000))
    drift_pool_key_name = "drift_pool" if drift_active else None
    drift_pool_balance = drift_pool_funding if drift_active else None

    # Extend manifest with RBHC controller configuration so the artifact
    # directory records exactly which controller mode and risk-budget knobs
    # produced the outputs.
    rbhc_manifest = {
        "controller_mode": controller_mode,
        "baseline_mode": baseline_mode,
        "concentration_cap_ppm": concentration_cap_ppm,
        "risk_budget": {
            "enabled": risk_enabled,
            "epsilon": risk_epsilon,
            "theta_ppm": risk_theta_ppm,
            "coalition_mode": risk_coalition_mode,
            "coalition_share_ppm": risk_coalition_ppm,
            "coalition_top_k": risk_top_k,
            "grid_step_ppm": risk_grid_step_ppm,
        },
        "scenario_kind": scenario_kind,
    }
    if drift_active:
        rbhc_manifest["concentration_drift"] = {
            "migration_rate_ppm": drift_migration_rate_ppm,
            "start_epoch": drift_start_epoch,
            "end_epoch": drift_end_epoch,
            "donor_top_h": drift_donor_top_h,
            "receiver_top_r": drift_receiver_top_r,
            "scenario_seed": drift_scenario_seed,
            "drift_pool_funding": drift_pool_funding,
        }
    try:
        existing = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    existing.update(rbhc_manifest)
    manifest_json.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{Path.home()}/go/bin:{Path.home()}/.local/go/bin:{Path.home()}/.local/bin:" + env.get("PATH", "")
    env["TRACKED_VALIDATOR_PREFIX"] = tracked_prefix
    env["ADAPTIVE_CONTROLLER_MODE"] = controller_mode
    env["RISK_BUDGET_ENABLED"] = "1" if risk_enabled else "0"
    env["RISK_BUDGET_EPS"] = f"{risk_epsilon:.6e}"
    env["RISK_BUDGET_THETA_PPM"] = str(risk_theta_ppm)
    env["RISK_BUDGET_COALITION_MODE"] = risk_coalition_mode
    env["RISK_BUDGET_COALITION_SHARE_PPM"] = str(risk_coalition_ppm)
    env["RISK_BUDGET_TOP_K"] = str(risk_top_k)
    env["RISK_BUDGET_GRID_STEP_PPM"] = str(risk_grid_step_ppm)
    env["BASELINE_MODE"] = baseline_mode
    env["CONCENTRATION_CAP_PPM"] = str(concentration_cap_ppm)
    # Prefer caller-provided POC_CHAIND; otherwise fall back to repo-built binaries.
    pinned53 = (repo / "cosmos" / "chain53" / "chain-five-three" / "build" / "chain-five-threed").resolve()
    pinned = (repo / "cosmos" / "chain" / "build" / "chaind").resolve()
    if not env.get("POC_CHAIND"):
        if pinned53.exists():
            env["POC_CHAIND"] = str(pinned53)
        elif pinned.exists():
            env["POC_CHAIND"] = str(pinned)

    try:
        ver = sh(["chaind", "version"], env=env, capture=True, check=False)
        print(f"[debug] using chaind={env.get('POC_CHAIND')} rc={ver.returncode} stdout={ver.stdout.strip()} stderr={ver.stderr.strip()}")
    except Exception:
        print("chaind not found; build chain first", file=sys.stderr)
        return 2

    # Kill any leftover PoC node processes (binary name may differ when POC_CHAIND is pinned).
    subprocess.run(["bash", "-lc", "pkill -f 'start --home /tmp/poc_epoch_' 2>/dev/null || true"], check=False)
    subprocess.run(["bash", "-lc", "pkill -f 'start --home /tmp/poc_sybil_' 2>/dev/null || true"], check=False)
    subprocess.run(["bash", "-lc", "pkill -f '/tmp/poc_epoch_' 2>/dev/null || true"], check=False)
    subprocess.run(["bash", "-lc", "pkill -f '/tmp/poc_sybil_' 2>/dev/null || true"], check=False)
    # Small grace period so the OS can release listeners.
    time.sleep(0.8)

    with draws_csv.open("w", newline="", encoding="utf-8") as fd, draws_debug_csv.open("w", newline="", encoding="utf-8") as fdd, summary_csv.open("w", newline="", encoding="utf-8") as fs, validator_metrics_csv.open("w", newline="", encoding="utf-8") as fvm:
        dw = csv.DictWriter(fd, fieldnames=draws_core_cols)
        dwd = csv.DictWriter(fdd, fieldnames=draws_cols)
        sw = csv.DictWriter(fs, fieldnames=summary_cols)
        vmw = csv.DictWriter(fvm, fieldnames=validator_metric_cols)
        dw.writeheader(); fd.flush()
        dwd.writeheader(); fdd.flush()
        sw.writeheader(); fs.flush()
        vmw.writeheader(); fvm.flush()
        validator_metric_rows: List[Dict[str, str]] = []

        for committee_size in committee_size_values:
            print(f"[sweep] committee_mode={committee_mode} committee_size={committee_size}")
            for k in sybil_k_values:
                # Build pre-attack checkpoint once per k and reuse for each lambda.
                ckpt_root = Path(f"/tmp/poc_epoch_checkpoint_k{k}")
                prep_root = Path(f"/tmp/poc_epoch_prep_k{k}")
                # Checkpoint at near-genesis; pre-attack epochs are now executed in the draw loop
                # so per-draw telemetry starts from the beginning of the run.
                target_pre = 1

                # Checkpoint validity signature. The cached checkpoint embeds a
                # complete genesis (chain-id, account set, bonded distribution,
                # and — critically — the optional drift_pool account). Reusing a
                # checkpoint whose genesis was built for a DIFFERENT scenario
                # silently restores the wrong chain: e.g. a burst checkpoint has
                # no drift_pool key/account, so every drift delegation fails with
                # rc=1 and empty output, or a previous seed's checkpoint carries a
                # different chain-id than the txs we sign. The cache key used to be
                # only `k`, which collided across scenarios and seeds. We now tag
                # the checkpoint with a signature of every genesis-affecting
                # parameter and rebuild on any mismatch.
                ckpt_sig = "|".join(str(x) for x in [
                    "v2", chain_id, denom, honest_nodes, k,
                    f"{beta:.6f}", attack_mode, f"{whale_share:.6f}",
                    sybil_at_genesis,
                    drift_pool_key_name or "-",
                    drift_pool_balance if drift_pool_balance is not None else "-",
                ])
                ckpt_sig_file = ckpt_root / ".ckpt_sig"

                should_build = True
                if reuse_checkpoint and ckpt_root.exists() and not rebuild_checkpoint:
                    cached_sig = ""
                    try:
                        cached_sig = ckpt_sig_file.read_text(encoding="utf-8").strip()
                    except OSError:
                        cached_sig = ""
                    if cached_sig == ckpt_sig:
                        print(f"[k={k}] reusing existing checkpoint: {ckpt_root}")
                        should_build = False
                    else:
                        print(f"[k={k}] stale/foreign checkpoint signature "
                              f"(cached={cached_sig!r} != current={ckpt_sig!r}); rebuilding")

                if should_build:
                    homes: List[Path] = []
                    procs: List[NodeProc] = []
                    try:
                        homes, procs = build_localnet(
                            env=env,
                            tmp_root=prep_root,
                            chain_id=chain_id,
                            denom=denom,
                            honest_nodes=honest_nodes,
                            sybil_k=k,
                            beta=beta,
                            p2p_base=p2p_base,
                            rpc_base=rpc_base,
                            api_base=api_base,
                            grpc_base=grpc_base,
                            from_acct_base=from_acct_base,
                            keyring=keyring,
                            sybil_active_at_genesis=sybil_at_genesis,
                            attack_mode=attack_mode,
                            whale_share=whale_share,
                            drift_pool_key=drift_pool_key_name,
                            drift_pool_balance=drift_pool_balance,
                        )

                        print(f"[k={k}] building pre-attack checkpoint to height {target_pre}...")
                        h_pre = wait_height(rpc_base + 0, target_pre, timeout_s=2500.0)
                        print(f"[k={k}] checkpoint ready at height {h_pre}")
                    finally:
                        stop_all(procs)

                    if ckpt_root.exists():
                        shutil.rmtree(ckpt_root)
                    shutil.copytree(prep_root, ckpt_root)
                    # Stamp the signature so a later run (different scenario or
                    # seed) does not silently reuse this genesis. Written last so a
                    # crash mid-build leaves an unsigned (hence rejected) dir.
                    ckpt_sig_file.write_text(ckpt_sig, encoding="utf-8")

                for lam_i in lambda_vals:
                    tmp_root = Path(f"/tmp/poc_epoch_k{k}_lam{lam_i}")
                    if tmp_root.exists():
                        shutil.rmtree(tmp_root)
                    shutil.copytree(ckpt_root, tmp_root)

                    homes = [tmp_root / f"node{i}" for i in range(honest_nodes + (1 if is_additive else 0) + k)]
                    procs = []
                    try:
                        procs = start_existing_network(
                            env=env,
                            tmp_root=tmp_root,
                            homes=homes,
                            p2p_base=p2p_base,
                            rpc_base=rpc_base,
                            api_base=api_base,
                            grpc_base=grpc_base,
                            denom=denom,
                        )
                        node_rpc = f"tcp://127.0.0.1:{rpc_base+0}"
                        h_live = wait_height(rpc_base + 0, target_pre, timeout_s=300.0)

                        if sybil_at_genesis:
                            ok_inject = k
                            h_attack = h_live
                            print(f"[k={k}, lam={lam_i}] sybil_at_genesis: no injection; starting at height {h_attack}")
                        else:
                            ok_inject = 0
                            h_attack = 0
                            print(f"[k={k}, lam={lam_i}] late-injection mode: pre_attack_epochs={pre_attack_epochs}; running draws from genesis first")

                        injection_schedule = []
                        if (not sybil_at_genesis):
                            injection_schedule = build_attack_injection_schedule(attacker_profile, k, post_attack_epochs)
                            print(f"[k={k}, lam={lam_i}] attacker_profile={attacker_profile} injection_schedule={injection_schedule}")

                        # NOTE: SDK v0.53 switched `config` UX to `config get/set/...`.
                        # We pass explicit tx/query flags instead of mutating client config here.
                        node_args = ["--node", node_rpc]

                        from_addr = sh([
                            "chaind", "keys", "show", payer_key, "-a",
                            "--keyring-backend", keyring,
                            "--home", str(homes[0]),
                        ], env=env, capture=True).stdout.strip()

                        # Concentration-drift bootstrap (Fix 1): drift_pool
                        # delegates PROPORTIONAL to each validator's current
                        # bonded stake so the pre-drift concentration profile is
                        # preserved (an equal delegation would flatten the Gini).
                        # The total pool stake injected is half the funding,
                        # leaving the other half as redelegation headroom.
                        if drift_active and drift_pool_key_name:
                            v_initial = _query_validators_sorted(env, homes[0], node_args)
                            if not v_initial:
                                raise RuntimeError("drift bootstrap: validator query returned empty set")
                            ops_initial = [op for op, _ in v_initial]
                            stakes_initial = [s for _, s in v_initial]
                            total_stake_initial = sum(stakes_initial) or 1
                            inject_budget = drift_pool_funding // 2
                            amounts_initial = [
                                int(inject_budget * s / total_stake_initial) for s in stakes_initial
                            ]
                            ok_count = _drift_initial_delegations(
                                env=env, home=homes[0], node_args=node_args,
                                drift_pool_key=drift_pool_key_name,
                                validator_ops=ops_initial,
                                amounts=amounts_initial,
                                denom=denom, chain_id=chain_id, keyring=keyring,
                                fees=fees, gas=tx_gas, broadcast=broadcast_mode,
                            )
                            print(f"[drift] proportional initial delegations {ok_count}/{len(ops_initial)} "
                                  f"(budget={inject_budget}{denom}, preserves start profile)")
                            # Strict success criterion: the proportional bootstrap
                            # only preserves the declared start profile if EVERY
                            # validator receives its share. A partial bootstrap
                            # leaves some validators with zero pool delegation
                            # (no redelegation capacity), silently distorting the
                            # drift trajectory. Allow at most one failed tx as an
                            # RPC-flap tolerance, otherwise abort.
                            n_targets = len(ops_initial)
                            if ok_count < n_targets - 1:
                                raise RuntimeError(
                                    f"drift bootstrap: only {ok_count}/{n_targets} initial "
                                    f"delegations succeeded; proportional start profile not "
                                    f"established — aborting to avoid silently distorted drift"
                                )
                            # Verify the realized profile matches the intended one:
                            # the post-bootstrap Gini must be within tolerance of the
                            # pre-bootstrap Gini (proportional injection preserves it).
                            gini_before = _gini_from_stakes(stakes_initial)
                            v_after = _query_validators_sorted(env, homes[0], node_args)
                            gini_after = _gini_from_stakes([s for _, s in v_after]) if v_after else gini_before
                            if abs(gini_after - gini_before) > 0.05:
                                raise RuntimeError(
                                    f"drift bootstrap: Gini drifted during bootstrap "
                                    f"({gini_before:.4f} -> {gini_after:.4f}, tol=0.05); "
                                    f"start profile not preserved — aborting"
                                )
                            print(f"[drift] start profile preserved: Gini {gini_before:.4f} -> {gini_after:.4f}")

                        if skip_set_lambda:
                            print(f"[k={k}, lam={lam_i}] set-lambda skipped by config (tx.skip_set_lambda=true)")
                        else:
                            seq0 = get_sequence(env, homes[0], node_args, from_addr)
                            print(f"[k={k}, lam={lam_i}] set-lambda start λ={lam_i} prev_seq={seq0}")
                            r = sh([
                                "chaind", "tx", "adaptivecommittee", "set-lambda", str(lam_i),
                                "--from", payer_key,
                                "--keyring-backend", keyring,
                                "--fees", fees,
                                "--gas", tx_gas,
                                "--broadcast-mode", broadcast_mode,
                                "--chain-id", chain_id,
                                "-y", "-o", "json",
                                "--home", str(homes[0]),
                            ] + node_args, env=env, capture=True, check=False)
                            txj = parse_json_output(r, "set-lambda")
                            if r.returncode != 0 or int(txj.get("code", 0)) != 0:
                                raise RuntimeError(f"set-lambda failed: {r.stdout}\n{r.stderr}")
                            txh_lam = txj.get("txhash", "")
                            print(f"[k={k}, lam={lam_i}] set-lambda accepted tx={txh_lam}")
                            # Sequence confirmation is best-effort (some SDK v0.53 account query paths are flaky).
                            try:
                                wait_sequence_increase(env, homes[0], node_args, from_addr, seq0, timeout_s=10.0)
                            except Exception as exc:
                                print(f"[warn] set-lambda sequence confirm skipped: {exc}")
                            if txh_lam:
                                txq = wait_tx_inclusion(env, homes[0], node_args, txh_lam, timeout_s=8.0)
                                # If sequence already advanced, missing tx index is treated as indexer lag, not tx failure.
                                if txq and int(txq.get("code", 0)) != 0:
                                    raise RuntimeError(f"set-lambda tx included but failed: tx={txh_lam} q={txq}")

                            # Hard barrier before draw loop: ensure at least one new block after set-lambda.
                            try:
                                hb = wait_height(rpc_base + 0, h_attack + 1, timeout_s=60.0)
                                print(f"[k={k}, lam={lam_i}] post-lambda barrier reached height={hb}")
                            except Exception as exc:
                                print(f"[warn] post-lambda barrier skipped: {exc}")

                        draws_done_total = 0
                        draw_idx_global = 0
                        prev_lambda_auto_ppm = None
                        stop_early = False
                        injected_so_far = (k if sybil_at_genesis else 0)
                        total_epochs = pre_attack_epochs + post_attack_epochs
                        for e in range(1, total_epochs + 1):
                            phase = "pre_attack" if (e <= pre_attack_epochs and not sybil_at_genesis) else "post_attack"

                            # Concentration-drift per-epoch migration step: shift
                            # a fraction of total bonded power from a donor
                            # validator to a receiver via the drift_pool's
                            # delegations. Runs in the [start, end] window
                            # regardless of pre/post phase.
                            if drift_active and drift_pool_key_name and drift_start_epoch <= e <= drift_end_epoch:
                                v_now = _query_validators_sorted(env, homes[0], node_args)
                                total_bonded_now = sum(t for _, t in v_now) if v_now else 0
                                migrate_amount = int(round(total_bonded_now * drift_migration_rate_ppm / 1_000_000.0))
                                donor_op, receiver_op = _drift_pick_donor_receiver(
                                    v_now, drift_donor_top_h, drift_receiver_top_r,
                                    epoch_idx=e, scenario_seed=drift_scenario_seed,
                                )
                                if donor_op and receiver_op and migrate_amount > 0:
                                    ok = _drift_migrate_step(
                                        env=env, home=homes[0], node_args=node_args,
                                        drift_pool_key=drift_pool_key_name,
                                        donor_op=donor_op, receiver_op=receiver_op,
                                        amount=migrate_amount, denom=denom,
                                        chain_id=chain_id, keyring=keyring,
                                        fees=fees, gas=tx_gas, broadcast=broadcast_mode,
                                    )
                                    print(f"[drift] e={e} migrate {migrate_amount}{denom} "
                                          f"{donor_op[:20]}... -> {receiver_op[:20]}... ok={ok}")
                                    # brief settle pause so the redelegate is committed before draws
                                    time.sleep(0.2)
                                else:
                                    print(f"[drift] e={e} skip (donor={donor_op}, receiver={receiver_op}, amount={migrate_amount})")

                            if (not sybil_at_genesis) and e > pre_attack_epochs and not drift_active:
                                post_e = e - pre_attack_epochs
                                need = injection_schedule[post_e - 1] if (post_e - 1) < len(injection_schedule) else 0
                                if need > 0 and injected_so_far < k:
                                    take = min(need, k - injected_so_far)
                                    got = inject_sybils(
                                        env, homes, honest_nodes, k, beta, chain_id, denom,
                                        from_acct_base, keyring, fees, node_rpc,
                                        start_sybil_idx=injected_so_far, inject_count=take,
                                        moniker_prefix=tracked_prefix,
                                        attack_mode=attack_mode, whale_share=whale_share,
                                    )
                                    injected_so_far += got
                                    ok_inject = injected_so_far
                                    h_attack = wait_height(rpc_base + 0, max(h_live, _rpc_latest_height(node_args)) + 2, timeout_s=180.0)
                                    print(f"[k={k}, lam={lam_i}] attack step post_epoch={post_e} added={got}/{take} cumulative={ok_inject}/{k} at height={h_attack}")
                                    if got < take:
                                        raise RuntimeError(
                                            f"[k={k}, lam={lam_i}] partial sybil injection step: {got}/{take} "
                                            f"(cumulative={ok_inject}/{k}). Aborting to avoid invalid scenario metrics."
                                        )

                                syb_cnt = count_validators_by_prefix(env, homes[0], node_args, tracked_prefix)
                                if post_e == 1 and syb_cnt == 0:
                                    raise RuntimeError(
                                        f"[k={k}, lam={lam_i}] no {tracked_prefix} validators found after first injection step "
                                        f"(injected={ok_inject}/{k}). Aborting to avoid all-zero fake results."
                                    )
                                if post_e == 1:
                                    print(f"[k={k}, lam={lam_i}] begin post-attack run: h_attack={h_attack}, post_epochs={post_attack_epochs}, draws_per_epoch={draws_per_epoch}, committee_mode={committee_mode}, committee={committee_size}, syb_count={syb_cnt}")

                            if phase == "pre_attack":
                                target_h = h_live + e * epoch_blocks
                                e_phase = e
                                e_total_phase = pre_attack_epochs
                            else:
                                post_e = e - pre_attack_epochs if not sybil_at_genesis else e
                                target_h = h_attack + post_e * epoch_blocks
                                e_phase = post_e
                                e_total_phase = post_attack_epochs

                            cur_h = wait_height(rpc_base + 0, target_h, timeout_s=1000.0)
                            print(f"[k={k}, lam={lam_i}] {phase} epoch {e_phase}/{e_total_phase} target_h={target_h} reached_h={cur_h}")

                            epoch_shares: List[float] = []
                            epoch_w: List[float] = []
                            epoch_a: List[float] = []
                            epoch_s: List[float] = []
                            epoch_lam_auto: List[float] = []
                            epoch_gini: List[float] = []
                            epoch_fresh: List[float] = []
                            epoch_stake_indep: List[float] = []
                            epoch_seat_minus_weight: List[float] = []
                            epoch_seat_minus_stake_indep: List[float] = []
                            epoch_member_prefixes: List[str] = []
                            # RBHC paper diagnostics accumulators.
                            epoch_lam_signal: List[float] = []
                            epoch_lam_risk: List[float] = []
                            epoch_lam_target: List[float] = []
                            epoch_risk_alpha: List[float] = []
                            epoch_risk_beta: List[float] = []
                            epoch_risk_b0: List[float] = []
                            epoch_risk_ba: List[float] = []
                            epoch_risk_sat: List[float] = []
                            epoch_rb_mode_observed: str = controller_mode

                            # Compute seats via current validator set + sybil prefix mapping once per epoch.
                            vals = json.loads(sh([
                                "chaind", "query", "staking", "validators", "-o", "json", "--home", str(homes[0])
                            ] + node_args, env=env, capture=True).stdout)
                            arr = vals.get("validators", vals) if isinstance(vals, dict) else vals

                            vset_n = len(arr)
                            if committee_mode == "all":
                                effective_committee_size = vset_n
                            else:
                                effective_committee_size = committee_size

                            if effective_committee_size > vset_n:
                                # The chain enforces this; fail early with a clear error to avoid confusion.
                                raise RuntimeError(
                                    f"committee_size={effective_committee_size} exceeds validator set size {vset_n}. "
                                    f"Lower committee_size in poc_config.yaml (or increase honest_nodes/sybil_k)."
                                )

                            def norm_addr(s: str) -> str:
                                return (s or "").strip()

                            attacker_ops = set(
                                norm_addr(v.get("operator_address"))
                                for v in arr
                                if (v.get("description", {}) or {}).get("moniker", "").startswith(tracked_prefix)
                            )
                            attacker_ops.discard("")

                            # Concentration-drift has no injected attacker cohort
                            # (k=0), so the moniker-prefix set above is empty and
                            # every tracked_* seat metric would be zero. In drift
                            # the entity worth tracking is the monitored top-k
                            # stake coalition — the same coalition the risk path
                            # scores — so its realized committee seat share is the
                            # empirical capture the controller is meant to reduce.
                            # Recompute the tracked set as the current top-k by
                            # bonded stake at this epoch; membership tracks the
                            # drift just like the risk certificate does.
                            if drift_active:
                                ranked_ops = sorted(
                                    (
                                        (norm_addr(v.get("operator_address")),
                                         int(v.get("tokens", "0") or 0))
                                        for v in arr
                                    ),
                                    key=lambda kv: kv[1], reverse=True,
                                )
                                attacker_ops = {op for op, _ in ranked_ops[:max(1, risk_top_k)] if op}

                            # Independent stake share check (from staking tokens).
                            tok_total = 0
                            tok_att = 0
                            for v in arr:
                                op = norm_addr(v.get("operator_address"))
                                try:
                                    t = int(v.get("tokens", "0"))
                                except Exception:
                                    t = 0
                                tok_total += t
                                if op in attacker_ops:
                                    tok_att += t
                            stake_share_indep_epoch = (tok_att / tok_total) if tok_total > 0 else 0.0

                            attacker_stakes_pairs: List[Tuple[str, int]] = []
                            for v in arr:
                                mon = (v.get("description", {}) or {}).get("moniker", "")
                                if not mon.startswith(tracked_prefix):
                                    continue
                                try:
                                    t = int(v.get("tokens", "0"))
                                except Exception:
                                    t = 0
                                attacker_stakes_pairs.append((mon, t))
                            attacker_stakes_pairs.sort(key=lambda kv: kv[0])
                            attacker_stakes_csv = ";".join(f"{m}:{t}" for m, t in attacker_stakes_pairs)

                            all_ops_sorted = [
                                norm_addr(v.get("operator_address"))
                                for v in sorted(arr, key=lambda x: int(x.get("tokens", "0")), reverse=True)
                            ]

                            for di in range(draws_per_epoch):
                                if phase == "post_attack" and post_attack_draw_limit > 0 and draws_done_total >= post_attack_draw_limit:
                                    stop_early = True
                                    break
                                raw_tag = f"k{k}_c{effective_committee_size}_lam{lam_i}_e{e}_d{di}"
                                tag = f"{policy_mode}__{raw_tag}"
                                h_now = _rpc_latest_height(node_args)
                                h_disp = h_now if h_now > 0 else cur_h
                                print(f"[k={k}, lam={lam_i}, e={e}] draw {di+1}/{draws_per_epoch} tag={tag} h_now={h_disp}")

                                # Detect stale/non-updating data: does this tag already exist BEFORE we draw?
                                pre_exists = False
                                pre_members_len = 0
                                qpre = sh([
                                    "chaind", "query", "adaptivecommittee", "last-draw", tag,
                                    "-o", "json",
                                    "--home", str(homes[0]),
                                ] + node_args, env=env, capture=True, check=False)
                                if qpre.returncode == 0 and (qpre.stdout or "").strip():
                                    try:
                                        pj = json.loads(qpre.stdout)
                                    except Exception:
                                        pj = {}
                                    payload_pre = pj.get("membersCsv") or pj.get("members_csv") or ""
                                    members_csv_pre, _meta_pre = parse_last_draw_payload(payload_pre)
                                    pre_members = [m.strip() for m in members_csv_pre.split(",") if m.strip()] if members_csv_pre else []
                                    if pre_members:
                                        pre_exists = True
                                        pre_members_len = len(pre_members)

                                seq1 = get_sequence(env, homes[0], node_args, from_addr)
                                r = sh([
                                    "chaind", "tx", "adaptivecommittee", "draw-committee", str(effective_committee_size), tag,
                                    "--from", payer_key,
                                    "--keyring-backend", keyring,
                                    "--fees", fees,
                                    "--gas", tx_gas,
                                    "--broadcast-mode", broadcast_mode,
                                    "--chain-id", chain_id,
                                    "-y", "-o", "json",
                                    "--home", str(homes[0]),
                                ] + node_args, env=env, capture=True, check=False)
                                txj = parse_json_output(r, "draw-committee")
                                if r.returncode != 0 or int(txj.get("code", 0)) != 0:
                                    raise RuntimeError(f"draw failed: {r.stdout}\n{r.stderr}")
                                txh = txj.get("txhash", "")

                                # Commit gate: require either sequence bump OR indexed tx success.
                                seq_confirmed = False
                                try:
                                    wait_sequence_increase(env, homes[0], node_args, from_addr, seq1, timeout_s=12.0)
                                    seq_confirmed = True
                                except Exception as exc:
                                    print(f"[warn] draw sequence confirm missed for {tag}: {exc}")

                                txq = wait_tx_inclusion(env, homes[0], node_args, txh, timeout_s=20.0)
                                tx_confirmed = bool(txq) and int(txq.get("code", 0)) == 0
                                if txq:
                                    print(f"[debug] draw tx commit: tag={tag} tx={txh[:12]}... h={txq.get('height','?')} code={txq.get('code','?')}")
                                if txq and int(txq.get("code", 0)) != 0:
                                    raise RuntimeError(f"draw tx included but failed: tx={txh} q={txq}")

                                if not seq_confirmed and not tx_confirmed:
                                    qtx = query_tx_by_hash(env, homes[0], node_args, txh)
                                    raise RuntimeError(
                                        f"draw tx commit unconfirmed for tag={tag}, tx={txh}; "
                                        f"seq_confirmed={seq_confirmed} tx_confirmed={tx_confirmed}; "
                                        f"query_tx_rc={qtx.returncode} out={(qtx.stdout or '')[:300]} err={(qtx.stderr or '')[:180]}"
                                    )

                                # last-draw can lag behind tx commit/indexing; poll longer before failing.
                                members: List[str] = []
                                attrs_from_state: Dict[str, str] = {}
                                last_draw_raw = ""
                                for _ in range(50):
                                    qr = sh([
                                        "chaind", "query", "adaptivecommittee", "last-draw", tag,
                                        "-o", "json",
                                        "--home", str(homes[0]),
                                    ] + node_args, env=env, capture=True, check=False)
                                    last_draw_raw = (qr.stdout or "").strip()
                                    if qr.returncode == 0 and last_draw_raw:
                                        try:
                                            qj = json.loads(last_draw_raw)
                                        except Exception:
                                            qj = {}
                                        payload = qj.get("membersCsv") or qj.get("members_csv") or ""
                                        members_csv, attrs_from_state = parse_last_draw_payload(payload)
                                        # Normalize to avoid whitespace mismatches.
                                        members = [m.strip() for m in (members_csv.split(",") if members_csv else []) if m.strip()]
                                        if members:
                                            break
                                    time.sleep(0.2)

                                if not members:
                                    # One last hard sync: wait +1 block, then re-query last-draw.
                                    try:
                                        wait_height(rpc_base + 0, cur_h + 1, timeout_s=30.0)
                                    except Exception:
                                        pass

                                    for _ in range(25):
                                        qr = sh([
                                            "chaind", "query", "adaptivecommittee", "last-draw", tag,
                                            "-o", "json",
                                            "--home", str(homes[0]),
                                        ] + node_args, env=env, capture=True, check=False)
                                        last_draw_raw = (qr.stdout or "").strip()
                                        if qr.returncode == 0 and last_draw_raw:
                                            try:
                                                qj = json.loads(last_draw_raw)
                                            except Exception:
                                                qj = {}
                                            payload = qj.get("membersCsv") or qj.get("members_csv") or ""
                                            members_csv, attrs_from_state = parse_last_draw_payload(payload)
                                            members = [m.strip() for m in (members_csv.split(",") if members_csv else []) if m.strip()]
                                            if members:
                                                break
                                        time.sleep(0.2)

                                if not members:
                                    # Extra diagnostics for stuck draw state.
                                    qtx = query_tx_by_hash(env, homes[0], node_args, txh)
                                    print(f"[debug] empty-members tx-query rc={qtx.returncode} out={(qtx.stdout or '')[:400]} err={(qtx.stderr or '')[:200]}")
                                    print(f"[debug] empty-members draw-tx-broadcast txh={txh} txj={txj}")
                                    print(f"[warn] skip draw due to empty committee members tag={tag} tx={txh}")
                                    continue

                                # Stable state-based attrs from last-draw payload.
                                attrs = attrs_from_state
                                if not attrs:
                                    attrs = {
                                        "attacker_stake_ppm": str(int(round(stake_share_indep_epoch * 1_000_000))),
                                        "attacker_age_ppm": "",
                                        "attacker_weight_ppm": str(int(round(stake_share_indep_epoch * 1_000_000))),
                                        "attacker_validators": str(len(attacker_ops)),
                                    }
                                    print(f"[warn] empty state attrs in last-draw payload for tag={tag}; using stake-based fallback attrs")

                                # Normalize members tokens (defensive): some JSON emits spaces after commas.
                                members = [m.strip() for m in members if m and m.strip()]

                                attacker_seats = sum(1 for m in members if m in attacker_ops)
                                attacker_seats_share = attacker_seats / max(1, len(members))

                                # Light heuristics to understand member address format.
                                pref = ""
                                if members:
                                    m0 = members[0]
                                    for p in ("valoper", "valcons", "cosmosvaloper", "cosmosvalcons"):
                                        if p in m0:
                                            pref = p
                                            break
                                    if not pref:
                                        pref = (m0[:12] + "…") if len(m0) > 12 else m0

                                w_share = float(attrs.get("attacker_weight_ppm") or 0.0) / 1e6
                                seat_minus_weight = attacker_seats_share - w_share
                                seat_minus_stake_indep = attacker_seats_share - stake_share_indep_epoch

                                uniq = set(members)
                                has_dupes = (len(uniq) != len(members))
                                attacker_unique = sum(1 for m in uniq if m in attacker_ops)

                                lam_auto_raw = attrs.get("lambda_auto_ppm", "")
                                lam_auto_i = None
                                try:
                                    lam_auto_i = int(str(lam_auto_raw)) if str(lam_auto_raw).strip() != "" else None
                                except Exception:
                                    lam_auto_i = None
                                lam_prev_i = prev_lambda_auto_ppm
                                lam_delta_i = (lam_auto_i - lam_prev_i) if (lam_auto_i is not None and lam_prev_i is not None) else None

                                row = {
                                    "k": str(k),
                                    "committee_size": str(effective_committee_size),
                                    "lambda_ppm": str(lam_i),
                                    "lambda_auto_ppm": str(lam_auto_i) if lam_auto_i is not None else "",
                                    "lambda_prev_auto_ppm": str(lam_prev_i) if lam_prev_i is not None else "",
                                    "lambda_auto_delta_ppm": str(lam_delta_i) if lam_delta_i is not None else "",
                                    "lambda_manual_ppm": attrs.get("lambda_manual_ppm", ""),
                                    "policy_mode": attrs.get("policy_mode", policy_mode),
                                    "gini_ppm": attrs.get("gini_ppm", ""),
                                    "fresh_pressure_ppm": attrs.get("fresh_pressure_ppm", ""),
                                    "phase": phase,
                                    "epoch_idx": str(e),
                                    "draw_i": str(di),
                                    "draw_idx_global": str(draw_idx_global + 1),
                                    "draw_idx_post_attack": str(draws_done_total + 1) if phase == "post_attack" else "",
                                    "height": str(cur_h),
                                    "attack_height": str(h_attack) if h_attack > 0 else "",
                                    "tag": tag,
                                    "tag_preexists": str(pre_exists).lower(),
                                    "tag_preexists_members_len": str(pre_members_len),
                                    "vset_n": str(vset_n),
                                    "attacker_seats": str(attacker_seats),
                                    "attacker_seats_share": f"{attacker_seats_share:.6f}",
                                    "attacker_stake_ppm": attrs.get("attacker_stake_ppm", ""),
                                    "attacker_age_ppm": attrs.get("attacker_age_ppm", ""),
                                    "attacker_weight_ppm": attrs.get("attacker_weight_ppm", ""),
                                    "attacker_validators": attrs.get("attacker_validators", ""),
                                    "members_len": str(len(members)),
                                    "unique_members_len": str(len(uniq)),
                                    "has_duplicate_members": str(has_dupes).lower(),
                                    "attacker_unique_members_len": str(attacker_unique),
                                    "members_prefix": pref,
                                    "attacker_ops_len": str(len(attacker_ops)),
                                    "attacker_stakes_csv": attacker_stakes_csv,
                                    "stake_share_indep": f"{stake_share_indep_epoch:.6f}",
                                    "attacker_tokens": str(tok_att),
                                    "total_tokens": str(tok_total),
                                    "seat_minus_weight": f"{seat_minus_weight:.6f}",
                                    "seat_minus_stake_indep": f"{seat_minus_stake_indep:.6f}",
                                    # RBHC paper diagnostics (pass-through from chain payload).
                                    "lambda_signal_target_ppm": attrs.get("lambda_signal_target_ppm", ""),
                                    "lambda_risk_target_ppm": attrs.get("lambda_risk_target_ppm", ""),
                                    "lambda_target_ppm": attrs.get("lambda_target_ppm", ""),
                                    "risk_alpha_ppm": attrs.get("risk_alpha_ppm", ""),
                                    "risk_beta_ppm": attrs.get("risk_beta_ppm", ""),
                                    "risk_budget_satisfied": attrs.get("risk_budget_satisfied", ""),
                                    "risk_coalition_size": attrs.get("risk_coalition_size", ""),
                                    "risk_bound0_log10e6": attrs.get("risk_bound0_log10e6", ""),
                                    "risk_bound_auto_log10e6": attrs.get("risk_bound_auto_log10e6", ""),
                                    "risk_controller_mode": attrs.get("risk_controller_mode", controller_mode),
                                }

                                counts: Dict[str, int] = {}
                                for m in members:
                                    counts[m] = counts.get(m, 0) + 1
                                for kk in topk_vals:
                                    topk_ops = all_ops_sorted[: int(kk)]
                                    row[f"top{kk}_seats"] = str(sum(counts.get(op, 0) for op in topk_ops))

                                row = enrich_tracked_row(row, tracked_meta)

                                dw.writerow({k: row.get(k, "") for k in draws_core_cols})
                                dwd.writerow(row)

                                vmetrics = attrs.get("validator_metrics", "")
                                if vmetrics:
                                    for part in [p for p in vmetrics.split(";") if p]:
                                        bits = part.split(":")
                                        if len(bits) != 4:
                                            continue
                                        vaddr, vstake, vage, vw = bits
                                        vm_row = {
                                            "k": str(k),
                                            "committee_size": str(effective_committee_size),
                                            "lambda_ppm": str(lam_i),
                                            "epoch_idx": str(e),
                                            "draw_i": str(di),
                                            "draw_idx_post_attack": str(draws_done_total + 1) if phase == "post_attack" else "",
                                            "tag": tag,
                                            "validator_address": vaddr,
                                            "validator_stake": vstake,
                                            "validator_age_score": vage,
                                            "validator_weight_score": vw,
                                            "is_attacker": str(vaddr in attacker_ops).lower(),
                                        }
                                        vmw.writerow(vm_row)
                                        validator_metric_rows.append(vm_row)

                                fd.flush(); fdd.flush(); fvm.flush()

                                epoch_shares.append(attacker_seats_share)
                                if row["attacker_weight_ppm"]:
                                    epoch_w.append(float(row["attacker_weight_ppm"]) / 1e6)
                                if row["attacker_age_ppm"]:
                                    epoch_a.append(float(row["attacker_age_ppm"]) / 1e6)
                                if row["attacker_stake_ppm"]:
                                    epoch_s.append(float(row["attacker_stake_ppm"]) / 1e6)
                                if row.get("lambda_auto_ppm", ""):
                                    epoch_lam_auto.append(float(row["lambda_auto_ppm"]))
                                if row.get("gini_ppm", ""):
                                    epoch_gini.append(float(row["gini_ppm"]))
                                if row.get("fresh_pressure_ppm", ""):
                                    epoch_fresh.append(float(row["fresh_pressure_ppm"]))

                                epoch_stake_indep.append(stake_share_indep_epoch)
                                epoch_seat_minus_weight.append(seat_minus_weight)
                                epoch_seat_minus_stake_indep.append(seat_minus_stake_indep)
                                epoch_member_prefixes.append(pref)
                                # RBHC: pass through the parsed diagnostics for aggregation.
                                if row.get("lambda_signal_target_ppm", ""):
                                    try: epoch_lam_signal.append(float(row["lambda_signal_target_ppm"]))
                                    except ValueError: pass
                                if row.get("lambda_risk_target_ppm", ""):
                                    try: epoch_lam_risk.append(float(row["lambda_risk_target_ppm"]))
                                    except ValueError: pass
                                if row.get("lambda_target_ppm", ""):
                                    try: epoch_lam_target.append(float(row["lambda_target_ppm"]))
                                    except ValueError: pass
                                if row.get("risk_alpha_ppm", ""):
                                    try: epoch_risk_alpha.append(float(row["risk_alpha_ppm"]))
                                    except ValueError: pass
                                if row.get("risk_beta_ppm", ""):
                                    try: epoch_risk_beta.append(float(row["risk_beta_ppm"]))
                                    except ValueError: pass
                                if row.get("risk_bound0_log10e6", ""):
                                    try: epoch_risk_b0.append(float(row["risk_bound0_log10e6"]))
                                    except ValueError: pass
                                if row.get("risk_bound_auto_log10e6", ""):
                                    try: epoch_risk_ba.append(float(row["risk_bound_auto_log10e6"]))
                                    except ValueError: pass
                                if row.get("risk_budget_satisfied", ""):
                                    try: epoch_risk_sat.append(float(row["risk_budget_satisfied"]))
                                    except ValueError: pass
                                if row.get("risk_controller_mode", ""):
                                    epoch_rb_mode_observed = row["risk_controller_mode"]
                                draw_idx_global += 1
                                if lam_auto_i is not None:
                                    prev_lambda_auto_ppm = lam_auto_i
                                if phase == "post_attack":
                                    draws_done_total += 1

                            def mean(xs: List[float]) -> float:
                                return (sum(xs) / len(xs)) if xs else 0.0

                            # modal-ish prefix for quick eyeballing of address type
                            pref_mode = ""
                            if epoch_member_prefixes:
                                counts = {}
                                for p in epoch_member_prefixes:
                                    counts[p] = counts.get(p, 0) + 1
                                pref_mode = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

                            if not epoch_shares:
                                print(f"[k={k}, lam={lam_i}] epoch {e}/{post_attack_epochs} has 0 draws (likely early-stop boundary); skipping summary row")
                                if stop_early:
                                    print(f"[k={k}, lam={lam_i}] early-stop reached: post_attack_draw_limit={post_attack_draw_limit}")
                                    break
                                continue

                            if phase != "post_attack":
                                # Keep historical behavior of summary/comparison artifacts: post-attack only.
                                continue

                            summary = {
                                "k": str(k),
                                "committee_size": str(effective_committee_size),
                                "lambda_ppm": str(lam_i),
                                "epoch_idx": str(e_phase),
                                "height": str(cur_h),
                                "draws": str(len(epoch_shares)),
                                "mean_attacker_share": f"{mean(epoch_shares):.6f}",
                                "min_attacker_share": f"{min(epoch_shares):.6f}",
                                "max_attacker_share": f"{max(epoch_shares):.6f}",
                                "mean_attacker_weight_share": f"{mean(epoch_w):.6f}" if epoch_w else "",
                                "mean_attacker_age_share": f"{mean(epoch_a):.6f}" if epoch_a else "",
                                "mean_attacker_stake_share": f"{mean(epoch_s):.6f}" if epoch_s else "",
                                "mean_attacker_weight_ppm": str(int(round(mean(epoch_w) * 1_000_000))) if epoch_w else "",
                                "mean_attacker_age_ppm": str(int(round(mean(epoch_a) * 1_000_000))) if epoch_a else "",
                                "mean_attacker_stake_ppm": str(int(round(mean(epoch_s) * 1_000_000))) if epoch_s else "",
                                "mean_lambda_auto_ppm": str(int(round(mean(epoch_lam_auto)))) if epoch_lam_auto else "",
                                "mean_gini_ppm": str(int(round(mean(epoch_gini)))) if epoch_gini else "",
                                "mean_fresh_pressure_ppm": str(int(round(mean(epoch_fresh)))) if epoch_fresh else "",
                                "mean_stake_share_indep": f"{mean(epoch_stake_indep):.6f}" if epoch_stake_indep else "",
                                "mean_seat_minus_weight": f"{mean(epoch_seat_minus_weight):.6f}" if epoch_seat_minus_weight else "",
                                "mean_seat_minus_stake_indep": f"{mean(epoch_seat_minus_stake_indep):.6f}" if epoch_seat_minus_stake_indep else "",
                                "members_prefix_mode": pref_mode,
                                # RBHC paper diagnostics — epoch means.
                                "mean_lambda_signal_target_ppm": str(int(round(mean(epoch_lam_signal)))) if epoch_lam_signal else "",
                                "mean_lambda_risk_target_ppm": str(int(round(mean(epoch_lam_risk)))) if epoch_lam_risk else "",
                                "mean_lambda_target_ppm": str(int(round(mean(epoch_lam_target)))) if epoch_lam_target else "",
                                "mean_risk_alpha_ppm": str(int(round(mean(epoch_risk_alpha)))) if epoch_risk_alpha else "",
                                "mean_risk_beta_ppm": str(int(round(mean(epoch_risk_beta)))) if epoch_risk_beta else "",
                                "mean_risk_bound0_log10e6": str(int(round(mean(epoch_risk_b0)))) if epoch_risk_b0 else "",
                                "mean_risk_bound_auto_log10e6": str(int(round(mean(epoch_risk_ba)))) if epoch_risk_ba else "",
                                # Fraction of draws in the epoch where the budget was met.
                                # Drops from 1.0 to 0.0 at the saturation point where
                                # even λ_max can no longer keep B_t(λ) ≤ ε (Issue 1).
                                "risk_budget_satisfied_frac": f"{mean(epoch_risk_sat):.4f}" if epoch_risk_sat else "",
                                "risk_controller_mode": epoch_rb_mode_observed,
                            }
                            summary = enrich_tracked_row(summary, tracked_meta)
                            sw.writerow(summary)
                            fs.flush()

                            print(
                                f"[k={k}, lam={lam_i}] epoch {e}/{post_attack_epochs} "
                                f"mean_{tracked_entity_mode}_share={summary['mean_tracked_share']}"
                            )
                            if stop_early:
                                print(f"[k={k}, lam={lam_i}] early-stop reached: post_attack_draw_limit={post_attack_draw_limit}")
                                break

                    finally:
                        stop_all(procs)

    latest_draws = art_dir / "results" / "epoch_draws_latest.csv"
    latest_draws_debug = art_dir / "results" / "epoch_draws_debug_latest.csv"
    latest_validator_metrics = art_dir / "results" / "validator_metrics_latest.csv"
    latest_summary = art_dir / "results" / "epoch_summary_latest.csv"
    latest_compare = art_dir / "results" / "epoch_lambda_comparison_latest.csv"
    latest_draw_summary = art_dir / "results" / "early_draws_summary_latest.csv"
    latest_final_table = art_dir / "results" / "epoch_final_table_latest.csv"
    latest_final_table_single = art_dir / "results" / "epoch_final_table_single_row_latest.csv"
    latest_final_policy_table = art_dir / "results" / "epoch_final_policy_table_latest.csv"
    latest_final_epoch_table = art_dir / "results" / "epoch_final_epoch_table_latest.csv"
    latest_manifest = art_dir / "results" / "run_manifest_latest.json"
    latest_run_dir_txt = art_latest_dir / "results" / "latest_run_dir.txt"
    latest_run_id_txt = art_latest_dir / "results" / "latest_run_id.txt"
    latest_draws.write_bytes(draws_csv.read_bytes())
    latest_draws_debug.write_bytes(draws_debug_csv.read_bytes())
    latest_validator_metrics.write_bytes(validator_metrics_csv.read_bytes())
    latest_summary.write_bytes(summary_csv.read_bytes())
    latest_manifest.write_bytes(manifest_json.read_bytes())
    latest_run_dir_txt.write_text(str(art_dir), encoding="utf-8")
    latest_run_id_txt.write_text(f"{run_id}\n", encoding="utf-8")

    def _mirror_latest_result(src: Path, latest_name: str):
        if art_latest_dir == art_dir:
            return
        (art_latest_dir / "results" / latest_name).write_bytes(src.read_bytes())

    def _mirror_latest_plot(src: Path, latest_name: str):
        if art_latest_dir == art_dir:
            return
        (art_latest_dir / "plots" / latest_name).write_bytes(src.read_bytes())

    _mirror_latest_result(manifest_json, "run_manifest_latest.json")
    _mirror_latest_result(draws_csv, "epoch_draws_latest.csv")
    _mirror_latest_result(draws_debug_csv, "epoch_draws_debug_latest.csv")
    _mirror_latest_result(validator_metrics_csv, "validator_metrics_latest.csv")
    _mirror_latest_result(summary_csv, "epoch_summary_latest.csv")

    # Build draw-index summary (post-injection index instead of epoch grouping).
    draw_rows = list(csv.DictReader(draws_csv.open("r", encoding="utf-8")))
    dcols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "k", "committee_size", "lambda_ppm", "draw_idx_post_attack",
        "mean_attacker_share", "mean_attacker_weight_share", "mean_seat_minus_stake_indep", "mean_seat_minus_weight",
        "mean_tracked_share", "mean_tracked_weight_share", "mean_tracked_minus_stake_indep", "mean_tracked_minus_weight",
    ]
    dgrp: Dict[Tuple[int, int, int, int], List[Dict[str, str]]] = {}
    for r in draw_rows:
        try:
            key = (int(r.get("k", "0")), int(r.get("committee_size", "0")), int(r.get("lambda_ppm", "0")), int(r.get("draw_idx_post_attack", "0")))
        except Exception:
            continue
        dgrp.setdefault(key, []).append(r)

    def _m(rows_, field):
        vals = []
        for rr in rows_:
            v = rr.get(field, "")
            if v == "":
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        return (sum(vals) / len(vals)) if vals else 0.0

    with draw_summary_csv.open("w", newline="", encoding="utf-8") as fds:
        wds = csv.DictWriter(fds, fieldnames=dcols)
        wds.writeheader()
        for (kk, cc, lam, di), arr in sorted(dgrp.items()):
            drow = {
                "k": str(kk), "committee_size": str(cc), "lambda_ppm": str(lam), "draw_idx_post_attack": str(di),
                "mean_attacker_share": f"{_m(arr, 'attacker_seats_share'):.6f}",
                "mean_attacker_weight_share": f"{_m(arr, 'attacker_weight_ppm')/1_000_000:.6f}",
                "mean_seat_minus_stake_indep": f"{_m(arr, 'seat_minus_stake_indep'):.6f}",
                "mean_seat_minus_weight": f"{_m(arr, 'seat_minus_weight'):.6f}",
            }
            wds.writerow(enrich_tracked_row(drow, tracked_meta))
    latest_draw_summary.write_bytes(draw_summary_csv.read_bytes())
    _mirror_latest_result(draw_summary_csv, "early_draws_summary_latest.csv")

    # Build lambda-vs-baseline comparison table (per k).
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    comp_cols = [
        "k", "committee_size", "lambda_ppm",
        "mean_attacker_share_overall", "delta_vs_baseline_share",
        "mean_attacker_weight_share_overall", "delta_vs_baseline_weight_share",
        "mean_attacker_age_share_overall", "mean_attacker_stake_share_overall",
        "mean_attacker_weight_ppm_overall", "delta_vs_baseline_weight_ppm",
    ]
    comp_rows = []
    by_key = {}
    for r in rows:
        try:
            kk = int(r.get("k", "0"))
            cc = int(r.get("committee_size", "0"))
            lam = int(r.get("lambda_ppm", "0"))
        except Exception:
            continue
        by_key.setdefault((kk, cc, lam), []).append(r)

    by_k = {}
    for (kk, cc, lam), arr in by_key.items():
        def m(field):
            vals = []
            for x in arr:
                v = x.get(field, "")
                if v == "":
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals) / len(vals)) if vals else 0.0
        rec = {
            "mean_attacker_share_overall": m("mean_attacker_share"),
            "mean_attacker_weight_share_overall": m("mean_attacker_weight_share"),
            "mean_attacker_age_share_overall": m("mean_attacker_age_share"),
            "mean_attacker_stake_share_overall": m("mean_attacker_stake_share"),
            "mean_attacker_weight_ppm_overall": m("mean_attacker_weight_ppm"),
        }
        by_k.setdefault((kk, cc), {})[lam] = rec

    for (kk, cc), lam_map in sorted(by_k.items()):
        base = lam_map.get(0)
        base_share = base["mean_attacker_share_overall"] if base else 0.0
        base_wshare = base["mean_attacker_weight_share_overall"] if base else 0.0
        base_wppm = base["mean_attacker_weight_ppm_overall"] if base else 0.0
        for lam in sorted(lam_map.keys()):
            rec = lam_map[lam]
            comp_rows.append({
                "k": str(kk),
                "committee_size": str(cc),
                "lambda_ppm": str(lam),
                "mean_attacker_share_overall": f"{rec['mean_attacker_share_overall']:.6f}",
                "delta_vs_baseline_share": f"{(rec['mean_attacker_share_overall'] - base_share):.6f}",
                "mean_attacker_weight_share_overall": f"{rec['mean_attacker_weight_share_overall']:.6f}",
                "delta_vs_baseline_weight_share": f"{(rec['mean_attacker_weight_share_overall'] - base_wshare):.6f}",
                "mean_attacker_age_share_overall": f"{rec['mean_attacker_age_share_overall']:.6f}",
                "mean_attacker_stake_share_overall": f"{rec['mean_attacker_stake_share_overall']:.6f}",
                "mean_attacker_weight_ppm_overall": f"{rec['mean_attacker_weight_ppm_overall']:.0f}",
                "delta_vs_baseline_weight_ppm": f"{(rec['mean_attacker_weight_ppm_overall'] - base_wppm):.0f}",
            })

    with compare_csv.open("w", newline="", encoding="utf-8") as fc:
        cw = csv.DictWriter(fc, fieldnames=comp_cols)
        cw.writeheader()
        for r in comp_rows:
            cw.writerow(r)
    latest_compare.write_bytes(compare_csv.read_bytes())
    _mirror_latest_result(compare_csv, "epoch_lambda_comparison_latest.csv")

    # Final compact results table (most informative columns for reporting).
    final_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "attack_mode", "whale_share",
        "k", "committee_size", "lambda_init_ppm",
        "pre_draws", "post_draws", "attack_draw_idx", "draws_per_epoch_cfg",
        "pre_lambda_mean_ppm", "post_lambda_mean_ppm", "post_lambda_peak_ppm", "post_lambda_peak_draw_idx", "post_lambda_half_life_draws",
        "post_lambda_rise_time_draws", "post_lambda_settle_time_draws", "post_lambda_overshoot_pct",
        "post_lambda_chatter_rms_ppm", "post_lambda_control_effort",
        "post_attacker_weight_mean", "post_attacker_seat_mean", "post_stake_baseline_mean", "auc_gain_vs_stake", "time_to_95pct_baseline_draws",
        "post_tracked_weight_mean", "post_tracked_seat_mean", "post_tracked_stake_baseline_mean", "auc_tracked_gain_vs_stake", "time_to_95pct_tracked_baseline_draws",
        "post_capture_ge_1_3_emp_pct", "post_capture_ge_1_2_emp_pct",
        "post_chernoff_ge_1_3_adaptive", "post_chernoff_ge_1_3_baseline",
        "post_chernoff_ge_1_2_adaptive", "post_chernoff_ge_1_2_baseline",
        "chernoff_bound_reduction_ge_1_3_pct", "chernoff_bound_reduction_ge_1_2_pct",
        "reduction_vs_baseline_full_pct",
        "reduction_vs_baseline_1ep_pct", "reduction_vs_baseline_2ep_pct", "reduction_vs_baseline_3ep_pct", "reduction_vs_baseline_5ep_pct",
        "tracked_vs_baseline_full_pct", "tracked_vs_baseline_1ep_pct", "tracked_vs_baseline_2ep_pct", "tracked_vs_baseline_3ep_pct", "tracked_vs_baseline_5ep_pct",
    ]
    policy_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "k", "committee_size", "lambda_init_ppm", "policy", "lambda_policy_ppm",
        "post_draws", "draws_per_epoch_cfg",
        "post_attacker_weight_mean", "post_attacker_seat_mean_model", "post_stake_baseline_mean",
        "post_tracked_weight_mean", "post_tracked_seat_mean_model", "post_tracked_stake_baseline_mean",
        "post_capture_ge_1_3_model_pct", "post_capture_ge_1_2_model_pct",
        "reduction_vs_baseline_full_pct", "reduction_vs_baseline_1ep_pct", "reduction_vs_baseline_2ep_pct", "reduction_vs_baseline_3ep_pct", "reduction_vs_baseline_5ep_pct",
        "tracked_vs_baseline_full_pct", "tracked_vs_baseline_1ep_pct", "tracked_vs_baseline_2ep_pct", "tracked_vs_baseline_3ep_pct", "tracked_vs_baseline_5ep_pct",
        "time_to_95pct_baseline_draws", "time_to_95pct_tracked_baseline_draws",
    ]

    dcore_rows = list(csv.DictReader(draws_csv.open("r", encoding="utf-8")))
    ddebug_rows = list(csv.DictReader(draws_debug_csv.open("r", encoding="utf-8")))
    grouped = {}
    for r in dcore_rows:
        try:
            key = (int(r.get("k", "0") or 0), int(r.get("committee_size", "0") or 0), int(r.get("lambda_ppm", "0") or 0))
        except Exception:
            continue
        grouped.setdefault(key, []).append(r)

    debug_group = {}
    for r in ddebug_rows:
        try:
            key = (int(r.get("k", "0") or 0), int(r.get("committee_size", "0") or 0), int(r.get("lambda_ppm", "0") or 0))
        except Exception:
            continue
        debug_group.setdefault(key, []).append(r)

    sampler_baseline_by_tag: Dict[str, Dict[str, float]] = {}
    if drift_active and validator_metric_rows:
        vm_group: Dict[str, List[Dict[str, str]]] = {}
        for r in validator_metric_rows:
            tag = str(r.get("tag", "") or "")
            if not tag:
                continue
            vm_group.setdefault(tag, []).append(r)
        for tag, rows in vm_group.items():
            try:
                committee_size = int(rows[0].get("committee_size", "0") or 0)
            except Exception:
                committee_size = 0
            stakes = []
            flags = []
            for row in rows:
                try:
                    stakes.append(int(row.get("validator_stake", "0") or 0))
                except Exception:
                    stakes.append(0)
                flags.append(str(row.get("is_attacker", "false") or "false").strip().lower() == "true")
            mean_share, cap13, cap12 = _estimate_ppswor_coalition_baseline(
                stakes, flags, committee_size, trials=2000, seed_text=tag
            )
            sampler_baseline_by_tag[tag] = {
                "seat_share": mean_share,
                "cap13_pct": cap13,
                "cap12_pct": cap12,
            }

    def _f(v, d=0.0):
        try:
            return float(v)
        except Exception:
            return d

    final_rows = []
    policy_rows = []
    for (kk, cc, lam_init), arr in sorted(grouped.items()):
        arr_sorted = sorted(arr, key=lambda x: int(x.get("draw_idx_global", "0") or 0))
        pre = [r for r in arr_sorted if r.get("phase", "") == "pre_attack"]
        post = [r for r in arr_sorted if r.get("phase", "") == "post_attack"]

        pre_lambda = [_f(r.get("lambda_auto_ppm", ""), 0.0) for r in pre if r.get("lambda_auto_ppm", "") != ""]
        post_lambda = [_f(r.get("lambda_auto_ppm", ""), 0.0) for r in post if r.get("lambda_auto_ppm", "") != ""]
        post_weight = [_f(r.get("attacker_weight_ppm", ""), 0.0) / 1_000_000.0 for r in post if r.get("attacker_weight_ppm", "") != ""]
        post_seat = [_f(r.get("attacker_seats_share", ""), 0.0) for r in post if r.get("attacker_seats_share", "") != ""]
        if drift_active and sampler_baseline_by_tag:
            post_base = [
                sampler_baseline_by_tag[str(r.get("tag", "") or "")]["seat_share"]
                for r in post
                if str(r.get("tag", "") or "") in sampler_baseline_by_tag
            ]
        else:
            post_base = [_f(r.get("stake_share_indep", ""), 0.0) for r in post if r.get("stake_share_indep", "") != ""]

        pre_lambda_mean = (sum(pre_lambda) / len(pre_lambda)) if pre_lambda else 0.0
        post_lambda_mean = (sum(post_lambda) / len(post_lambda)) if post_lambda else 0.0

        peak_ppm = 0.0
        peak_draw = ""
        half_life = ""
        rise_time = ""
        settle_time = ""
        overshoot_pct = ""
        chatter_rms = ""
        control_effort = ""
        if post and post_lambda:
            i_peak = max(range(len(post_lambda)), key=lambda i: post_lambda[i])
            peak_ppm = post_lambda[i_peak]
            peak_draw = post[i_peak].get("draw_idx_global", "")
            if peak_ppm > 0:
                for j in range(i_peak, len(post_lambda)):
                    if post_lambda[j] <= (0.5 * peak_ppm):
                        try:
                            half_life = str(int(post[j].get("draw_idx_global", "0")) - int(post[i_peak].get("draw_idx_global", "0")))
                        except Exception:
                            half_life = ""
                        break

            # Rise time: draws from attack start (post[0]) until first time lambda >= 0.9*peak.
            if peak_ppm > 0:
                thr_rise = 0.9 * peak_ppm
                for j in range(len(post_lambda)):
                    if post_lambda[j] >= thr_rise:
                        try:
                            rise_time = str(int(post[j].get("draw_idx_global", "0")) - int(post[0].get("draw_idx_global", "0")))
                        except Exception:
                            rise_time = ""
                        break

            # Settle time: draws from peak until lambda returns to <= 0.1*peak (near-zero band).
            if peak_ppm > 0:
                thr_settle = 0.1 * peak_ppm
                for j in range(i_peak, len(post_lambda)):
                    if post_lambda[j] <= thr_settle:
                        try:
                            settle_time = str(int(post[j].get("draw_idx_global", "0")) - int(post[i_peak].get("draw_idx_global", "0")))
                        except Exception:
                            settle_time = ""
                        break

            # Overshoot: peak vs tail-mean (last 20% of post window). If tail ~ 0, report peak/scale.
            tail_n = max(1, len(post_lambda) // 5)
            tail_mean = sum(post_lambda[-tail_n:]) / tail_n
            if tail_mean > 1.0:
                overshoot_pct = f"{((peak_ppm - tail_mean) / tail_mean) * 100.0:.3f}"
            elif peak_ppm > 0:
                # Tail collapsed to ~0; absolute overshoot relative to full scale (ppm/1e6).
                overshoot_pct = f"{(peak_ppm / 1_000_000.0) * 100.0:.3f}"

            # Chatter (RMS of first differences) and control effort (integral under lambda curve).
            if len(post_lambda) >= 2:
                diffs = [post_lambda[j] - post_lambda[j - 1] for j in range(1, len(post_lambda))]
                chatter_rms = f"{math.sqrt(sum(d * d for d in diffs) / len(diffs)):.3f}"
            control_effort = f"{sum(post_lambda) / 1_000_000.0:.3f}"

        post_cmp = post_seat if drift_active else post_weight

        post_weight_mean = (sum(post_weight) / len(post_weight)) if post_weight else 0.0
        post_base_mean = (sum(post_base) / len(post_base)) if post_base else 0.0
        n = min(len(post_cmp), len(post_base))
        auc_gain = sum((post_base[i] - post_cmp[i]) for i in range(n)) if n > 0 else 0.0

        def _red_vs_baseline_pct_for(series: List[float], baseline_series: List[float], n_take: int) -> str:
            if n_take <= 0:
                return ""
            nn = min(n_take, len(series), len(baseline_series))
            if nn <= 0:
                return ""
            b = sum(baseline_series[:nn]) / nn
            w = sum(series[:nn]) / nn
            if b <= 0:
                return ""
            return f"{((b - w) / b) * 100.0:.3f}"

        def _t95_for(series: List[float], baseline_series: List[float]) -> str:
            nn = min(len(series), len(baseline_series))
            if nn <= 0:
                return ""
            bmean = sum(baseline_series[:nn]) / nn
            if bmean <= 0:
                return ""
            thr = 0.95 * bmean
            for i in range(nn):
                if series[i] >= thr:
                    try:
                        d0 = int(post[0].get("draw_idx_global", "0"))
                        di = int(post[i].get("draw_idx_global", "0"))
                        return str(di - d0 + 1)
                    except Exception:
                        return ""
            return ""

        def _binom_tail_prob(p: float, m: int, q: int) -> float:
            if m <= 0 or q <= 0:
                return 0.0
            if p <= 0:
                return 0.0
            if p >= 1:
                return 1.0
            s = 0.0
            for x in range(q, m + 1):
                s += math.comb(m, x) * (p ** x) * ((1.0 - p) ** (m - x))
            return s

        # Binary KL divergence D(a || p) in nats, with clamping to avoid log(0).
        def _binary_kl(a: float, p: float) -> float:
            eps = 1e-12
            a = min(max(a, eps), 1.0 - eps)
            p = min(max(p, eps), 1.0 - eps)
            return a * math.log(a / p) + (1.0 - a) * math.log((1.0 - a) / (1.0 - p))

        # Chernoff upper bound on Pr[X >= q] for X ~ Binomial(m, p):
        #   Pr[X >= q] <= exp(-m * D(q/m || p))  when q/m > p; else trivially <= 1.
        # Directly implements the bound from Ch2 (Theorem \ref{thm:ch2-chernoff}, binomial corollary).
        def _chernoff_upper_tail(p: float, m: int, q: int) -> float:
            if m <= 0 or q <= 0:
                return 0.0
            if q > m:
                return 0.0
            if p <= 0:
                return 0.0
            if p >= 1:
                return 1.0
            a = q / m
            if a <= p:
                return 1.0
            return math.exp(-m * _binary_kl(a, p))

        def _chernoff_mean(weights: List[float], m: int, q: int) -> float:
            if not weights:
                return 0.0
            vals = [_chernoff_upper_tail(max(0.0, min(1.0, w)), m, q) for w in weights]
            return sum(vals) / len(vals) if vals else 0.0

        def _capture_model_pct(weights: List[float], q: int) -> str:
            if not weights:
                return ""
            m = max(1, cc)
            vals = [_binom_tail_prob(max(0.0, min(1.0, w)), m, q) for w in weights]
            if not vals:
                return ""
            return f"{(sum(vals) / len(vals)) * 100.0:.3f}"

        def _capture_emp_pct_from_rows(rows_: List[dict], q: int) -> str:
            if not rows_:
                return ""
            hits = 0
            nrows = 0
            for rr in rows_:
                try:
                    seats = int(rr.get("attacker_seats", "0") or 0)
                    nrows += 1
                    if seats >= q:
                        hits += 1
                except Exception:
                    pass
            if nrows <= 0:
                return ""
            return f"{(hits / nrows) * 100.0:.3f}"

        red_full = _red_vs_baseline_pct_for(post_cmp, post_base, n)
        red_1ep = _red_vs_baseline_pct_for(post_cmp, post_base, 1 * max(1, draws_per_epoch))
        red_2ep = _red_vs_baseline_pct_for(post_cmp, post_base, 2 * max(1, draws_per_epoch))
        red_3ep = _red_vs_baseline_pct_for(post_cmp, post_base, 3 * max(1, draws_per_epoch))
        red_5ep = _red_vs_baseline_pct_for(post_cmp, post_base, 5 * max(1, draws_per_epoch))

        t95 = _t95_for(post_cmp, post_base)

        attack_draw_idx = ""
        if post:
            attack_draw_idx = post[0].get("draw_idx_global", "")

        post_seat_mean = (sum(post_seat) / len(post_seat)) if post_seat else 0.0
        q_13 = max(1, int(math.ceil(cc / 3.0)))
        q_12 = max(1, int(math.ceil(cc / 2.0)))
        # Empirical capture must use debug/full rows that carry integer attacker_seats.
        post_dbg_emp = [r for r in sorted(debug_group.get((kk, cc, lam_init), []), key=lambda x: int(x.get("draw_idx_global", "0") or 0)) if (r.get("phase", "") == "post_attack")]
        cap_emp_13 = _capture_emp_pct_from_rows(post_dbg_emp, q_13)
        cap_emp_12 = _capture_emp_pct_from_rows(post_dbg_emp, q_12)

        # Chernoff upper-bound certificates (Ch2 Theorem): compare adaptive-mix trajectory
        # (post_weight) against pure-stake baseline (post_base) for thresholds q_13, q_12.
        # chernoff_bound_reduction_*_pct gives % tightening of the certificate under adaptive.
        cb_adapt_13 = _chernoff_mean(post_weight, cc, q_13)
        cb_base_13 = _chernoff_mean(post_base, cc, q_13)
        cb_adapt_12 = _chernoff_mean(post_weight, cc, q_12)
        cb_base_12 = _chernoff_mean(post_base, cc, q_12)

        def _cb_reduction_pct(base: float, adapt: float) -> str:
            if base <= 0:
                return ""
            return f"{((base - adapt) / base) * 100.0:.3f}"

        final_rows.append(enrich_tracked_row({
            "attack_mode": attack_mode,
            "whale_share": f"{whale_share:.4f}" if is_additive else "0.0000",
            "k": str(kk),
            "committee_size": str(cc),
            "lambda_init_ppm": str(lam_init),
            "pre_draws": str(len(pre)),
            "post_draws": str(len(post)),
            "attack_draw_idx": str(attack_draw_idx),
            "draws_per_epoch_cfg": str(draws_per_epoch),
            "pre_lambda_mean_ppm": str(int(round(pre_lambda_mean))),
            "post_lambda_mean_ppm": str(int(round(post_lambda_mean))),
            "post_lambda_peak_ppm": str(int(round(peak_ppm))),
            "post_lambda_peak_draw_idx": str(peak_draw),
            "post_lambda_half_life_draws": str(half_life),
            "post_lambda_rise_time_draws": rise_time,
            "post_lambda_settle_time_draws": settle_time,
            "post_lambda_overshoot_pct": overshoot_pct,
            "post_lambda_chatter_rms_ppm": chatter_rms,
            "post_lambda_control_effort": control_effort,
            "post_attacker_weight_mean": f"{post_weight_mean:.6f}",
            "post_attacker_seat_mean": f"{post_seat_mean:.6f}",
            "post_stake_baseline_mean": f"{post_base_mean:.6f}",
            "auc_gain_vs_stake": f"{auc_gain:.6f}",
            "time_to_95pct_baseline_draws": str(t95),
            "post_capture_ge_1_3_emp_pct": cap_emp_13,
            "post_capture_ge_1_2_emp_pct": cap_emp_12,
            "post_chernoff_ge_1_3_adaptive": f"{cb_adapt_13:.6e}",
            "post_chernoff_ge_1_3_baseline": f"{cb_base_13:.6e}",
            "post_chernoff_ge_1_2_adaptive": f"{cb_adapt_12:.6e}",
            "post_chernoff_ge_1_2_baseline": f"{cb_base_12:.6e}",
            "chernoff_bound_reduction_ge_1_3_pct": _cb_reduction_pct(cb_base_13, cb_adapt_13),
            "chernoff_bound_reduction_ge_1_2_pct": _cb_reduction_pct(cb_base_12, cb_adapt_12),
            "reduction_vs_baseline_full_pct": red_full,
            "reduction_vs_baseline_1ep_pct": red_1ep,
            "reduction_vs_baseline_2ep_pct": red_2ep,
            "reduction_vs_baseline_3ep_pct": red_3ep,
            "reduction_vs_baseline_5ep_pct": red_5ep,
        }, tracked_meta))

        # Policy comparison rows for this same scenario:
        # - baseline_stake: primary reference policy (lambda=0)
        # - static_uniform_id: fixed lambda against identity-uniform baseline (u_i=1/N)
        # - adaptive: measured adaptive trajectory from chain events
        post_dbg = [r for r in sorted(debug_group.get((kk, cc, lam_init), []), key=lambda x: int(x.get("draw_idx_global", "0") or 0)) if (r.get("phase", "") == "post_attack")]
        dbg_stake = [_f(r.get("attacker_stake_ppm", ""), 0.0) / 1_000_000.0 for r in post_dbg if r.get("attacker_stake_ppm", "") != ""]
        dbg_vset_n = [_f(r.get("vset_n", ""), 0.0) for r in post_dbg if r.get("vset_n", "") != ""]
        dbg_att_ops = [_f(r.get("attacker_ops_len", ""), 0.0) for r in post_dbg if r.get("attacker_ops_len", "") != ""]

        lam_s = float(static_lambda_ppm) / 1_000_000.0
        n_static_id = min(len(dbg_stake), len(dbg_vset_n), len(dbg_att_ops), len(post_base))
        static_uniform_id_w = []
        for i in range(n_static_id):
            n_id = dbg_vset_n[i]
            att_id = dbg_att_ops[i]
            id_share = (att_id / n_id) if n_id > 0 else 0.0
            static_uniform_id_w.append(((1.0 - lam_s) * dbg_stake[i]) + (lam_s * id_share))

        def _mean(xs: List[float]) -> float:
            return (sum(xs) / len(xs)) if xs else 0.0

        if drift_active and sampler_baseline_by_tag:
            base_cap_13_vals = [
                sampler_baseline_by_tag[str(r.get("tag", "") or "")]["cap13_pct"]
                for r in post if str(r.get("tag", "") or "") in sampler_baseline_by_tag
            ]
            base_cap_12_vals = [
                sampler_baseline_by_tag[str(r.get("tag", "") or "")]["cap12_pct"]
                for r in post if str(r.get("tag", "") or "") in sampler_baseline_by_tag
            ]
            base_cap_13 = f"{((sum(base_cap_13_vals) / len(base_cap_13_vals)) if base_cap_13_vals else 0.0):.3f}"
            base_cap_12 = f"{((sum(base_cap_12_vals) / len(base_cap_12_vals)) if base_cap_12_vals else 0.0):.3f}"
        else:
            base_cap_13 = _capture_model_pct(post_base, q_13)
            base_cap_12 = _capture_model_pct(post_base, q_12)
        policy_rows.append(enrich_tracked_row({
            "k": str(kk),
            "committee_size": str(cc),
            "lambda_init_ppm": str(lam_init),
            "policy": "baseline_stake",
            "lambda_policy_ppm": "0",
            "post_draws": str(n),
            "draws_per_epoch_cfg": str(draws_per_epoch),
            "post_attacker_weight_mean": f"{post_base_mean:.6f}",
            "post_attacker_seat_mean_model": f"{post_base_mean:.6f}",
            "post_stake_baseline_mean": f"{post_base_mean:.6f}",
            "post_capture_ge_1_3_model_pct": base_cap_13,
            "post_capture_ge_1_2_model_pct": base_cap_12,
            "reduction_vs_baseline_full_pct": "0.000",
            "reduction_vs_baseline_1ep_pct": "0.000",
            "reduction_vs_baseline_2ep_pct": "0.000",
            "reduction_vs_baseline_3ep_pct": "0.000",
            "reduction_vs_baseline_5ep_pct": "0.000",
            "time_to_95pct_baseline_draws": "1",
        }, tracked_meta))

        static_uid_mean = _mean(static_uniform_id_w)
        policy_rows.append(enrich_tracked_row({
            "k": str(kk),
            "committee_size": str(cc),
            "lambda_init_ppm": str(lam_init),
            "policy": "static_uniform_id",
            "lambda_policy_ppm": str(static_lambda_ppm),
            "post_draws": str(n_static_id),
            "draws_per_epoch_cfg": str(draws_per_epoch),
            "post_attacker_weight_mean": f"{static_uid_mean:.6f}",
            "post_attacker_seat_mean_model": f"{static_uid_mean:.6f}",
            "post_stake_baseline_mean": f"{post_base_mean:.6f}",
            "post_capture_ge_1_3_model_pct": _capture_model_pct(static_uniform_id_w, q_13),
            "post_capture_ge_1_2_model_pct": _capture_model_pct(static_uniform_id_w, q_12),
            "reduction_vs_baseline_full_pct": _red_vs_baseline_pct_for(static_uniform_id_w, post_base, n_static_id),
            "reduction_vs_baseline_1ep_pct": _red_vs_baseline_pct_for(static_uniform_id_w, post_base, 1 * max(1, draws_per_epoch)),
            "reduction_vs_baseline_2ep_pct": _red_vs_baseline_pct_for(static_uniform_id_w, post_base, 2 * max(1, draws_per_epoch)),
            "reduction_vs_baseline_3ep_pct": _red_vs_baseline_pct_for(static_uniform_id_w, post_base, 3 * max(1, draws_per_epoch)),
            "reduction_vs_baseline_5ep_pct": _red_vs_baseline_pct_for(static_uniform_id_w, post_base, 5 * max(1, draws_per_epoch)),
            "time_to_95pct_baseline_draws": _t95_for(static_uniform_id_w, post_base),
        }, tracked_meta))

        policy_rows.append(enrich_tracked_row({
            "k": str(kk),
            "committee_size": str(cc),
            "lambda_init_ppm": str(lam_init),
            "policy": "adaptive",
            "lambda_policy_ppm": "auto",
            "post_draws": str(n),
            "draws_per_epoch_cfg": str(draws_per_epoch),
            "post_attacker_weight_mean": f"{post_weight_mean:.6f}",
            "post_attacker_seat_mean_model": f"{post_weight_mean:.6f}",
            "post_stake_baseline_mean": f"{post_base_mean:.6f}",
            "post_capture_ge_1_3_model_pct": _capture_model_pct(post_weight, q_13),
            "post_capture_ge_1_2_model_pct": _capture_model_pct(post_weight, q_12),
            "reduction_vs_baseline_full_pct": red_full,
            "reduction_vs_baseline_1ep_pct": red_1ep,
            "reduction_vs_baseline_2ep_pct": red_2ep,
            "reduction_vs_baseline_3ep_pct": red_3ep,
            "reduction_vs_baseline_5ep_pct": red_5ep,
            "time_to_95pct_baseline_draws": str(t95),
        }, tracked_meta))

    # Always keep full comparison table (all scenarios/rows).
    with final_table_csv.open("w", newline="", encoding="utf-8") as fft:
        fw = csv.DictWriter(fft, fieldnames=final_cols)
        fw.writeheader()
        for rr in final_rows:
            fw.writerow(rr)

    latest_final_table.write_bytes(final_table_csv.read_bytes())
    _mirror_latest_result(final_table_csv, "epoch_final_table_latest.csv")

    # Policy comparison table (baseline/static/adaptive) for each scenario.
    with final_policy_csv.open("w", newline="", encoding="utf-8") as fpt:
        pw = csv.DictWriter(fpt, fieldnames=policy_cols)
        pw.writeheader()
        for rr in policy_rows:
            pw.writerow(rr)
    latest_final_policy_table.write_bytes(final_policy_csv.read_bytes())
    _mirror_latest_result(final_policy_csv, "epoch_final_policy_table_latest.csv")

    # Optional one-row export for thesis insertion convenience.
    if final_table_single_row and final_rows:
        one = sorted(final_rows, key=lambda r: int(r.get("post_draws", "0") or 0), reverse=True)[0]
        with final_table_single_csv.open("w", newline="", encoding="utf-8") as fts:
            sw = csv.DictWriter(fts, fieldnames=final_cols)
            sw.writeheader()
            sw.writerow(one)
        latest_final_table_single.write_bytes(final_table_single_csv.read_bytes())
        _mirror_latest_result(final_table_single_csv, "epoch_final_table_single_row_latest.csv")
    else:
        try:
            if latest_final_table_single.exists():
                latest_final_table_single.unlink()
        except Exception:
            pass

    # Final per-epoch compact table (presentation-friendly, not single-row aggregated only).
    epoch_cols = [
        "tracked_entity_mode", "tracked_entity_label", "tracked_baseline_label", "baseline_comparison_mode",
        "k", "committee_size", "lambda_init_ppm", "phase", "epoch_idx",
        "draws", "mean_lambda_auto_ppm", "mean_attacker_weight_share", "mean_stake_share_indep",
        "mean_tracked_weight_share",
        "gain_vs_stake", "gain_vs_model",
    ]
    epoch_group = {}
    for r in dcore_rows:
        ph = (r.get("phase", "") or "").strip() or "unknown"
        try:
            key = (
                int(r.get("k", "0") or 0),
                int(r.get("committee_size", "0") or 0),
                int(r.get("lambda_ppm", "0") or 0),
                ph,
                int(r.get("epoch_idx", "0") or 0),
            )
        except Exception:
            continue
        epoch_group.setdefault(key, []).append(r)

    with final_epoch_csv.open("w", newline="", encoding="utf-8") as fep:
        ew = csv.DictWriter(fep, fieldnames=epoch_cols)
        ew.writeheader()
        for (kk, cc, lam_init, ph, eidx), arr in sorted(epoch_group.items()):
            lam_vals = [float(x.get("lambda_auto_ppm", "0") or 0.0) for x in arr if x.get("lambda_auto_ppm", "") != ""]
            w_vals = [float(x.get("attacker_weight_ppm", "0") or 0.0)/1_000_000.0 for x in arr if x.get("attacker_weight_ppm", "") != ""]
            b_vals = [float(x.get("stake_share_indep", "0") or 0.0) for x in arr if x.get("stake_share_indep", "") != ""]
            g_stake_vals = [float(x.get("seat_minus_stake_indep", "0") or 0.0) for x in arr if x.get("seat_minus_stake_indep", "") != ""]
            g_model_vals = [float(x.get("seat_minus_weight", "0") or 0.0) for x in arr if x.get("seat_minus_weight", "") != ""]

            mlam = (sum(lam_vals)/len(lam_vals)) if lam_vals else 0.0
            mw = (sum(w_vals)/len(w_vals)) if w_vals else 0.0
            mb = (sum(b_vals)/len(b_vals)) if b_vals else 0.0
            mgs = (sum(g_stake_vals)/len(g_stake_vals)) if g_stake_vals else 0.0
            mgm = (sum(g_model_vals)/len(g_model_vals)) if g_model_vals else 0.0

            ew.writerow(enrich_tracked_row({
                "k": str(kk),
                "committee_size": str(cc),
                "lambda_init_ppm": str(lam_init),
                "phase": ph,
                "epoch_idx": str(eidx),
                "draws": str(len(arr)),
                "mean_lambda_auto_ppm": str(int(round(mlam))),
                "mean_attacker_weight_share": f"{mw:.6f}",
                "mean_stake_share_indep": f"{mb:.6f}",
                "gain_vs_stake": f"{mgs:.6f}",
                "gain_vs_model": f"{mgm:.6f}",
            }, tracked_meta))

    latest_final_epoch_table.write_bytes(final_epoch_csv.read_bytes())
    _mirror_latest_result(final_epoch_csv, "epoch_final_epoch_table_latest.csv")

    # Plots from summary CSV
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))

    def plot_metric(metric_col: str, out_name: str, ylabel: str):
        plt.figure(figsize=(8.0, 4.6))
        keys = sorted(set((int(r["k"]), int(r.get("committee_size", "0")), int(r["lambda_ppm"])) for r in rows))
        for k, c, lam in keys:
            xs = [int(r["epoch_idx"]) for r in rows if int(r["k"]) == k and int(r.get("committee_size", "0")) == c and int(r["lambda_ppm"]) == lam]
            ys = [float(r[metric_col]) for r in rows if int(r["k"]) == k and int(r.get("committee_size", "0")) == c and int(r["lambda_ppm"]) == lam and r[metric_col] != ""]
            if xs and ys:
                plt.plot(xs[:len(ys)], ys, label=f"k={k}, c={c}, λ={lam/1_000_000:.2f}")
        plt.xlabel("post-attack epoch")
        plt.ylabel(ylabel)
        plt.title(f"Epoch simulation: {ylabel} vs epoch")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        out = art_dir / "plots" / out_name
        plt.savefig(out, dpi=170, bbox_inches="tight")
        return out

    p1 = plot_metric("mean_attacker_share", f"epoch_attacker_seat_share_{run_id}.png", f"mean {tracked_entity_label} seats share")

    def plot_weight_with_baseline(out_name: str):
        plt.figure(figsize=(8.0, 4.8))
        keys = sorted(set((int(r["k"]), int(r.get("committee_size", "0")), int(r["lambda_ppm"])) for r in rows))
        for k, c, lam in keys:
            srows = [r for r in rows if int(r["k"]) == k and int(r.get("committee_size", "0")) == c and int(r["lambda_ppm"]) == lam]
            xs = [int(r["epoch_idx"]) for r in srows if r.get("mean_attacker_weight_share", "") != ""]
            ys = [float(r["mean_attacker_weight_share"]) for r in srows if r.get("mean_attacker_weight_share", "") != ""]
            bs = [float(r.get("mean_stake_share_indep", "0") or 0.0) for r in srows if r.get("mean_attacker_weight_share", "") != ""]
            if not xs or not ys:
                continue

            lbl = f"k={k}, c={c}, λ_init={lam/1_000_000:.2f}"
            plt.plot(xs[:len(ys)], ys, label=lbl)

            # Baseline (stake-only) and win/loss shading.
            if bs:
                b0 = sum(bs) / len(bs)
                xb = xs[:len(ys)]
                yb = [b0 for _ in xb]
                plt.plot(xb, yb, linestyle="--", alpha=0.6, label=f"{tracked_baseline_label} k={k},c={c}")

                # Green area: adaptive better than baseline.
                plt.fill_between(xb, ys, yb, where=[y <= b for y, b in zip(ys, yb)], alpha=0.18, color="green")
                # Red area: adaptive worse than baseline.
                plt.fill_between(xb, ys, yb, where=[y > b for y, b in zip(ys, yb)], alpha=0.12, color="red")

        plt.xlabel("post-attack epoch")
        plt.ylabel(f"mean {tracked_entity_label} model weight share")
        plt.title(f"{tracked_entity_label_title} model weight: adaptive vs {tracked_baseline_label}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        out = art_dir / "plots" / out_name
        plt.savefig(out, dpi=170, bbox_inches="tight")
        return out

    p2 = plot_weight_with_baseline(f"epoch_attacker_weight_share_{run_id}.png")
    p3 = plot_metric("mean_seat_minus_stake_indep", f"epoch_attacker_gain_vs_stake_{run_id}.png", f"{tracked_entity_label} gain vs {tracked_baseline_label}")
    p4 = plot_metric("mean_seat_minus_weight", f"epoch_attacker_gain_vs_weight_{run_id}.png", f"{tracked_entity_label} gain vs model weight")

    # Draw-index chart (no epoch concept).
    p7 = art_dir / "plots" / f"draw_attacker_weight_share_{run_id}.png"
    drows = list(csv.DictReader(draw_summary_csv.open("r", encoding="utf-8")))
    plt.figure(figsize=(8.0, 4.6))
    dkeys = sorted(set((int(r["k"]), int(r["committee_size"]), int(r["lambda_ppm"])) for r in drows))
    for k, c, lam in dkeys:
        xs = [int(r["draw_idx_post_attack"]) for r in drows if int(r["k"]) == k and int(r["committee_size"]) == c and int(r["lambda_ppm"]) == lam]
        ys = [float(r["mean_attacker_weight_share"]) for r in drows if int(r["k"]) == k and int(r["committee_size"]) == c and int(r["lambda_ppm"]) == lam]
        if xs and ys:
            plt.plot(xs, ys, label=f"k={k}, c={c}, λ_init={lam/1_000_000:.2f}")
    plt.xlabel("draw index since attack")
    plt.ylabel(f"mean {tracked_entity_label} weight share")
    plt.title(f"Post-injection: {tracked_entity_label} weight share vs draw index")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.savefig(p7, dpi=170, bbox_inches="tight")

    # Draw-index charts for adaptive controller internals.
    p8 = art_dir / "plots" / f"draw_lambda_gini_{run_id}.png"
    p9 = art_dir / "plots" / f"draw_lambda_auto_trace_{run_id}.png"
    plt.figure(figsize=(8.2, 4.8))
    dcore = list(csv.DictReader(draws_csv.open("r", encoding="utf-8")))
    x8 = [int(r.get("draw_idx_post_attack", "0") or 0) for r in dcore if (r.get("draw_idx_post_attack", "") != "")]
    ylam = [float(r.get("lambda_auto_ppm", "0") or 0) / 1_000_000 for r in dcore if (r.get("draw_idx_post_attack", "") != "")]
    ygini = [float(r.get("gini_ppm", "0") or 0) / 1_000_000 for r in dcore if (r.get("draw_idx_post_attack", "") != "")]
    if x8 and ylam:
        plt.plot(x8[:len(ylam)], ylam, label="lambda_auto")
    if x8 and ygini:
        plt.plot(x8[:len(ygini)], ygini, label="gini")
    plt.xlabel("draw index since attack")
    plt.ylabel("value")
    plt.title("Adaptive controller signals vs draw index")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.savefig(p8, dpi=170, bbox_inches="tight")

    # Lambda-only per-tx trace from genesis (global draw index): full + post-attack zoom.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.8), sharex=False, constrained_layout=True)
    x9 = [int(r.get("draw_idx_global", "0") or 0) for r in dcore if (r.get("draw_idx_global", "") != "")]
    y9 = [float(r.get("lambda_auto_ppm", "0") or 0) for r in dcore if (r.get("draw_idx_global", "") != "")]
    xa = [int(r.get("draw_idx_global", "0") or 0) for r in dcore if (r.get("attack_height", "") != "")]

    def _smooth(vals, w=3):
        if not vals or w <= 1:
            return vals
        out = []
        for i in range(len(vals)):
            j0 = max(0, i-w+1)
            seg = vals[j0:i+1]
            out.append(sum(seg)/len(seg))
        return out

    x_attack = min(xa) if xa else None
    peak_txt = ""

    if x9 and y9:
        ys = _smooth(y9, w=3)
        ax1.plot(x9[:len(ys)], ys, linewidth=2.0, label="lambda_auto")
        if x_attack is not None:
            ax1.axvline(x=x_attack, linestyle="--", alpha=0.6, label=f"attack@draw={x_attack}")

        # Trim full-view axis to informative region (drop long terminal zero tail).
        nz = [x for x, y in zip(x9, y9) if y > 0]
        if nz:
            x_last_nz = max(nz)
            x_min = max(0, min(x9) - 2)
            x_max = min(max(x9), x_last_nz + 20)
            if x_max > x_min:
                ax1.set_xlim(x_min, x_max)

        # Summary stats for caption-like annotation.
        y_peak = max(y9)
        i_peak = y9.index(y_peak)
        x_peak = x9[i_peak]
        x_half = None
        for x, y in zip(x9[i_peak:], y9[i_peak:]):
            if y <= (0.5 * y_peak):
                x_half = x
                break
        half_life = (x_half - x_peak) if x_half is not None else None
        peak_txt = f"peak={int(y_peak)} ppm @draw={x_peak}; half-life={'n/a' if half_life is None else str(half_life)+' draws'}"

    ax1.set_ylabel("lambda_auto_ppm")
    ax1.set_title("Adaptive λ trace (full run)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    if peak_txt:
        ax1.text(0.01, 0.97, peak_txt, transform=ax1.transAxes, va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"))

    if x9 and y9:
        if x_attack is not None:
            post = [(x, y) for x, y in zip(x9, y9) if x >= x_attack]
        else:
            post = list(zip(x9, y9))
        if post:
            xpost = [p[0] for p in post]
            ypost = [p[1] for p in post]
            yps = _smooth(ypost, w=3)
            ax2.plot(xpost[:len(yps)], yps, linewidth=2.0, label="lambda_auto (post)")
            if x_attack is not None:
                ax2.set_xlim(x_attack, x_attack + 200)
                ax2.axvline(x=x_attack, linestyle="--", alpha=0.6)
            elif xpost:
                ax2.set_xlim(min(xpost), min(max(xpost), min(xpost) + 200))
            ymax = max(ypost) if ypost else 1
            ax2.set_ylim(0, max(1.0, ymax*1.10))

    ax2.set_xlabel("global draw index (from genesis)")
    ax2.set_ylabel("lambda_auto_ppm")
    ax2.set_title("Adaptive λ trace (post-attack zoom)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.savefig(p9, dpi=170, bbox_inches="tight")

    # Post-attack tracked-entity residual vs model weight (smoothed + cumulative, post-only).
    p10 = art_dir / "plots" / f"draw_attacker_gain_vs_weight_full_{run_id}.png"
    plt.figure(figsize=(8.6, 4.9))
    post_pairs = []
    for r in dcore:
        if (r.get("phase", "") or "") != "post_attack":
            continue
        xs = r.get("draw_idx_post_attack", "")
        if xs == "":
            continue
        try:
            x = int(xs)
            y = float(r.get("seat_minus_weight", "0") or 0.0)
        except Exception:
            continue
        post_pairs.append((x, y))
    post_pairs.sort(key=lambda t: t[0])
    x10 = [p[0] for p in post_pairs]
    y10 = [p[1] for p in post_pairs]

    def _rolling_mean(vals, w=50):
        if not vals:
            return vals
        out = []
        for i in range(len(vals)):
            j0 = max(0, i-w+1)
            seg = vals[j0:i+1]
            out.append(sum(seg)/len(seg))
        return out

    def _cumulative_mean(vals):
        out = []
        s = 0.0
        for i, v in enumerate(vals, start=1):
            s += v
            out.append(s / i)
        return out

    if x10 and y10:
        w_rm = max(5, min(50, 2 * max(1, draws_per_epoch)))
        y_rm = _rolling_mean(y10, w=w_rm)
        y_cm = _cumulative_mean(y10)
        plt.plot(x10[:len(y_rm)], y_rm, linewidth=2.0, label=f"rolling mean (post, w={w_rm})")
        plt.plot(x10[:len(y_cm)], y_cm, linewidth=1.8, alpha=0.9, label="cumulative mean (post)")
        plt.axhline(y=0.0, linestyle="--", alpha=0.5, color="gray")
        plt.axvline(x=1, linestyle="--", alpha=0.45, label="post-attack start")

    plt.xlabel("draw index since attack")
    plt.ylabel("seat share - model weight")
    plt.title(f"{tracked_entity_label_title} seat-model residual (post-attack only, smoothed)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.savefig(p10, dpi=170, bbox_inches="tight")

    # Scenario comparison bars: tracked-entity baseline comparison in early windows (epoch-based).
    # For concentration-drift runs these reduction-vs-baseline bars are degenerate
    # at k=0 and should not be emitted as meaningful artifacts.
    p11 = art_dir / "plots" / f"scenario_reduction_vs_baseline_bars_{run_id}.png"
    if scenario_kind == "concentration_drift":
        print("[drift] skipped scenario_reduction_vs_baseline_bars (degenerate at k=0)")
        p11 = None
    elif final_rows:
        try:
            rows_sorted = sorted(
                final_rows,
                key=lambda r: (int(r.get("k", "0") or 0), int(r.get("committee_size", "0") or 0), int(r.get("lambda_init_ppm", "0") or 0)),
            )
            scen_labels = [f"k={r['k']},c={r['committee_size']}" for r in rows_sorted]
            series = [
                ("1 ep", "reduction_vs_baseline_1ep_pct"),
                ("2 ep", "reduction_vs_baseline_2ep_pct"),
                ("3 ep", "reduction_vs_baseline_3ep_pct"),
                ("5 ep", "reduction_vs_baseline_5ep_pct"),
                ("full post", "reduction_vs_baseline_full_pct"),
            ]
            xs = list(range(len(rows_sorted)))
            bw = 0.82 / max(1, len(series))
            plt.figure(figsize=(10.2, 5.2))
            for j, (sname, scol) in enumerate(series):
                vals = []
                for r in rows_sorted:
                    try:
                        vals.append(float(r.get(scol, "") or 0.0))
                    except Exception:
                        vals.append(0.0)
                offs = [x - 0.41 + (j + 0.5) * bw for x in xs]
                plt.bar(offs, vals, width=bw, label=sname)
            plt.xticks(xs, scen_labels, rotation=20)
            plt.ylabel(tracked_vs_baseline_axis_label)
            plt.xlabel("scenario")
            plt.title(f"Early-window {tracked_entity_label} baseline comparison by scenario (1 epoch = {draws_per_epoch} draws)")
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend(fontsize=8, ncol=3)
            plt.tight_layout()
            plt.savefig(p11, dpi=170, bbox_inches="tight")
        finally:
            try:
                plt.close()
            except Exception:
                pass
    else:
        p11 = None

    # Policy bars (baseline/static/adaptive): early (1 epoch) vs full-post reduction.
    p12 = art_dir / "plots" / f"policy_reduction_vs_baseline_bars_{run_id}.png"
    if scenario_kind == "concentration_drift":
        print("[drift] skipped policy_reduction_vs_baseline_bars (degenerate at k=0)")
        p12 = None
    elif policy_rows:
        try:
            pref = [r for r in policy_rows if r.get("policy") == "adaptive"]
            scen_keys = sorted({(int(r.get("k", "0") or 0), int(r.get("committee_size", "0") or 0), str(r.get("lambda_init_ppm", "0"))) for r in pref})
            scen_labels = [f"k={k},c={c}" for (k, c, _lam) in scen_keys]
            policy_order = ["baseline_stake", "static_uniform_id", "adaptive"]
            policy_names = {
                "baseline_stake": "baseline",
                "static_uniform_id": f"static uniform-id λ={static_lambda_ppm/1_000_000:.2f}",
                "adaptive": "adaptive",
            }

            def _lookup(policy: str, k: int, c: int, lam: str, col: str) -> float:
                for r in policy_rows:
                    if r.get("policy") == policy and int(r.get("k", "0") or 0) == k and int(r.get("committee_size", "0") or 0) == c and str(r.get("lambda_init_ppm", "0")) == lam:
                        try:
                            return float(r.get(col, "") or 0.0)
                        except Exception:
                            return 0.0
                return 0.0

            xs = list(range(len(scen_keys)))
            bw = 0.78 / max(1, len(policy_order))
            fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10.2, 7.0), sharex=True, constrained_layout=True)
            for j, pol in enumerate(policy_order):
                vals_1ep = [_lookup(pol, k, c, lam, "reduction_vs_baseline_1ep_pct") for (k, c, lam) in scen_keys]
                vals_full = [_lookup(pol, k, c, lam, "reduction_vs_baseline_full_pct") for (k, c, lam) in scen_keys]
                offs = [x - 0.39 + (j + 0.5) * bw for x in xs]
                ax_a.bar(offs, vals_1ep, width=bw, label=policy_names.get(pol, pol))
                ax_b.bar(offs, vals_full, width=bw, label=policy_names.get(pol, pol))

            ax_a.set_ylabel(tracked_vs_baseline_axis_label)
            ax_a.set_title(f"Early window (1 epoch = {draws_per_epoch} draws)")
            ax_a.grid(True, axis="y", alpha=0.3)
            ax_a.legend(fontsize=8, ncol=3)

            ax_b.set_ylabel(tracked_vs_baseline_axis_label)
            ax_b.set_title("Full post-attack window")
            ax_b.grid(True, axis="y", alpha=0.3)
            ax_b.set_xticks(xs)
            ax_b.set_xticklabels(scen_labels, rotation=20)
            ax_b.set_xlabel("scenario")

            plt.savefig(p12, dpi=170, bbox_inches="tight")
        finally:
            try:
                plt.close()
            except Exception:
                pass
    else:
        p12 = None

    # Early-window bar chart across lambda values.
    # Skip when only one initial lambda exists (adaptive run: chart is non-informative).
    p5 = art_dir / "plots" / f"early_ab_attacker_weight_{run_id}.png"
    if len(lambda_vals) > 1:
        try:
            plt.figure(figsize=(8.0, 4.8))
            groups = sorted(by_k.keys())
            lam_all = sorted({lam for _kc, lmap in by_k.items() for lam in lmap.keys()})
            if groups and lam_all:
                xs = list(range(len(groups)))
                w = 0.78 / max(1, len(lam_all))
                for j, lam in enumerate(lam_all):
                    vals = []
                    for g in groups:
                        rec = by_k[g].get(lam)
                        vals.append(rec["mean_attacker_weight_share_overall"] if rec else 0.0)
                    offs = [x - 0.39 + (j + 0.5) * w for x in xs]
                    plt.bar(offs, vals, width=w, label=f"λ={lam/1_000_000:.2f}")

                labels = [f"k={kk}, c={cc}" for (kk, cc) in groups]
                plt.xticks(xs, labels)
                plt.ylabel(f"mean {tracked_entity_label} weight share (early window)")
                plt.title(f"Early-window: {tracked_entity_label} weight share across λ")
                plt.grid(True, axis="y", alpha=0.3)
                plt.legend(fontsize=8, ncol=2)
                plt.savefig(p5, dpi=170, bbox_inches="tight")
            else:
                plt.close()
                p5 = None
        finally:
            try:
                plt.close()
            except Exception:
                pass
    else:
        p5 = None

    # Stake-over-draw plot removed by request (stake is fixed in this experiment).

    (art_dir / "plots" / "epoch_attacker_seat_share_latest.png").write_bytes(p1.read_bytes())
    _mirror_latest_plot(p1, "epoch_attacker_seat_share_latest.png")
    (art_dir / "plots" / "epoch_attacker_weight_share_latest.png").write_bytes(p2.read_bytes())
    _mirror_latest_plot(p2, "epoch_attacker_weight_share_latest.png")
    (art_dir / "plots" / "epoch_attacker_gain_vs_stake_latest.png").write_bytes(p3.read_bytes())
    _mirror_latest_plot(p3, "epoch_attacker_gain_vs_stake_latest.png")
    (art_dir / "plots" / "epoch_attacker_gain_vs_weight_latest.png").write_bytes(p4.read_bytes())
    _mirror_latest_plot(p4, "epoch_attacker_gain_vs_weight_latest.png")
    if p5 and p5.exists():
        (art_dir / "plots" / "early_ab_attacker_weight_latest.png").write_bytes(p5.read_bytes())
        _mirror_latest_plot(p5, "early_ab_attacker_weight_latest.png")
    (art_dir / "plots" / "draw_attacker_weight_share_latest.png").write_bytes(p7.read_bytes())
    _mirror_latest_plot(p7, "draw_attacker_weight_share_latest.png")
    (art_dir / "plots" / "draw_lambda_gini_latest.png").write_bytes(p8.read_bytes())
    _mirror_latest_plot(p8, "draw_lambda_gini_latest.png")
    (art_dir / "plots" / "draw_lambda_auto_trace_latest.png").write_bytes(p9.read_bytes())
    _mirror_latest_plot(p9, "draw_lambda_auto_trace_latest.png")
    (art_dir / "plots" / "draw_attacker_gain_vs_weight_full_latest.png").write_bytes(p10.read_bytes())
    _mirror_latest_plot(p10, "draw_attacker_gain_vs_weight_full_latest.png")
    if p11 and p11.exists():
        (art_dir / "plots" / "scenario_reduction_vs_baseline_bars_latest.png").write_bytes(p11.read_bytes())
        _mirror_latest_plot(p11, "scenario_reduction_vs_baseline_bars_latest.png")
    if p12 and p12.exists():
        (art_dir / "plots" / "policy_reduction_vs_baseline_bars_latest.png").write_bytes(p12.read_bytes())
        _mirror_latest_plot(p12, "policy_reduction_vs_baseline_bars_latest.png")

    print(str(art_dir))
    if art_latest_dir != art_dir:
        print(str(art_latest_dir))
        print(str(latest_run_dir_txt))
    print(str(manifest_json))
    print(str(draws_csv))
    print(str(draws_debug_csv))
    print(str(validator_metrics_csv))
    print(str(summary_csv))
    print(str(draw_summary_csv))
    print(str(compare_csv))
    print(str(final_table_csv))
    print(str(final_policy_csv))
    if final_table_single_row and final_table_single_csv.exists():
        print(str(final_table_single_csv))
    print(str(final_epoch_csv))
    print(str(p1))
    print(str(p2))
    print(str(p3))
    print(str(p4))
    print(str(p7))
    print(str(p8))
    print(str(p9))
    print(str(p10))
    if p11:
        print(str(p11))
    if p12:
        print(str(p12))
    if p5:
        print(str(p5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
