#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snapshot collector for Sui + Cosmos-SDK chains, with progress & debug logging.

Outputs (per UTC date folder):
  - data/YYYY-MM-DD/summary.csv
  - data/YYYY-MM-DD/<chain>.jsonl
  - data/YYYY-MM-DD/<chain>_weights.csv   (ranked validator shares for simulations)

Logging:
  - INFO: high-level progress, always printed to stdout
  - DEBUG: detailed steps, enabled with env DEBUG=1
"""

import os, sys, json, csv, time
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import requests

# ----------------------------- configuration --------------------------------
UA = {"User-Agent": "consensus-snapshots/1.3 (+progress+debug)"}
TIMEOUT = 30
RETRIES = 2
BACKOFF = 1.6
DEBUG = os.environ.get("DEBUG", "0") == "1"

CHAINS = [
    {"key": "sui",        "family": "sui",    "rpc": "https://fullnode.mainnet.sui.io:443"},

    # Cosmos SDK family (already supported)
    {"key": "cosmoshub",  "family": "cosmos"},
    {"key": "osmosis",    "family": "cosmos"},
    {"key": "injective",  "family": "cosmos"},
    {"key": "celestia",   "family": "cosmos"},
    {"key": "sei",        "family": "cosmos"},

    # NEW: NEAR
    # override with env NEAR_RPC if needed
    {"key": "near",       "family": "near",   "rpc": os.environ.get("NEAR_RPC", "https://rpc.mainnet.near.org")},

    # NEW: Avalanche (primary network validators via Info API)
    # override with env AVAX_INFO_URL if needed
    {"key": "avalanche",  "family": "avax",   "rpc": os.environ.get("AVAX_PCHAIN_URL", "https://api.avax.network/ext/bc/P")},
    {"key": "tezos",      "family": "tezos",  "rpc": os.environ.get("TEZOS_API_URL", "https://api.tzkt.io")},
    {"key": "solana",     "family": "solana", "rpc": os.environ.get("SOLANA_RPC", "https://api.mainnet-beta.solana.com")}
]
COSMOS_DIRECTORY_BASE = "https://cosmos.directory"

# Optional per-chain overrides (JSON string in env), e.g.:
# ENDPOINT_OVERRIDES_JSON={"injective":"https://sentry.lcd.injective.network:443"}
ENDPOINT_OVERRIDES = {}
try:
    ENDPOINT_OVERRIDES = json.loads(os.environ.get("ENDPOINT_OVERRIDES_JSON", "{}"))
except Exception:
    print("[WARN] Failed to parse ENDPOINT_OVERRIDES_JSON; ignoring.", file=sys.stderr)

# Human-decimal mapping (used for bonded_tokens_human where applicable)
DECIMALS = {
    "cosmoshub": 6, "osmosis": 6, "celestia": 6, "sei": 6, "injective": 18,
    "near": 24,        # yoctoNEAR
    "avalanche": 9,    # nAVAX (stake amounts returned in nano-AVAX units)
    "tezos": 6,        # mutez
    "solana": 9,       # lamports
}

# ----------------------------- logging helpers ------------------------------
def log(level: str, msg: str):
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(f"[{ts}] {level} {msg}")

def dbg(msg: str):
    if DEBUG:
        log("DEBUG", msg)

# ----------------------------- HTTP helpers ---------------------------------
_session = requests.Session()
_session.headers.update(UA)

def http_get_json(url: str, params: dict = None, timeout: int = TIMEOUT) -> dict:
    last_err = None
    for attempt in range(1, RETRIES + 2):
        try:
            dbg(f"GET {url} params={params} attempt={attempt}")
            r = _session.get(url, params=params, timeout=timeout)
            dbg(f" <- {r.status_code} {r.reason}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt <= RETRIES:
                sleep_s = BACKOFF ** (attempt - 1)
                dbg(f"   retrying in {sleep_s:.1f}s due to: {e}")
                time.sleep(sleep_s)
    raise last_err

def http_post_json(url: str, body: dict, timeout: int = TIMEOUT) -> dict:
    last_err = None
    for attempt in range(1, RETRIES + 2):
        try:
            dbg(f"POST {url} attempt={attempt}")
            r = _session.post(url, json=body, timeout=timeout)
            dbg(f" <- {r.status_code} {r.reason}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt <= RETRIES:
                sleep_s = BACKOFF ** (attempt - 1)
                dbg(f"   retrying in {sleep_s:.1f}s due to: {e}")
                time.sleep(sleep_s)
    raise last_err

# ----------------------------- metric helpers -------------------------------
def gini(shares: List[float]) -> float:
    n = len(shares)
    if n == 0:
        return 0.0
    xs = sorted(s for s in shares if s > 0.0)
    s = sum(xs)
    if s <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n

def hhi(shares: List[float]) -> float:
    return sum(s * s for s in shares if s > 0.0)

def k_for_threshold(sorted_desc: List[float], thr: float) -> int:
    s = 0.0
    for i, v in enumerate(sorted_desc, 1):
        s += v
        if s >= thr:
            return i
    return len(sorted_desc)

def top_share(sorted_desc: List[float], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(sorted_desc[:min(k, len(sorted_desc))])

def to_human_amount(raw: str, decimals: int) -> str:
    try:
        val = int(raw)
        return f"{val / (10 ** decimals):.6f}"
    except Exception:
        return ""

# ----------------------------- Sui fetchers ---------------------------------
def fetch_sui_committee(rpc_url: str) -> Dict[str, Any]:
    log("INFO", "Sui: fetching committee via JSON-RPC …")
    body = {"jsonrpc": "2.0", "id": 1, "method": "suix_getCommitteeInfo", "params": []}
    j = http_post_json(rpc_url, body)
    res = j.get("result")
    if not res:
        raise RuntimeError(f"Sui RPC {rpc_url} returned no 'result'. Payload: {j}")
    epoch = int(res["epoch"])
    vals, shares_raw = [], []
    for pk, power in res["validators"]:
        p_int = int(power)
        vals.append({
            "validator_id": pk,
            "voting_power": p_int,
            "stake_tokens": None,
            "commission": None,
            "jailed": False,
        })
        shares_raw.append(p_int)
    log("INFO", f"Sui: epoch={epoch}, validators={len(vals)}")
    return {"epoch_or_height": epoch, "validators": vals, "shares_raw": shares_raw}

# ----------------------------- Cosmos fetchers ------------------------------
def rest_proxy_fallback(chain_key: str) -> str:
    return f"https://rest.cosmos.directory/{chain_key}"

def is_lcd_healthy(lcd_base: str, errors: List[str]) -> bool:
    try:
        url = f"{lcd_base}/cosmos/base/tendermint/v1beta1/node_info"
        j = http_get_json(url, timeout=10)
        ok = isinstance(j, dict) and ("default_node_info" in json.dumps(j))
        if not ok:
            dbg(f"health-check response (truncated): {str(j)[:200]}")
        return ok
    except Exception as e:
        errors.append(f"{lcd_base}: {e}")
        return False

def autodiscover_lcd(chain_key: str) -> Tuple[Optional[str], List[str], List[str]]:
    log("INFO", f"{chain_key}: resolving LCD endpoint …")
    tried, errors = [], []

    # Special candidates for certain networks
    special = []
    if chain_key == "injective":
        special = ["https://sentry.lcd.injective.network:443"]

    # 1) per-chain env override: LCD_<CHAIN>
    env_key = f"LCD_{chain_key.upper()}"
    if os.environ.get(env_key):
        cand = os.environ[env_key].rstrip("/")
        tried.append(cand)
        if is_lcd_healthy(cand, errors):
            log("INFO", f"{chain_key}: using LCD from env {env_key}: {cand}")
            return cand, tried, errors
        dbg(f"{chain_key}: env LCD not healthy: {cand}")

    # 2) JSON overrides map
    if chain_key in ENDPOINT_OVERRIDES:
        cand = str(ENDPOINT_OVERRIDES[chain_key]).rstrip("/")
        tried.append(cand)
        if is_lcd_healthy(cand, errors):
            log("INFO", f"{chain_key}: using LCD from overrides: {cand}")
            return cand, tried, errors
        dbg(f"{chain_key}: override LCD not healthy: {cand}")

    # 3) special candidates
    for cand in special:
        tried.append(cand)
        if is_lcd_healthy(cand, errors):
            log("INFO", f"{chain_key}: using LCD special: {cand}")
            return cand, tried, errors

    # 4) cosmos.directory <chain>.json discovery
    try:
        url = f"{COSMOS_DIRECTORY_BASE}/{chain_key}.json"
        j = http_get_json(url)
        lcds = [x.get("address", "").rstrip("/") for x in j.get("apis", {}).get("rest", []) if x.get("address")]
        dbg(f"{chain_key}: cosmos.directory candidates={lcds}")
        for cand in lcds:
            tried.append(cand)
            if is_lcd_healthy(cand, errors):
                log("INFO", f"{chain_key}: using LCD from cosmos.directory: {cand}")
                return cand, tried, errors
    except Exception as e:
        errors.append(f"cosmos.directory fetch failed: {e}")

    # 5) proxy fallback
    fb = rest_proxy_fallback(chain_key).rstrip("/")
    tried.append(fb)
    if is_lcd_healthy(fb, errors):
        log("INFO", f"{chain_key}: using LCD proxy fallback: {fb}")
        return fb, tried, errors

    return None, tried, errors

def fetch_cosmos_validators_paged(lcd_base: str) -> Tuple[List[Dict[str, Any]], int]:
    url = f"{lcd_base}/cosmos/staking/v1beta1/validators"
    out, next_key, page = [], None, 0
    while True:
        page += 1
        params = {"status": "BOND_STATUS_BONDED", "pagination.limit": "200"}
        if next_key:
            params["pagination.key"] = next_key
        j = http_get_json(url, params=params)
        vals = j.get("validators", [])
        out.extend(vals)
        next_key = (j.get("pagination") or {}).get("next_key")
        log("INFO", f"Cosmos: validators page {page}: got={len(vals)} total={len(out)} next_key={'yes' if next_key else 'no'}")
        if not next_key:
            break
    return out, page

def fetch_staking_params(lcd_base: str) -> dict:
    j = http_get_json(f"{lcd_base}/cosmos/staking/v1beta1/params")
    return j.get("params", {}) if isinstance(j, dict) else {}

def fetch_staking_pool(lcd_base: str) -> dict:
    j = http_get_json(f"{lcd_base}/cosmos/staking/v1beta1/pool")
    return j.get("pool", {}) if isinstance(j, dict) else {}

def fetch_cosmos(lcd_base: str):
    log("INFO", f"Cosmos: querying validator set and staking data from {lcd_base} …")
    vs = http_get_json(f"{lcd_base}/cosmos/base/tendermint/v1beta1/validatorsets/latest")
    height = int(vs["block_height"])
    vp_list = [int(v.get("voting_power", 0)) for v in vs.get("validators", [])]

    vals_raw, pages = fetch_cosmos_validators_paged(lcd_base)
    vals = [{
        "validator_id": v.get("operator_address"),
        "stake_tokens": int(v.get("tokens","0")),
        "commission": float(v.get("commission",{}).get("commission_rates",{}).get("rate","0")),
        "jailed": bool(v.get("jailed", False)),
    } for v in vals_raw]

    params = fetch_staking_params(lcd_base)
    pool = fetch_staking_pool(lcd_base)

    log("INFO", f"Cosmos: height={height} bonded_total={len(vals)} vp_entries={len(vp_list)} "
               f"pages={pages} max_validators={params.get('max_validators')} "
               f"bonded_tokens={pool.get('bonded_tokens')}")
    return height, vals, vp_list, params, pool

# ----------------------------- NEAR fetcher ----------------------------------
def fetch_near_validators(rpc_url: str) -> Dict[str, Any]:
    log("INFO", "NEAR: fetching 'validators' via JSON-RPC …")
    body = {"jsonrpc":"2.0","id":"snap","method":"validators","params":[None]}
    j = http_post_json(rpc_url, body)
    res = j.get("result", {})
    current = res.get("current_validators", []) or []
    epoch_height = res.get("epoch_height", 0)
    vals, shares_raw = [], []
    for v in current:
        # stake is a decimal string in yoctoNEAR
        stake = int(v.get("stake", "0"))
        if stake <= 0:
            continue
        vals.append({
            "validator_id": v.get("account_id", ""),
            "stake_tokens": stake,
            "commission": None,
            "jailed": False,
        })
        shares_raw.append(stake)
    log("INFO", f"NEAR: epoch_height={epoch_height} validators={len(vals)}")
    return {"epoch_or_height": int(epoch_height), "validators": vals, "shares_raw": shares_raw}

# ----------------------------- Avalanche fetcher -----------------------------
# ----------------------------- Avalanche (P-chain) ---------------------------
AVAX_PRIMARY_SUBNET_ID = "11111111111111111111111111111111LpoYY"

def fetch_avax_validators(pchain_url: str) -> Dict[str, Any]:
    """
    Avalanche P-chain API:
      POST {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "platform.getCurrentValidators",
        "params": { }  # omit subnetID -> Primary Network
      }
    For Primary Network the response includes 'weight' (validator stake) and
    'delegatorWeight' (sum of delegations). Use total = weight + delegatorWeight.
    """
    log("INFO", "Avalanche: fetching current validators via P-chain …")
    body = {"jsonrpc": "2.0", "id": 1, "method": "platform.getCurrentValidators", "params": {}}
    j = http_post_json(pchain_url, body)

    if "error" in j:
        err = j.get("error", {})
        raise RuntimeError(f"P-chain RPC error: code={err.get('code')} msg={err.get('message')}")

    res = j.get("result", {})
    validators = res.get("validators", []) or []
    if not isinstance(validators, list):
        raise RuntimeError(f"P-chain returned unexpected payload: {str(j)[:200]}")

    vals, shares_raw = [], []
    for v in validators:
        # fields are strings; default to 0 if missing
        w  = int(v.get("weight", "0"))
        dw = int(v.get("delegatorWeight", "0"))
        total = w + dw
        if total <= 0:
            continue
        node_id = v.get("nodeID", "")
        vals.append({
            "validator_id": node_id,
            "stake_tokens": total,   # nAVAX effective weight
            "commission": None,
            "jailed": False,
        })
        shares_raw.append(total)

    if not shares_raw:
        log("WARN", "Avalanche: non-empty RPC but zero total weight — check endpoint / rate limits.")

    # Optionally get P-chain height:
    # h = http_post_json(pchain_url, {"jsonrpc":"2.0","id":1,"method":"platform.getHeight","params":{}})
    # epoch_or_height = int(h.get("result", {}).get("height", "0"))
    epoch_or_height = 0
    return {"epoch_or_height": epoch_or_height, "validators": vals, "shares_raw": shares_raw}


# ----------------------------- Tezos (TzKT) ----------------------------------
def fetch_tezos_delegates(api_base: str) -> Dict[str, Any]:
    """
    TzKT public API:
      GET {api_base}/v1/delegates?active=true&limit=...&offset=...
    Field 'stakingBalance' is in mutez (1e6). We page until empty.
    """
    base = api_base.rstrip("/")
    url = f"{base}/v1/delegates"
    log("INFO", "Tezos: fetching active delegates (bakers) from TzKT …")
    vals, shares_raw = [], []
    offset = 0
    page = 0
    limit = 1000  # TzKT supports large limits; adjust if needed
    while True:
        page += 1
        params = {"active": "true", "limit": str(limit), "offset": str(offset)}
        j = http_get_json(url, params=params)
        if not isinstance(j, list):
            raise RuntimeError(f"TzKT returned non-list at page {page}: {str(j)[:200]}")
        if not j:
            break
        for d in j:
            stake = int(d.get("stakingBalance", 0))
            if stake <= 0:
                continue
            vals.append({
                "validator_id": d.get("address", ""),
                "stake_tokens": stake,
                "commission": None,
                "jailed": False,
            })
            shares_raw.append(stake)
        offset += len(j)
        log("INFO", f"Tezos: page={page} got={len(j)} total={len(vals)}")
    if not shares_raw:
        raise RuntimeError("Tezos: no active delegates with positive stakingBalance (check API / rate limits)")
    # For epoch/height we could call /v1/head; not strictly required here
    return {"epoch_or_height": 0, "validators": vals, "shares_raw": shares_raw}

# ----------------------------- Solana fetcher --------------------------------
def fetch_solana_vote_accounts(rpc_url: str) -> Dict[str, Any]:
    """
    Solana JSON-RPC:
      getVoteAccounts -> returns 'current' list with activatedStake (lamports).
    """
    log("INFO", "Solana: fetching vote accounts …")
    body = {"jsonrpc":"2.0","id":1,"method":"getVoteAccounts","params":[]}
    j = http_post_json(rpc_url, body)
    res = j.get("result", {})
    current = res.get("current", []) or []
    vals, shares_raw = [], []
    for va in current:
        stake = int(va.get("activatedStake", 0))
        if stake <= 0:
            continue
        vals.append({
            "validator_id": va.get("votePubkey",""),
            "stake_tokens": stake,
            "commission": None,
            "jailed": False,
        })
        shares_raw.append(stake)
    log("INFO", f"Solana: validators with stake={len(vals)}")
    return {"epoch_or_height": 0, "validators": vals, "shares_raw": shares_raw}


# ----------------------------- export helpers -------------------------------
def export_weights_csv(chain_key: str, out_dir: str, rows: List[Dict[str, Any]]):
    """
    rows must contain:
      rank, validator_id, share, cumulative_share, stake_tokens (opt), voting_power (opt)
    """
    p = os.path.join(out_dir, f"{chain_key}_weights.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        cols = ["rank", "validator_id", "share", "cumulative_share", "stake_tokens", "voting_power"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            for k in cols:
                r.setdefault(k, "")
            w.writerow(r)
    log("INFO", f"Wrote per-validator weights: {p}")

# ----------------------------- main per-chain -------------------------------
def snapshot_chain(chain: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    key = chain["key"]; fam = chain["family"]
    log("INFO", f"==== Processing chain: {key} (family={fam}) ====")

    if fam == "sui":
        data = fetch_sui_committee(chain["rpc"])
        epoch = data["epoch_or_height"]
        shares_raw = data["shares_raw"]
        total = sum(shares_raw) or 1
        shares = [x / total for x in shares_raw if x > 0]
        n = len(shares)
        max_validators = ""
        bonded_tokens = ""
        bonded_human = ""

        # build per-validator export
        pairs = []
        for v in data["validators"]:
            share = (v["voting_power"] or 0) / total
            pairs.append({
                "validator_id": v["validator_id"],
                "share": share,
                "stake_tokens": "",
                "voting_power": v["voting_power"],
            })
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum = 0.0
        for i, r in enumerate(pairs, 1):
            cum += r["share"]
            r["rank"] = i
            r["cumulative_share"] = cum
        export_weights_csv(key, out_dir, pairs)

    elif fam == "cosmos":
        lcd, tried, errs = autodiscover_lcd(key)
        if not lcd:
            err_msg = (f"No LCD for {key}. Tried={tried}. Errors={errs}. "
                       f"Hint: set ENDPOINT_OVERRIDES_JSON or LCD_{key.upper()} "
                       f"(e.g. https://rest.cosmos.directory/{key})")
            raise RuntimeError(err_msg)
        height, vals, vp, params, pool = fetch_cosmos(lcd)
        epoch = height
        total_stake = sum(v.get("stake_tokens", 0) for v in vals) or 1
        shares = [(v.get("stake_tokens", 0) / total_stake) for v in vals if v.get("stake_tokens", 0) > 0]
        n = len(vals)
        max_validators = int(params.get("max_validators", 0) or 0)
        bonded_tokens = str(pool.get("bonded_tokens", ""))
        bonded_human = to_human_amount(bonded_tokens, DECIMALS.get(key, 6))

        if n == 100:
            log("WARN", f"{key}: received exactly 100 bonded validators — check pagination or override LCD.")

        # per-validator export
        pairs = []
        for v in vals:
            st = v.get("stake_tokens", 0)
            share = st / total_stake if total_stake else 0.0
            pairs.append({
                "validator_id": v.get("validator_id"),
                "share": share,
                "stake_tokens": st,
                "voting_power": "",
            })
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum = 0.0
        for i, r in enumerate(pairs, 1):
            cum += r["share"]
            r["rank"] = i
            r["cumulative_share"] = cum
        export_weights_csv(key, out_dir, pairs)

    elif fam == "near":
        data = fetch_near_validators(chain["rpc"])
        epoch = data["epoch_or_height"]
        shares_raw = data["shares_raw"]
        total = sum(shares_raw) or 1
        shares = [x / total for x in shares_raw if x > 0]
        n = len(shares)
        max_validators = ""   # not applicable here
        bonded_tokens = str(total)
        bonded_human = to_human_amount(bonded_tokens, DECIMALS.get("near",24))

        # export per-validator weights
        pairs = []
        for v in data["validators"]:
            st = v["stake_tokens"]; share = st/total if total else 0.0
            pairs.append({"validator_id": v["validator_id"], "share": share,
                          "stake_tokens": st, "voting_power": ""})
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum=0.0
        for i,r in enumerate(pairs,1): cum+=r["share"]; r["rank"]=i; r["cumulative_share"]=cum
        export_weights_csv(key, out_dir, pairs)

    elif fam == "avax":
        data = fetch_avax_validators(chain["rpc"])
        epoch = data["epoch_or_height"]
        shares_raw = data["shares_raw"]
        total = sum(shares_raw) or 1
        shares = [x / total for x in shares_raw if x > 0]
        n = len(shares)
        max_validators = ""
        bonded_tokens = str(total)
        bonded_human = to_human_amount(bonded_tokens, DECIMALS.get("avalanche",9))

        pairs=[]
        for v in data["validators"]:
            st=v["stake_tokens"]; share = st/total if total else 0.0
            pairs.append({"validator_id": v["validator_id"], "share": share,
                          "stake_tokens": st, "voting_power": ""})
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum=0.0
        for i,r in enumerate(pairs,1): cum+=r["share"]; r["rank"]=i; r["cumulative_share"]=cum
        export_weights_csv(key, out_dir, pairs)

    elif fam == "tezos":
        data = fetch_tezos_delegates(chain["rpc"])
        epoch = data["epoch_or_height"]
        shares_raw = data["shares_raw"]
        total = sum(shares_raw) or 1
        shares = [x / total for x in shares_raw if x > 0]
        n = len(shares)
        max_validators = ""
        bonded_tokens = str(total)
        bonded_human = to_human_amount(bonded_tokens, DECIMALS.get("tezos",6))

        pairs=[]
        for v in data["validators"]:
            st=v["stake_tokens"]; share = st/total if total else 0.0
            pairs.append({"validator_id": v["validator_id"], "share": share,
                          "stake_tokens": st, "voting_power": ""})
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum=0.0
        for i,r in enumerate(pairs,1): cum+=r["share"]; r["rank"]=i; r["cumulative_share"]=cum
        export_weights_csv(key, out_dir, pairs)

    elif fam == "solana":
        data = fetch_solana_vote_accounts(chain["rpc"])
        epoch = data["epoch_or_height"]
        shares_raw = data["shares_raw"]
        total = sum(shares_raw) or 1
        shares = [x / total for x in shares_raw if x > 0]
        n = len(shares)
        max_validators = ""
        bonded_tokens = str(total)
        bonded_human = to_human_amount(bonded_tokens, DECIMALS.get("solana",9))

        pairs=[]
        for v in data["validators"]:
            st=v["stake_tokens"]; share = st/total if total else 0.0
            pairs.append({"validator_id": v["validator_id"], "share": share,
                          "stake_tokens": st, "voting_power": ""})
        pairs.sort(key=lambda x: x["share"], reverse=True)
        cum=0.0
        for i,r in enumerate(pairs,1): cum+=r["share"]; r["rank"]=i; r["cumulative_share"]=cum
        export_weights_csv(key, out_dir, pairs)
    else:
        raise RuntimeError(f"Unknown family: {fam}")
    if not shares:
        raise RuntimeError(f"{key}: empty validator set or zero total stake — skipping metrics")

    shares_desc = sorted(shares, reverse=True)

    g = round(gini(shares_desc), 6)
    h = round(hhi(shares_desc), 6)
    k33 = k_for_threshold(shares_desc, 0.33)
    k50 = k_for_threshold(shares_desc, 0.50)
    k66 = k_for_threshold(shares_desc, 0.66)

    rec = {
        "chain": key,
        "snapshot_time": dt.datetime.utcnow().isoformat() + "Z",
        "epoch_or_height": epoch,
        "n": n,
        "gini": g,
        "hhi": h,
        "top1_share": round(top_share(shares_desc, 1), 6),
        "top5_share": round(top_share(shares_desc, 5), 6),
        "top10_share": round(top_share(shares_desc, 10), 6),
        "top20_share": round(top_share(shares_desc, 20), 6),
        "k33": k33,
        "k50": k50,
        "k66": k66,
        "k33_share": round(k33 / n if n else 0.0, 6),
        "k50_share": round(k50 / n if n else 0.0, 6),
        "k66_share": round(k66 / n if n else 0.0, 6),
        "max_validators": max_validators,
        "bonded_tokens": bonded_tokens,
        "bonded_tokens_human": bonded_human,
    }

    # Persist JSONL per-chain
    with open(os.path.join(out_dir, f"{key}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log("INFO", f"{key}: metrics computed (n={n}, gini={g:.3f}, hhi={h:.3f}, k66={k66})")

    return rec

# ----------------------------- main -----------------------------------------
def run_snapshots(chains):
    day = dt.datetime.utcnow().date().isoformat()
    out_dir = os.path.join("data", day)
    os.makedirs(out_dir, exist_ok=True)

    log("INFO", f"=== Snapshot run started (UTC day={day}) ===")
    log("INFO", f"Chains to process: {len(chains)}")

    summary, failed = [], []
    for idx, ch in enumerate(chains, 1):
        log("INFO", f"Progress: chain {idx}/{len(chains)} → {ch['key']}")
        try:
            res = snapshot_chain(ch, out_dir)
            log("INFO", f"[OK] {res['chain']} n={res['n']} gini={res['gini']:.3f} hhi={res['hhi']:.3f} k66={res['k66']}")
            summary.append(res)
        except Exception as e:
            log("ERROR", f"[ERR] {ch['key']}: {e}")
            failed.append({"chain": ch["key"], "error": str(e)})

    if summary:
        fsum = os.path.join(out_dir, "summary.csv")
        fieldnames = [
            "chain", "epoch_or_height", "n",
            "gini", "hhi",
            "top1_share", "top5_share", "top10_share", "top20_share",
            "k33", "k50", "k66", "k33_share", "k50_share", "k66_share",
            "max_validators", "bonded_tokens", "bonded_tokens_human",
            "snapshot_time"
        ]
        with open(fsum, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in summary:
                for k in fieldnames:
                    r.setdefault(k, "")
                w.writerow(r)
        log("INFO", f"Wrote summary: {fsum}")

    if failed:
        log("WARN", f"Completed with failures: {[f['chain'] for f in failed]}")
    else:
        log("INFO", "Completed with no failures.")

    log("INFO", f"=== Snapshot run finished. Output dir: {out_dir} ===")

def main():
    run_snapshots(CHAINS)

# --- CLI glue ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated chain keys to run (e.g., 'avalanche,solana')", default=None)
    parser.add_argument("--skip", help="comma-separated chain keys to skip", default=None)
    args = parser.parse_args()

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    skip = set(x.strip() for x in args.skip.split(",")) if args.skip else set()

    selected = []
    for ch in CHAINS:
        k = ch["key"]
        if only is not None and k not in only:
            continue
        if k in skip:
            continue
        selected.append(ch)

    run_snapshots(selected)