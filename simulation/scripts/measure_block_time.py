#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone


def rpc_get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_rfc3339(s: str) -> datetime:
    # Cosmos often returns: 2026-02-12T19:12:34.123456789Z
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # Trim nanoseconds to microseconds for Python datetime
    if "." in s:
        main, rest = s.split(".", 1)
        if "+" in rest:
            frac, tz = rest.split("+", 1)
            frac = (frac + "000000")[:6]
            s = f"{main}.{frac}+{tz}"
        elif "-" in rest:
            frac, tz = rest.split("-", 1)
            frac = (frac + "000000")[:6]
            s = f"{main}.{frac}-{tz}"
        else:
            frac = (rest + "000000")[:6]
            s = f"{main}.{frac}"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def main():
    ap = argparse.ArgumentParser(description="Measure real block interval from Cosmos RPC")
    ap.add_argument("--rpc", default="http://127.0.0.1:36657", help="RPC base URL")
    ap.add_argument("--window", type=int, default=200, help="How many latest blocks to sample")
    args = ap.parse_args()

    status = rpc_get(f"{args.rpc}/status")
    latest = int(status["result"]["sync_info"]["latest_block_height"])
    if latest < 3:
        print(f"Too few blocks yet (latest={latest}). Wait a bit and rerun.")
        return

    start = max(1, latest - args.window + 1)
    times = []

    for h in range(start, latest + 1):
        blk = rpc_get(f"{args.rpc}/block?height={h}")
        ts = blk["result"]["block"]["header"]["time"]
        times.append(parse_rfc3339(ts))

    deltas = []
    for i in range(1, len(times)):
        d = (times[i] - times[i - 1]).total_seconds()
        if d > 0:
            deltas.append(d)

    if len(deltas) < 5:
        print("Not enough intervals to analyze.")
        return

    sd = sorted(deltas)
    p50 = quantile(sd, 0.50)
    p95 = quantile(sd, 0.95)
    p99 = quantile(sd, 0.99)
    dmin = sd[0]
    dmax = sd[-1]
    mean = statistics.mean(sd)

    print(f"RPC: {args.rpc}")
    print(f"Sampled heights: {start}..{latest} ({len(deltas)} intervals)")
    print("Block interval stats (seconds):")
    print(f"  min : {dmin:.3f}")
    print(f"  p50 : {p50:.3f}")
    print(f"  p95 : {p95:.3f}")
    print(f"  p99 : {p99:.3f}")
    print(f"  max : {dmax:.3f}")
    print(f"  mean: {mean:.3f}")

    # Conservative suggestions for localnet
    timeout_commit = max(0.3, 1.8 * p95)
    timeout_propose = max(0.15, 0.9 * p50)
    timeout_prevote = max(0.08, 0.4 * p50)
    timeout_precommit = max(0.08, 0.4 * p50)

    print("\nSuggested config.toml timeouts:")
    print(f'  timeout_propose   = "{int(timeout_propose * 1000)}ms"')
    print(f'  timeout_prevote   = "{int(timeout_prevote * 1000)}ms"')
    print(f'  timeout_precommit = "{int(timeout_precommit * 1000)}ms"')
    print(f'  timeout_commit    = "{int(timeout_commit * 1000)}ms"')

    print("\nIf chain misses commits or loops rounds, increase timeout_commit first.")


if __name__ == "__main__":
    main()
