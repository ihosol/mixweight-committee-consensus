#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    cfg = repo / "cosmos" / "poc_config_adaptive_2p2.yaml"
    if len(sys.argv) > 1 and sys.argv[1].strip():
        p = Path(sys.argv[1].strip())
        cfg = p if p.is_absolute() else (repo / p)

    runner = repo / "cosmos" / "scripts" / "epochrun.py"
    cmd = [sys.executable, str(runner), str(cfg)]
    env = dict(**__import__("os").environ)
    env.setdefault("POC_CHAIND", str(repo / "cosmos" / "chain53" / "chain-five-three" / "build" / "chain-five-threed"))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
