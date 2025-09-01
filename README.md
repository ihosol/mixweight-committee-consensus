# Mixweight Committee Consensus

Data snapshots and risk evaluation for mixed-weight committee selection
(λ-mixing of stake with uniform/reputation term) across multiple PoS networks.

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DEBUG=1 python snapshots.py
DEBUG=1 RB_M_STRATEGY=fraction:0.4 RB_ALPHAS=0.20,0.25,0.30,0.33 RB_LAMBDAS=0,0.1,0.2,0.3 python risk_batch.py
Artifacts appear under data/YYYY-MM-DD/.
Configure endpoints via .env (see .env.example).
