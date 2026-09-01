#!/usr/bin/env python3
from pathlib import Path

CFGDIR = Path(__file__).resolve().parents[1] / 'configs'
CFGDIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    # Main realistic case
    dict(name='FP_A1_main_12h_c9_k6_b033_burst', honest=12, committee=9, k=6, beta='0.33', profile='burst', attack_extra=''),
    # Companion robustness case
    dict(name='FP_A2_companion_12h_c9_k6_b033_trickle', honest=12, committee=9, k=6, beta='0.33', profile='trickle', attack_extra=''),
    # Stronger but still defendable sensitivity
    dict(name='FP_B1_sensitivity_12h_c9_k8_b033_burst', honest=12, committee=9, k=8, beta='0.33', profile='burst', attack_extra=''),
    # Stress / robustness appendix case
    dict(name='FP_B2_stress_12h_c9_k8_b040_burst', honest=12, committee=9, k=8, beta='0.40', profile='burst', attack_extra=''),
    # Honest-newcomer fairness baseline case
    dict(name='FP_H1_honest_newcomer_12h_c9_k1_b010_burst', honest=12, committee=9, k=1, beta='0.10', profile='burst', attack_extra='  entry_kind: honest_newcomer\n'),
    # Honest-newcomer robustness case
    dict(name='FP_H2_honest_newcomer_12h_c9_k1_b010_trickle', honest=12, committee=9, k=1, beta='0.10', profile='trickle', attack_extra='  entry_kind: honest_newcomer\n'),
]

TEMPLATE = '''chain:
  chain_id: poc-1
  denom: stake

localnet:
  nodes: {nodes}
  p2p_port_base: 26680
  rpc_port_base: 36657
  api_port_base: 31317
  grpc_port_base: 39090

experiment:
  honest_nodes: {honest}
  artifacts_subdir: {name}

attack:
  beta: {beta}
  sybil_k_values: [{k}]
  attacker_profile: {profile}
{attack_extra}

workload:
  committee_mode: fixed
  committee_size_values: [{committee}]
  lambda_ppm_values: [0]

tx:
  from: alice
  keyring_backend: test
  fees: 2000stake
  gas: 2000000
  broadcast_mode: sync
  skip_set_lambda: true

coalition:
  topk_values: [1, 2, 3, 4, 5]

epoch:
  epoch_blocks: 4
  pre_attack_epochs: 20
  post_attack_epochs: 30
  draws_per_epoch: 8
  post_attack_draw_limit: 0
  reuse_checkpoint: false
  rebuild_checkpoint: true
  sybil_at_genesis: false

report:
  final_table_single_row: false
  static_lambda_ppm: 300000
  preserve_run_history: true
'''

for sc in SCENARIOS:
    text = TEMPLATE.format(nodes=sc['honest'] + sc['k'], **sc)
    (CFGDIR / f"{sc['name']}.yaml").write_text(text)
    print(CFGDIR / f"{sc['name']}.yaml")
