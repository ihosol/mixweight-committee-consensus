#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def main():
    repo = Path(__file__).resolve().parents[2]
    csv_path = repo / "cosmos" / "artifacts" / "results" / "sybil_seats_vs_lambda.csv"
    out_dir = repo / "cosmos" / "artifacts" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    # Data structure: data[k][lambda] = [share1, share2, ...]
    data = defaultdict(lambda: defaultdict(list))
    
    # Metadata for title
    meta = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row['k'])
            lam = int(row['lambda_ppm']) / 1_000_000.0
            share = float(row['attacker_seats_share']) # Normalized [0, 1]
            # Or use raw seats? Text discusses "seats" often, but plot usually normalized.
            # Let's use avg seats (share * committee_size)
            m = int(row['committee_size'])
            seats = float(row['attacker_seats'])
            
            data[k][lam].append(seats)
            
            if not meta:
                meta['beta'] = row.get('beta', '?')
                meta['m'] = m

    if not data:
        print("No data found in CSV.")
        sys.exit(0)

    # Extract k values and define colors
    ks = sorted(data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(ks)))

    # 1. Plot Average Attacker Seats
    plot_avg_seats(data, out_dir, meta, colors, ks)
    
    # 2. Plot Risk Probability (Safety Violation)
    # Threshold: >= 33% of committee. For m=4, >= 2 seats.
    plot_risk_prob(data, out_dir, meta, colors, ks, threshold=2)

    # 3. Plot Distribution (Boxplot) at critical Lambdas
    plot_distribution(data, out_dir, ks)

def plot_avg_seats(data, out_dir, meta, colors, ks):
    plt.figure(figsize=(7, 4.5))
    
    for i, k in enumerate(ks):
        lams = sorted(data[k].keys())
        means = []
        for lam in lams:
            means.append(np.mean(data[k][lam]))
            
        plt.plot(lams, means, marker='o', linewidth=2, label=f"k={k} (Splitting)", color=colors[i])

    # Fair share line
    m = int(meta.get('m', 4))
    beta = float(meta.get('beta', 0.33))
    fair_share = m * beta
    plt.axhline(y=fair_share, color='gray', linestyle='--', alpha=0.5, label='Fair Share (Proportional)')

    plt.title(f"Sybil Defense: Average Attacker Representation\n(m={m}, beta={beta})")
    plt.xlabel(r"Mixing Parameter $\lambda$ (Age Penalty Strength)")
    plt.ylabel("Avg Attacker Seats")
    plt.ylim(bottom=-0.1, top=m+0.1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_path = out_dir / "sybil_avg_seats.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

def plot_risk_prob(data, out_dir, meta, colors, ks, threshold):
    plt.figure(figsize=(7, 4.5))
    
    for i, k in enumerate(ks):
        lams = sorted(data[k].keys())
        probs = []
        for lam in lams:
            vals = data[k][lam]
            # Probability of reaching or exceeding BFT threshold
            count = sum(1 for v in vals if v >= threshold)
            probs.append(count / len(vals))
            
        plt.plot(lams, probs, marker='s', linewidth=2, label=f"k={k}", color=colors[i])

    plt.axhline(y=0.0, color='k', linewidth=0.5)
    plt.title(f"Safety: Probability of Capture (Seats >= {threshold})\n(m={meta.get('m',4)}, beta={meta.get('beta',0.33)})")
    plt.xlabel(r"Mixing Parameter $\lambda$")
    plt.ylabel(f"P(Attacker >= {threshold} Seats)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    out_path = out_dir / "sybil_risk_prob.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

def plot_distribution(data, out_dir, ks):
    # Boxplot for specific lambdas: 0.0 (Unsafe), 0.5 (Mixed), 1.0 (Safe)
    target_lams = [0.0, 0.5, 1.0]
    
    plt.figure(figsize=(8, 5))
    
    # Prepare data for boxplot
    # We want to group by Lambda, then by K
    
    plot_data = []
    labels = []
    positions = []
    
    # Check which target lambdas actually exist in data
    # (Use k=1 to check keys, assume consistent)
    available_lams = sorted(data[ks[0]].keys())
    # Find closest matches
    selected_lams = []
    for t in target_lams:
        closest = min(available_lams, key=lambda x: abs(x - t))
        if closest not in selected_lams:
            selected_lams.append(closest)
            
    pos = 1
    for lam in selected_lams:
        for k in ks:
            vals = data[k][lam]
            plot_data.append(vals)
            labels.append(f"λ={lam}\nk={k}")
            positions.append(pos)
            pos += 1
        pos += 1 # Spacer between lambda groups

    plt.boxplot(plot_data, positions=positions, patch_artist=True, boxprops=dict(alpha=0.6))
    
    # Custom x-ticks
    # plt.xticks(positions, labels, rotation=45, ha='right')
    # Simplified labels
    plt.xticks(positions, [f"k={k}" for _ in selected_lams for k in ks], rotation=0)
    
    # Add Lambda group labels manually or via title
    # This is a bit quick-and-dirty, but effective
    
    plt.title("Distribution of Attacker Seats by Strategy and Mixing Level")
    plt.ylabel("Attacker Seats")
    plt.grid(True, axis='y', linestyle=':', alpha=0.6)
    
    # Add annotations for Lambda groups
    # Calculate center of each group
    group_len = len(ks)
    for i, lam in enumerate(selected_lams):
        center = (i * (group_len + 1)) + 1 + (group_len - 1) / 2
        plt.text(center, -0.8, f"λ ≈ {lam}", ha='center', fontweight='bold')

    out_path = out_dir / "sybil_boxplot.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
