#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
    repo = Path(__file__).resolve().parents[2]
    csv_path = repo / "cosmos" / "artifacts" / "results" / "sybil_seats_vs_lambda.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    # Data structure: data[k][lambda] = [seats1, seats2, ...]
    data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = int(row['k'])
            lam = int(row['lambda_ppm']) / 1_000_000.0
            seats = float(row['attacker_seats'])
            data[k][lam].append(seats)

    if not data:
        print("No data found.")
        sys.exit(0)

    # Define columns (Lambdas to show)
    # Check what's available, picking specific points for the table
    all_lams = sorted(list(data[list(data.keys())[0]].keys()))
    # Try to pick 0.0, 0.5, 1.0 if available, or just use what's there
    target_lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    display_lams = [l for l in target_lams if any(abs(l - al) < 0.01 for al in all_lams)]
    
    # Header
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{\textbf{Impact of Splitting and Aging.} comparison of attacker expected seats (mean) and safety violation probability ($P(\ge 2 \text{ seats})$) across Sybil strategies ($k$) and mixing levels ($\lambda$).}")
    print(r"\label{tab:sybil_results}")
    
    # Dynamic column setup
    # k | Lambda 1 (Mean / Prob) | Lambda 2 ...
    cols = "c" + "|cc" * len(display_lams)
    print(r"\begin{tabular}{" + cols + "}")
    print(r"\toprule")
    
    # Header Row 1: Multicolumns for Lambdas
    header1 = r"\textbf{Strategy}"
    for lam in display_lams:
        header1 += r" & \multicolumn{2}{c}{\textbf{$\lambda=" + f"{lam:.2g}" + r"$}}"
    print(header1 + r" \\")
    
    # Header Row 2: Metrics
    header2 = r"\textbf{Sybil Count ($k$)}"
    for _ in display_lams:
        header2 += r" & $\mathbb{E}[\text{Seats}]$ & $P(\text{fail})$"
    print(header2 + r" \\")
    print(r"\midrule")
    
    # Rows: For each K
    ks = sorted(data.keys())
    for k in ks:
        row_str = f"{k}"
        for lam in display_lams:
            # Find exact matching lambda key
            # (Assuming consistency, otherwise handle missing)
            closest = min(data[k].keys(), key=lambda x: abs(x - lam))
            
            vals = data[k][closest]
            mean = np.mean(vals)
            # Fail prob: seats >= 2 (for m=4, >33%)
            fails = sum(1 for v in vals if v >= 2)
            prob = fails / len(vals)
            
            # Formatting
            # Bold the safe outcomes (Prob < 0.05)?
            prob_str = f"{prob:.2f}"
            if prob < 0.05:
                prob_str = r"\textbf{" + prob_str + "}"
                
            row_str += f" & {mean:.2f} & {prob_str}"
            
        print(row_str + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
