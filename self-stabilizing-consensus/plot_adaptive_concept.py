import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
INPUT_FILE = './validation_reports/final_status.csv'
OUTPUT_DIR = './analysis_output'

# --- THE ADAPTIVE CONTROL LAW (The PhD Logic) ---
# lambda = k * (Gini - Target)
# If Gini is below Target (e.g., 0.4), the network is "Safe enough", lambda = 0.
GINI_SAFE_THRESHOLD = 0.40  
SLOPE_K = 1.0               
STATIC_LAMBDA_BENCHMARK = 0.30 

def plot_motivation():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Missing {INPUT_FILE}. Run 'python validate_integrity.py' first.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Apply The Control Law
    # Formula: lambda = max(0, (Gini - 0.40) * 1.0)
    df['lambda_adaptive'] = (df['avg_gini'] - GINI_SAFE_THRESHOLD) * SLOPE_K
    df['lambda_adaptive'] = df['lambda_adaptive'].clip(lower=0) # Lambda cannot be negative
    
    # 3. Sort by Gini (to make the curve pretty)
    df = df.sort_values('avg_gini')
    
    # 4. Visualization
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Bar Plot: The Adaptive Lambda (What the network NEEDS)
    # Color bars: Green (Safe) -> Red (Needs Regulation)
    norm = plt.Normalize(df['lambda_adaptive'].min(), df['lambda_adaptive'].max())
    colors = plt.cm.viridis(norm(df['lambda_adaptive']))
    
    bars = plt.bar(df['network'], df['lambda_adaptive'], color=colors, alpha=0.8, label='Proposed Adaptive λ')

    # Line Plot: The Static Lambda (The "Naive" Approach)
    plt.axhline(y=STATIC_LAMBDA_BENCHMARK, color='red', linestyle='--', linewidth=2, label=f'Static λ={STATIC_LAMBDA_BENCHMARK}')

    # 5. Annotations
    plt.title("The Case for Adaptive Governance: One Size Does Not Fit All", fontsize=15)
    plt.ylabel("Mixing Parameter (λ)", fontsize=12)
    plt.xlabel("Network (Sorted by Centralization)", fontsize=12)
    plt.ylim(0, 0.6)
    
    # Add text labels on bars
    for bar, gini in zip(bars, df['avg_gini']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'G={gini:.2f}',
                 ha='center', va='bottom', fontsize=9, rotation=0)

    # Add region labels
    plt.text(0, 0.55, "Zone 1: Over-Regulated\n(Static λ hurts efficiency)", color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.text(8, 0.55, "Zone 2: Under-Regulated\n(Static λ risks security)", color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    
    outfile = os.path.join(OUTPUT_DIR, "adaptive_motivation.png")
    plt.savefig(outfile, dpi=300)
    print(f"✅ Plot saved to {outfile}")
    print("\n--- ANALYSIS ---")
    print(df[['network', 'avg_gini', 'lambda_adaptive']])

if __name__ == "__main__":
    plot_motivation()