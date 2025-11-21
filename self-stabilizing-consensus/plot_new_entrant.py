import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
OUTPUT_DIR = './analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_convergence():
    # X-Axis: Focus on the first 2 years + a little extra to show the cap
    # 730 days = 2 years
    days = np.linspace(0, 800, 1000) 
    
    # Parameters
    TAU_MAX = 730 
    BETA = 0.5    # Changed to 0.5 (Square Root) for stronger visual effect
    
    # --- FORMULAS ---
    
    # 1. Soft Cap (The Standard Model - "Strict")
    # Linearly grows until 730, then flat.
    # This effectively overlaps with "Raw Age" until the cap.
    y_standard = np.clip(days, 0, TAU_MAX)
    
    # 2. Beta-Weighted (The Proposed Solution - "Fast Start")
    # Formula: y = TAU_MAX * (t / TAU_MAX)^beta
    # It curves UPWARDS early on.
    y_beta = TAU_MAX * np.power((days / TAU_MAX), BETA)
    
    # Cap the Beta curve too so they meet at the same maximum
    y_beta = np.where(days > TAU_MAX, TAU_MAX, y_beta)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot 1: Standard Linear Model (The Problem)
    plt.plot(days, y_standard, color='#c0392b', linestyle='--', linewidth=2, label='Standard Linear Model (Strict)')
    
    # Plot 2: Beta Smoothing (The Solution)
    plt.plot(days, y_beta, color='#27ae60', linewidth=3, label=f'Proposed $\\beta$-Smoothing (Fast Start, $\\beta={BETA}$)')
    
    # --- THE VISUAL FIX: Highlight the "Bonus" ---
    # Shade the area between the lines to show "Entrant Advantage"
    plt.fill_between(days, y_standard, y_beta, 
                     where=(days < TAU_MAX), 
                     color='#27ae60', alpha=0.15, 
                     label='New Entrant Advantage Area')

    # --- ANNOTATIONS ---
    
    # 1. Vertical Line for Maturity
    plt.axvline(x=TAU_MAX, color='black', linestyle=':', alpha=0.5)
    plt.text(TAU_MAX - 150, 20, "Max Reputation Cap (2 Years)", color='gray', fontsize=10)
    
    # 2. The "Probationary Boost" Highlight at 3 Months (Day 90)
    check_day = 90
    idx = (np.abs(days - check_day)).argmin()
    val_standard = y_standard[idx]
    val_beta = y_beta[idx]
    multiplier = val_beta / val_standard
    
    # Draw arrow
    plt.annotate('', xy=(check_day, val_beta), xytext=(check_day, val_standard),
                 arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    
    # Text label
    plt.text(check_day + 20, (val_beta + val_standard)/2, 
             f"Early Boost: {multiplier:.1f}x\n(at 90 Days)", 
             color='blue', fontsize=10, fontweight='bold', va='center')

    # Labels & Style
    plt.title("Reducing Entry Barriers: $\\beta$-Smoothed vs. Linear Reputation", fontsize=14)
    plt.xlabel("Validator Active Time (Days)", fontsize=12)
    plt.ylabel("Effective Baseline Weight (Score)", fontsize=12)
    
    # Set limits to focus on the curve
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    
    # Save
    outfile = os.path.join(OUTPUT_DIR, "entrant_convergence.png")
    plt.savefig(outfile, dpi=300)
    print(f"✅ Plot saved to {outfile}")

if __name__ == "__main__":
    plot_convergence()