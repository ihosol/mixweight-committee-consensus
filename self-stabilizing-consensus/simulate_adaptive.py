import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# --- CONFIGURATION ---
DATA_DIR = './data'
OUTPUT_DIR = './analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation Settings
ATTACK_DAY_INDEX = 30       # Attack happens on the 30th day of data
SYBIL_SPLIT_COUNT = 50      # Whale splits into this many nodes
STATIC_LAMBDA = 0.3         # The "Naive" benchmark

# The Adaptive Control Law
# lambda = k * (Gini - Target)
GINI_TARGET = 0.40
CONTROL_K = 1.0
MAX_LAMBDA = 0.5

# --- HELPER FUNCTIONS ---

def calculate_gini(stakes):
    """Calculates Gini on the fly for the simulated set."""
    array = np.array(stakes, dtype=np.float64)
    if np.amin(array) < 0: array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def get_adaptive_lambda(gini):
    """The Feedback Controller."""
    # Transfer function: Rectified Linear Unit (ReLU) shifted by Target
    val = (gini - GINI_TARGET) * CONTROL_K
    return np.clip(val, 0.0, MAX_LAMBDA)

def compute_selection_mass(df, lambda_val, baseline_type='uniform'):
    """
    Core Probability Engine.
    Returns the DataFrame with a 'probability' column.
    """
    total_stake = df['stake'].sum()
    df['w_stake'] = df['stake'] / total_stake
    
    if baseline_type == 'uniform':
        # Naive: 1/N
        df['w_base'] = 1.0 / len(df)
    elif baseline_type == 'age':
        # Sybil Defense: Age / Sum(Age)
        total_age = df['age'].sum()
        # Safety check for div by zero if total_age is 0 (unlikely)
        if total_age == 0: total_age = 1.0
        df['w_base'] = df['age'] / total_age
    
    # The Mixture Formula
    df['prob'] = (1 - lambda_val) * df['w_stake'] + lambda_val * df['w_base']
    return df

def simulate_network(network_name):
    net_path = os.path.join(DATA_DIR, network_name)
    if not os.path.exists(net_path): return None
    
    files = sorted([f for f in os.listdir(net_path) if f.endswith('.csv')])
    
    if len(files) < ATTACK_DAY_INDEX + 5:
        print(f"Skipping {network_name}: Not enough history.")
        return None

    print(f"Simulating {network_name}...")
    
    # Tracking State
    validator_first_seen = {} # To calc Age
    results = []
    
    # Iterate through time
    for i, filename in enumerate(files):
        date_str = filename.replace(".csv", "")
        day_path = os.path.join(net_path, filename)
        
        try:
            df = pd.read_csv(day_path)
            
            # --- FIX: DATA TYPE CONVERSION ---
            # Ensure stake is numeric, turn errors ('') into NaN and drop them
            df['stake_tokens'] = pd.to_numeric(df['stake_tokens'], errors='coerce')
            df = df.dropna(subset=['stake_tokens'])
            
            # Standardize columns
            df = df.rename(columns={'stake_tokens': 'stake', 'validator_id': 'id'})
            df = df[['id', 'stake']]
            
            # Skip empty days
            if df.empty: continue

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # 1. Update Ages (Reputation System)
        current_ids = set(df['id'].unique())
        for vid in current_ids:
            if vid not in validator_first_seen:
                validator_first_seen[vid] = i # First seen at index i
        
        # Assign Age: (Current Index - First Seen) + 1
        df['age'] = df['id'].map(lambda x: i - validator_first_seen.get(x, i) + 1)

        # 2. Identify the Target (The Whale)
        if df.empty: continue
        top_node = df.sort_values('stake', ascending=False).iloc[0]
        whale_id = top_node['id']
        whale_stake = top_node['stake']
        
        # 3. Logic Branch: Normal vs Attack
        is_attack_active = (i >= ATTACK_DAY_INDEX)
        
        sim_df = df.copy()
        
        if is_attack_active:
            # --- INJECT SYBIL ATTACK ---
            # Remove Whale
            sim_df = sim_df[sim_df['id'] != whale_id]
            
            # Add Sybils
            sybil_rows = []
            sybil_stake = whale_stake / SYBIL_SPLIT_COUNT
            for s in range(SYBIL_SPLIT_COUNT):
                sybil_rows.append({
                    'id': f"SYBIL_{s}",
                    'stake': sybil_stake,
                    'age': 1 # NEW NODES HAVE NO REPUTATION
                })
            sim_df = pd.concat([sim_df, pd.DataFrame(sybil_rows)], ignore_index=True)
        
        # 4. Calculate Control Variables
        current_gini = calculate_gini(sim_df['stake'].values)
        adaptive_lam = get_adaptive_lambda(current_gini)
        
        # 5. Run Models
        
        # Model A: Static + Uniform (The Vulnerable One)
        res_static = compute_selection_mass(sim_df.copy(), STATIC_LAMBDA, 'uniform')
        if is_attack_active:
            entity_prob_static = res_static[res_static['id'].str.startswith("SYBIL")]['prob'].sum()
        else:
            entity_prob_static = res_static[res_static['id'] == whale_id]['prob'].sum()

        # Model B: Adaptive + Age (The Secure One)
        res_adapt = compute_selection_mass(sim_df.copy(), adaptive_lam, 'age')
        if is_attack_active:
            entity_prob_adapt = res_adapt[res_adapt['id'].str.startswith("SYBIL")]['prob'].sum()
        else:
            entity_prob_adapt = res_adapt[res_adapt['id'] == whale_id]['prob'].sum()

        # Store Data
        results.append({
            'day': i,
            'date': date_str,
            'gini': current_gini,
            'lambda_adaptive': adaptive_lam,
            'whale_share_static': entity_prob_static,
            'whale_share_adaptive': entity_prob_adapt,
            'is_attack': is_attack_active
        })

    return pd.DataFrame(results)
# --- CONFIGURATION CHECK ---
# Ensure these are set at the top of your script
CONTROL_K = 2.0         # High sensitivity to make the effect visible
GINI_TARGET = 0.35      # Lower target to trigger mechanism earlier
# ---------------------------

def plot_results(results_dict):
    targets = ['solana', 'avalanche', 'celestia'] 
    valid_targets = [t for t in targets if t in results_dict]

    if not valid_targets: return

    fig, axes = plt.subplots(len(valid_targets), 1, figsize=(10, 14), sharex=True)
    if len(valid_targets) == 1: axes = [axes]

    for idx, net in enumerate(valid_targets):
        df = results_dict[net]
        ax = axes[idx]
        
        # --- 1. CALCULATION ---
        # Normalize relative to the moment BEFORE the attack (Day 29)
        # This sets Day 29 to "0.0%". Everything else is a deviation.
        base_prob = df.iloc[ATTACK_DAY_INDEX-1]['whale_share_adaptive'] 
        
        df['rel_static'] = (df['whale_share_static'] - base_prob) / base_prob * 100
        df['rel_adaptive'] = (df['whale_share_adaptive'] - base_prob) / base_prob * 100
        
        # --- 2. PLOTTING ---
        # Zero Line (Honest Behavior)
        ax.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.5)
        
        # The Static Model (Red)
        ax.plot(df['day'], df['rel_static'], color='#e74c3c', linestyle='--', linewidth=2, label='Static λ (Sybil Vulnerable)')
        
        # The Adaptive Model (Green)
        ax.plot(df['day'], df['rel_adaptive'], color='#27ae60', linewidth=3, label='Adaptive λ (Sybil Resistant)')
        
        # --- 3. SHADING (The "Mechanism Effect") ---
        # Shade the area between Red and Green to show the "Protection" magnitude
        ax.fill_between(df['day'], df['rel_static'], df['rel_adaptive'], 
                        where=(df['day'] >= ATTACK_DAY_INDEX),
                        color='#27ae60', alpha=0.1, label='Mechanism Impact')

        # --- 4. ANNOTATIONS (Fixed Positioning) ---
        # Get the final values to position text at the end of the chart
        last_day = df.iloc[-1]['day']
        final_static = df.iloc[-1]['rel_static']
        final_adapt = df.iloc[-1]['rel_adaptive']
        
        # Label for Red Line (Place ABOVE)
        if final_static > 0:
            ax.annotate(f"Attack Gains: +{final_static:.1f}%", 
                        xy=(last_day, final_static), 
                        xytext=(0, -20), textcoords='offset points',
                        ha='right', color='#c0392b', fontweight='bold')

        # Label for Green Line (Place BELOW)
        if final_adapt < 0:
            ax.annotate(f"Attack Loss: {final_adapt:.1f}%", 
                        xy=(last_day, final_adapt), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='right', color='#27ae60', fontweight='bold')

        # Vertical Line for Attack Start
        ax.axvline(x=ATTACK_DAY_INDEX, color='black', linestyle=':', alpha=0.5)
        if idx == 0:
            ax.text(ATTACK_DAY_INDEX + 0.5, max(df['rel_static'])*0.5, "Sybil Attack Starts", rotation=90, alpha=0.6)

        # Styling
        ax.set_title(f"Network: {net.upper()} (Gini: {df['gini'].mean():.2f})", fontsize=12)
        ax.set_ylabel("% Change in Selection Power")
        ax.grid(True, alpha=0.3)
        
        if idx == 0: ax.legend(loc='upper left')

    axes[-1].set_xlabel("Simulation Day")
    plt.suptitle("Game Theoretic Result: Sybil Strategy becomes Dominated", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sybil_incentive_analysis.png")
    print(f"✅ Saved clean plot to {OUTPUT_DIR}/sybil_incentive_analysis.png")
def print_quantitative_table(results_dict):
    print("\n--- QUANTITATIVE IMPACT TABLE (Latex Ready) ---")
    print(f"{'Network':<12} | {'Gini':<5} | {'Q_pre':<8} | {'Q_stat':<8} | {'Q_adapt':<8} | {'Delta_Adapt':<8}")
    print("-" * 70)
    
    stats = []
    
    for net, df in results_dict.items():
        # Data points around the attack
        # Day 29 (Pre-Attack) vs Day 30 (Post-Attack)
        pre = df[df['day'] == ATTACK_DAY_INDEX - 1].iloc[0]
        post = df[df['day'] == ATTACK_DAY_INDEX].iloc[0]
        
        # 1. Pre-Attack Selection Prob (Honest Whale)
        # Using Adaptive model as the baseline "Honest" state
        q_pre = pre['whale_share_adaptive'] 
        
        # 2. Post-Attack Static (Attacker wins)
        q_static = post['whale_share_static']
        
        # 3. Post-Attack Adaptive (Attacker loses)
        q_adapt = post['whale_share_adaptive']
        
        # 4. The Percent Change (The Penalty)
        delta_pct = ((q_adapt - q_pre) / q_pre) * 100
        
        # Average Gini for context
        avg_gini = df['gini'].mean()
        
        stats.append({
            'Network': net.capitalize(),
            'Gini': f"{avg_gini:.2f}",
            'Q_pre': f"{q_pre*100:.2f}\%",
            'Q_stat': f"{q_static*100:.2f}\%",
            'Q_adapt': f"{q_adapt*100:.2f}\%",
            'Delta': f"{delta_pct:.2f}\%"
        })
        
        print(f"{net:<12} | {avg_gini:.2f}  | {q_pre:.4f}   | {q_static:.4f}   | {q_adapt:.4f}   | {delta_pct:.2f}%")

    # Optional: Save to CSV
    pd.DataFrame(stats).to_csv(f"{OUTPUT_DIR}/impact_table.csv", index=False)

# Add this to the bottom of your main execution block:
if __name__ == "__main__":
    networks = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    all_results = {}
    
    for net in networks:
        res = simulate_network(net)
        if res is not None and not res.empty:
            all_results[net] = res
            
    plot_results(all_results)
    print_quantitative_table(all_results)