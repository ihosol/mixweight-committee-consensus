import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Base path to the folders (replace with your local path)
base_path = './data'  # e.g., '/Users/yourname/mixweight-data'

# List all date folders in year-month-day format (e.g., '2025-09-05' to '2025-10-09')
dates = []
for folder in os.listdir(base_path):
    if os.path.isdir(os.path.join(base_path, folder)) and len(folder) == 10 and folder.startswith('2025-'):
        dates.append(folder)
dates.sort()  # Sort chronologically

# 1. Aggregate all risk_results.csv into one DataFrame
all_data = []
for date in dates:
    file_path = os.path.join(base_path, date, 'risk_results.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = date
        df['network'] = df['file'].str.replace('_weights.csv', '')  # Extract network name
        all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)

# 2. Summarize key metrics (e.g., for α=0.25, λ=0 vs λ=0.3)
summary = []
for net in full_df['network'].unique():
    for alpha in [0.25]:  # Focus on α=0.25; extend as needed
        df_net_alpha = full_df[(full_df['network'] == net) & (full_df['alpha'] == alpha)]
        
        # Average over days for λ=0 and λ=0.3
        mu_0 = df_net_alpha[df_net_alpha['lambda'] == 0.0]['mu'].mean()
        mu_03 = df_net_alpha[df_net_alpha['lambda'] == 0.3]['mu'].mean()
        bound_0 = df_net_alpha[df_net_alpha['lambda'] == 0.0]['chernoff_bound'].mean()
        bound_03 = df_net_alpha[df_net_alpha['lambda'] == 0.3]['chernoff_bound'].mean()
        risk_0 = df_net_alpha[df_net_alpha['lambda'] == 0.0]['empirical_risk'].mean()
        risk_03 = df_net_alpha[df_net_alpha['lambda'] == 0.3]['empirical_risk'].mean()
        
        # Computations
        pct_mu_red = ((mu_0 - mu_03) / mu_0 * 100) if mu_0 != 0 else 0
        orders_drop = -np.log10(bound_03 / bound_0) if bound_0 > 0 and bound_03 > 0 else 0
        
        summary.append({
            'network': net,
            'alpha': alpha,
            'mu_0_avg': mu_0,
            'mu_03_avg': mu_03,
            'pct_mu_red': pct_mu_red,
            'bound_0_avg': bound_0,
            'bound_03_avg': bound_03,
            'orders_drop': orders_drop,
            'risk_0_avg': risk_0,
            'risk_03_avg': risk_03
        })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('chapter5_summary_table.csv', index=False)  # For paper table
print(summary_df.to_latex(index=False))  # Generate LaTeX for paper

# 3. Build Plots (save as PNG/SVG for paper)
# Plot 1: μ vs λ for α=0.25 (averaged over days, per network)
df_plot = full_df[full_df['alpha'] == 0.25].groupby(['network', 'lambda'])['mu'].mean().reset_index()
for net in df_plot['network'].unique():
    df_net = df_plot[df_plot['network'] == net]
    plt.plot(df_net['lambda'], df_net['mu'], label=net)
plt.xlabel('λ')
plt.ylabel('Expected Adversarial Seats μ(λ)')
plt.title('Averaged μ(λ) vs λ (α=0.25)')
plt.legend()
plt.savefig('mu_vs_lambda_alpha025.png')
plt.savefig('mu_vs_lambda_alpha025.svg')  # Vector for paper
plt.clf()

# Plot 2: Log Chernoff Bound vs λ for α=0.25 (averaged)
df_bound = full_df[full_df['alpha'] == 0.25].groupby(['network', 'lambda'])['chernoff_bound'].mean().reset_index()
for net in df_bound['network'].unique():
    df_net = df_bound[df_bound['network'] == net]
    plt.plot(df_net['lambda'], np.log10(df_net['chernoff_bound']), label=net)
plt.xlabel('λ')
plt.ylabel('log10 Chernoff Bound')
plt.title('Averaged log Chernoff Bound vs λ (α=0.25)')
plt.legend()
plt.savefig('log_chernoff_vs_lambda_alpha025.png')
plt.savefig('log_chernoff_vs_lambda_alpha025.svg')
plt.clf()

# Plot 3: Bar Chart of % mu Reduction (λ=0.3, α=0.25)
df_bar = summary_df[summary_df['alpha'] == 0.25]
plt.bar(df_bar['network'], df_bar['pct_mu_red'])
plt.xlabel('Network')
plt.ylabel('% mu Reduction')
plt.title('% mu Reduction at λ=0.3 (α=0.25)')
plt.xticks(rotation=45)
plt.savefig('pct_mu_reduction_bar.png')
plt.savefig('pct_mu_reduction_bar.svg')
plt.clf()

# Plot 4: Time Series of mu Reduction % for λ=0.3 α=0.25 (per network over dates)
df_time = full_df[(full_df['alpha'] == 0.25) & (full_df['lambda'].isin([0.0, 0.3]))]
df_time_pivot = df_time.pivot_table(index=['date', 'network'], columns='lambda', values='mu', aggfunc='mean').reset_index()
df_time_pivot['pct_mu_red'] = ((df_time_pivot[0.0] - df_time_pivot[0.3]) / df_time_pivot[0.0]) * 100
df_time_pivot['date'] = pd.to_datetime(df_time_pivot['date'])
for net in df_time_pivot['network'].unique():
    df_net_time = df_time_pivot[df_time_pivot['network'] == net].sort_values('date')
    plt.plot(df_net_time['date'], df_net_time['pct_mu_red'], label=net)
plt.xlabel('Date')
plt.ylabel('% mu Reduction')
plt.title('Time Series of % mu Reduction (λ=0.3, α=0.25)')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('mu_reduction_time_series.png')
plt.savefig('mu_reduction_time_series.svg')
plt.clf()

print("Summary table saved as chapter5_summary_table.csv and LaTeX printed above.")
print("Plots generated: mu_vs_lambda_alpha025.png/svg, log_chernoff_vs_lambda_alpha025.png/svg, pct_mu_reduction_bar.png/svg, mu_reduction_time_series.png/svg.")