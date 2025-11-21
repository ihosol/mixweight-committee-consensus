import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = './data'
METRICS_FILE = './data/global_metrics_history.csv'
OUTPUT_DIR = './validation_reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_continuity():
    print("🔍 Checking Data Continuity...")
    
    # 1. Load Global Metrics
    if not os.path.exists(METRICS_FILE):
        print("❌ Master metrics file missing! Run ETL first.")
        return

    df = pd.read_csv(METRICS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Check Date Gaps per Network
    report = []
    for net, group in df.groupby('chain'):
        group = group.sort_values('date')
        dates = group['date']
        
        # Calculate missing days
        full_range = pd.date_range(start=dates.min(), end=dates.max())
        missing = full_range.difference(dates)
        
        # Check Gini Stability (Standard Deviation)
        gini_std = group['gini'].std()
        
        status = "PASS"
        if len(missing) > 0: status = "GAPS DETECTED"
        if gini_std > 0.1: status = "VOLATILE GINI"
        
        report.append({
            'network': net,
            'start': dates.min().date(),
            'end': dates.max().date(),
            'total_snapshots': len(group),
            'missing_days': len(missing),
            'avg_gini': round(group['gini'].mean(), 3),
            'gini_stability': round(gini_std, 4),
            'status': status
        })
        
        # Plot Gini History
        plt.figure(figsize=(10,4))
        sns.lineplot(data=group, x='date', y='gini', marker='o')
        plt.title(f"Metric Stability: {net.upper()}")
        plt.savefig(f"{OUTPUT_DIR}/{net}_continuity.png")
        plt.close()

    # 3. Print Report
    rep_df = pd.DataFrame(report)
    print("\nData Integrity Report:")
    print(rep_df.to_string(index=False))
    rep_df.to_csv(f"{OUTPUT_DIR}/final_status.csv", index=False)
    
    # 4. Verify ID Persistence (Critical for Age)
    print("\n🔍 Verifying Identity Persistence (Validator Age)...")
    for net in rep_df['network']:
        net_path = os.path.join(DATA_DIR, net)
        csvs = sorted([f for f in os.listdir(net_path) if f.endswith('.csv')])
        if not csvs: continue
        
        # Load first and last snapshot
        first = pd.read_csv(os.path.join(net_path, csvs[0]))
        last = pd.read_csv(os.path.join(net_path, csvs[-1]))
        
        # Check intersection of IDs
        ids_start = set(first['validator_id'])
        ids_end = set(last['validator_id'])
        retention = len(ids_start.intersection(ids_end)) / len(ids_start)
        
        print(f"   - {net}: {len(csvs)} days. ID Retention Rate: {retention:.1%} (Target > 80%)")

if __name__ == "__main__":
    check_continuity()


    