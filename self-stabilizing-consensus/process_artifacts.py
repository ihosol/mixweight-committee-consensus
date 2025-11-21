import os
import json
import shutil
import pandas as pd

# CONFIGURATION
SOURCE_DIR = './artifacts'      # Where your unzipped folders are
TARGET_DIR = './data'           # Where clean CSVs will go
METRICS_FILE = './data/global_metrics_history.csv'

def parse_iso_date(date_str):
    """Extracts YYYY-MM-DD from ISO string."""
    try:
        return date_str.split('T')[0]
    except (AttributeError, IndexError):
        return None

def run_etl():
    print(f"🚀 Starting ETL from {SOURCE_DIR}...")
    
    # Prepare output
    if os.path.exists(TARGET_DIR): shutil.rmtree(TARGET_DIR)
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    all_summaries = []
    files_processed = 0

    for root, dirs, files in os.walk(SOURCE_DIR):
        print(f" files in {files}")

        # --- PART A: Harvest Global Metrics (Gini) ---
        if 'summary.csv' in files:
            try:
                s_path = os.path.join(root, 'summary.csv')
                df = pd.read_csv(s_path)
                # Standardize Date
                if 'snapshot_time' in df.columns:
                    df['date'] = df['snapshot_time'].apply(parse_iso_date)
                    all_summaries.append(df)
            except Exception as e:
                print(f"⚠️ Skipped summary in {root}: {e}")

        # --- PART B: Organize Validator Lists ---
        # We pair .jsonl (metadata) with .csv (data)
        jsonl_files = [f for f in files if f.endswith('.jsonl')]
        print(f" jsonl_files in {jsonl_files}")

        for j_file in jsonl_files:
            network = j_file.replace('.jsonl', '') # e.g., 'solana'
            csv_file = f"{network}_weights.csv"
            
            if csv_file not in files: continue # Skip incomplete pairs
            
            try:
                # 1. Get Date from JSONL
                with open(os.path.join(root, j_file), 'r') as f:
                    meta = json.loads(f.readline())
                
                date_str = parse_iso_date(meta.get('snapshot_time'))
                if not date_str: continue

                # 2. Create Network Folder
                net_dir = os.path.join(TARGET_DIR, network)
                os.makedirs(net_dir, exist_ok=True)
                
                # 3. Copy & Rename CSV
                src = os.path.join(root, csv_file)
                dst = os.path.join(net_dir, f"{date_str}.csv")
                shutil.copy2(src, dst)
                
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {network}: {e}")

    # --- PART C: Save Master Metrics ---
    if all_summaries:
        full_df = pd.concat(all_summaries, ignore_index=True)
        # Keep only useful columns for Part 2
        keep_cols = ['date', 'chain', 'n', 'gini', 'hhi', 'k33']
        final_df = full_df[[c for c in keep_cols if c in full_df.columns]]
        
        final_df = final_df.sort_values(['chain', 'date']).drop_duplicates()
        final_df.to_csv(METRICS_FILE, index=False)
        print(f"📊 Master metrics saved to {METRICS_FILE}")

    print(f"✅ Organized {files_processed} validator snapshots into '{TARGET_DIR}/'")

if __name__ == "__main__":
    run_etl()
