import pandas as pd
import requests
import os
import io

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Main CSV URL found by the agent
csv_url = "https://data.nrel.gov/system/files/305/1765565264-combined_wind_experiments_0.csv"
csv_path = os.path.join(DATA_DIR, "nrel_305_wind.csv")

print(f"Downloading {csv_url}...")
try:
    response = requests.get(csv_url, timeout=30)
    response.raise_for_status()
    with open(csv_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
    
    # Validation
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    print("Columns:", df.columns.tolist())
    print(df.describe())
    
    if len(df) < 500:
        print("WARNING: Data row count low.")
    
except Exception as e:
    print(f"Failed to download/process NREL data: {e}")
