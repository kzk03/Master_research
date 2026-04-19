#!/usr/bin/env python3
"""10プロジェクトのCSVを統合して combined_raw.csv を生成する。"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECTS = [
    "nova", "cinder", "neutron", "ironic", "glance",
    "keystone", "horizon", "swift", "heat", "octavia",
]

dfs = []
for proj in PROJECTS:
    csv_path = DATA_DIR / f"{proj}_raw.csv"
    df = pd.read_csv(csv_path)
    dfs.append(df)
    print(f"{proj}: {len(df)} rows")

combined = pd.concat(dfs, ignore_index=True)
output_path = DATA_DIR / "combined_raw.csv"
combined.to_csv(output_path, index=False)
print(f"\nCombined: {len(combined)} rows -> {output_path}")
