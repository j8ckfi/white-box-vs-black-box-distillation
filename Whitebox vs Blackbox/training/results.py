from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

import config


def save_results(records: List[Dict[str, float]]) -> None:
    if not records:
        print("No records to save.")
        return
    df = pd.DataFrame(records)
    output_dir = Path(config.OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "results_summary.csv"
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(records)} record(s) to {output_csv}")

