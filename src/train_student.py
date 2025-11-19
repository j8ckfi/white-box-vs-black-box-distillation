"""
Thin orchestration script for white-box vs. black-box KD sweeps.

All heavy lifting lives under `training/`, so this file now only handles
argument parsing, loop orchestration, and result collation.

IMPROVEMENTS:
- Resumable training: Checks if a (distill_type, seed) pair is already done.
- Single-pair execution: Can be run with specific distill_type/seed for granular control.
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import config
from training import run_trial, save_results

os.environ.setdefault("OMP_NUM_THREADS", str(config.MAX_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(config.MAX_CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(config.MAX_CPU_THREADS))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local KD training (no Ray / no DeepSpeed).")
    parser.add_argument(
        "--distill-types",
        type=str,
        default="black_box,hidden_state,attention,combined",
        help="Comma-separated list of distillation types to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated list of integer seeds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Override learning rate (defaults to config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size per optimizer step (defaults to config).",
    )
    parser.add_argument(
        "--single-trial",
        action="store_true",
        help="Run only the first distill_type/seed pair for a smoke test.",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing results and re-run everything.",
    )
    return parser.parse_args()


def _load_completed_runs() -> set[Tuple[str, int]]:
    """Load the set of (distill_type, seed) pairs that have already finished."""
    output_csv = Path(config.OUTPUT_PATH) / "results_summary.csv"
    completed = set()
    if output_csv.exists():
        try:
            df = pd.read_csv(output_csv)
            # Consider a run "complete" if it has reached the final epoch?
            # For simplicity, if it appears in the results CSV, we assume it's done 
            # or at least has some data. If we want strict completion check, we'd check epochs.
            # Let's assume if any entry exists for (type, seed) with epoch == NUM_EPOCHS - 1, it's done.
            
            # Check if 'epoch' column exists
            if "epoch" in df.columns and "distill_type" in df.columns and "seed" in df.columns:
                # Filter for max epoch
                max_epoch = config.NUM_EPOCHS - 1
                done_df = df[df["epoch"] == max_epoch]
                for _, row in done_df.iterrows():
                    completed.add((row["distill_type"], int(row["seed"])))
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    return completed


def _expand_combos(distill_types: List[str], seeds: List[int], single_trial: bool) -> List[Tuple[str, int]]:
    combos = [(distill_type, seed) for distill_type in distill_types for seed in seeds]
    if single_trial:
        print("Running in single-trial mode.")
        combos = combos[:1]
    return combos


def main() -> None:
    args = _parse_args()

    distill_types = [t.strip() for t in args.distill_types.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not distill_types:
        raise ValueError("At least one distillation type must be specified.")
    if not seeds:
        raise ValueError("At least one seed must be specified.")

    all_combos = _expand_combos(distill_types, seeds, args.single_trial)
    
    # Filter out completed runs unless forced
    completed_runs = _load_completed_runs() if not args.force_restart else set()
    
    combos_to_run = []
    for dt, seed in all_combos:
        if (dt, seed) in completed_runs:
            print(f"Skipping {dt} seed {seed} (already completed).")
        else:
            combos_to_run.append((dt, seed))
    
    if not combos_to_run:
        print("All requested combinations are already completed!")
        return

    print(f"Scheduled to run {len(combos_to_run)} trials.")

    for distill_type, seed in combos_to_run:
        print(f"\n{'='*40}")
        print(f"Starting: {distill_type} | Seed: {seed}")
        print(f"{'='*40}\n")
        
        try:
            trial_records = run_trial(
                distill_type=distill_type,
                seed=seed,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
            )
            # Save immediately after each trial to prevent data loss
            save_results(trial_records)
        except Exception as e:
            print(f"ERROR in trial {distill_type} seed {seed}: {e}")
            # Continue to next trial? Or crash?
            # For long sweeps, better to continue and log error.
            import traceback
            traceback.print_exc()

    print("\nAll scheduled trials finished.")


if __name__ == "__main__":
    main()
