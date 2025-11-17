"""
Thin orchestration script for white-box vs. black-box KD sweeps.

All heavy lifting lives under `training/`, so this file now only handles
argument parsing, loop orchestration, and result collation.
"""

from __future__ import annotations

import argparse
import os
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
    return parser.parse_args()


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

    combos = _expand_combos(distill_types, seeds, args.single_trial)

    all_records: List[Dict[str, float]] = []
    for distill_type, seed in combos:
        trial_records = run_trial(
            distill_type=distill_type,
            seed=seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        all_records.extend(trial_records)

    save_results(all_records)
    print("Training run complete.")


if __name__ == "__main__":
    main()

