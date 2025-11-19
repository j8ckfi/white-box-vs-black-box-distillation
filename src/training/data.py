from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

import config
from distillation_core import OfflineDistillationDataset, list_parquet_files


def build_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_offline_dataset(tokenizer: AutoTokenizer) -> OfflineDistillationDataset:
    parquet_files = list_parquet_files(config.OFFLINE_DATA_PATH)
    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found under {config.OFFLINE_DATA_PATH}. "
            "Please generate offline teacher data before training."
        )
    print(f"Loading offline dataset from {len(parquet_files)} files found in {config.OFFLINE_DATA_PATH}")
    return OfflineDistillationDataset(parquet_files, tokenizer, config.MAX_SEQ_LENGTH)


def split_dataset(dataset, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    val_size = max(1, int(len(dataset) * config.VALIDATION_SPLIT))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(
            f"Dataset too small ({len(dataset)} samples) for validation "
            f"split {config.VALIDATION_SPLIT}"
        )
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def create_dataloader(subset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=getattr(config, "DATALOADER_NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )

