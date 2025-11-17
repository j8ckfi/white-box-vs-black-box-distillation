from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch

import config


def set_random_seed(seed: int) -> None:
    """Keep every framework on the same RNG seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(config.MAX_CPU_THREADS, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_teacher_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move teacher tensors that exist in the batch onto the target device."""
    teacher_data: Dict[str, torch.Tensor] = {}
    for key in ("teacher_logits", "teacher_hidden_state", "teacher_attention_map"):
        tensor = batch.get(key)
        if tensor is not None:
            teacher_data[key] = tensor.to(device)
    return teacher_data

