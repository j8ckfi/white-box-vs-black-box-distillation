from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from distillation_core import compute_loss

from .metrics import answers_match, normalize_text
from .utils import prepare_teacher_batch


def compute_batch_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: AutoTokenizer,
    task_names: List[str],
    task_totals: Dict[str, int],
    task_correct: Dict[str, int],
) -> Tuple[int, int]:
    preds = logits.argmax(dim=-1)
    total = 0
    correct = 0
    for pred_seq, label_seq, task_name in zip(preds, labels, task_names):
        mask = label_seq != -100
        if mask.sum() == 0:
            continue
        pred_tokens = pred_seq[mask]
        label_tokens = label_seq[mask]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        target_text = tokenizer.decode(label_tokens, skip_special_tokens=True)
        pred_norm = normalize_text(pred_text)
        target_norm = normalize_text(target_text)
        if answers_match(pred_norm, target_norm):
            correct += 1
            task_correct[task_name] = task_correct.get(task_name, 0) + 1
        task_totals[task_name] = task_totals.get(task_name, 0) + 1
        total += 1
    return correct, total


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    distill_type: str,
    tokenizer: AutoTokenizer,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    evaluated = 0
    num_batches = 0
    task_totals: Dict[str, int] = {}
    task_correct: Dict[str, int] = {}
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_data = prepare_teacher_batch(batch, device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden_states=distill_type in ["hidden_state", "combined"],
                return_attention=distill_type in ["attention", "combined"],
                output_attentions=distill_type in ["attention", "combined"],
            )
            losses = compute_loss(outputs, labels, teacher_data, distill_type)
            total_loss += losses["task_loss"].item()
            batch_correct, batch_total = compute_batch_accuracy(
                outputs["logits"],
                labels,
                tokenizer,
                batch.get("task_name", ["unknown"] * labels.size(0)),
                task_totals,
                task_correct,
            )
            correct += batch_correct
            evaluated += batch_total
            num_batches += 1
    model.train()
    if num_batches == 0:
        return 0.0, 0.0, {}
    accuracy = (correct / evaluated) if evaluated else 0.0
    per_task = {
        task: task_correct.get(task, 0) / total if total else 0.0
        for task, total in task_totals.items()
    }
    return total_loss / num_batches, accuracy, per_task

