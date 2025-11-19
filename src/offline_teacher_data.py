"""
Offline Teacher Data Pre-computation Script

This script runs the teacher model (Llama-2-7b) once on all datasets
and saves the outputs (logits, hidden states, attention maps) to disk.
This enables efficient offline distillation without running the teacher
during student training.

CRITICAL CHANGE:
- Now processes (Prompt + Answer) concatenated.
- Saves 'prompt_length' to allow masking the prompt during training.
- Uses Llama-2 tokenizer (compatible with TinyLlama).
- Added tqdm progress bar.
"""

import os
import sys
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import time
from datetime import datetime, timedelta
import signal
import gc
from tqdm import tqdm  # Added tqdm

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config

TOP_K_LOGITS = getattr(config, "TOP_K_LOGITS", 128)
HIDDEN_STRIDE = max(1, getattr(config, "HIDDEN_STRIDE", 1))
ATTENTION_STRIDE = max(1, getattr(config, "ATTENTION_STRIDE", 1))

TASK_NAME_SST2 = "sst2"
TASK_NAME_MMLU = "mmlu"
TASK_NAME_GSM8K = "gsm8k"


def configure_temp_directories():
    """Pin temporary directories to the high-capacity drive when available."""
    temp_dir = getattr(config, "SYSTEM_TEMP_DIR", None)
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        os.environ["TMP"] = temp_dir
        os.environ["TEMP"] = temp_dir


configure_temp_directories()

# Module-level cache for model and tokenizer
_model_cache = {}
_tokenizer_cache = {}


def _to_py(value):
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        return [_to_py(elem) for elem in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_py(elem) for elem in value]
    return value


def _to_numpy_array(value, dtype):
    python_value = _to_py(value)
    try:
        return np.array(python_value, dtype=dtype)
    except (ValueError, TypeError):
        return np.stack([np.array(elem, dtype=dtype) for elem in python_value])


def _downsample_sequence(data: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return data
    return data[::stride]


def _downsample_attention(data: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return data
    return data[:, ::stride, ::stride]


def _extract_topk_logits(logits_tensor: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab = logits_tensor.size(-1)
    effective_k = min(top_k, vocab)
    values, indices = torch.topk(logits_tensor, k=effective_k, dim=-1)
    return values, indices


def get_model_and_tokenizer(model_name: str, hf_token: str = None):
    cache_key = model_name
    if cache_key in _model_cache and cache_key in _tokenizer_cache:
        return _model_cache[cache_key], _tokenizer_cache[cache_key]
    
    if not hf_token:
        hf_token = getattr(config, 'HUGGING_FACE_TOKEN', None) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    model_kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        **({"token": hf_token} if hf_token else {})
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.eval()
    
    _tokenizer_cache[cache_key] = tokenizer
    _model_cache[cache_key] = model
    return model, tokenizer


def validate_tensor_data(data: np.ndarray, name: str) -> bool:
    if data is None:
        raise ValueError(f"{name} is None")
    if np.isnan(data).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(data).any():
        raise ValueError(f"{name} contains Inf values")
    if data.size == 0:
        raise ValueError(f"{name} is empty")
    return True


def process_batch_with_teacher(batch: Dict[str, List[Any]], model_name: str, hf_token: str = None) -> Dict[str, List]:
    """
    Process a batch of prompts + answers through the teacher model.
    """
    model, tokenizer = get_model_and_tokenizer(model_name, hf_token)
    
    prompts = batch.get("prompt", [])
    answers = batch.get("answer", [])
    raw_task_names = batch.get("task_name", ["unknown"] * len(prompts))
    
    full_texts = []
    prompt_lens = []
    valid_indices = []

    # Pre-process strings
    for i, (p, a) in enumerate(zip(prompts, answers)):
        p_str = str(p) if p else ""
        a_str = str(a) if a else ""
        if not p_str.strip() or not a_str.strip():
            continue
        
        full_text = f"{p_str} {a_str}"
        full_texts.append(full_text)
        
        p_tokens = tokenizer(p_str, add_special_tokens=True)["input_ids"]
        prompt_lens.append(len(p_tokens))
        
        valid_indices.append(i)

    if not full_texts:
        return {}

    # Tokenize Full Texts
    inputs = tokenizer(
        full_texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )

    # Extract data
    topk_values, topk_indices = _extract_topk_logits(outputs.logits, TOP_K_LOGITS)
    topk_values = topk_values.detach().cpu().numpy().astype(np.float16)
    topk_indices = topk_indices.detach().cpu().numpy().astype(np.int32)
    
    teacher_hidden_state = outputs.hidden_states[-1].cpu().numpy().astype(np.float16)
    
    if outputs.attentions:
        teacher_attention_map = outputs.attentions[-1].cpu().numpy().astype(np.float16)
    else:
        raise RuntimeError("No attention maps returned from teacher model.")

    batch_size = input_ids.shape[0]
    result = {
        "prompt": [prompts[i] for i in valid_indices],
        "answer": [answers[i] for i in valid_indices],
        "task_name": [raw_task_names[i] for i in valid_indices],
        "prompt_length": prompt_lens,
        "teacher_topk_indices": [],
        "teacher_topk_values": [],
        "teacher_hidden_state": [],
        "teacher_attention_map": [],
        "input_ids": [],
        "attention_mask": [],
    }

    for i in range(batch_size):
        seq_len = int(attention_mask[i].sum().item())
        
        # Logits
        result["teacher_topk_indices"].append(topk_indices[i, :seq_len, :].tolist())
        result["teacher_topk_values"].append(topk_values[i, :seq_len, :].tolist())
        
        # Hidden States (strided)
        hidden = teacher_hidden_state[i, :seq_len, :]
        result["teacher_hidden_state"].append(_downsample_sequence(hidden, HIDDEN_STRIDE).tolist())
        
        # Attention (strided)
        attn = teacher_attention_map[i, :, :seq_len, :seq_len]
        result["teacher_attention_map"].append(_downsample_attention(attn, ATTENTION_STRIDE).tolist())
        
        # Input IDs
        result["input_ids"].append(input_ids[i, :seq_len].cpu().numpy().tolist())
        result["attention_mask"].append(attention_mask[i, :seq_len].cpu().numpy().tolist())

    # Cleanup
    del topk_values, topk_indices, teacher_hidden_state, teacher_attention_map, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return result


def load_and_preprocess_datasets() -> List[Dict[str, str]]:
    all_data = []
    
    # SST-2
    print("Loading SST-2 dataset...")
    sst2 = load_dataset(config.SST2_DATASET, config.SST2_CONFIG, split="train")
    SST2_SUBSET = 5000
    for ex in sst2.select(range(min(SST2_SUBSET, len(sst2)))):
        prompt = str(ex.get("sentence", "")).strip()
        ans = "positive" if ex.get("label") == 1 else "negative"
        if prompt:
            all_data.append({"prompt": prompt, "answer": ans, "task_name": TASK_NAME_SST2})

    # MMLU
    print("Loading MMLU dataset...")
    try:
        mmlu = load_dataset(config.MMLU_DATASET, "all", split="auxiliary_train")
        for ex in mmlu.select(range(min(1000, len(mmlu)))):
            q = str(ex.get("question", "")).strip()
            choices = ex.get("choices", [])
            c_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            p = f"Question: {q}\nChoices:\n{c_str}\nAnswer:"
            a = str(ex.get("answer", ""))
            if q:
                all_data.append({"prompt": p, "answer": a, "task_name": TASK_NAME_MMLU})
    except Exception as e:
        print(f"MMLU load failed: {e}")

    # GSM8K
    print("Loading GSM8K dataset...")
    try:
        gsm = load_dataset(config.GSM8K_DATASET, "main", split="train")
        for ex in gsm.select(range(min(1000, len(gsm)))):
            p = str(ex.get("question", "")).strip()
            a = str(ex.get("answer", "")).strip()
            if p and a:
                all_data.append({"prompt": p, "answer": a, "task_name": TASK_NAME_GSM8K})
    except Exception as e:
        print(f"GSM8K load failed: {e}")

    print(f"Total examples: {len(all_data)}")
    return all_data


def _flatten_result_rows(result_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    rows = []
    num_items = len(result_dict.get("prompt", []))
    for i in range(num_items):
        rows.append({k: result_dict[k][i] for k in result_dict})
    return rows


def _batch_iterator(data: List[Dict[str, str]], batch_size: int):
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        yield {
            "prompt": [x["prompt"] for x in batch],
            "answer": [x["answer"] for x in batch],
            "task_name": [x.get("task_name", "unknown") for x in batch]
        }


def _append_to_parquet(rows, writer, output_file):
    if not rows:
        return writer, 0
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(
            output_file,
            table.schema,
            compression=getattr(config, "PARQUET_COMPRESSION", "zstd"),
            compression_level=getattr(config, "PARQUET_COMPRESSION_LEVEL", 3),
            use_dictionary=True,
        )
    writer.write_table(table)
    return writer, len(rows)


def main():
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    
    hf_token = getattr(config, 'HUGGING_FACE_TOKEN', None) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run on a tiny subset for verification")
    args = parser.parse_args()

    all_data = load_and_preprocess_datasets()
    
    if args.smoke_test:
        print("\n⚠️ SMOKE TEST MODE: Using only 10 examples")
        all_data = all_data[:10]

    batch_size = int(os.getenv("OFFLINE_BATCH_SIZE", "4")) # Increased default slightly for Llama-2-7b on big GPU
    
    output_dir = config.OFFLINE_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "offline_teacher_data.parquet")
    if os.path.exists(output_file):
        os.remove(output_file)

    writer = None
    rows_written = 0
    start_time = time.time()
    
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    
    try:
        # WRAPPED IN TQDM
        with tqdm(total=len(all_data), desc="Generating Data", unit="ex") as pbar:
            for batch in _batch_iterator(all_data, batch_size):
                res = process_batch_with_teacher(batch, config.TEACHER_MODEL_NAME, hf_token)
                rows = _flatten_result_rows(res)
                writer, n = _append_to_parquet(rows, writer, output_file)
                rows_written += n
                pbar.update(n)
                
    finally:
        if writer:
            writer.close()
            
    print(f"Done. Wrote {rows_written} examples to {output_file}")

if __name__ == "__main__":
    main()
