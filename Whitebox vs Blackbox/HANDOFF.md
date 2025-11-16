## Whitebox vs Blackbox KD – Handoff Notes

### Current Status (2025-11-16)
- Mixed-task offline teacher data (SST-2 + MMLU + GSM8K) is verified and ready under `config.OFFLINE_DATA_PATH`.
- `train_student.py` now supports optional **DeepSpeed ZeRO‑3 with CPU offload** (`--use-deepspeed --deepspeed-config ds_config_zero3.json`) so we can fully exploit Colab Pro+/A100 (40 GB VRAM + 150 GB host RAM) for much larger batches.
- Dataloader worker counts are configurable via `config.DATALOADER_NUM_WORKERS`.
- `requirements.txt` installs DeepSpeed; the default DS config (`ds_config_zero3.json`) ships next to the training script.

### Recommended Workflow
1. **Regenerate teacher data only if necessary**
   ```bash
   %cd "/content/wbvb/Whitebox vs Blackbox"
   !python offline_teacher_data.py
   ```
   Outputs land in `/content/drive/MyDrive/wbvb/output/offline_teacher_data.parquet`. Verification samples 20 random rows and fails loudly on malformed tensors.

2. **Colab Pro+ / A100 training loop**
   1. `git clone` → `%cd /content/wbvb/Whitebox vs Blackbox`
   2. `pip install -r requirements.txt && pip install deepspeed`
   3. (Optional) `virtualenv .wbvb-env && source .wbvb-env/bin/activate`
   4. Point the run at Drive:
      ```python
      import os, pathlib
      os.environ["WBVB_OFFLINE_DATA_PATH"] = "/content/drive/MyDrive/wbvb/output"
      os.environ["WBVB_OUTPUT_PATH"] = "/content/drive/MyDrive/wbvb/results"
      pathlib.Path(os.environ["WBVB_OUTPUT_PATH"]).mkdir(parents=True, exist_ok=True)
      ```
   5. For the A100, bump `config.py`: `BATCH_SIZE = 40`, `WHITEBOX_BATCH_SIZE = 10`, `DATALOADER_NUM_WORKERS = 4`.
   6. Launch:
      ```bash
      %cd "/content/wbvb/Whitebox vs Blackbox"
      !PYTHONUNBUFFERED=1 python -u train_student.py \
          --use-deepspeed \
          --deepspeed-config ds_config_zero3.json \
          --distill-types black_box,hidden_state,attention,combined \
          --seeds 0,1,2,3,4,5,6
      ```
   7. Summarize results:
      ```bash
      %cd "/content/wbvb"
      !python scripts/summarize_results.py --csv /content/drive/MyDrive/wbvb/results/results_summary.csv
      ```

3. **Local CPU/GPU runs (no DeepSpeed)**
   - Keep `BATCH_SIZE=8`, `WHITEBOX_BATCH_SIZE=2`, `DATALOADER_NUM_WORKERS=0`.
   - Launch `python train_student.py --distill-types ... --seeds ...`.
   - AMP + GradScaler kick in automatically on CUDA; CPU falls back to fp32.

### Notes / Caveats
- When DeepSpeed is active the script unwraps `engine.module` for evaluation, so per-task accuracies remain correct.
- `ds_config_zero3.json` sets ZeRO‑3 with CPU offload and bf16. Tweak it if you prefer pure-GPU training or different accumulation.
- Console `val_acc` is the blended metric; check the `validation_accuracy_<task>` columns in the CSVs for SST-2/MMLU/GSM8K.
- `results_summary.csv` grows quickly—archive or filter before presenting numbers.

Ping me if you need more detail on DeepSpeed or the Colab workflow. Good luck!

