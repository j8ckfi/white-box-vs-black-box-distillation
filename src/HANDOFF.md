## Whitebox vs Blackbox KD – Handoff Notes

### Current Status (2025-11-19)
- **Refactored for Fairness:**
  - **Effective Batch Size:** Now normalized to **64** for all methods. `config.py` defines `TARGET_EFFECTIVE_BATCH_SIZE`, and `trainer.py` dynamically calculates gradient accumulation steps.
    - Black Box: Batch 16 * 4 accum = 64
    - White Box: Batch 8 * 8 accum = 64
  - **Hidden State Projector:** Restored the learnable linear projector (2048 -> 4096) in `distillation_student.py`. This allows using **MSE Loss** instead of the problematic CKA loss for hidden state alignment.

- **Data:** Mixed-task offline teacher data (SST-2 + MMLU + GSM8K) lives in `/content/drive/MyDrive/wbvb/output/offline_teacher_data.parquet`.
- **Codebase:** Standard PyTorch training loop (no DeepSpeed). Fits comfortably on Colab Pro+ (A100) or even T4.

### Recommended Workflow (Colab Pro+)
The repo structure is flat (or close to it). Use this block to clone, patch, and run:

```python
# 1. Mount Drive
from google.colab import drive
import os
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2. Clone & Locate Code
%cd /content
!rm -rf white-box-vs-black-box-kd-llms
!git clone https://github.com/j8ck1632/white-box-vs-black-box-kd-llms.git

# Auto-detect directory
import glob
found = glob.glob("white-box-vs-black-box-kd-llms/**/train_student.py", recursive=True)
if not found: raise RuntimeError("Repo structure changed!")
target_dir = os.path.dirname(found[0])
os.chdir(target_dir)
!pip install -r requirements.txt

# 3. Apply Patches (Projector + Dynamic Batch Size)
# [Insert the long Python patch script provided by the agent here]
# See: distillation_student.py (Projector restored) and config.py (Target Batch Size)

# 4. Run Training
os.environ["WBVB_OFFLINE_DATA_PATH"] = "/content/drive/MyDrive/wbvb/output"
os.environ["WBVB_OUTPUT_PATH"] = "/content/drive/MyDrive/wbvb/results"

!PYTHONUNBUFFERED=1 python -u train_student.py \
    --distill-types black_box,hidden_state,attention,combined \
    --seeds 0,1,2,3,4,5,6
```

### Key Fixes Implemented
1.  **Restored Projector:** `DistillationStudent` now has a `nn.Linear(2048, 4096)`.
2.  **Dynamic Gradient Accumulation:** `trainer.py` ensures `batch_size * accum_steps ≈ 64`.
3.  **Removed DeepSpeed:** The code is standard PyTorch AMP; DeepSpeed flags are not supported and were causing confusion.

### Notes / Caveats
- **Repo Structure:** The directory structure might vary. The script above uses `glob` to find the correct folder.
- **Results:** `results_summary.csv` will now reflect fair comparisons. Expect White Box methods to perform significantly better than before.
- **Memory:** `WHITEBOX_BATCH_SIZE` defaults to 8. If you hit OOM on T4, lower this in `config.py` (the script will automatically increase accumulation steps to compensate).