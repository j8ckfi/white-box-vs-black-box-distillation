# Investigation Report: Why Values Are So Low

## Executive Summary
The extremely low validation accuracy (and high loss) is caused by two fundamental implementation errors:
1.  **Critical Training Logic Error**: The model is being trained to predict the *Answer* sequence given the *Prompt* sequence position-by-position, which is mathematically impossible.
2.  **Tokenizer Mismatch**: The Student (TinyLlama) and Teacher (Mistral) use different tokenizers. The KD implementation blindly compares their outputs, resulting in the minimization of divergence between completely unrelated tokens (e.g., forcing the student's "The" embedding to match the teacher's "apple" embedding).

## Detailed Analysis

### 1. The Training Logic Error (The "Nonsense" Objective)
In a standard Causal Language Model (like Llama/Mistral), training is done via "Teacher Forcing" on a concatenated sequence of `[Prompt, Answer]`. The model sees tokens `0...t` and predicts `t+1`.

**Your Implementation:**
- **Input**: `Prompt` only (e.g., "What is 2+2?")
- **Target**: `Answer` only (e.g., "The answer is 4")

The code calculates Cross-Entropy Loss between the logits of the *Prompt* and the tokens of the *Answer*.
- At position 0: The model sees "What". You ask it to predict "The".
- At position 1: The model sees "is". You ask it to predict "answer".
- At position 2: The model sees "2". You ask it to predict "is".

This destroys the model's language understanding because there is no causal link between the *n-th* token of the prompt and the *n-th* token of the answer. They are sequential, not parallel.

### 2. Tokenizer Mismatch (The "White Box" Failure)
Knowledge Distillation (KD) typically minimizes the KL Divergence between Student logits and Teacher logits:
$$ L_{KD} = KL(Student(x)_i || Teacher(x)_i) $$

This assumes that index $i$ in the Student's output corresponds to the same semantic unit (word/subword) as index $i$ in the Teacher's output.

**Verification Results:**
We verified the tokenizers for `Mistral-7B` and `TinyLlama-1.1B`:
- **Text**: "The quick brown fox..."
- **Teacher Tokens**: `[1, 415, 2936, ...]`
- **Student Tokens**: `[1, 450, 4996, ...]`

They are completely different.
- When the code calculates `kd_loss`, it penalizes the Student for not having the same probability distribution as the Teacher at index $i$.
- Since index $i$ represents different words for each model, the Student is being actively confused by the Teacher's signals.
- This applies to `hidden_state` and `attention` distillation as well—we are aligning features of different words.

## Accepted Knowledge Distillation Methods for LLMs

When Tokenizers Mismatch (as is common with different model families), you cannot use standard Logit/Hidden-State matching.

**1. Sequence-Level KD (SeqKD) / Supervised Fine-Tuning on Teacher Outputs**
- **Method**: The Teacher generates complete textual answers for the prompts. The Student is then trained on these `(Prompt, Teacher_Answer)` pairs using standard Supervised Fine-Tuning (SFT).
- **Why it works**: It ignores the internal representations (logits/hidden states) and focuses on the final output text.
- **Category**: Black Box (Output-only).

**2. Vocabulary Alignment (Complex)**
- **Method**: Construct a mapping matrix between Teacher vocab and Student vocab.
- **Pros/Cons**: Computationally expensive and imprecise for subword tokenizers. Generally not used in favor of SeqKD.

**3. Cross-Modal / Feature Alignment (Advanced White Box)**
- **Method**: Treat the models as having different modalities. Use contrastive learning or complex alignment heads that don't assume 1:1 sequence matching.
- **Pros/Cons**: Very complex to implement and tune.

## Recommendations

### Phase 1: Fix the Baseline (SFT)
Before fixing KD, we must fix the training loop to learn *anything* at all.
1.  **Data Loading**: Modify `OfflineDistillationDataset` to return `input_ids` as `[Prompt, Answer]` concatenated.
2.  **Labels**: Set `labels` to `[Prompt, Answer]`, but mask out the `Prompt` part (set to -100) so we only compute loss on the `Answer`.

### Phase 2: Fix Knowledge Distillation
Given the tokenizer mismatch, we have two paths:

**Path A: Switch to Compatible Teacher (Enables White Box)**
- **Action**: Change Teacher to `meta-llama/Llama-2-7b-hf` (or `13b`).
- **Why**: TinyLlama uses the Llama-2 tokenizer. The sequences will align 1:1.
- **Result**: `black_box` (Logits), `hidden_state`, and `attention` distillation will become valid and likely effective.

**Path B: Switch to Sequence-Level KD (Black Box Only)**
- **Action**: Use the current Mistral Teacher. Generate text answers instead of saving logits. Train Student on `(Prompt, Mistral_Answer)`.
- **Why**: Robust to tokenizer mismatch.
- **Result**: Effective "Black Box" distillation. "White Box" methods must be abandoned for this pair.

**Recommendation**: Since your project compares "White Box vs Black Box", **Path A** is better. It allows you to keep the current architecture and actually compare the methods fairly.

