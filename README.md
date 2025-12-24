# Attention Alignment Outperforms Logit Distillation for LLM Compression

[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A systematic comparison of white-box and black-box knowledge distillation for compressing **Llama-2-7B** into **TinyLlama-1.1B** (6.4× compression).

## Key Findings

| Method | Accuracy | vs Baseline |
|--------|:--------:|:-----------:|
| **Attention Distillation** | **95.56%** | **+1.58%** |
| Combined (Attn + Hidden) | 95.56% | +1.58% |
| Hidden State Only | 94.71% | +0.73% |
| Black-Box (Logits) | 93.98% | — |

**TL;DR:** Attention alignment provides +1.58% accuracy over standard logit distillation. Adding hidden states provides no additional benefit—attention is the key signal.

## Paper

The full paper is in [`paper/main.pdf`](paper/main.pdf).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate figures from data
python generate_figures.py

# Run statistical analysis
python statistical_analysis.py
```

## Repository Structure

```
├── paper/
│   ├── main.tex          # LaTeX source
│   ├── main.pdf          # Compiled paper
│   └── references.bib    # Bibliography
├── figures/              # Publication-quality plots
├── src/                  # Experiment code
│   ├── distillation_core.py
│   ├── distillation_student.py
│   ├── train_student.py
│   └── offline_teacher_data.py
├── latestsummary.csv     # Final experiment results
└── generate_figures.py   # Figure generation script
```

## Methodology

We compare four distillation approaches:

1. **Black-Box**: KL divergence on output logits only
2. **Hidden State**: Logits + final-layer hidden state alignment (MSE)
3. **Attention**: Logits + final-layer attention map alignment (MSE)
4. **Combined**: All signals

Evaluated on SST-2 (sentiment), MMLU (reasoning), and GSM8K (math) with N=7 seeds per method.

## Citation

```bibtex
@article{large2024attention,
  title={Attention Alignment Outperforms Logit Distillation for LLM Compression},
  author={Large, Jack},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT
