# Finetuning Experiment 1 — Llama-3.2-1B-Instruct on Banking77

Fine-tuning pipeline for `unsloth/Llama-3.2-1B-Instruct` on the `mteb/banking77` 77-class intent classification benchmark.

**Two environments supported:**
- **Google Colab (T4 GPU)** — primary, recommended. See [`colab/`](colab/).
- **Local CPU (Intel Iris Xe)** — fallback. See [`src/`](src/).

---

## Quick Start (Colab)

1. Open [`colab/banking77_finetune.ipynb`](colab/banking77_finetune.ipynb) in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Add `WANDB_API_KEY` to Colab Secrets (🔑 sidebar)
4. Runtime → Run all → authorise Drive when prompted

Full instructions: [`colab/run_instruct.txt`](colab/run_instruct.txt)

---

## Results (Colab T4, 3 epochs, ~50 min)

| Setting | Accuracy | Correct / Total |
|---|---|---|
| Base model (zero-shot) | 0.00% | 0 / 3,076 |
| Fine-tuned (free generation) | **90.21%** | 2,775 / 3,076 |
| Fine-tuned (constrained decoding) | **90.28%** | 2,777 / 3,076 |

Full analysis: [`colab/README.md`](colab/README.md)

---

## Repository Structure

```
colab/
├── banking77_finetune.ipynb   # Self-contained Colab notebook (primary)
├── README.md                  # Detailed methodology, results, and decisions
└── run_instruct.txt           # Step-by-step run guide + troubleshooting

src/                           # Local CPU scripts (Intel Iris Xe fallback)
├── 1_prepare_data.py          # Download & format dataset → data/
├── 2_train_intel.py           # Train LoRA adapter → models/
└── 3_resume_and_infer.py      # Auto-resume + inference test

setup_env.sh                   # Environment setup (uv, Unsloth source install)
CLAUDE.md                      # Developer notes and constraints
```

---

## Training Stack

| Component | Library |
|---|---|
| Base model + LoRA | Unsloth (Colab) / PEFT (local) |
| SFT training loop | TRL `SFTTrainer` |
| Quantisation | bitsandbytes (4-bit NF4, Colab only) |
| Experiment tracking | Weights & Biases |
| Dataset | `mteb/banking77` |
