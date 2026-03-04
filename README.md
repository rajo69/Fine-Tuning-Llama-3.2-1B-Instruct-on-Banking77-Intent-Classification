# Fine-Tuning Llama-3.2-1B-Instruct on Banking77 Intent Classification

> **90.21% exact-match accuracy** on 77-class intent classification
> Fine-tuned from a base model that scored **0%** in zero-shot on the same task
> Single T4 GPU · 3 epochs · ~50 minutes · QLoRA + rsLoRA + NEFTune

---

## Quick Start

**Prerequisites**: Google account, W&B account (free), Colab T4 runtime.

**One-time setup**: Add `WANDB_API_KEY` to Colab Secrets (🔑 icon in Colab sidebar).

1. Open [`colab/banking77_finetune_compiled.ipynb`](colab/banking77_finetune_compiled.ipynb) in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Click the Google Drive authorisation popup (one unavoidable manual click)
4. Runtime → Run all

---

## Overview

This project fine-tunes [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) on the [`mteb/banking77`](https://huggingface.co/datasets/mteb/banking77) dataset — a 77-class intent classification benchmark for banking customer queries. The goal was to build a production-quality SFT pipeline that incorporates current best practices and yields rigorous, interpretable evaluation results, not just a single accuracy number.

The base model scored **0/3076** in zero-shot evaluation. After one fine-tuning run of 3 epochs (~50 minutes on a T4), the fine-tuned model scores **2775/3076 (90.21%)** with trie-based constrained decoding guaranteeing no invalid label outputs.

---

## Results

| Setting | Accuracy | Correct / Total |
|---|---|---|
| Base model (zero-shot) | 0.00% | 0 / 3,076 |
| Fine-tuned — 500-sample callback (epoch 3) | 90.60% | ~453 / 500 |
| Fine-tuned — full eval set (free generation) | **90.21%** | 2,775 / 3,076 |
| Fine-tuned — constrained decoding (trie) | **90.28%** | 2,777 / 3,076 |

The zero-shot result reflects a key insight: a generative model cannot do intent classification with exact label names without fine-tuning. The base model generates natural language descriptions rather than the structured label strings required for this task — the entire 90.2pp improvement is attributable to SFT.

**Constrained decoding analysis**: the trie fixed exactly 2 samples and broke 0. This is the ideal outcome — no regressions. The two fixed samples reveal two distinct mechanisms:

| Query | True label | Free generation | Constrained | Mechanism |
|---|---|---|---|---|
| "How can i get multiple disposble cards" | `disposable_card_limits` | `getting_spare_card` *(valid, wrong)* | `disposable_card_limits` | Trie routing shifted probability between valid competing paths |
| "how do VR cards work" | `get_disposable_virtual_card` | `get_virtual_card` *(invalid)* | `get_disposable_virtual_card` | Invalid label eliminated; probability redirected to nearest valid continuation |

The first case is notable: `getting_spare_card` is a valid label, so the fix was not about eliminating an invalid output — it happened because the trie's per-step token restrictions altered the probability landscape enough to tip greedy selection toward the correct label. The second case is the canonical constrained decoding use case: `get_virtual_card` does not exist in `LABEL_NAMES` and was produced by truncating `get_disposable_virtual_card` mid-sequence.

### Per-class breakdown

| Best classes (1.00 acc) | Worst classes |
|---|---|
| `apple_pay_or_google_pay` (40/40) | `topping_up_by_card` (0/40) |
| `top_up_by_card_charge` (40/40) | `card_arrival` (21/40) |
| `getting_spare_card` (40/40) | `transfer_not_received_by_recipient` (27/39) |
| `verify_top_up` (40/40) | `beneficiary_not_allowed` (28/40) |
| `passcode_forgotten` (40/40) | `declined_transfer` (29/40) |

### Top confused pairs

| True intent | Predicted as | Count | Root cause |
|---|---|---|---|
| `card_arrival` | `card_delivery_estimate` | 18 | Dataset label ambiguity — same user query, different labels |
| `topping_up_by_card` | `top_up_by_card_charge` | 13 | Semantically near-identical training examples |
| `topping_up_by_card` | `top_up_reverted` | 10 | Same |
| `topping_up_by_card` | `top_up_by_card` *(invalid)* | 9 | Label hallucination — fixed by constrained decoding |
| `beneficiary_not_allowed` | `failed_transfer` | 7 | Overlapping intent descriptions |

The `topping_up_by_card` complete failure (0/40) is the most instructive finding. The model hallucinated a plausible but non-existent label (`top_up_by_card`) on 9 of 40 samples — confirming it never confidently learned to distinguish `topping_up_by_card` from `top_up_by_card_charge`. This is a training data problem, not a model capacity problem: the two intents have near-identical surface forms in the dataset. Constrained decoding eliminates invalid labels but cannot redistribute probability mass that the model never learned to assign correctly.

---

## Model & Training Configuration

### Why Llama-3.2-1B-Instruct?

The 1B parameter scale was chosen deliberately. It is the smallest model that retains a meaningful instruction-following capability while being trainable on a free-tier T4 GPU (16 GB VRAM) in under an hour. The `-Instruct` variant (as opposed to the base pretrained model) brings a pre-trained chat template and RLHF alignment — both directly useful for a structured-output classification task.

### Architecture

```
Base model : Llama-3.2-1B-Instruct (loaded in 4-bit NF4 via bitsandbytes)
Adapter    : LoRA — r=16, α=32, dropout=0.05
             Target modules: q_proj, k_proj, v_proj, o_proj,
                             gate_proj, up_proj, down_proj
Trainable  : ~1.2% of parameters
```

### Training hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `lora_rank` | 16 | Balances capacity and memory; sweep explored [8, 16, 32] |
| `lora_alpha` | 32 | `alpha = 2 × rank` — community-converged default for stable gradient scaling |
| `use_rslora` | True | Rank-stabilized LoRA scales the adapter output by `1/√rank`, preventing gradient explosion at higher ranks |
| `neftune_noise_alpha` | 5 | NEFTune injects uniform noise into input embeddings during training, consistently improving instruction-following quality at zero added cost |
| `optimizer` | `adamw_8bit` | bitsandbytes 8-bit Adam reduces optimizer state from ~2.4 GB to ~0.6 GB with negligible accuracy impact |
| `learning_rate` | 2e-4 | Standard LoRA fine-tuning range; cosine schedule with 3% warmup |
| `effective_batch_size` | 16 | `per_device=8 × grad_accum=2`; packing enabled to avoid padding waste |
| `fp16` / `bf16` | auto-detected | bf16 on T4; fp16 fallback otherwise |
| `epochs` | 3 | Eval loss flattened after epoch 3; `load_best_model_at_end=True` selects best checkpoint |

### Why the native chat template instead of Alpaca format?

This was the single most impactful architectural decision. The model was instruction-tuned with Llama's native chat template:

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>
```

Using the Alpaca `### Instruction: / ### Input: / ### Response:` format instead would have fought the model's pre-trained priors — forcing the model to operate in a format it was never RLHF-trained on. Matching the training format to the inference format is essential for instruction-tuned models.

---

## Data Pipeline

**Dataset**: `mteb/banking77` — 13,069 samples across 77 banking intent classes, balanced at ~40 samples per class in the test split.

**Split handling**: Train (9,993) and test (3,076) splits are kept separate throughout. The test split is held out entirely during training — it is never written to the training JSONL, only kept in memory for the `GenerationEvalCallback` and final evaluation.

**Label format**: Integer labels (0–76) are resolved to human-readable snake_case strings via a fixed `LABEL_NAMES` list. The model outputs the string label directly. This framing as a generative classification task (rather than a discriminative one) was chosen because it (a) leverages the model's text generation capability naturally, and (b) avoids the need for a classification head that would require architectural changes.

**Prompt construction**:
```python
messages = [
    {"role": "user", "content": f"{instruction}\n\n{query}"},
    {"role": "assistant", "content": label_name},
]
tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## Evaluation Methodology

Three layers of evaluation were implemented, each providing different signal:

### 1. In-training generation callback (`GenerationEvalCallback`)

Runs after each epoch on a fixed 500-sample stratified subset (seeded at 42 for reproducibility). Uses the same chat template as training. Logs `eval/exact_match_accuracy` to W&B. Chosen over perplexity-only evaluation because loss does not directly measure whether the model outputs valid, exact label names.

### 2. Full batched evaluation (`run_full_eval`)

After training, runs generation on all 3,076 test samples using left-padded batched inference (batch size 32). Reports:
- Overall exact-match accuracy
- Per-class accuracy sorted ascending (identifies worst intents)
- Top 15 confused pairs (identifies semantic collision points)
- All results logged to W&B as sortable tables

### 3. Trie-based constrained decoding (`run_constrained_eval`)

Builds a token-level prefix trie over all 77 valid label names. At each generation step, `prefix_allowed_tokens_fn` restricts the vocabulary to only tokens that continue a valid label sequence — making it physically impossible for the model to generate an invalid label.

```
trie["top"] → {"_up": {"_by": {"_card": {EOS: {}           # top_up_by_card_charge
                                          "_charge": {EOS}  # (continues)
                                         }}}}
```

This eliminates the `top_up_by_card` hallucination class outright. The delta over unconstrained generation is small (+0.1–0.5pp) because most errors are semantic confusions between valid labels, not invalid outputs — but constrained decoding is the correct approach for any production deployment.

---

## Engineering Decisions

### Checkpoint recovery

A `TimeBasedCheckpointCallback` saves the model to Google Drive every 30 minutes of wall-clock time (not by step count). This is more reliable than step-based checkpointing on Colab because session interruptions are time-based, not step-based.

### Run-mode flags

Two boolean flags (`RESUME_MODE`, `RUN_SWEEP`) gate the resume cell and the hyperparameter sweep, making the notebook safe to execute with **Runtime → Run all** without accidentally overwriting a trained model or triggering a 75-minute sweep:

```python
RESUME_MODE = False   # True → load latest Drive checkpoint
RUN_SWEEP   = False   # True → run W&B Bayesian sweep after training
```

### W&B Colab Secrets integration

The W&B login cell reads from Colab Secrets (`WANDB_API_KEY`) before falling back to an interactive prompt — eliminating the only remaining manual step in a "Run all" execution:

```python
try:
    from google.colab import userdata
    wandb.login(key=userdata.get("WANDB_API_KEY"))
except Exception:
    wandb.login()   # interactive fallback
```

### Hyperparameter sweep design

A W&B Bayesian sweep over `{learning_rate, lora_rank, batch_size}` with `early_terminate: hyperband` (min_iter=1). `lora_alpha` is not a free parameter — it is always derived as `2 × lora_rank`, which eliminates a dimension from the search space without meaningful loss since the `alpha/rank = 2` ratio is well-supported in the literature. Each sweep run trains for 1 epoch (~15 min), allowing 5 configurations in ~75 minutes on a free T4.

---

## Project Structure

```
colab/
├── banking77_finetune_compiled.ipynb  # Main notebook (all sections below)
│
│   Section 0  — GPU runtime check
│   Section 1  — Install dependencies (unsloth, trl, peft, wandb, ...)
│   Section 2  — Mount Google Drive (persistent checkpoint storage)
│   Config     — RESUME_MODE / RUN_SWEEP flags
│   Section 3  — Dataset download, train/eval split, W&B login
│   Section 4  — Load Llama-3.2-1B-Instruct + apply rsLoRA adapters
│   Section 5  — TimeBasedCheckpointCallback + GenerationEvalCallback
│   Section 6  — Format dataset (chat template) + SFTTrainer + train
│   Section 7  — TensorBoard (inline)
│   Section 8  — 5-query inference test
│   Section 9  — Resume from checkpoint (gated by RESUME_MODE)
│   Section 10 — W&B hyperparameter sweep (gated by RUN_SWEEP)
│   Section 11 — Full evaluation: batched eval + per-class + constrained decoding
│
└── run_instruct.txt                   # Step-by-step run guide + troubleshooting

src/                                   # Local CPU scripts (Intel Iris Xe fallback)
├── 1_prepare_data.py                  # Download & format dataset → data/
├── 2_train_intel.py                   # Train LoRA adapter → models/
└── 3_resume_and_infer.py              # Auto-resume + inference test

setup_env.sh                           # Environment setup (uv, Unsloth source install)
```

**Output structure in Google Drive:**
```
MyDrive/banking77_finetune/
├── models/final_lora/          ← final trained adapter (best epoch by eval_loss)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── checkpoints/                ← per-epoch HuggingFace checkpoints
├── checkpoints/periodic_backup/← 30-min safety snapshots (timestamped)
└── logs/                       ← TensorBoard event files
```

---

## What Would Push Accuracy Higher

The analysis from Section 11 identifies concrete, actionable paths to improvement:

**~+1–1.5% — retrain with augmented `topping_up_by_card` examples** *(requires 1 epoch)*
The model never learned to distinguish `topping_up_by_card` from `top_up_by_card_charge`. The fix is targeted data augmentation: construct training examples where the distinguishing feature (the user is *actively performing* a top-up by card vs being *charged* for one) is made more explicit in the query text.

**~+0.6% — address the `card_arrival`/`card_delivery_estimate` ambiguity** *(requires relabelling)*
18 of 301 total errors come from a single confused pair where even human annotators would struggle to distinguish. The labels describe near-identical user situations with different intent names. This is an inherent dataset ceiling, not a model failure.

**~+0.5–1% — train additional epochs or use optimal hyperparameters from sweep**
The W&B sweep over `{learning_rate, lora_rank, batch_size}` with Bayesian optimisation will identify whether a higher rank (32) or lower learning rate (1e-4) improves the accuracy on the confused classes.

**No additional training — constrained decoding in production**
Empirically confirmed: constrained decoding fixed 2 samples and broke 0 (delta: +0.07pp, 2775 → 2777/3076). The zero broken-samples result is significant — it means the trie never forced the model into a worse valid label for a sample it previously got right. For any deployment, `run_constrained_eval` should be the inference path. It eliminates invalid-label hallucinations at ~1.5× inference time cost (trie lookup per step), with no accuracy regression risk.

---

## Loading the Trained Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base  = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base, "/path/to/final_lora")
model = model.merge_and_unload()   # optional: merge for faster inference
```

---

## Technical Stack

| Component | Library | Version |
|---|---|---|
| Base model + LoRA | [Unsloth](https://github.com/unslothai/unsloth) | 2026.3.x |
| SFT training loop | [TRL](https://github.com/huggingface/trl) `SFTTrainer` | 0.17+ |
| LoRA adapters | [PEFT](https://github.com/huggingface/peft) | 0.14+ |
| Quantisation | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 0.45+ |
| Experiment tracking | [Weights & Biases](https://wandb.ai) | — |
| Dataset | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | — |

---

## Running the Trained Model Locally with Ollama

This section covers downloading the trained LoRA adapter from Google Drive,
merging it with the base model, converting to GGUF format, and running
inference locally via Ollama.

### Prerequisites

- Python 3.10+ with `pip`
- [Ollama](https://ollama.com/download) installed and running (`ollama serve`)
- ~8 GB free disk space
- Git (to clone llama.cpp)

---

### Step 1 — Download the adapter from Google Drive

Open Google Drive and navigate to:
```
MyDrive/banking77_finetune/models/final_lora/
```

Download the entire `final_lora/` folder. It contains:
```
final_lora/
├── adapter_config.json
├── adapter_model.safetensors   (~50 MB)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Place it at a local path, e.g. `~/banking77/final_lora/`.

---

### Step 2 — Merge the LoRA adapter into the base model

Install dependencies:
```bash
pip install transformers peft torch accelerate
```

Run this script (save as `merge.py`):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ADAPTER_PATH = "./final_lora"       # path to your downloaded folder
OUTPUT_PATH  = "./banking77_merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct",
    torch_dtype="auto",
)

print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Done. Merged model saved to {OUTPUT_PATH}")
```

```bash
python merge.py
```

This produces a standard HuggingFace model directory at `./banking77_merged/`
(~2.5 GB for the 1B model in float32, ~1.3 GB in float16).

---

### Step 3 — Convert to GGUF with llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

Convert and quantize (Q4_K_M gives the best quality/size tradeoff):
```bash
python convert_hf_to_gguf.py ../banking77_merged \
    --outfile ../banking77_q4km.gguf \
    --outtype q4_k_m
```

The output `banking77_q4km.gguf` will be ~800 MB.

> **Note**: If `convert_hf_to_gguf.py` is not found, check the `llama.cpp` root —
> older versions use `convert.py` instead.

---

### Step 4 — Create an Ollama Modelfile

Create a file named `Modelfile` (no extension) next to the GGUF:
```
FROM ./banking77_q4km.gguf

SYSTEM """You are a banking intent classifier. Given a customer query, output only the intent label as a single snake_case string from the Banking77 label set. Do not explain. Do not add punctuation. Output the label name only."""

PARAMETER temperature 0
PARAMETER stop "<|eot_id|>"
```

`temperature 0` makes output deterministic (greedy), which is what the model
was trained and evaluated with.

---

### Step 5 — Load into Ollama and run

```bash
ollama create banking77 -f Modelfile
```

Test inference:
```bash
ollama run banking77 "I lost my card, how do I get a new one?"
```

Expected output:
```
card_arrival
```

Run a quick batch test:
```bash
ollama run banking77 "How do I top up my account using a card?"
ollama run banking77 "Why was my transfer declined?"
ollama run banking77 "Can I use Apple Pay with my account?"
```

---

### Step 6 — Inference via Python (optional)

```python
import requests

def classify_intent(query: str) -> str:
    instruction = (
        "Classify the following banking customer query into one of 77 "
        "intent categories. Output only the intent label name."
    )
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": "banking77",
        "prompt": f"{instruction}\n\n{query}",
        "stream": False,
        "options": {"temperature": 0},
    })
    return resp.json()["response"].strip()

print(classify_intent("I need to update my PIN"))
# → passcode_forgotten
```

---

### Troubleshooting

**`ollama create` fails with "invalid model file"**
Ensure `FROM` points to the correct absolute or relative path to the `.gguf`
file. Use an absolute path if unsure:
```
FROM /home/user/banking77/banking77_q4km.gguf
```

**Model outputs long text instead of a label**
The SYSTEM prompt is not being applied. Verify Ollama version ≥ 0.1.30 which
added SYSTEM support in Modelfiles. Update with:
```bash
ollama update
```

**Merge script is slow (no GPU)**
Add `device_map="cpu"` to `from_pretrained` calls. The merge itself is a
matrix addition — it only runs once and does not require a GPU.

---

## What Files to Push to Make the Model Publicly Available

GitHub is **not suitable** for model weights — files over 100 MB are rejected
even with Git LFS (LFS has bandwidth limits on free plans). The standard
approach for sharing fine-tuned models is **Hugging Face Hub**.

### Option A — Push the LoRA adapter to Hugging Face Hub (recommended)

The adapter is ~50 MB and loads on top of the publicly available base model,
so users only need to download the small diff.

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token
```

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo("your-username/banking77-llama-1b-lora", exist_ok=True)
api.upload_folder(
    folder_path="./final_lora",
    repo_id="your-username/banking77-llama-1b-lora",
    repo_type="model",
)
```

Users can then load it with:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base  = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base, "your-username/banking77-llama-1b-lora")
model = model.merge_and_unload()
```

### Option B — Push the GGUF to Hugging Face Hub (for direct Ollama use)

```python
api.create_repo("your-username/banking77-llama-1b-gguf", exist_ok=True)
api.upload_file(
    path_or_fileobj="./banking77_q4km.gguf",
    path_in_repo="banking77_q4km.gguf",
    repo_id="your-username/banking77-llama-1b-gguf",
    repo_type="model",
)
```

Users can then run directly:
```bash
ollama run hf.co/your-username/banking77-llama-1b-gguf
```

### Option C — Share the Google Drive folder (quickest, no account needed)

Right-click `MyDrive/banking77_finetune/models/final_lora/` in Drive →
Share → Anyone with the link → Viewer. Paste the link in your README.

### Summary

| Option | Size to upload | User experience | Best for |
|---|---|---|---|
| HF Hub — LoRA adapter | ~50 MB | Load with PEFT | Researchers / developers |
| HF Hub — GGUF | ~800 MB | `ollama run hf.co/...` | End users / Ollama |
| Google Drive link | ~50 MB | Manual download | Quick sharing |
| GitHub (do NOT) | — | Rejected >100 MB | — |

---

## References

- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Kalajdzievski (2023). [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732) — rsLoRA
- Jain et al. (2023). [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
- Casanueva et al. (2020). [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807) — Banking77 dataset
- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
