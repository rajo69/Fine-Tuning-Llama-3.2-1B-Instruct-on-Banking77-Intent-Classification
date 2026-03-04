"""
src/3_resume_and_infer.py
=========================
Recovery and inference script using standard HuggingFace (no Unsloth).

Automatically inspects `checkpoints/` and `models/` to find the best
available model, then runs intent classification inference.

Priority order:
  1. models/final_lora           — completed training (most preferred)
  2. checkpoints/hourly_backup/  — most recent hourly checkpoint (by name)
  3. checkpoints/checkpoint-*    — latest HF Trainer checkpoint (by step)
  4. base model (no fine-tuning) — last resort

Run:
    python src/3_resume_and_infer.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "3_resume_and_infer.log"), mode="w"),
    ],
)
logger = logging.getLogger("resume_and_infer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
FINAL_MODEL_DIR = Path("models/final_lora")
HOURLY_CKPT_DIR = Path("checkpoints/hourly_backup")
CHECKPOINT_DIR  = Path("checkpoints")
MAX_SEQ_LENGTH  = 256
MAX_NEW_TOKENS  = 32

ALPACA_PROMPT = """\
Below is an instruction that describes a task, paired with an input. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

INSTRUCTION = (
    "Classify the following banking customer query into one of 77 intent categories. "
    "Respond with only the intent label name."
)

TEST_QUERIES = [
    "I lost my card, what should I do?",
    "Can I use my card abroad?",
    "How do I change my PIN?",
    "My transfer hasn't arrived yet.",
    "What are the limits for topping up?",
]


# =============================================================================
# Model discovery
# =============================================================================

def is_valid_model_dir(path: Path) -> bool:
    """Check for adapter_config.json (LoRA) or config.json (full model)."""
    return path.is_dir() and (
        (path / "adapter_config.json").exists() or
        (path / "config.json").exists()
    )


def find_latest_hourly_checkpoint() -> Path | None:
    if not HOURLY_CKPT_DIR.is_dir():
        return None
    candidates = sorted(
        [p for p in HOURLY_CKPT_DIR.iterdir() if is_valid_model_dir(p)],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_hf_checkpoint() -> Path | None:
    if not CHECKPOINT_DIR.is_dir():
        return None
    candidates = []
    for p in CHECKPOINT_DIR.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-") and is_valid_model_dir(p):
            try:
                step = int(p.name.split("-")[1])
                candidates.append((step, p))
            except (IndexError, ValueError):
                pass
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def discover_model() -> tuple[str, str, bool]:
    """
    Returns (model_path, description, is_adapter).
    is_adapter=True  → load base model + PeftModel
    is_adapter=False → load model directly (base model fallback)
    """
    if is_valid_model_dir(FINAL_MODEL_DIR):
        is_adapter = (FINAL_MODEL_DIR / "adapter_config.json").exists()
        return str(FINAL_MODEL_DIR), "final trained model (models/final_lora)", is_adapter

    hourly = find_latest_hourly_checkpoint()
    if hourly is not None:
        is_adapter = (hourly / "adapter_config.json").exists()
        return str(hourly), f"hourly checkpoint ({hourly})", is_adapter

    hf_ckpt = find_latest_hf_checkpoint()
    if hf_ckpt is not None:
        is_adapter = (hf_ckpt / "adapter_config.json").exists()
        return str(hf_ckpt), f"HF Trainer checkpoint ({hf_ckpt})", is_adapter

    logger.warning("No fine-tuned checkpoint found. Using base model.")
    return BASE_MODEL_NAME, "base model (untuned)", False


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, tokenizer, queries: list[str]) -> list[str]:
    import torch

    model.eval()
    predictions: list[str] = []

    with torch.no_grad():
        for i, query in enumerate(queries, start=1):
            prompt = ALPACA_PROMPT.format(instruction=INSTRUCTION, input=query)

            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                )

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Strip the prompt tokens — decode only generated part
                new_ids   = outputs[0][inputs["input_ids"].shape[1]:]
                prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                predictions.append(prediction)

                logger.info("[%d/%d] Query : %s", i, len(queries), query)
                logger.info("[%d/%d] Intent: %s", i, len(queries), prediction)
                logger.info("-" * 50)

            except Exception as e:
                logger.error("Inference error for query %d: %s", i, e, exc_info=True)
                predictions.append("ERROR")

    return predictions


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Resume & Inference (HuggingFace stack)")
    logger.info("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ------------------------------------------------------------------
    # 1. Discover best available model
    # ------------------------------------------------------------------
    model_path, source_desc, is_adapter = discover_model()
    logger.info("Model source : %s", source_desc)
    logger.info("Model path   : %s", model_path)
    logger.info("Is LoRA adapter: %s", is_adapter)

    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    # If it's an adapter, load the tokenizer from the adapter dir (it was
    # saved there by trainer.save_model()); fall back to base model name.
    tok_source = model_path if is_adapter else model_path
    logger.info("Loading tokenizer from: %s", tok_source)
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    if is_adapter:
        # Load base model first, then overlay LoRA adapter weights
        from peft import PeftModel
        logger.info("Loading base model: %s", BASE_MODEL_NAME)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        logger.info("Loading LoRA adapter from: %s", model_path)
        model = PeftModel.from_pretrained(base, model_path)
        # Merge adapter weights into the base model for faster inference
        logger.info("Merging LoRA weights for inference...")
        model = model.merge_and_unload()
    else:
        logger.info("Loading model directly from: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    logger.info("Model ready.")

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Running inference on %d test queries...", len(TEST_QUERIES))
    logger.info("=" * 60)

    predictions = run_inference(model, tokenizer, TEST_QUERIES)

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Query':<45} {'Predicted Intent':<25}")
    print("-" * 70)
    for query, pred in zip(TEST_QUERIES, predictions):
        q = (query[:42] + "...") if len(query) > 45 else query
        print(f"{q:<45} {pred:<25}")
    print("=" * 70 + "\n")

    logger.info("Inference complete. Source: %s", source_desc)


if __name__ == "__main__":
    main()
