"""
Local inference — loads merged models from data/processed/ and runs on MPS/CPU.
Used as a fallback when SageMaker endpoints are unavailable.

Set USE_LOCAL_MODELS=true in .env to force local mode.
"""
import json
import os
import torch
from functools import lru_cache
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "classifier": ROOT / "data" / "processed" / "merged_classifier",
    "generator":  ROOT / "data" / "processed" / "merged_generator",
}

# Use MPS on Apple Silicon, CUDA on GPU servers, else CPU
def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    model_dir = str(MODEL_PATHS[model_name])
    device = _get_device()
    print(f"[LocalInference] Loading {model_name} from {model_dir} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    print(f"[LocalInference] {model_name} ready on {device}")
    return model, tokenizer


def call_local(model_name: str, prompt: str, params: dict) -> str:
    """Run local inference. Returns generated_text (new tokens only)."""
    model, tokenizer = _load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **params,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


def use_local_models() -> bool:
    return os.getenv("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")
