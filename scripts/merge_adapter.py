"""
Step 1 — Merge LoRA adapters into base model.

Produces two standalone models (no adapter dependency at runtime):
  data/processed/merged_classifier/
  data/processed/merged_generator/

Run once before deploying to SageMaker.
Requires ~12 GB free disk space and a HuggingFace token with LLaMA access.

Usage:
    python scripts/merge_adapter.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

JOBS = [
    {
        "name": "classifier",
        "adapter": ROOT / "data/processed/symptom_classifier_adapter/checkpoint-1569",
        "output": ROOT / "data/processed/merged_classifier",
    },
    {
        "name": "generator",
        "adapter": ROOT / "data/processed/response_generator_adapter/final_adapter",
        "output": ROOT / "data/processed/merged_generator",
    },
]


def merge(job: dict):
    name = job["name"]
    adapter_path = str(job["adapter"])
    output_path = str(job["output"])

    print(f"\n[merge] === {name} ===")
    print(f"[merge] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",   # merge on CPU to avoid VRAM limits
        token=HF_TOKEN,
    )

    print(f"[merge] Attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print(f"[merge] Merging weights …")
    model = model.merge_and_unload()

    print(f"[merge] Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[merge] Done: {name}")


if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env")
        sys.exit(1)
    for job in JOBS:
        merge(job)
    print("\n[merge] All done. Next step: python scripts/deploy_sagemaker.py")
