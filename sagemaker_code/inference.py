"""
SageMaker inference script — runs inside the HuggingFace DLC container.
Loaded once when the endpoint starts; predict_fn called for every request.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_fn(model_dir: str):
    """Load tokenizer and model from /opt/ml/model (= the unpacked tar.gz)."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def input_fn(request_body: str, content_type: str) -> dict:
    return json.loads(request_body)


def predict_fn(data: dict, model_and_tokenizer) -> list:
    """
    Expected input:
        {
            "inputs": "<prompt string>",
            "parameters": { "max_new_tokens": 32, "do_sample": false, ... }
        }
    Returns:
        [{"generated_text": "<generated string>"}]
    """
    model, tokenizer = model_and_tokenizer
    prompt = data["inputs"]
    params = data.get("parameters", {})

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
    return [{"generated_text": generated}]


def output_fn(prediction: list, accept: str) -> str:
    return json.dumps(prediction)
