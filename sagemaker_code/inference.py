"""
SageMaker inference script — runs inside the HuggingFace DLC container.
Loaded once when the endpoint starts; predict_fn called for every request.
"""
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Output cleaning — strips medical Q&A sign-offs and doctor signature lines
# that leak from training data into Agent 3 outputs.
# ---------------------------------------------------------------------------
_SIGNOFF_PATTERNS = re.compile(
    r"(thanks\s+for\s+(posting|your\s+question)"
    r"|hope\s+i\s+have\s+solved"
    r"|i\s+will\s+be\s+happy\s+to\s+help"
    r"|wishing\s+(you\s+)?good\s+health"
    r"|wish\s+you\s+(good\s+health|well)"
    r"|feel\s+free\s+to\s+contact"
    r"|do\s+not\s+hesitate\s+to\s+contact"
    r"|regards[,.]?\s*$)",
    re.IGNORECASE,
)
_DOCTOR_LINE = re.compile(
    r"^Dr\.\s+\w[\w\s]+\.\s*(cardiologist|physician|specialist|surgeon|neurologist"
    r"|dermatologist|gastroenterologist|endocrinologist|pulmonologist"
    r"|orthopedist|urologist|internist)?\.?\s*$",
    re.IGNORECASE,
)


def _clean_output(text: str) -> str:
    """Truncate at first sign-off sentence or doctor-signature line."""
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        if _SIGNOFF_PATTERNS.search(s) or _DOCTOR_LINE.match(s.strip()):
            break
        kept.append(s)
    cleaned = " ".join(kept).strip()
    return cleaned if cleaned else text.split(".")[0].strip()


def model_fn(model_dir: str, context=None):
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

    # Apply sign-off cleaning only for generator endpoint (longer outputs)
    if len(generated) > 100:
        generated = _clean_output(generated)

    return [{"generated_text": generated}]


def output_fn(prediction: list, accept: str) -> str:
    return json.dumps(prediction)
