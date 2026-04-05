"""
Local inference — uses OpenAI API as fallback when SageMaker endpoints are unavailable.
Set USE_LOCAL_MODELS=true in .env to force this mode.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def call_local(model_name: str, prompt: str, params: dict) -> str:
    """Use OpenAI gpt-4o-mini as local inference fallback."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    max_tokens  = params.get("max_new_tokens", 256)
    temperature = float(params.get("temperature", 0.1))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def use_local_models() -> bool:
    return os.getenv("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")
