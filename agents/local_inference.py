"""
Local inference — uses OpenAI API as fallback when SageMaker endpoints are unavailable.
Set USE_LOCAL_MODELS=true in .env to force this mode.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Model-specific system prompts
# Classifier: replicate the urgency calibration baked into the fine-tuned model
# Generator:  replicate the output format the fine-tuned model was trained on
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a medical triage assistant. Classify patient symptoms into a department "
    "and urgency level.\n\n"
    "Urgency rules — apply strictly:\n"
    "• Emergency: immediately life-threatening. Examples: heart attack signs "
    "(chest pain + breathlessness + sweating + arm/jaw pain), stroke signs "
    "(sudden face drooping, arm weakness, speech difficulty), paralysis, "
    "loss of consciousness, severe uncontrolled bleeding, severe breathing difficulty.\n"
    "  Rule: if 4 or more moderate-to-severe symptoms co-occur, classify Emergency.\n"
    "• Urgent: significant, needs medical attention within hours. Examples: "
    "moderate-to-severe chest pain alone, high fever (>38.5 °C), "
    "3 or more moderate symptoms co-occurring (e.g. pain + nausea + sweating), "
    "symptoms severely limiting daily function.\n"
    "• Routine: mild or stable. Examples: minor aches, mild fever, skin rash, "
    "digestive discomfort, symptoms present for days but stable and not worsening.\n\n"
    "Respond ONLY in this exact format, nothing else:\n"
    "Department: <department> | Urgency: <Routine|Urgent|Emergency>"
)

_GENERATOR_SYSTEM = (
    "You are a clinic scheduling system. Write a brief, formal appointment confirmation.\n"
    "Do NOT use greetings like 'Hi' or 'Dear'. Do NOT mention any website or platform.\n"
    "Do NOT sign with a doctor name. Do NOT ask follow-up questions.\n"
    "Output exactly two parts separated by '---':\n"
    "Part 1: One sentence confirming the appointment (doctor, department, time).\n"
    "Part 2: 3 short pre-visit instructions relevant to the patient's symptoms.\n"
    "Start Part 1 with: Confirmation: Your appointment with"
)

_SYSTEM_PROMPTS = {
    "classifier": _CLASSIFIER_SYSTEM,
    "generator":  _GENERATOR_SYSTEM,
}


def call_local(model_name: str, prompt: str, params: dict) -> str:
    """Use OpenAI gpt-4o-mini as local inference fallback."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    max_tokens  = params.get("max_new_tokens", 256)
    temperature = float(params.get("temperature", 0.1))

    system_prompt = _SYSTEM_PROMPTS.get(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def use_local_models() -> bool:
    return os.getenv("USE_LOCAL_MODELS", "").lower() in ("1", "true", "yes")
