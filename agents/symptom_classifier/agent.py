"""
Agent 1 — Symptom Classifier
Calls the SageMaker endpoint (medi-agent-classifier) for inference.
No local model loading required.

Input : AgentState with 'patient_text'
Output: AgentState updated with 'department' and 'urgency'
"""
import json
import os
import re
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from schemas import AgentState, ClassifierOutput

AWS_REGION    = os.getenv("AWS_REGION", "us-west-2")
ENDPOINT_NAME = os.getenv("AGENT1_ENDPOINT_NAME", "medi-agent-classifier")

_sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

DEPARTMENTS = [
    "Cardiology", "Neurology", "Orthopedics", "Gastroenterology",
    "Pulmonology", "Dermatology", "Endocrinology", "Infectious Disease",
    "Urology", "General Medicine",
]
URGENCIES = ["Routine", "Urgent", "Emergency"]


def _build_prompt(patient_text: str,
                  user_age: int = None,
                  user_gender: str = None,
                  blood_type: str = None,
                  allergies: list = None,
                  height_cm: int = None,
                  weight_kg: float = None) -> str:
    context_parts = []
    if user_age:
        context_parts.append(f"Age: {user_age}")
    if user_gender:
        context_parts.append(f"Gender: {user_gender}")
    if blood_type:
        context_parts.append(f"Blood type: {blood_type}")
    if allergies:
        context_parts.append(f"Allergies: {', '.join(str(a) for a in allergies)}")
    if height_cm and weight_kg:
        bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
        context_parts.append(f"BMI: {bmi} ({height_cm}cm, {weight_kg}kg)")
    elif height_cm:
        context_parts.append(f"Height: {height_cm}cm")
    elif weight_kg:
        context_parts.append(f"Weight: {weight_kg}kg")
    context_line = f"Patient info: {', '.join(context_parts)}\n" if context_parts else ""

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a medical triage assistant. Given patient symptoms, output "
        "the most appropriate department and urgency level.\n"
        "Respond ONLY in the format: Department: <dept> | Urgency: <urgency>\n"
        f"Valid departments: {', '.join(DEPARTMENTS)}\n"
        f"Valid urgency levels: {', '.join(URGENCIES)}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{context_line}"
        f"Patient symptoms: {patient_text}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _call_endpoint(prompt: str) -> str:
    params = {"max_new_tokens": 32, "do_sample": False}

    from agents.local_inference import use_local_models, call_local
    if use_local_models():
        return call_local("classifier", prompt, params)

    try:
        payload = {"inputs": prompt, "parameters": params}
        response = _sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read())
        return result[0]["generated_text"]
    except Exception as e:
        print(f"[Agent1] SageMaker unavailable ({e}), falling back to local model")
        return call_local("classifier", prompt, params)


def _parse_output(text: str) -> ClassifierOutput:
    dept_match = re.search(r"Department:\s*([^|]+)", text, re.IGNORECASE)
    urg_match  = re.search(r"Urgency:\s*(\w+)", text, re.IGNORECASE)

    department = "General Medicine"
    urgency    = "Routine"

    if dept_match:
        raw = dept_match.group(1).strip()
        for d in DEPARTMENTS:
            if d.lower() in raw.lower():
                department = d
                break

    if urg_match:
        raw = urg_match.group(1).strip().capitalize()
        if raw in URGENCIES:
            urgency = raw

    return ClassifierOutput(department=department, urgency=urgency)


def run_classifier(state: AgentState) -> AgentState:
    """LangGraph node — calls SageMaker, writes 'department' and 'urgency'."""
    prompt    = _build_prompt(
        state["patient_text"],
        state.get("user_age"),
        state.get("user_gender"),
        state.get("blood_type"),
        state.get("allergies"),
        state.get("height_cm"),
        state.get("weight_kg"),
    )
    generated = _call_endpoint(prompt)
    result    = _parse_output(generated)

    print(f"[Agent1] → department={result.department}, urgency={result.urgency}")
    return {**state, "department": result.department, "urgency": result.urgency}
