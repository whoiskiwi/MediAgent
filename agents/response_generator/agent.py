"""
Agent 3 — Response Generator
Calls the SageMaker endpoint (medi-agent-generator) for inference.
No local model loading required.

IMPORTANT: Agent 3 was only trained on Routine / Urgent.
           Emergency is normalised to Urgent via normalize_urgency().

Input : AgentState with patient_text, department, doctor, time_slot, urgency
Output: AgentState updated with 'confirmation' and 'instructions'
"""
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import boto3
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from schemas import AgentState, normalize_urgency

AWS_REGION    = os.getenv("AWS_REGION", "us-west-2")
ENDPOINT_NAME = os.getenv("AGENT3_ENDPOINT_NAME", "medi-agent-generator")

_sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def _build_prompt(
    patient_text: str,
    department: str,
    doctor: str,
    time_slot: str,
    urgency: str,
    user_age: int = None,
    user_gender: str = None,
) -> str:
    context_parts = []
    if user_age:
        context_parts.append(f"Age: {user_age}")
    if user_gender:
        context_parts.append(f"Gender: {user_gender}")
    context_line = f"Patient info: {', '.join(context_parts)}\n" if context_parts else ""

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful medical assistant. Generate a warm appointment "
        "confirmation message and specific pre-visit instructions for the patient.\n"
        "Respond in exactly two parts separated by '---':\n"
        "Part 1: Appointment confirmation (2-3 sentences)\n"
        "Part 2: Pre-visit instructions (3-5 bullet points)\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{context_line}"
        f"Patient complaint: {patient_text}\n"
        f"Department: {department}\n"
        f"Doctor: {doctor}\n"
        f"Time slot: {time_slot}\n"
        f"Urgency: {urgency}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _call_endpoint(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    response = _sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    return result[0]["generated_text"]


def _parse_output(text: str) -> Tuple[str, str]:
    if "---" in text:
        parts = text.split("---", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), "Please arrive 15 minutes early and bring a photo ID."


def run_generator(state: AgentState) -> AgentState:
    """LangGraph node — calls SageMaker, writes 'confirmation' and 'instructions'."""
    safe_urgency = normalize_urgency(state.get("urgency", "Routine"))

    prompt    = _build_prompt(
        patient_text=state["patient_text"],
        department=state.get("department", "General Practice"),
        doctor=state.get("doctor", "your doctor"),
        time_slot=state.get("time_slot", "your scheduled time"),
        urgency=safe_urgency,
        user_age=state.get("user_age"),
        user_gender=state.get("user_gender"),
    )
    generated = _call_endpoint(prompt)
    confirmation, instructions = _parse_output(generated)

    print(f"[Agent3] → confirmation generated ({len(confirmation)} chars)")
    return {**state, "confirmation": confirmation, "instructions": instructions}
