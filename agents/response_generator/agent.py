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
import re
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
        "You are a clinic scheduling system. Write a brief, formal appointment confirmation.\n"
        "Do NOT use greetings like 'Hi' or 'Dear'. Do NOT mention any website or platform.\n"
        "Do NOT sign with a doctor name. Do NOT ask follow-up questions.\n"
        "Output exactly two parts separated by '---':\n"
        "Part 1: One sentence confirming the appointment (doctor, department, time).\n"
        "Part 2: 3 short pre-visit instructions relevant to the patient's symptoms and demographics.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{context_line}"
        f"Symptoms: {patient_text}\n"
        f"Department: {department}\n"
        f"Doctor: {doctor}\n"
        f"Time: {time_slot}\n"
        f"Urgency: {urgency}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "Confirmation: Your appointment with"
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


_STOP_PATTERNS = re.compile(
    r"(thanks\s+for\s+(posting|your\s+question|writing)"
    r"|hope\s+i\s+have\s+(solved|answered)"
    r"|i\s+will\s+be\s+happy\s+to\s+help"
    r"|wishing\s+(you\s+)?good\s+health"
    r"|feel\s+free\s+to\s+contact"
    r"|do\s+not\s+hesitate\s+to\s+contact"
    r"|visit\s+our\s+website"
    r"|welcome\s+to\s+hcm"
    r"|let\s+me\s+know\s+if\s+i\s+can"
    r"|contact\s*:\s*\d"
    r"|regards[,.]"
    r"|available\s+for\s+direct)",
    re.IGNORECASE,
)
_APPT_KEYWORDS = re.compile(
    r"(appointment|confirmed|department|doctor|time slot|friday|monday|tuesday|wednesday|thursday|saturday|sunday|\d{2}:\d{2})",
    re.IGNORECASE,
)


def _extract_confirmation(text: str, max_sentences: int = 2) -> str:
    """Keep only the first sentences that contain appointment info."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        stripped = s.strip()
        if not stripped:
            continue
        if _STOP_PATTERNS.search(stripped):
            break
        if len(kept) >= max_sentences:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _extract_instructions(text: str) -> str:
    """Extract pre-visit instructions from the generated text."""
    # Try explicit Instructions: label
    m = re.search(r"Instructions?\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        raw = m.group(1).strip()
        sentences = re.split(r"(?<=[.!?])\s+", raw)
        kept = []
        seen = set()
        for s in sentences:
            stripped = s.strip()
            if not stripped:
                continue
            if _STOP_PATTERNS.search(stripped):
                break
            if re.match(r"^(hi[,\s]|hello[,\s]|dear\s)", stripped, re.IGNORECASE):
                continue
            if stripped in seen:
                break
            seen.add(stripped)
            kept.append(stripped)
            if len(kept) >= 4:
                break
        if kept:
            return " ".join(kept)
    return "Please arrive 15 minutes early and bring a valid photo ID."


def _parse_output(raw: str, prompt_prefix: str = "Confirmation: Your appointment with") -> Tuple[str, str]:
    text = (prompt_prefix + raw) if not raw.strip().startswith("Confirmation") else raw

    if "---" in text:
        parts = text.split("---", 1)
        confirmation = _extract_confirmation(parts[0].strip(), max_sentences=3)
        raw_instructions = parts[1].strip()
        instructions = raw_instructions if raw_instructions else _extract_instructions(text)
    else:
        confirmation = _extract_confirmation(text, max_sentences=2)
        instructions = _extract_instructions(text)

    return confirmation, instructions


def _build_first_aid_prompt(patient_text: str, department: str, urgency: str,
                             user_age: int = None, user_gender: str = None) -> str:
    context_parts = []
    if user_age:
        context_parts.append(f"Age: {user_age}")
    if user_gender:
        context_parts.append(f"Gender: {user_gender}")
    context_line = f"Patient info: {', '.join(context_parts)}\n" if context_parts else ""

    level = "emergency" if urgency == "Emergency" else "urgent"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a medical first aid advisor. The patient has a {level} condition.\n"
        "List 3-5 specific, actionable first aid steps the patient or bystander should take RIGHT NOW "
        "based on the symptoms described, before reaching a doctor.\n"
        "Be concise and specific to the symptoms. Do not give general advice.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{context_line}"
        f"Symptoms: {patient_text}\n"
        f"Department: {department}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "Immediate steps:\n"
    )


def run_generator(state: AgentState) -> AgentState:
    """LangGraph node — calls SageMaker, writes 'confirmation', 'instructions', and optionally 'first_aid'."""
    urgency      = state.get("urgency", "Routine")
    safe_urgency = normalize_urgency(urgency)

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
    confirmation, instructions = _parse_output(generated, prompt_prefix="Confirmation: Your appointment with")

    first_aid = None
    if urgency in ("Urgent", "Emergency"):
        fa_prompt = _build_first_aid_prompt(
            patient_text=state["patient_text"],
            department=state.get("department", "General Practice"),
            urgency=urgency,
            user_age=state.get("user_age"),
            user_gender=state.get("user_gender"),
        )
        fa_raw   = _call_endpoint(fa_prompt)
        first_aid = ("Immediate steps:\n" + fa_raw).strip()

    print(f"[Agent3] → confirmation generated ({len(confirmation)} chars), first_aid={'yes' if first_aid else 'no'}")
    return {**state, "confirmation": confirmation, "instructions": instructions, "first_aid": first_aid}
