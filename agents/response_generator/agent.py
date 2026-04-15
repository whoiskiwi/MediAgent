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


def _fetch_rag_context(patient_text: str, department: str) -> str:
    """Retrieve top-2 MedlinePlus snippets relevant to the symptoms. Returns empty string on failure."""
    try:
        from agents.rag.retriever import get_retriever
        query = f"{patient_text} {department}"
        docs  = get_retriever().query(query, n_results=2)
        if not docs:
            return ""
        lines = [f"- {d['title']}: {d['text'][:200].strip()}" for d in docs]
        return "Medical references:\n" + "\n".join(lines) + "\n"
    except Exception as e:
        print(f"[Agent3] RAG fetch failed (non-critical): {e}")
        return ""


def _build_prompt(
    patient_text: str,
    department: str,
    doctor: str,
    time_slot: str,
    urgency: str,
    user_age: int = None,
    user_gender: str = None,
    blood_type: str = None,
    allergies: list = None,
    height_cm: int = None,
    weight_kg: float = None,
    rag_context: str = "",
) -> str:
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
        "You are a clinic scheduling system. Write a brief, formal appointment confirmation.\n"
        "Do NOT use greetings like 'Hi' or 'Dear'. Do NOT mention any website or platform.\n"
        "Do NOT sign with a doctor name. Do NOT ask follow-up questions.\n"
        "Output exactly two parts separated by '---':\n"
        "Part 1: One sentence confirming the appointment (doctor, department, time).\n"
        "Part 2: 3 short pre-visit instructions relevant to the patient's symptoms and demographics.\n"
        "Use the medical references below (if provided) to make the instructions more specific and accurate.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{context_line}"
        f"{rag_context}"
        f"Symptoms: {patient_text}\n"
        f"Department: {department}\n"
        f"Doctor: {doctor}\n"
        f"Time: {time_slot}\n"
        f"Urgency: {urgency}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "Confirmation: Your appointment with"
    )


def _call_endpoint(prompt: str) -> str:
    params = {"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_p": 0.9}

    from agents.local_inference import use_local_models, call_local
    if use_local_models():
        return call_local("generator", prompt, params)

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
        print(f"[Agent3] SageMaker unavailable ({e}), falling back to local model")
        return call_local("generator", prompt, params)


_STOP_PATTERNS = re.compile(
    r"(thanks\s+for\s+(posting|your\s+question|writing|asking)"
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
    r"|available\s+for\s+direct"
    r"|healthcaremagic"
    r"|in\s+(healthcaremagic|hcm)\s+forum"
    r"|asking\s+in\s+\w+\s+forum)",
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


def _call_first_aid_llm(patient_text: str, department: str, urgency: str,
                         user_age: int = None, user_gender: str = None,
                         blood_type: str = None, allergies: list = None,
                         height_cm: int = None, weight_kg: float = None) -> str:
    """Call DeepSeek (or OpenAI fallback) for first aid steps."""
    from openai import OpenAI

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
        context_parts.append(f"BMI: {bmi}")
    context_line = f"Patient info: {', '.join(context_parts)}\n" if context_parts else ""

    level = "emergency" if urgency == "Emergency" else "urgent"
    prompt = (
        f"{context_line}"
        f"Symptoms: {patient_text}\n"
        f"Department: {department}\n\n"
        f"The patient has a {level} condition. "
        "List exactly 3 to 5 specific, actionable first aid steps the patient or bystander "
        "should take RIGHT NOW, before reaching a doctor. "
        "Be concise and specific to the symptoms. Do not list diagnostic tests or medications."
    )

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key   = os.getenv("OPENAI_API_KEY")

    _TIMEOUT = 20
    if deepseek_key:
        try:
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com", timeout=_TIMEOUT)
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Agent3] DeepSeek first aid failed: {e}, falling back to OpenAI")

    if openai_key:
        try:
            client = OpenAI(api_key=openai_key, timeout=_TIMEOUT)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Agent3] OpenAI failed: {e}")

    return "Please seek immediate medical attention."


def run_generator(state: AgentState) -> AgentState:
    """LangGraph node — calls SageMaker, writes 'confirmation', 'instructions', and optionally 'first_aid'."""
    urgency      = state.get("urgency", "Routine")
    safe_urgency = normalize_urgency(urgency)

    rag_context = _fetch_rag_context(state["patient_text"], state.get("department", ""))

    prompt    = _build_prompt(
        patient_text=state["patient_text"],
        department=state.get("department", "General Practice"),
        doctor=state.get("doctor", "your doctor"),
        time_slot=state.get("time_slot", "your scheduled time"),
        urgency=safe_urgency,
        user_age=state.get("user_age"),
        user_gender=state.get("user_gender"),
        blood_type=state.get("blood_type"),
        allergies=state.get("allergies"),
        height_cm=state.get("height_cm"),
        weight_kg=state.get("weight_kg"),
        rag_context=rag_context,
    )
    generated = _call_endpoint(prompt)
    confirmation, instructions = _parse_output(generated, prompt_prefix="Confirmation: Your appointment with")

    first_aid = None
    if urgency in ("Urgent", "Emergency") and state.get("logged_in"):
        fa_raw = _call_first_aid_llm(
            patient_text=state["patient_text"],
            department=state.get("department", "General Practice"),
            urgency=urgency,
            user_age=state.get("user_age"),
            user_gender=state.get("user_gender"),
            blood_type=state.get("blood_type"),
            allergies=state.get("allergies"),
            height_cm=state.get("height_cm"),
            weight_kg=state.get("weight_kg"),
        )
        first_aid = fa_raw.strip()

    print(f"[Agent3] → confirmation generated ({len(confirmation)} chars), first_aid={'yes' if first_aid else 'no'}")
    return {**state, "confirmation": confirmation, "instructions": instructions, "first_aid": first_aid}
