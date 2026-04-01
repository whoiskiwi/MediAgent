"""
Causes Generator — uses DeepSeek (primary) or OpenAI (fallback)
to produce a list of possible causes for the patient's symptoms.

Input : AgentState with 'patient_text' and 'department'
Output: AgentState updated with 'possible_causes' (list[str])
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))
from schemas import AgentState

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


def _call_llm(patient_text: str, department: str) -> list[str]:
    prompt = (
        f"A patient in the {department} department reports: \"{patient_text}\"\n\n"
        "List 3 to 5 possible medical causes for these symptoms.\n"
        "Reply with a numbered list only, one cause per line, no extra explanation.\n"
        "Example:\n1. Muscle strain\n2. Herniated disc\n3. Kidney stone"
    )

    # Try DeepSeek first
    if DEEPSEEK_API_KEY:
        try:
            client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            return _parse_causes(resp.choices[0].message.content)
        except Exception as e:
            print(f"[CausesGenerator] DeepSeek failed: {e}, falling back to OpenAI")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        return _parse_causes(resp.choices[0].message.content)

    print("[CausesGenerator] No API key configured, skipping causes generation")
    return []


def _parse_causes(text: str) -> list[str]:
    causes = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbers/bullets: "1.", "1)", "-", "•"
        import re
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        if line:
            causes.append(line)
    return causes[:5]


def run_causes_generator(state: AgentState) -> AgentState:
    """LangGraph node — generates possible causes for patient symptoms."""
    causes = _call_llm(
        patient_text=state.get("patient_text", ""),
        department=state.get("department", "General Medicine"),
    )
    print(f"[CausesGenerator] → {len(causes)} possible causes")
    return {**state, "possible_causes": causes}
