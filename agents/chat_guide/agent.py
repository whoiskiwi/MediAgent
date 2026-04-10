"""
Chat Guide Agent — generates follow-up questions and summaries for the chatbox intake flow.

One API call generates all questions upfront; a second call produces the final summary.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

_FALLBACK_QUESTIONS = [
    {
        "q": "How long have you had this symptom?",
        "options": ["Less than 1 day", "1–3 days", "4–7 days", "More than a week"],
    },
    {
        "q": "How would you rate the severity?",
        "options": ["Mild — barely noticeable", "Moderate — affects daily activity", "Severe — hard to function", "Extreme — unbearable"],
    },
    {
        "q": "Do you have any other symptoms alongside this?",
        "options": ["Fever or chills", "Nausea or vomiting", "Pain elsewhere", "None of the above"],
    },
]


def generate_questions(symptom: str) -> list[dict]:
    """
    One API call — returns 3 follow-up questions with multiple-choice options.
    Falls back to generic questions if the API fails or returns invalid JSON.
    """
    prompt = (
        "You are a medical intake assistant at a hospital appointment booking system.\n\n"
        f'A patient says: "{symptom}"\n\n'
        "Generate exactly 3 follow-up questions to gather enough information to route them "
        "to the right department. Cover: duration, severity, and associated symptoms.\n\n"
        "Rules:\n"
        "- Each question must have exactly 4 short, specific options.\n"
        "- Options must be mutually exclusive and cover the realistic range.\n"
        "- Do NOT include an 'Other' option — the user can type freely.\n"
        "- Respond ONLY with a valid JSON array, no markdown, no extra text.\n\n"
        "Format:\n"
        '[{"q": "...", "options": ["...", "...", "...", "..."]}, ...]'
    )
    try:
        raw = _call_llm(prompt, max_tokens=500)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text.strip())
        if isinstance(questions, list) and len(questions) >= 1:
            return questions[:4]
    except Exception as e:
        print(f"[ChatGuide] generate_questions failed: {e}")
    return _FALLBACK_QUESTIONS


def generate_summary(symptom: str, qa_pairs: list) -> str:
    """
    One API call — returns a warm 2-sentence summary of the patient's condition
    followed by a confirmation prompt asking if they want to book.
    """
    qa_text = "\n".join(f"  - {qa['q']}: {qa['a']}" for qa in qa_pairs)
    prompt = (
        "You are a compassionate medical intake assistant.\n\n"
        f"Patient's main complaint: {symptom}\n"
        f"Follow-up answers:\n{qa_text}\n\n"
        "Write exactly 2–3 sentences:\n"
        "1. A warm, clear summary of what the patient is experiencing.\n"
        "2. Ask whether they would like you to book an appointment for them.\n"
        "Do not diagnose. Be empathetic and concise."
    )
    try:
        return _call_llm(prompt, max_tokens=150)
    except Exception as e:
        print(f"[ChatGuide] generate_summary failed: {e}")
        return (
            "Thank you for sharing those details. "
            "Based on what you've described, I'd recommend seeing a doctor. "
            "Would you like me to book an appointment for you?"
        )


def _call_llm(prompt: str, max_tokens: int = 512) -> str:
    from openai import OpenAI

    if DEEPSEEK_API_KEY:
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ChatGuide] DeepSeek failed: {e}, falling back to OpenAI")

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    raise RuntimeError("No LLM API key available")
