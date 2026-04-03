"""
Causes Generator — uses DeepSeek (primary) or OpenAI (fallback).
RAG-enhanced: retrieves relevant MedlinePlus documents as context.

Input : AgentState with 'patient_text' and 'department'
Output: AgentState updated with 'possible_causes' (list of structured dicts)
        Each item: {cause, reason, reference: {title, url} | None}
"""
import json
import os
import re
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


def _get_rag_docs(patient_text: str) -> list[dict]:
    """Retrieve relevant MedlinePlus docs. Returns [] if index not ready."""
    try:
        from agents.rag.retriever import get_retriever
        return get_retriever().query(patient_text, n_results=4)
    except Exception as e:
        print(f"[CausesGenerator] RAG unavailable: {e}")
        return []


def _build_prompt(patient_text: str, department: str, rag_docs: list[dict],
                  age: int = None, gender: str = None) -> str:
    ref_block = ""
    if rag_docs:
        parts = []
        for i, doc in enumerate(rag_docs):
            parts.append(f"[REF{i+1}] Title: {doc['title']}\n{doc['text'][:500]}")
        ref_block = (
            "The following MedlinePlus references may be relevant:\n\n"
            + "\n\n".join(parts)
            + "\n\n---\n"
        )

    demographics = []
    if age is not None:
        demographics.append(f"Age: {age}")
    if gender:
        demographics.append(f"Gender: {gender}")
    demo_line = f"Patient demographics: {', '.join(demographics)}\n" if demographics else ""

    return (
        f"{ref_block}"
        f"{demo_line}"
        f"Patient in {department} department reports: \"{patient_text}\"\n\n"
        "List 3 to 5 possible causes. For each cause provide:\n"
        "1. The cause name\n"
        "2. A one-sentence reason why it fits the symptoms\n"
        "3. Which reference supports it (REF1, REF2, etc.), or null if none\n\n"
        "Reply ONLY with a JSON array, no markdown, no extra text. Example:\n"
        '[\n'
        '  {"cause": "Muscle strain", "reason": "Lifting heavy objects can overstretch back muscles.", "ref": "REF1"},\n'
        '  {"cause": "Herniated disc", "reason": "Disc pressure on nerves causes sharp back pain.", "ref": null}\n'
        ']'
    )


def _call_llm(prompt: str) -> str:
    if DEEPSEEK_API_KEY:
        try:
            client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com",
            )
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[CausesGenerator] DeepSeek failed: {e}, falling back to OpenAI")

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    return "[]"


def _parse_output(raw: str, rag_docs: list[dict]) -> list[dict]:
    """Parse LLM JSON output and attach RAG references."""
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        items = json.loads(raw)
    except Exception:
        # Fallback: try to extract JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except Exception:
            return []

    # Build ref index: "REF1" → rag_docs[0], etc.
    ref_map = {f"REF{i+1}": doc for i, doc in enumerate(rag_docs)}

    results = []
    for item in items[:5]:
        ref_key = item.get("ref")
        doc = ref_map.get(ref_key) if ref_key else None
        results.append({
            "cause":  item.get("cause", ""),
            "reason": item.get("reason", ""),
            "reference": {"title": doc["title"], "url": doc["url"]} if doc else None,
        })

    return results


def run_causes_generator(state: AgentState) -> AgentState:
    """LangGraph node — RAG-enhanced structured possible causes generation."""
    patient_text = state.get("patient_text", "")
    department   = state.get("department", "General Medicine")

    rag_docs = _get_rag_docs(patient_text)
    if rag_docs:
        print(f"[CausesGenerator] {len(rag_docs)} RAG docs retrieved")

    prompt = _build_prompt(patient_text, department, rag_docs,
                           age=state.get("age"), gender=state.get("gender"))
    raw    = _call_llm(prompt)
    causes = _parse_output(raw, rag_docs)

    print(f"[CausesGenerator] → {len(causes)} possible causes")
    return {**state, "possible_causes": causes}
