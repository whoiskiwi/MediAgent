"""
Medical Q&A — RAG-powered question answering over MedlinePlus + user docs.

Drug lookup tries OpenFDA first (free, no API key needed), falls back to RAG.
"""
import os
from pathlib import Path

import requests as http_requests
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]   # medi-agent/
load_dotenv(ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Medical Q&A (MedlinePlus + user docs)
# ---------------------------------------------------------------------------

def answer_question(question: str, n_docs: int = 4) -> dict:
    """
    Returns:
        {
          "answer": str,
          "sources": [{"title": str, "url": str}]
        }
    """
    from agents.rag.retriever import get_retriever

    docs = get_retriever().query(question, n_results=n_docs)

    if not docs:
        return {
            "answer": "Sorry, I couldn't find relevant information in the knowledge base.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[{d['title']}]\n{d['text'][:800]}" for d in docs
    )
    prompt = (
        "You are a compassionate medical information assistant helping patients understand "
        "their conditions and take better care of themselves.\n\n"
        "Using ONLY the references provided, answer the patient's question by covering these angles "
        "(where relevant):\n"
        "1. Relief: What can they do right now to ease discomfort or symptoms?\n"
        "2. Prevention: How can they avoid this condition or stop it from getting worse?\n"
        "3. Reducing recurrence: Lifestyle, diet, or habits that lower the chance of it happening again?\n"
        "4. Reducing suffering: Home care tips, what to watch out for, when to see a doctor?\n\n"
        "Be practical and specific. If the references don't cover a particular angle, skip it. "
        "Do not invent information not found in the references.\n\n"
        f"References:\n{context}\n\n"
        f"Patient question: {question}\n\nAnswer:"
    )

    answer = _call_llm(prompt)
    sources = [{"title": d["title"], "url": d["url"]} for d in docs]
    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# Drug lookup — FDA first, RAG fallback
# ---------------------------------------------------------------------------

def lookup_drug(drug_name: str) -> dict:
    """
    Look up drug information. Tries OpenFDA API first, then RAG fallback.

    Returns:
        {
          "source": "FDA" | "RAG",
          "answer": str,
          "sources": [{"title": str, "url": str}],
          "fda": {indications, warnings, side_effects, dosage} | None
        }
    """
    fda_data = _lookup_drug_fda(drug_name)
    if fda_data:
        summary = _format_fda_summary(drug_name, fda_data)
        fda_url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=BasicSearch.process&query={drug_name.replace(' ', '+')}"
        return {
            "source": "FDA",
            "answer": summary,
            "sources": [{"title": f"FDA Drug Label — {drug_name}", "url": fda_url}],
            "fda": fda_data,
        }

    # FDA had nothing → fall back to MedlinePlus RAG
    rag_result = answer_question(
        f"What is {drug_name}? What is it used for, what are the side effects, "
        f"and what precautions should patients know?"
    )
    return {"source": "RAG", "fda": None, **rag_result}


def _lookup_drug_fda(drug_name: str) -> dict | None:
    """Query OpenFDA drug label API. Returns structured dict or None."""
    try:
        url = "https://api.fda.gov/drug/label.json"
        # Try brand name first, then generic name
        for field in ("openfda.brand_name", "openfda.generic_name"):
            params = {"search": f'{field}:"{drug_name}"', "limit": 1}
            resp = http_requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    r = results[0]
                    return {
                        "indications": _first(r.get("indications_and_usage")),
                        "warnings":    _first(r.get("warnings") or r.get("boxed_warning")),
                        "side_effects": _first(r.get("adverse_reactions")),
                        "dosage":      _first(r.get("dosage_and_administration")),
                    }
    except Exception as e:
        print(f"[QA] OpenFDA lookup failed: {e}")
    return None


def _first(field) -> str:
    """Extract first element from FDA list field, clean noise, or 'N/A'."""
    if not field:
        return "N/A"
    text = field[0] if isinstance(field, list) else str(field)
    if not text:
        return "N/A"
    text = _clean_fda_text(text)
    return text[:600]


def _clean_fda_text(text: str) -> str:
    """Remove FDA label formatting noise: section numbers, inline citations."""
    import re
    # Remove leading section header like "1 INDICATIONS & USAGE " or "6 ADVERSE REACTIONS "
    text = re.sub(r'^\d+(\.\d+)*\s+[A-Z][A-Z &]+\s*', '', text)
    # Remove inline citation numbers like "( 1 )", "( 2.1 )", "(1)", "(2.1)"
    text = re.sub(r'\(\s*\d+(\.\d+)*\s*\)', '', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _format_fda_summary(drug_name: str, fda: dict) -> str:
    parts = [f"**{drug_name}** (FDA Drug Label)\n"]
    if fda.get("indications") and fda["indications"] != "N/A":
        parts.append(f"**Indications:** {fda['indications']}")
    if fda.get("dosage") and fda["dosage"] != "N/A":
        parts.append(f"**Dosage:** {fda['dosage']}")
    if fda.get("warnings") and fda["warnings"] != "N/A":
        parts.append(f"**Warnings:** {fda['warnings']}")
    if fda.get("side_effects") and fda["side_effects"] != "N/A":
        parts.append(f"**Side Effects:** {fda['side_effects']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Shared LLM caller
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    if DEEPSEEK_API_KEY:
        try:
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[QA] DeepSeek failed: {e}, falling back to OpenAI")

    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    return "No LLM available to answer the question."
