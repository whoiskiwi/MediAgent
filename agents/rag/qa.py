"""
Medical Q&A — RAG-powered question answering over MedlinePlus.

Retrieves relevant documents, then asks DeepSeek (or OpenAI) to answer
based on the retrieved context.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]   # medi-agent/
load_dotenv(ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


def answer_question(question: str, n_docs: int = 4) -> dict:
    """
    Returns:
        {
          "answer": str,
          "sources": [{"title": str, "url": str}]
        }
    """
    from agents.rag.retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.query(question, n_results=n_docs)

    if not docs:
        return {"answer": "Sorry, I couldn't find relevant information in the knowledge base.", "sources": []}

    context = "\n\n".join(
        f"[{d['title']}]\n{d['text'][:800]}" for d in docs
    )

    prompt = (
        f"You are a helpful medical information assistant. "
        f"Use ONLY the following MedlinePlus references to answer the question. "
        f"Be concise and accurate. If the references don't contain enough information, say so.\n\n"
        f"References:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    answer = _call_llm(prompt)
    sources = [{"title": d["title"], "url": d["url"]} for d in docs]
    return {"answer": answer, "sources": sources}


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
