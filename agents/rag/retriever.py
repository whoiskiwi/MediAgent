"""
RAG Retriever — wraps ChromaDB for semantic search over MedlinePlus.

Usage:
    from agents.rag.retriever import get_retriever
    retriever = get_retriever()
    docs = retriever.query("back pain", n_results=3)
"""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

ROOT       = Path(__file__).resolve().parents[2]   # medi-agent/
CHROMA_DIR = ROOT / "data" / "chroma_db"
load_dotenv(ROOT / ".env")
COLLECTION = "medlineplus"


@lru_cache(maxsize=1)
def get_retriever() -> "MedRetriever":
    return MedRetriever()


class MedRetriever:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        embed_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._col = client.get_collection(
            name=COLLECTION,
            embedding_function=embed_fn,
        )

    def query(self, text: str, n_results: int = 3) -> list[dict]:
        """Return top-n relevant documents as list of {title, text, url}."""
        results = self._col.query(
            query_texts=[text],
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        docs = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({
                "title": meta.get("title", ""),
                "url":   meta.get("url", ""),
                "text":  doc,
            })
        return docs

    def is_ready(self) -> bool:
        try:
            return self._col.count() > 0
        except Exception:
            return False
