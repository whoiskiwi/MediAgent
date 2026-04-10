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
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

ROOT       = Path(__file__).resolve().parents[2]   # medi-agent/
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION = "medlineplus"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def get_retriever() -> "MedRetriever":
    return MedRetriever()


@lru_cache(maxsize=1)
def _make_embed_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


class MedRetriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._col = client.get_collection(
            name=COLLECTION,
            embedding_function=_make_embed_fn(),
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
