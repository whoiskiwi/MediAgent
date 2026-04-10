"""
PDF Ingestion — parse, chunk, embed, and store user-uploaded PDFs in ChromaDB.

Usage:
    from agents.rag.ingest import ingest_pdf
    chunks_added = ingest_pdf(file_bytes, "my_document.pdf")
"""
import io
import os
import uuid
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

ROOT            = Path(__file__).resolve().parents[2]
CHROMA_DIR      = ROOT / "data" / "chroma_db"
USER_COLLECTION = "user_docs"
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"

load_dotenv(ROOT / ".env")


def _get_user_collection():
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=USER_COLLECTION,
        embedding_function=embed_fn,
    )


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """
    Parse a PDF, chunk it, embed, and store in user_docs collection.

    Returns the number of chunks added.
    Raises RuntimeError on parse or embedding failure.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed — run: pip install pypdf")

    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages_text)

    if not full_text.strip():
        raise RuntimeError("No text could be extracted from the PDF")

    raw = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 40]
    chunks: list[str] = []
    current = ""
    for para in raw:
        if len(current) + len(para) + 2 <= 400:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            current = para[:400]
    if current:
        chunks.append(current)

    if not chunks:
        raise RuntimeError("No usable text chunks found in PDF")

    col = _get_user_collection()
    doc_id = str(uuid.uuid4())[:8]
    ids       = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "doc_id": doc_id} for _ in chunks]
    col.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def list_user_docs() -> list[dict]:
    """Return a list of distinct uploaded documents: [{doc_id, source, chunks}]."""
    col = _get_user_collection()
    if col.count() == 0:
        return []
    results = col.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "")
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "source": meta.get("source", ""), "chunks": 0}
        seen[doc_id]["chunks"] += 1
    return list(seen.values())


def delete_user_doc(doc_id: str) -> int:
    """Delete all chunks for a given doc_id. Returns chunks deleted."""
    col = _get_user_collection()
    results = col.get(where={"doc_id": doc_id}, include=[])
    ids = results.get("ids", [])
    if ids:
        col.delete(ids=ids)
    return len(ids)
