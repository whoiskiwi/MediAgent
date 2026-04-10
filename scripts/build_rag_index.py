"""
Parse MedlinePlus XML and build a ChromaDB vector index.

Usage:
    python scripts/build_rag_index.py

The index is saved to data/chroma_db/ and reused on subsequent runs.
Only English topics with a non-empty full-summary are indexed.
"""
import html
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

XML_PATH    = ROOT / "data" / "mplus_topics_2026-04-01.xml"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
COLLECTION  = "medlineplus"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE  = 64   # smaller batch — bge-large is heavier than text-embedding-3-small


def clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_topics(xml_path: Path) -> list[dict]:
    print(f"Parsing {xml_path} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    topics = []
    for topic in root.findall("health-topic"):
        if topic.attrib.get("language", "English") != "English":
            continue

        title   = topic.attrib.get("title", "")
        url     = topic.attrib.get("url", "")
        summary_el = topic.find("full-summary")
        if summary_el is None or not summary_el.text:
            continue

        summary = clean_html(summary_el.text)
        if len(summary) < 50:
            continue

        groups = [g.text for g in topic.findall("group") if g.text]
        also_called = [a.text for a in topic.findall("also-called") if a.text]

        # Build a rich text chunk: title + aliases + summary
        text_parts = [f"Title: {title}"]
        if also_called:
            text_parts.append(f"Also known as: {', '.join(also_called)}")
        text_parts.append(summary)

        topics.append({
            "id":     url or title,
            "text":  "\n".join(text_parts),
            "title": title,
            "groups": ", ".join(groups),
            "url":   url,
        })

    print(f"  → {len(topics)} English topics with summaries")
    return topics


def build_index(topics: list[dict]):
    print(f"Loading embedding model: {EMBED_MODEL} (first run downloads ~1.3 GB) ...")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to rebuild cleanly
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(topics)
    for i in range(0, total, BATCH_SIZE):
        batch = topics[i : i + BATCH_SIZE]
        collection.add(
            ids        = [t["id"] for t in batch],
            documents  = [t["text"] for t in batch],
            metadatas  = [{"title": t["title"], "groups": t["groups"], "url": t["url"]} for t in batch],
        )
        print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total}", end="\r")

    print(f"\nDone. {total} documents indexed in {CHROMA_DIR}")


if __name__ == "__main__":
    topics = parse_topics(XML_PATH)
    build_index(topics)
