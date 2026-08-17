#!/usr/bin/env python3
"""
Ingest local documents into Pinecone.

Reads all .txt files under data/raw/, chunks them, embeds with SentenceTransformer,
and upserts to Pinecone. Run this once before starting the app.

Usage:
    uv run python scripts/ingest.py
"""

import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config.settings import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 100


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def main():
    data_dir = Path("data/raw")
    txt_files = sorted(data_dir.rglob("*.txt"))
    if not txt_files:
        logger.error(f"No .txt files found under {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(txt_files)} files: {[f.name for f in txt_files]}")

    chunks, meta = [], []
    for f in txt_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.md5(f"{f.name}:{i}".encode()).hexdigest()
            chunks.append(chunk)
            meta.append({"id": chunk_id, "source": f.name, "chunk_index": i})

    logger.info(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)

    vectors = [
        {"id": m["id"], "values": e.tolist(), "metadata": {"text": c, "source": m["source"]}}
        for c, e, m in zip(chunks, embeddings, meta)
    ]

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        logger.info(f"  Upserted {min(i + BATCH_SIZE, len(vectors))}/{len(vectors)}")

    stats = index.describe_index_stats()
    logger.info(f"Done. {stats.total_vector_count} vectors in index '{PINECONE_INDEX}'.")


if __name__ == "__main__":
    main()
