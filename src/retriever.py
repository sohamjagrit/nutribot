"""Pinecone retriever — swap this module to change retrieval strategy."""

import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langsmith import traceable
from config.settings import PINECONE_API_KEY, PINECONE_INDEX, EMBED_MODEL, RERANK_CANDIDATES

logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None
_index = None

# BGE models perform better with this prefix on queries (not on documents).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = pc.Index(PINECONE_INDEX)
        stats = _index.describe_index_stats()
        logger.info(f"Pinecone index '{PINECONE_INDEX}': {stats.total_vector_count} vectors")
    return _index


@traceable(name="pinecone-retrieve", run_type="retriever")
def retrieve(query: str, top_k: int = RERANK_CANDIDATES) -> list[str]:
    """Return top-k document texts for a query."""
    prefix = _BGE_QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
    vec = _get_embedder().encode(prefix + query).tolist()
    results = _get_index().query(vector=vec, top_k=top_k, include_metadata=True)
    return [m["metadata"]["text"] for m in results["matches"] if "text" in m["metadata"]]
