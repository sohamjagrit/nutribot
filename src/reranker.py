"""Cross-encoder reranker — scores query-doc pairs and returns top-k by relevance."""

import logging
from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL, RERANK_TOP_K, MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, docs: list[str], top_k: int = RERANK_TOP_K) -> tuple[list[str], float]:
    """Score each doc against the query, return (top_k reranked docs, best score).

    ms-marco scores are logits: positive = relevant, strongly negative = off-topic.
    Best score is used as a relevance gate against MIN_RELEVANCE_SCORE.
    """
    if not docs:
        return [], float("-inf")

    scores = _get_model().predict([(query, doc) for doc in docs])
    ranked = sorted(zip(scores, docs), reverse=True)[:top_k]
    best_score = float(ranked[0][0])

    logger.info(f"Reranked {len(docs)} → {len(ranked)} docs | best score: {best_score:.2f} | threshold: {MIN_RELEVANCE_SCORE}")
    return [doc for _, doc in ranked], best_score
