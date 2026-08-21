"""LangChain tools exposed to the ReAct agent."""

import re
import logging
from langchain_core.tools import tool
from src.retriever import retrieve as pinecone_retrieve
from src.reranker import rerank
from config.settings import RERANK_CANDIDATES, MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)

_REFUSAL = (
    "I couldn't find relevant information in my knowledge base to answer that. "
    "Try asking about nutrients, foods, vitamins, diet, or weight management."
)


@tool
def retrieve_nutrition_info(query: str) -> str:
    """Search the nutrition knowledge base. Use for ANY question about foods, nutrients,
    vitamins, minerals, diet, weight management, or health — even if you think you know
    the answer. Always retrieve before responding."""
    docs = pinecone_retrieve(query, top_k=RERANK_CANDIDATES)
    if not docs:
        return _REFUSAL
    reranked, best_score = rerank(query, docs)
    if best_score < MIN_RELEVANCE_SCORE:
        return _REFUSAL
    return "\n\n".join(f"[Source {i+1}] {d}" for i, d in enumerate(reranked))


@tool
def calculate_macros(
    calories: int,
    protein_pct: float = 30,
    carbs_pct: float = 40,
    fat_pct: float = 30,
) -> str:
    """Calculate macronutrient gram targets from a daily calorie goal.
    Protein and carbs = 4 kcal/g; fat = 9 kcal/g. Percentages should sum to 100."""
    protein_g = round(calories * protein_pct / 100 / 4)
    carbs_g = round(calories * carbs_pct / 100 / 4)
    fat_g = round(calories * fat_pct / 100 / 9)
    return (
        f"{calories} kcal/day → "
        f"Protein: {protein_g}g ({protein_pct:.0f}%) | "
        f"Carbs: {carbs_g}g ({carbs_pct:.0f}%) | "
        f"Fat: {fat_g}g ({fat_pct:.0f}%)"
    )


def parse_sources_from_tool_output(text: str) -> list[str]:
    """Extract individual source texts from a retrieve_nutrition_info result."""
    if not text or "[Source" not in text:
        return []
    parts = re.split(r"\[Source \d+\]\s*", text)
    return [p.strip() for p in parts if p.strip()]
