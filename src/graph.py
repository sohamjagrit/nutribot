"""LangGraph RAG pipeline: retrieve → generate."""

import logging
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import GROQ_API_KEY, GROQ_MODEL
from src.retriever import retrieve as pinecone_retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are NutriBot, a nutrition assistant backed by a knowledge base.

Rules:
- Answer ONLY using the provided context. Do not use outside knowledge.
- Cite sources inline like [Source 1], [Source 2].
- If the context doesn't contain the answer, or the question is not about nutrition, say:
  "I don't have enough information in my knowledge base to answer that."
- Never invent facts or numbers."""


class RAGState(TypedDict):
    question: str
    documents: list[str]
    answer: str


_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)


def retrieve(state: RAGState) -> dict:
    docs = pinecone_retrieve(state["question"])
    logger.info(f"Retrieved {len(docs)} documents")
    return {"documents": docs}


def generate(state: RAGState) -> dict:
    context = "\n\n".join(
        f"[Source {i + 1}] {doc}" for i, doc in enumerate(state["documents"])
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
    ]
    response = _llm.invoke(messages)
    return {"answer": response.content}


_builder = StateGraph(RAGState)
_builder.add_node("retrieve", retrieve)
_builder.add_node("generate", generate)
_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()
