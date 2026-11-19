"""Multi-agent LangGraph: router -> retrieve -> grade -> (rewrite -> retrieve) -> answer.

Each LLM node has one role. Fast model: router, grader, rewriter. Main model: answer.
"""

import logging
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config.settings import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, FAST_MODEL, MAX_RETRIEVAL_RETRIES,
    MIN_RELEVANCE_SCORE, RERANK_CANDIDATES,
)
from src.reranker import rerank
from src.retriever import retrieve as pinecone_retrieve
from src.tools import calculate_macros


logger = logging.getLogger(__name__)

_REFUSAL = (
    "I couldn't find relevant information in my knowledge base to answer that. "
    "Try asking about nutrients, foods, vitamins, diet, or weight management."
)
_OFF_TOPIC = (
    "I'm NutriBot, a nutrition assistant. I can help with foods, nutrients, vitamins, "
    "diet and macro targets. For personal medical concerns, please see a healthcare professional."
)

_ANSWER_SYSTEM = """\
You are NutriBot, a personalized nutrition assistant backed by a science knowledge base.

Rules:
- Answer ONLY from the context below. Never invent facts or numbers outside it.
- Cite retrieved sources inline as [Source 1], [Source 2], etc. (macro results need no citation).
- Use the conversation history for personalized, context-aware answers.
- Recommend consulting a healthcare professional for personal medical concerns.

Context:
{context}"""


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    query: str
    docs: list[str]
    macros: str
    supported: bool
    retries: int


class Route(BaseModel):
    route: Literal["nutrition", "macros", "off_topic"] = Field(
        description="nutrition: food/nutrient/diet/health question; macros: wants gram targets "
        "from a calorie goal; off_topic: unrelated to nutrition."
    )
    query: str = Field(description="Standalone search query, resolving references to earlier turns.")
    calories: int | None = Field(default=None, description="Daily calories, macros route only.")
    protein_pct: float = 30
    carbs_pct: float = 40
    fat_pct: float = 30


class Grade(BaseModel):
    supported: bool = Field(description="True if the context contains information that answers the question.")


_llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=CLAUDE_MODEL)
_fast = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=FAST_MODEL, temperature=0)
_router = _fast.with_structured_output(Route)
_grader = _fast.with_structured_output(Grade)


def _text(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c


def _history(state: State, n: int = 6) -> str:
    return "\n".join(f"{m.type}: {_text(m)}" for m in state["messages"][-n:])


def router(state: State) -> dict:
    r = _router.invoke(
        "Classify the latest user message and write a standalone search query.\n\n" + _history(state)
    )
    macros = ""
    if r.route == "macros":
        if r.calories:
            macros = calculate_macros.invoke(
                {"calories": r.calories, "protein_pct": r.protein_pct,
                 "carbs_pct": r.carbs_pct, "fat_pct": r.fat_pct}
            )
        else:
            r.route = "nutrition"  # no calorie number to compute with
    logger.info(f"router: {r.route} | query: {r.query!r}")
    # Reset per-turn state (checkpointer persists it across turns).
    return {"route": r.route, "query": r.query, "macros": macros, "docs": [], "supported": False, "retries": 0}


def retrieve(state: State) -> dict:
    docs = pinecone_retrieve(state["query"], top_k=RERANK_CANDIDATES)
    if not docs:
        return {"docs": []}
    reranked, best = rerank(state["query"], docs)
    return {"docs": reranked if best >= MIN_RELEVANCE_SCORE else []}


def grade(state: State) -> dict:
    if not state["docs"]:
        return {"supported": False}
    context = "\n\n".join(state["docs"])
    g = _grader.invoke(f"Question: {state['query']}\n\nContext:\n{context}\n\nDoes the context answer the question?")
    logger.info(f"grader: supported={g.supported}")
    return {"supported": g.supported}


def rewrite(state: State) -> dict:
    resp = _fast.invoke(
        f"Rewrite this search query to retrieve better nutrition documents. "
        f"Use different keywords or more specific terms. Return only the query.\n\nQuery: {state['query']}"
    )
    return {"query": _text(resp).strip(), "retries": state["retries"] + 1}


def answer(state: State) -> dict:
    parts = []
    if state["macros"]:
        parts.append(f"Macro calculation: {state['macros']}")
    parts += [f"[Source {i + 1}] {d}" for i, d in enumerate(state["docs"])]
    system = SystemMessage(_ANSWER_SYSTEM.format(context="\n\n".join(parts)))
    resp = _llm.invoke([system, *state["messages"]])
    return {"messages": [resp]}


def refuse(state: State) -> dict:
    return {"messages": [AIMessage(_OFF_TOPIC if state["route"] == "off_topic" else _REFUSAL)]}


def after_router(state: State) -> str:
    return {"nutrition": "retrieve", "macros": "answer", "off_topic": "refuse"}[state["route"]]


def after_grade(state: State) -> str:
    if state["supported"]:
        return "answer"
    return "rewrite" if state["retries"] < MAX_RETRIEVAL_RETRIES else "refuse"


def create_graph(checkpointer):
    g = StateGraph(State)
    for fn in (router, retrieve, grade, rewrite, answer, refuse):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, ["retrieve", "answer", "refuse"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["answer", "rewrite", "refuse"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer)
