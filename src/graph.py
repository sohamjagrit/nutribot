"""ReAct agent: tool-calling loop with conversation memory."""

import logging
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_anthropic import ChatAnthropic
from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, SQLITE_DB_PATH
from src.tools import retrieve_nutrition_info, calculate_macros

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are NutriBot, a personalized nutrition assistant backed by a science knowledge base.

Rules:
- Always call retrieve_nutrition_info before answering questions about foods, nutrients,
  vitamins, minerals, diet, or health. Never answer from memory alone.
- Call calculate_macros when a user wants protein/carbs/fat gram targets from a calorie goal.
- Cite retrieved sources inline as [Source 1], [Source 2], etc.
- Never invent facts or numbers outside the retrieved context.
- You have full conversation history — use it to give personalized, context-aware answers.
- Recommend consulting a healthcare professional for personal medical concerns."""

_llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=CLAUDE_MODEL)

checkpointer = AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH)

graph = create_react_agent(
    _llm,
    [retrieve_nutrition_info, calculate_macros],
    checkpointer=checkpointer,
    prompt=_SYSTEM,
)
