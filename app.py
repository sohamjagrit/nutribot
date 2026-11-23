"""NutriBot FastAPI application."""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel
from config.settings import CLAUDE_MODEL, ENV, FAST_MODEL, GIT_SHA, MCP_NUTRITION_URL, SQLITE_DB_PATH
from langsmith import Client
from src.graph import create_graph
from src.monitoring import RequestMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

graph = None
mcp_tools: list = []  # OpenNutrition food-lookup tools; empty if the MCP server is unreachable


async def load_mcp_tools() -> list:
    """Load food tools from the MCP container once. Degrades to [] so the app still starts."""
    try:
        client = MultiServerMCPClient(
            {"opennutrition": {"transport": "streamable_http", "url": MCP_NUTRITION_URL}}
        )
        tools = await client.get_tools()
        logging.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])
        return tools
    except Exception:
        logging.warning("MCP server unreachable at %s; food lookup disabled", MCP_NUTRITION_URL, exc_info=True)
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, mcp_tools
    mcp_tools = await load_mcp_tools()
    os.makedirs(os.path.dirname(SQLITE_DB_PATH) or ".", exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
        graph = create_graph(checkpointer, mcp_tools)
        yield


app = FastAPI(title="NutriBot", description="Personalized RAG Chatbot for Nutrition Q&A", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChatRequest(BaseModel):
    question: str = ""
    user_context: str = ""
    thread_id: str = "default"
    resume: dict | bool | None = None  # answer to a pending interrupt (fields dict or confirm bool)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    needs_clarification: bool
    interrupt: dict | None = None
    run_id: str | None = None


class FeedbackRequest(BaseModel):
    run_id: str
    score: int  # 1 = thumbs up, 0 = thumbs down
    comment: str | None = None


def _graph_input(req: ChatRequest):
    if req.resume is not None:
        return Command(resume=req.resume)
    return {"messages": [_make_message(req)]}


def _make_message(req: ChatRequest) -> HumanMessage:
    content = req.question
    if req.user_context:
        content = f"[User context: {req.user_context}]\n{content}"
    return HumanMessage(content=content)


def _run_config(req: ChatRequest) -> dict:
    return {
        "run_name": "nutribot-chat",
        "run_id": uuid.uuid4(),
        "tags": [ENV, "nutribot"],
        "metadata": {
            "thread_id": req.thread_id, "git_sha": GIT_SHA,
            "claude_model": CLAUDE_MODEL, "fast_model": FAST_MODEL,
        },
        "configurable": {"thread_id": req.thread_id},
    }


def _monitor(req: ChatRequest, config: dict, endpoint: str) -> RequestMonitor:
    mon = RequestMonitor(str(config["run_id"]), req.thread_id, endpoint, resumed=req.resume is not None)
    config["callbacks"] = [mon.callback]
    return mon


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.resume is None and not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    config = _run_config(req)
    mon = _monitor(req, config, "/chat")
    try:
        result = await graph.ainvoke(_graph_input(req), config=config)
    except Exception as e:
        mon.log(status=500, error=type(e).__name__)
        raise
    mon.log(result, interrupted="__interrupt__" in result)
    if "__interrupt__" in result:
        return ChatResponse(
            question=req.question, answer="", sources=[], needs_clarification=True,
            interrupt=result["__interrupt__"][0].value, run_id=str(config["run_id"]),
        )
    answer = result["messages"][-1].content
    return ChatResponse(
        question=req.question,
        answer=answer,
        sources=result["docs"] if result["supported"] else [],
        needs_clarification=False,
        run_id=str(config["run_id"]),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    Client().create_feedback(req.run_id, key="user_score", score=req.score, comment=req.comment)
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint — streams tokens as Claude generates."""
    if req.resume is None and not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    config = _run_config(req)
    mon = _monitor(req, config, "/chat/stream")

    async def event_stream():
        meta: dict = {"sources": [], "needs_clarification": False, "run_id": str(config["run_id"])}
        spoke = False
        try:
            async for event in graph.astream_events(
                _graph_input(req),
                version="v2",
                config=config,
            ):
                kind = event["event"]

                meta_ev = event.get("metadata", {})
                node = meta_ev.get("langgraph_node")
                in_calories = meta_ev.get("langgraph_checkpoint_ns", "").startswith("calories:")

                # The agent may speak before and after tool calls; keep those paragraphs apart.
                if kind == "on_chat_model_start" and in_calories and spoke:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chr(10) * 2})}\n\n"
                    spoke = False

                # Only the answer node and the calorie agent's tokens are shown;
                # router/grader output is internal.
                if kind == "on_chat_model_stream" and (node == "answer" or in_calories):
                    content = event["data"]["chunk"].content
                    if isinstance(content, list):
                        content = "".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if content:
                        spoke = in_calories
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                # Refusals are static text, so emit them whole when the node finishes.
                elif kind == "on_chain_end" and event["name"] == "refuse" and node == "refuse":
                    text = event["data"]["output"]["messages"][-1].content
                    yield f"data: {json.dumps({'type': 'chunk', 'content': text})}\n\n"

            final = await graph.aget_state(config)
            pending = [i.value for t in final.tasks for i in t.interrupts]
            if pending:
                mon.log(final.values, interrupted=True)
                yield f"data: {json.dumps({'type': 'interrupt', **pending[0]})}\n\n"
                return
            if final.values.get("route") == "nutrition" and final.values.get("supported"):
                meta["sources"] = final.values["docs"]

            mon.log(final.values)
            yield f"data: {json.dumps({'type': 'done', **meta})}\n\n"
        except Exception as e:
            logging.exception("Streaming error")
            mon.log(status=500, error=type(e).__name__)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
