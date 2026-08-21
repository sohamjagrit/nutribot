"""NutriBot FastAPI application."""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from src.graph import graph
from src.tools import parse_sources_from_tool_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="NutriBot", description="Personalized RAG Chatbot for Nutrition Q&A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class ChatRequest(BaseModel):
    question: str
    user_context: str = ""
    thread_id: str = "default"


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    needs_clarification: bool


def _make_message(req: ChatRequest) -> HumanMessage:
    content = req.question
    if req.user_context:
        content = f"[User context: {req.user_context}]\n{content}"
    return HumanMessage(content=content)


def _run_config(req: ChatRequest) -> dict:
    return {
        "run_name": "nutribot-chat",
        "configurable": {"thread_id": req.thread_id},
    }


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = await graph.ainvoke(
        {"messages": [_make_message(req)]},
        config=_run_config(req),
    )
    answer = result["messages"][-1].content
    sources = []
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_nutrition_info":
            sources = parse_sources_from_tool_output(msg.content)
            break
    return ChatResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        needs_clarification=False,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint — streams tokens as Claude generates."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_stream():
        meta: dict = {"sources": [], "needs_clarification": False}
        try:
            async for event in graph.astream_events(
                {"messages": [_make_message(req)]},
                version="v2",
                config=_run_config(req),
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if isinstance(content, list):
                        content = "".join(
                            b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    if content:
                        yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

                elif kind == "on_tool_end" and event["name"] == "retrieve_nutrition_info":
                    output = event["data"].get("output")
                    text = output.content if hasattr(output, "content") else str(output or "")
                    sources = parse_sources_from_tool_output(text)
                    if sources:
                        meta["sources"] = sources

            yield f"data: {json.dumps({'type': 'done', **meta})}\n\n"
        except Exception:
            logging.exception("Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Internal server error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
