import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

import app
from config.settings import CLAUDE_MODEL, FAST_MODEL
from src import monitoring
from src.graph import _calorie_prompt
from src.monitoring import RequestMonitor, cost_usd


@pytest.fixture
def lines(caplog):
    """JSON log lines emitted by the request logger (it does not propagate, so attach caplog directly)."""
    monitoring._log.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="nutribot.request")
    yield lambda: [r.getMessage() for r in caplog.records if r.name == "nutribot.request"]
    monitoring._log.removeHandler(caplog.handler)


def _llm_result(input_tokens, output_tokens, **details):
    msg = AIMessage(
        "x",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": details,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_cost_counts_cache_reads_and_writes_at_their_rates():
    # Sonnet $3/$15: 1000 uncached + 2000 read (0.1x) + 400 write (1.25x) + 500 out
    c = cost_usd(CLAUDE_MODEL, 1000, 2000, 400, 500)
    assert abs(c - (3000 + 600 + 1500 + 7500) / 1e6) < 1e-9
    assert cost_usd(FAST_MODEL, 1000, 0, 0, 0) == 1000 / 1e6  # Haiku $1 in


def test_usage_handler_tracks_tokens_cost_and_node_latency():
    mon = RequestMonitor("r1", "t1", "/chat")
    n, m = uuid.uuid4(), uuid.uuid4()
    mon.callback.on_chain_start(
        {}, {}, run_id=n, metadata={"langgraph_node": "router"}, name="router"
    )
    mon.callback.on_chat_model_start({}, [], run_id=m, metadata={"ls_model_name": FAST_MODEL})
    mon.callback.on_llm_end(_llm_result(3000, 100, cache_read=2000, cache_creation=500), run_id=m)
    mon.callback.on_chain_end({}, run_id=n)
    assert mon.usage.tokens == {
        "input": 3000,
        "output": 100,
        "cache_read": 2000,
        "cache_write": 500,
    }
    assert abs(mon.usage.cost - cost_usd(FAST_MODEL, 500, 2000, 500, 100)) < 1e-12
    assert "router" in mon.usage.node_ms


def test_log_is_one_json_line_with_route_refusal_and_interrupt(lines):
    mon = RequestMonitor("r1", "t1", "/chat/stream")
    rec = mon.log({"route": "nutrition", "supported": False, "retries": 1})
    (line,) = lines()
    assert "\n" not in line and json.loads(line) == rec
    assert rec["refused"] and rec["grader_supported"] is False and rec["status"] == 200
    assert not RequestMonitor("r", "t", "/chat").log({"route": "calories"}, interrupted=True)[
        "refused"
    ]
    assert RequestMonitor("r", "t", "/chat").log({"route": "off_topic"})["refused"]
    err = RequestMonitor("r", "t", "/chat").log(status=500, error="ValueError")
    assert err["status"] == 500 and err["route"] is None and not err["refused"]


def test_chat_error_logs_500(monkeypatch, lines):
    class Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("x")

    monkeypatch.setattr(app, "graph", Boom())
    r = TestClient(app.app, raise_server_exceptions=False).post("/chat", json={"question": "hi"})
    assert r.status_code == 500
    rec = json.loads(lines()[-1])
    assert rec["status"] == 500 and rec["error"] == "RuntimeError" and rec["endpoint"] == "/chat"


def test_calorie_prompt_caches_static_system_and_keeps_summary_outside_it():
    sys_msg = _calorie_prompt({"messages": [], "summary": "user is 30"})[0]
    assert isinstance(sys_msg, SystemMessage)
    static, summary = sys_msg.content
    assert static["cache_control"] == {"type": "ephemeral"} and "cache_control" not in summary
    assert "user is 30" in summary["text"]
    assert len(_calorie_prompt({"messages": []})[0].content) == 1
