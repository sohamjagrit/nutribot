from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import graph
from src.graph import _cut_index, needs_summary, summarize


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage("SUMMARY")


def _turns(n):
    msgs = []
    for i in range(n):
        msgs += [HumanMessage(f"q{i}", id=f"h{i}"), AIMessage(f"a{i}", id=f"a{i}")]
    return msgs


def test_under_budget_skips(monkeypatch):
    monkeypatch.setattr(graph, "CONTEXT_TOKEN_BUDGET", 10_000)
    assert needs_summary({"messages": _turns(3)}) == "router"


def test_over_budget_summarizes(monkeypatch):
    monkeypatch.setattr(graph, "CONTEXT_TOKEN_BUDGET", 5)
    assert needs_summary({"messages": _turns(10)}) == "summarize"


def test_over_budget_but_nothing_to_fold(monkeypatch):
    monkeypatch.setattr(graph, "CONTEXT_TOKEN_BUDGET", 5)
    assert needs_summary({"messages": _turns(2)}) == "router"  # 4 messages <= keep 6 -> cut at 0


def test_summarize_removes_old_keeps_last(monkeypatch):
    monkeypatch.setattr(graph, "_fast", FakeLLM())
    monkeypatch.setattr(graph, "KEEP_LAST_MESSAGES", 4)
    msgs = _turns(5)
    out = summarize({"messages": msgs, "summary": ""})
    assert out["summary"] == "SUMMARY"
    assert [m.id for m in out["messages"]] == [m.id for m in msgs[:6]]


def test_cut_never_splits_tool_pair():
    msgs = [
        HumanMessage("q0", id="h0"),
        AIMessage("a0", id="a0"),
        HumanMessage("q1", id="h1"),
        AIMessage("", id="t1", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]),
        ToolMessage("food data", tool_call_id="c1", id="r1"),
        AIMessage("done", id="a1"),
    ]
    # keep=2 would start at the ToolMessage; it must back up to the human message.
    assert _cut_index(msgs, 2) == 2


def test_existing_summary_is_folded_in(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(graph, "_fast", fake)
    monkeypatch.setattr(graph, "KEEP_LAST_MESSAGES", 2)
    summarize({"messages": _turns(4), "summary": "user weighs 80kg"})
    assert "user weighs 80kg" in fake.prompts[0]
