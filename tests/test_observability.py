from fastapi.testclient import TestClient

import app


def test_run_config_has_tags_metadata_and_run_id():
    cfg = app._run_config(app.ChatRequest(question="hi", thread_id="t1"))
    assert "nutribot" in cfg["tags"]
    assert cfg["metadata"]["thread_id"] == "t1"
    assert {"git_sha", "claude_model", "fast_model"} <= cfg["metadata"].keys()
    assert cfg["run_id"] and cfg["configurable"]["thread_id"] == "t1"
    assert app._run_config(app.ChatRequest(question="hi"))["run_id"] != cfg["run_id"]


def test_feedback_calls_create_feedback(monkeypatch):
    calls = []

    class FakeClient:
        def create_feedback(self, run_id, **kw):
            calls.append((run_id, kw))

    monkeypatch.setattr(app, "Client", FakeClient)
    r = TestClient(app.app).post("/feedback", json={"run_id": "abc", "score": 1, "comment": "ok"})
    assert r.status_code == 200
    assert calls == [("abc", {"key": "user_score", "score": 1, "comment": "ok"})]
