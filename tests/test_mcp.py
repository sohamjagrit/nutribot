import asyncio

import app


def test_load_mcp_tools_degrades_when_unreachable(monkeypatch):
    monkeypatch.setattr(app, "MCP_NUTRITION_URL", "http://127.0.0.1:1/mcp")
    assert asyncio.run(app.load_mcp_tools()) == []
