"""Per-request usage monitoring: node latency, tokens and cost, emitted as one JSON log line.

`RequestMonitor.callback` is added to the run config; `log()` writes the line CloudWatch filters read.
`request_id` equals the LangSmith run_id, so a log line links straight to its trace.
"""

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

from config.settings import ENV, FAST_MODEL, GIT_SHA, PRICING_FAST, PRICING_MAIN

_log = logging.getLogger("nutribot.request")
if not _log.handlers:  # raw JSON only (no logging prefix) so CloudWatch can parse it
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


def cost_usd(model: str, uncached: int, cache_read: int, cache_write: int, output: int) -> float:
    """Anthropic pricing: cache reads 0.1x input, 5-minute cache writes 1.25x input. USD per 1M tokens."""
    inp, out = PRICING_FAST if model == FAST_MODEL else PRICING_MAIN
    return (
        uncached * inp + cache_read * inp * 0.1 + cache_write * inp * 1.25 + output * out
    ) / 1_000_000


class _Usage(BaseCallbackHandler):
    def __init__(self):
        self.node_ms: dict[str, int] = defaultdict(int)
        self.tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.cost = 0.0
        self._started: dict = {}
        self._model: dict = {}

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, name=None, **kw):
        if name and name != "__start__" and metadata and name == metadata.get("langgraph_node"):
            self._started[run_id] = (name, time.perf_counter())

    def _chain_done(self, run_id):
        if run_id in self._started:
            name, t0 = self._started.pop(run_id)
            self.node_ms[name] += round((time.perf_counter() - t0) * 1000)

    def on_chain_end(self, outputs, *, run_id, **kw):
        self._chain_done(run_id)

    def on_chain_error(self, error, *, run_id, **kw):
        self._chain_done(run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        self._model[run_id] = (metadata or {}).get("ls_model_name", "")

    def on_llm_end(self, response, *, run_id, **kw):
        model = self._model.pop(run_id, "")
        for gens in response.generations:
            for g in gens:
                u = getattr(getattr(g, "message", None), "usage_metadata", None)
                if not u:
                    continue
                d = u.get("input_token_details") or {}
                read = d.get("cache_read") or 0
                write = (d.get("cache_creation") or 0) + (d.get("ephemeral_5m_input_tokens") or 0)
                uncached = u["input_tokens"] - read - write
                self.tokens["input"] += u["input_tokens"]
                self.tokens["output"] += u["output_tokens"]
                self.tokens["cache_read"] += read
                self.tokens["cache_write"] += write
                self.cost += cost_usd(model, uncached, read, write, u["output_tokens"])


class RequestMonitor:
    def __init__(self, request_id: str, thread_id: str, endpoint: str, resumed: bool = False):
        self.request_id, self.thread_id, self.endpoint, self.resumed = (
            request_id,
            thread_id,
            endpoint,
            resumed,
        )
        self.usage = _Usage()
        self.callback = self.usage
        self._t0 = time.perf_counter()

    def log(
        self,
        state: dict | None = None,
        interrupted: bool = False,
        status: int = 200,
        error: str | None = None,
    ):
        s = state or {}
        route = s.get("route")
        refused = route == "off_topic" or (route == "nutrition" and not s.get("supported"))
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "endpoint": self.endpoint,
            "env": ENV,
            "git_sha": GIT_SHA,
            "status": status,
            "error": error,
            "route": route,
            "grader_supported": s.get("supported") if route == "nutrition" else None,
            "retries": s.get("retries"),
            "refused": refused and not interrupted,
            "interrupted": interrupted,
            "resumed": self.resumed,
            "latency_ms": round((time.perf_counter() - self._t0) * 1000),
            "node_ms": dict(self.usage.node_ms),
            "tokens": self.usage.tokens,
            "cost_usd": round(self.usage.cost, 6),
        }
        _log.info(json.dumps(rec))
        return rec
