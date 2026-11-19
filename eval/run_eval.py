"""NutriBot eval harness — produces real, quotable numbers for the RAG pipeline.

Two metric groups:
  1. Retrieval (deterministic, no API cost): precision@k, recall, MRR, hit rate.
  2. End-to-end (runs the agent): latency, tokens & cost per query, and
     LLM-as-judge groundedness + answer correctness.

Usage:
  # Full run, save a baseline snapshot:
  python eval/run_eval.py run --tag baseline

  # Cheap retrieval-only sweep (no LLM calls, no cost):
  python eval/run_eval.py run --tag baseline_retrieval --retrieval-only

  # After you change chunking / top_k / model, re-run and compare:
  python eval/run_eval.py run --tag reranker_v2
  python eval/run_eval.py compare eval/results/baseline.json eval/results/reranker_v2.json

The delta printed by `compare` is your resume number
("reduced X by N%", "improved Y from A to B").
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Claude pricing, USD per 1M tokens. VERIFY current rates before quoting cost.
PRICING = {
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
    "opus": (15.00, 75.00),
}
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheap model to grade answers


def _price_for(model: str) -> tuple[float, float]:
    for key, rates in PRICING.items():
        if key in model.lower():
            return rates
    return PRICING["sonnet"]


def _text(msg) -> str:
    """Coalesce an LLM message's content (str or block list) to plain text."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def load_dataset() -> list[dict]:
    data = json.loads((ROOT / "eval" / "dataset.json").read_text())
    return data["items"]


# ----------------------------------------------------------------------------
# 1. Retrieval metrics — deterministic, no LLM
# ----------------------------------------------------------------------------
def _is_relevant(doc: str, keywords: list[str]) -> bool:
    low = doc.lower()
    return any(kw.lower() in low for kw in keywords)


def eval_retrieval(items: list[dict]) -> dict:
    from config.settings import RERANK_CANDIDATES
    from src.retriever import retrieve
    from src.reranker import rerank

    per_q = []
    for it in items:
        kws = it["relevant_keywords"]
        candidates = retrieve(it["question"], top_k=RERANK_CANDIDATES)
        reranked, _ = rerank(it["question"], candidates)

        rel_flags = [_is_relevant(d, kws) for d in reranked]
        k = len(reranked) or 1
        precision_at_k = sum(rel_flags) / k
        hit = 1.0 if any(rel_flags) else 0.0
        recall_pool = 1.0 if any(_is_relevant(d, kws) for d in candidates) else 0.0
        mrr = 0.0
        for rank, flag in enumerate(rel_flags, start=1):
            if flag:
                mrr = 1.0 / rank
                break

        per_q.append({
            "id": it["id"],
            "precision_at_k": round(precision_at_k, 3),
            "hit_at_k": hit,
            "recall_in_pool": recall_pool,
            "mrr": round(mrr, 3),
            "n_reranked": len(reranked),
        })

    def avg(field):
        return round(statistics.mean(q[field] for q in per_q), 3)

    return {
        "aggregate": {
            "precision_at_k": avg("precision_at_k"),
            "hit_at_k": avg("hit_at_k"),
            "recall_in_pool": avg("recall_in_pool"),
            "mrr": avg("mrr"),
        },
        "per_question": per_q,
    }


# ----------------------------------------------------------------------------
# 2. End-to-end metrics — runs the real agent
# ----------------------------------------------------------------------------
def _build_agent():
    from langgraph.checkpoint.memory import MemorySaver
    from src.graph import create_graph
    return create_graph(MemorySaver())


def _run_agent(agent, question: str, thread: str) -> dict:
    config = {"configurable": {"thread_id": thread}}
    t0 = time.perf_counter()
    result = agent.invoke({"messages": [("user", question)]}, config)
    latency = time.perf_counter() - t0

    msgs = result["messages"]
    in_tok = out_tok = 0  # answer-node tokens only; router/grader calls are not in state
    sources: list[str] = list(result["docs"]) if result["supported"] else []
    answer = ""
    for m in msgs:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            in_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
        if getattr(m, "type", None) == "ai":
            txt = _text(m).strip()
            if txt:
                answer = txt
    return {
        "answer": answer,
        "sources": sources,
        "latency_s": latency,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def _judge(judge_llm, prompt: str) -> bool:
    """Return True if the judge's verdict starts with YES."""
    resp = judge_llm.invoke(prompt)
    return _text(resp).strip().upper().startswith("YES")


def eval_e2e(items: list[dict], use_judge: bool) -> dict:
    from config.settings import CLAUDE_MODEL

    agent = _build_agent()
    judge_llm = None
    if use_judge:
        from langchain_anthropic import ChatAnthropic
        from config.settings import ANTHROPIC_API_KEY
        judge_llm = ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=JUDGE_MODEL, temperature=0)

    in_rate, out_rate = _price_for(CLAUDE_MODEL)
    per_q = []
    for it in items:
        run = _run_agent(agent, it["question"], thread=f"eval-{it['id']}")
        cost = (run["input_tokens"] * in_rate + run["output_tokens"] * out_rate) / 1_000_000

        grounded = correct = None
        if use_judge and run["answer"]:
            sources_block = "\n\n".join(run["sources"]) or "(no sources retrieved)"
            grounded = _judge(judge_llm, (
                "You are grading a nutrition assistant for GROUNDEDNESS.\n"
                "Answer YES if every factual claim in the ANSWER is supported by the SOURCES, "
                "otherwise NO. Reply with YES or NO first.\n\n"
                f"SOURCES:\n{sources_block}\n\nANSWER:\n{run['answer']}"
            ))
            correct = _judge(judge_llm, (
                "You are grading a nutrition answer for CORRECTNESS against a reference.\n"
                "Answer YES if the ANSWER captures the key facts of the REFERENCE, otherwise NO. "
                "Reply with YES or NO first.\n\n"
                f"REFERENCE:\n{it['expected_answer']}\n\nANSWER:\n{run['answer']}"
            ))

        per_q.append({
            "id": it["id"],
            "latency_s": round(run["latency_s"], 3),
            "input_tokens": run["input_tokens"],
            "output_tokens": run["output_tokens"],
            "cost_usd": round(cost, 6),
            "grounded": grounded,
            "correct": correct,
        })

    n = len(per_q)
    latencies = sorted(q["latency_s"] for q in per_q)
    total_tok = sum(q["input_tokens"] + q["output_tokens"] for q in per_q)
    agg = {
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "p95_latency_s": round(latencies[int(0.95 * (n - 1))], 3),
        "avg_tokens_per_query": round(total_tok / n, 1),
        "avg_cost_per_query_usd": round(statistics.mean(q["cost_usd"] for q in per_q), 6),
        "cost_per_1k_queries_usd": round(statistics.mean(q["cost_usd"] for q in per_q) * 1000, 3),
    }
    if use_judge:
        grounded_vals = [q["grounded"] for q in per_q if q["grounded"] is not None]
        correct_vals = [q["correct"] for q in per_q if q["correct"] is not None]
        agg["groundedness_rate"] = round(sum(grounded_vals) / len(grounded_vals), 3) if grounded_vals else None
        agg["hallucination_rate"] = round(1 - agg["groundedness_rate"], 3) if grounded_vals else None
        agg["answer_correctness_rate"] = round(sum(correct_vals) / len(correct_vals), 3) if correct_vals else None

    return {"aggregate": agg, "per_question": per_q}


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _print_aggregate(title: str, agg: dict):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in agg.items():
        print(f"  {k:<28} {v}")


def cmd_run(args):
    items = load_dataset()
    from config.settings import CLAUDE_MODEL, EMBED_MODEL, RERANKER_MODEL, RERANK_TOP_K
    print(f"Dataset: {len(items)} questions")
    print(f"Embed={EMBED_MODEL} | Reranker={RERANKER_MODEL} | top_k={RERANK_TOP_K} | LLM={CLAUDE_MODEL}")

    result = {"tag": args.tag, "n_questions": len(items),
              "config": {"embed_model": EMBED_MODEL, "reranker": RERANKER_MODEL,
                         "rerank_top_k": RERANK_TOP_K, "llm": CLAUDE_MODEL}}

    print("\n[1/2] Retrieval metrics (no API cost)...")
    result["retrieval"] = eval_retrieval(items)
    _print_aggregate("RETRIEVAL", result["retrieval"]["aggregate"])

    if not args.retrieval_only:
        print("\n[2/2] End-to-end metrics (running the agent)...")
        result["e2e"] = eval_e2e(items, use_judge=not args.no_judge)
        _print_aggregate("END-TO-END", result["e2e"]["aggregate"])

    out = ROOT / "eval" / "results" / f"{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out.relative_to(ROOT)}")


def _flatten(result: dict) -> dict:
    flat = {}
    if "retrieval" in result:
        flat.update({f"retrieval.{k}": v for k, v in result["retrieval"]["aggregate"].items()})
    if "e2e" in result:
        flat.update({f"e2e.{k}": v for k, v in result["e2e"]["aggregate"].items()})
    return flat


def cmd_compare(args):
    a = json.loads(Path(args.baseline).read_text())
    b = json.loads(Path(args.candidate).read_text())
    fa, fb = _flatten(a), _flatten(b)
    print(f"\nCompare: {a.get('tag')}  ->  {b.get('tag')}\n")
    print(f"  {'metric':<34}{'before':>12}{'after':>12}{'delta':>12}{'change':>10}")
    print("  " + "-" * 78)
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            pct = (delta / va * 100) if va else float("nan")
            print(f"  {key:<34}{va:>12.3f}{vb:>12.3f}{delta:>+12.3f}{pct:>+9.1f}%")
        else:
            print(f"  {key:<34}{str(va):>12}{str(vb):>12}")


def main():
    p = argparse.ArgumentParser(description="NutriBot eval harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the eval and save a snapshot")
    r.add_argument("--tag", required=True, help="label for this run (used as filename)")
    r.add_argument("--retrieval-only", action="store_true", help="skip agent/LLM (no API cost)")
    r.add_argument("--no-judge", action="store_true", help="run agent but skip LLM-judge scoring")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="print before->after deltas between two snapshots")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
