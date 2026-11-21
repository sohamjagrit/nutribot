# Eval history

One row per evaluated version, newest last. Raw snapshots live in `eval/results/<tag>.json`.
Add a row after every `uv run python eval/run_eval.py run --tag <tag>`; compare with
`uv run python eval/run_eval.py compare eval/results/<old>.json eval/results/<new>.json`.

Dataset: `eval/dataset.json`, 15 questions (one question = 6.7%, so judge metrics are noisy).
Retrieval config for all runs: BGE `bge-base-en-v1.5`, Pinecone top-15, reranker `ms-marco-MiniLM-L-6-v2`, top-5.

## Retrieval

| Tag | Date | Commit | precision@5 | hit@5 | recall (pool) | MRR |
|---|---|---|---|---|---|---|
| baseline | 2026-11 | 93771a6 (old ReAct) | 0.893 | 1.000 | 1.000 | 0.967 |
| multiagent | 2026-11-21 | 37e0d9d | 0.893 | 1.000 | 1.000 | 0.967 |

## End-to-end (LLM agent + judge)

| Tag | Correctness | Groundedness | Hallucination | Avg latency | p95 latency | Tokens/query | $/query |
|---|---|---|---|---|---|---|---|
| baseline | 80% | 13.3% | 86.7% | 16.8 s | 21.5 s | 4,839 | $0.0225 |
| multiagent | 80% | 66.7% | 33.3% | 10.4 s | 13.0 s | 2,526 | $0.0122 |

## Notes per version

**baseline** (old ReAct agent, what EC2 runs today). Groundedness 13% is probably a strict judge, not
real hallucination; verify by hand.

**multiagent** (router -> retrieve -> grade -> answer/refuse). Retrieval is identical, since the
retriever and reranker did not change. Cost -46%, latency -38%, groundedness +53 pts, correctness
unchanged. Groundedness gain not yet hand-verified. A LangSmith 403 appeared during the run
(invalid key); it did not affect the numbers.

## Pending

Add a row when each of these lands: calorie agent (step 3), context summarization (step 4).
Grow the dataset to 30-50 questions before treating small judge deltas as real.
