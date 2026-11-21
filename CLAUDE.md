# NutriBot

RAG nutrition chatbot. FastAPI + LangGraph + Pinecone + Claude. Deployed with Docker on AWS EC2.
Keep the codebase minimal: no new layers or abstractions unless a task needs them.
Plan work here, then wait for an explicit "go" before building.
Detailed tech spec (APIs, schemas, formulas, CI gates, deploy): `docs/TECH_SPEC.md`.

## Current state

```
app.py               FastAPI: /, /health, /chat, /chat/stream (SSE). AsyncSqliteSaver via lifespan.
config/settings.py   Env-var config (models, retrieval thresholds, SQLITE_DB_PATH).
src/graph.py         StateGraph: router -> retrieve -> grade -> (rewrite -> retrieve) -> answer | refuse
src/retriever.py     Pinecone query + BGE embeddings (lazy-loaded). Only node with @traceable.
src/reranker.py      Cross-encoder rerank, relevance gate (MIN_RELEVANCE_SCORE).
src/tools.py         calculate_macros (router calls it directly; no agent loop).
scripts/ingest.py    Embed data/raw/*.txt -> Pinecone (index currently ~1,345 vectors).
eval/                run_eval.py + dataset.json (15 Qs) + results/baseline*.json
static/index.html    Chat UI.
```

- Models: Haiku (`FAST_MODEL`) = router/grader/rewriter; Sonnet (`CLAUDE_MODEL`) = answer.
- Graph is a fixed workflow, not a tool-calling agent. Memory = SQLite checkpointer, unbounded history.
- Baseline (old ReAct agent): precision@5 0.893, MRR 0.967, 16.8s avg, $0.0225/query,
  correctness 80%, groundedness 13% (judge may be too strict; verify by hand).
- The multi-agent graph has NOT been evaluated or deployed. EC2 still runs the old version.
- No tests, no CI (`.github/workflows/` is empty), no MCP, no human-in-the-loop, no prod monitoring.
- `data/` is gitignored and NOT copied into the Docker image; food data will come from the MCP container.
- Never commit `.env`, `data/` or keys. Use `uv` for everything; Docker uses `uv sync --frozen`.

## Target (what each item must make true)

Run `eval --tag multiagent` and redeploy the current version first (step 0). Then, in order:

1. **Calculators + tests** (`src/tools.py`, `tests/`): pure `bmr` (Mifflin-St Jeor), `tdee` (activity
   1.2/1.375/1.55/1.725/1.9), `goal_calories` (lose -500, gain +300, floor 1500 male / 1200 female),
   `calculate_macros`. All math deterministic Python; unit-tested; no API calls.
2. **External MCP server** (https://github.com/deadletterq/mcp-opennutrition, MIT, TypeScript, 300k+ foods,
   local SQLite, no external API calls). Tools: `search-food-by-name`, `get-foods`, `get-food-by-id`,
   `get-food-by-ean13`. It is food lookup only; BMR/TDEE/macros stay as local calculators (step 1).
   - Run it as a second container (`deadletterq/mcp-opennutrition`, streamable-http) on a private Docker
     network with NutriBot; do NOT publish its port to the host (no auth). Pin the image by digest.
   - The README maps `9113:3000` but the source defaults to port 9113 unless `PORT` is set: set `PORT=3000`
     explicitly and verify before relying on it.
   - Connect with `langchain-mcp-adapters` (`MultiServerMCPClient`, transport `streamable_http`, URL from
     `MCP_NUTRITION_URL`). Load tools once in the FastAPI lifespan.
   - Its tool descriptions enter our agent's context: treat them as third-party text, keep our system prompt
     authoritative, and don't let it override refusal/safety rules.
   - Data is ODbL/DbCL: show "Food data: OpenNutrition (ODbL)" attribution in the UI and README.
   - Add a `docker-compose.yml` (nutribot + opennutrition + shared volume for memory) and update the EC2 steps.
3. **Calorie agent + human-in-the-loop**: a `calories` route to a `create_react_agent` node using the MCP
   food tools, the local calculators, and an `ask_user(fields)` tool that calls LangGraph `interrupt()`.
   `/chat/stream` emits an
   `interrupt` SSE event with the missing fields; the UI shows a small form; the client resumes with
   `Command(resume=answers)`. Also interrupt to confirm any plan that hit the calorie floor.
   Router stops calling `calculate_macros` itself.
4. **Context management**: add `summary` to State. A `summarize` node (Haiku) runs when messages exceed a
   token budget (`CONTEXT_TOKEN_BUDGET`): fold older messages into `summary`, delete them with
   `RemoveMessage`, keep the last N. The answer prompt includes the summary.
5. **LangSmith**: set tracing env on EC2; tag runs with `thread_id`, env, git sha, model; trace reranker
   and MCP calls; push `eval/dataset.json` to a LangSmith dataset; capture `run_id` so UI thumbs call
   `create_feedback`; online groundedness evaluator on sampled traces.
6. **Production monitoring**: one JSON log line per request (request_id, route, per-node latency, tokens,
   cost, grader result, refusal, interrupt) via a usage callback; Docker `awslogs` -> CloudWatch;
   alarms on 5xx, p95 latency, daily cost, refusal rate. No Prometheus/Grafana until needed.
7. **CI/CD** (`.github/workflows/`): PR = ruff, pytest, docker build, retrieval-only eval gated against
   the committed baseline. LLM e2e eval only when graph/prompts/models change, and nightly. Merge to
   main = build -> ECR -> deploy to EC2 -> `/health` check -> rollback on failure.
8. **Later, separate tasks**: more RAG data (then update the chunk count honestly), web-search fallback.

### Eval and regression rules
- Eval stays offline and out of the container (`eval/` is in `.dockerignore`).
- Gate deterministic checks exactly (calorie numbers, floor, off-topic refusal, missing-input interrupt).
- Gate judge metrics with a tolerance; 15 Qs is noisy (one question = 6.7%), so grow the set to 30-50.
- Baseline updates are deliberate PRs, never automatic.

### Done when (per step)
Tests pass, eval shows no regression on existing items, new behavior has an eval or test, and a manual
chat covers the happy path and the missing-input path. Update README and `.env.example` as you go.
