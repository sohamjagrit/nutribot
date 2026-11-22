# NutriBot

A RAG-powered nutrition Q&A chatbot built with LangGraph, Claude, and Pinecone. Answers questions about foods, nutrients, vitamins, and diet by retrieving from a curated knowledge base — never from model memory alone.

Deployed on **AWS EC2** and accessible via a streaming chat UI.

---

## Architecture

```
User → FastAPI → LangGraph pipeline
                      │
                   router (Haiku): classify + standalone query
        ┌─────────────┼───────────────┐
    nutrition       calories        off_topic
        │             │               │
    retrieve       ReAct agent        refuse
 Pinecone top-15   (Sonnet) ──→ END
 + cross-encoder
 rerank top-5
        │
     grade (Haiku)
     ┌───┴────┐
 supported  unsupported
     │        │ rewrite → retrieve (once) → refuse
     └────────┴──→ answer (Sonnet) → END
```

A fast model routes, grades and rewrites; the main model writes the final answer, strictly from retrieved context, with inline `[Source n]` citations.

The `calories` route is a ReAct agent for calorie targets, macro grams and the macros of specific meals. It decides when to use the OpenNutrition food lookup (MCP), the calculators (`calorie_plan`, `calculate_macros`, `meal_macros`; all math is deterministic Python) and `ask_user`. When details are missing it pauses via a LangGraph `interrupt`: `/chat/stream` emits `{"type":"interrupt","fields":[...]}` and the UI shows a form (kg/cm or lb/in). The client resumes by POSTing `{"resume": {...}, "thread_id": ...}`. A plan that hits the calorie floor (1500 male / 1200 female) asks for confirmation (`{"type":"interrupt","confirm":"..."}`, resume with `true`/`false`).

**Two-stage retrieval** separates recall (Pinecone ANN search) from precision (cross-encoder reranking with a relevance gate).

**Self-correcting retrieval:** if the grader decides the context can't answer the question, the query is rewritten and retried before the bot refuses.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLMs | Claude Sonnet (answers), Claude Haiku (router/grader/rewriter) via `langchain-anthropic` |
| Orchestration | LangGraph `StateGraph` |
| Vector store | Pinecone |
| Embeddings | `BAAI/bge-base-en-v1.5` (SentenceTransformer) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Conversation memory | LangGraph `AsyncSqliteSaver` (persistent, per thread) |
| Backend | FastAPI + uvicorn |
| Observability | LangSmith (optional) |
| Deployment | Docker on AWS EC2 |

---

## Project Structure

```
nutribot/
├── app.py                  # FastAPI app: /chat and /chat/stream endpoints
├── config/settings.py      # Env-var config
├── src/
│   ├── graph.py            # LangGraph pipeline (router → retrieve → grade → answer | calories agent)
│   ├── retriever.py        # Pinecone query + BGE embedding
│   ├── reranker.py         # Cross-encoder reranking
│   └── tools.py            # Calorie/macro calculators + ask_user (interrupt)
├── scripts/ingest.py       # Chunk, embed, and upsert docs to Pinecone
├── eval/                   # Golden set + eval harness (retrieval and end-to-end metrics)
├── data/raw/               # Source documents (not committed)
└── static/index.html       # Chat UI (streaming, markdown, source citations)
```

---

## Setup

### 1. Install dependencies

```bash
pip install uv
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in PINECONE_API_KEY, ANTHROPIC_API_KEY, and optionally LANGCHAIN_API_KEY
```

### 3. Ingest documents into Pinecone

```bash
uv run python scripts/ingest.py
```

### 4. Run the app

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

### Docker

```bash
cp .env.example .env   # fill in keys
docker compose up --build
```

Compose runs NutriBot plus the OpenNutrition MCP food-lookup server on a private network
(the MCP port is not published; it has no auth). NutriBot still starts if it is down,
with food lookup disabled.

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Sync response |
| `POST` | `/chat/stream` | SSE token streaming |

**Request body** (`/chat` and `/chat/stream`):
```json
{
  "question": "What foods are high in vitamin C?",
  "user_context": "",
  "thread_id": "default"
}
```

`thread_id` scopes conversation memory — each browser session gets its own thread.

---

## Evaluation

```bash
uv run python eval/run_eval.py run --tag my_run                      # full run
uv run python eval/run_eval.py run --tag my_run --retrieval-only     # free, no LLM calls
uv run python eval/run_eval.py compare eval/results/baseline.json eval/results/my_run.json
```

---

## Deploy on AWS EC2

```bash
# On the instance (Docker + compose plugin installed, security group open on port 80)
git clone https://github.com/sohamjagrit/NutriBot && cd NutriBot
cp .env.example .env   # fill in PINECONE_API_KEY and ANTHROPIC_API_KEY
docker compose up -d --build
docker compose logs nutribot | grep "MCP tools"   # confirm food tools loaded
```

The `nutribot-data` volume keeps SQLite conversation memory across container restarts.

---

## Data attribution

Food data: [OpenNutrition](https://www.opennutrition.app) (ODbL / DbCL), served by
[mcp-opennutrition](https://github.com/deadletterq/mcp-opennutrition) (MIT).
