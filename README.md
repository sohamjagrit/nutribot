# NutriBot

A RAG-powered nutrition Q&A chatbot built with LangGraph, Claude, and Pinecone. Answers questions about foods, nutrients, vitamins, and diet by retrieving from a curated knowledge base — never from model memory alone.

Deployed on **AWS EC2** and accessible via a streaming chat UI.

---

## Architecture

```
User → FastAPI → LangGraph ReAct Agent (Claude)
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
   retrieve_nutrition_info    calculate_macros
              │
   Pinecone (vector search)
      top-15 candidates
              │
   Cross-encoder reranker
      top-5 by relevance
              │
   Relevance gate → LLM context
```

**Two-stage retrieval** separates recall (Pinecone ANN search) from precision (cross-encoder reranking), keeping the LLM context tight and relevant.

**ReAct agent loop** means the model can call tools multiple times per turn and chain reasoning across them before responding.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude (Anthropic) via `langchain-anthropic` |
| Agent framework | LangGraph `create_react_agent` |
| Vector store | Pinecone |
| Embeddings | `BAAI/bge-base-en-v1.5` (SentenceTransformer) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Conversation memory | LangGraph `SqliteSaver` (persistent, per thread) |
| Backend | FastAPI + uvicorn |
| Observability | LangSmith |
| Deployment | Docker on AWS EC2 |

---

## Project Structure

```
nutribot/
├── app.py                  # FastAPI app — /chat and /chat/stream endpoints
├── config/settings.py      # Env-var config
├── src/
│   ├── graph.py            # LangGraph ReAct agent definition
│   ├── retriever.py        # Pinecone query + BGE embedding
│   ├── reranker.py         # Cross-encoder reranking + relevance gate
│   └── tools.py            # LangChain tools exposed to the agent
├── scripts/
│   └── ingest.py           # Chunk, embed, and upsert docs to Pinecone
├── data/raw/               # Source documents (articles + USDA CSV)
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
docker build -t nutribot .
docker run -p 8000:8000 --env-file .env nutribot
```

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
