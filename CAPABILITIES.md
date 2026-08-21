# NutriBot — Capabilities (as of 2026-08-20)

## Deployment
- **URL:** http://44.194.89.198
- **Host:** AWS EC2 (Docker container, port 80)
- **Status:** Live and healthy

---

## What NutriBot Can Do

### 1. Nutrition Q&A
Answers questions about foods, nutrients, vitamins, minerals, diet, and weight management by retrieving from a curated knowledge base. Never answers from model memory alone.

Examples:
- "How does vitamin C help with iron absorption?"
- "What foods are high in protein?"
- "What is the daily iron requirement for women?"
- "Is intermittent fasting effective?"

### 2. Macro Calculator
Computes protein, carbs, and fat gram targets from a daily calorie goal with custom percentage splits.

Example: "I eat 2000 calories a day — how should I split my macros?"

### 3. Streaming Responses
Responses stream token-by-token via SSE so users see output immediately rather than waiting for the full answer.

### 4. Source Citations
Retrieved knowledge base passages are shown as collapsible sources below each answer, so users can verify where the information came from.

### 5. Conversation Memory
Each browser session maintains its own conversation thread — NutriBot remembers earlier messages within a session to give context-aware follow-up answers.

### 6. Suggested Questions
The UI offers one-click starter questions to guide new users.

---

## Retrieval Pipeline

| Stage | Detail |
|---|---|
| Embedder | `BAAI/bge-base-en-v1.5` (768-dim) |
| Vector store | Pinecone — `nutrition-index` (1,345 vectors) |
| Retrieval mode | Semantic only |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq — `llama-3.3-70b-versatile` |

**Two-stage retrieval:** Pinecone returns the top candidates by vector similarity, the cross-encoder reranks them by relevance, and only the top results are passed to the LLM. Answers below the relevance threshold are refused rather than hallucinated.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns status, uptime, retriever type, embedder name |
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Sync Q&A |
| `POST` | `/chat/stream` | Token-streaming Q&A (SSE) |

---

## Current Limitations

- Knowledge base covers general nutrition topics only — no personalised medical advice
- Hybrid retrieval (semantic + keyword) is currently disabled due to missing S3 data
- No user authentication — all sessions are anonymous
- Answers are scoped to the knowledge base; very niche queries may return a refusal
