import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "nutrition-index")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
FAST_MODEL: str = os.getenv("FAST_MODEL", "claude-haiku-4-5-20251001")  # router, grader, rewriter
MAX_RETRIEVAL_RETRIES: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "1"))


EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES: int = int(os.getenv("RERANK_CANDIDATES", "15"))  # fetched from Pinecone
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))             # passed to LLM after rerank
MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "-4.0"))  # ms-marco logit threshold

# Context management: when history exceeds the token budget, older messages are folded into a summary.
CONTEXT_TOKEN_BUDGET: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "6000"))
KEEP_LAST_MESSAGES: int = int(os.getenv("KEEP_LAST_MESSAGES", "6"))

SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/memory.db")

# OpenNutrition MCP server (food lookup). Private Docker network only; no auth.
MCP_NUTRITION_URL: str = os.getenv("MCP_NUTRITION_URL", "http://localhost:3000/mcp")

# Run metadata for traces
ENV: str = os.getenv("ENV", "dev")
GIT_SHA: str = os.getenv("GIT_SHA", "dev")
