"""config/settings.py — Single source of truth for all constants."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_collection"

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "openai/gpt-oss-120b"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120

# ── Retrieval ──────────────────────────────────────────────────────────────────
BM25_WEIGHT     = 0.4    # keyword / sparse leg weight
SEMANTIC_WEIGHT = 0.6    # dense / semantic leg weight
CANDIDATE_K     = 10     # candidates fetched per retrieval leg
FINAL_TOP_K     = 5      # chunks kept after LLM reranking

# ── Document-type keyword map ──────────────────────────────────────────────────
DOC_TYPE_MAP = {
    "aircraft_specs": ["airbus", "boeing", "a350", "a380", "aircraft", "plane"],
    "policy":         ["allowance", "discount", "program", "armed forces", "policy"],
    "process":        ["boarding", "seat", "change", "process", "procedure"],
    "awards":         ["award", "recognition"],
}

