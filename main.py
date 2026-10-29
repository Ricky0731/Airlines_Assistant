"""
main.py — Entry point for the Advanced RAG Q&A application.

Usage:
    python main.py

First run  : builds the ChromaDB index automatically from data/.
Later runs : loads the existing index (fast, no re-embedding).

To add new PDFs:
    1. Drop PDF files into data/
    2. python ingestion/ingest.py --rebuild
    3. python main.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from config.settings import CHROMA_DIR, BM25_WEIGHT, SEMANTIC_WEIGHT, CANDIDATE_K, FINAL_TOP_K
from engine.rag_engine import build_pipeline, load_pipeline


def main() -> None:
    # ── API key ───────────────────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        sys.exit("❌  GROQ_API_KEY not set. Add it to .env as: GROQ_API_KEY=gsk_...")

    # ── Build or load the pipeline ────────────────────────────────────────────
    chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    if chroma_exists:
        print(f"\nFound existing index at '{CHROMA_DIR}'. Loading …\n")
        rag = load_pipeline(GROQ_API_KEY)
    else:
        print("\nNo existing index found. Building from scratch …\n")
        rag = build_pipeline(GROQ_API_KEY)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 58)
    print("  Advanced RAG — Interactive Q&A")
    print(f"  Retrieval : BM25 {BM25_WEIGHT:.0%} + Semantic {SEMANTIC_WEIGHT:.0%}")
    print(f"  Candidates: {CANDIDATE_K} per leg  →  top {FINAL_TOP_K} after reranking")
    print("  Modes     : FACTUAL_QA / DOCUMENT_SPECIFIC / DOCUMENT_SUMMARY")
    print("  Type 'quit' to exit.")
    print("=" * 58)

    # ── Q&A loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        result = rag.query(question)

        print("\n── Answer ──────────────────────────────────────────────")
        print(result["answer"])

        print(
            f"\n── Mode: {result['mode']}  "
            f"| Rewritten: \"{result['rewritten_query']}\""
        )

        if result["sources"]:
            print("\n── Sources ─────────────────────────────────────────────")
            print("\n".join(result["sources"]))
        else:
            print("\n── Sources: none")


if __name__ == "__main__":
    main()
