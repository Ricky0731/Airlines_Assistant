"""
ingestion/ingest.py — Standalone ingestion script.

Run this whenever you add new PDFs to the data/ folder:
    python ingestion/ingest.py

Options:
    --rebuild   wipe the existing ChromaDB index and rebuild from scratch
"""

import sys
import shutil
from pathlib import Path

# Allow running directly: python ingestion/ingest.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from config.settings import CHROMA_DIR
from ingestion.loader import load_pdfs
from ingestion.chunker import chunk_documents
from storage.vectorstore import get_embedding_fn, build_vectorstore


def run_ingestion(force_rebuild: bool = False) -> None:
    sep = "=" * 58

    if force_rebuild and Path(CHROMA_DIR).exists():
        print(f"  Wiping existing index at '{CHROMA_DIR}' …")
        shutil.rmtree(CHROMA_DIR)

    print(sep)
    print("  INGESTION PIPELINE")
    print(sep)

    print("\nSTEP 1 — Loading PDFs from data/")
    print("-" * 40)
    docs = load_pdfs()

    print("STEP 2 — Chunking + metadata enrichment")
    print("-" * 40)
    chunks = chunk_documents(docs)

    print("STEP 3 — Loading sentence-transformer embeddings")
    print("-" * 40)
    embedding_fn = get_embedding_fn()

    print("STEP 4 — Building ChromaDB vector store")
    print("-" * 40)
    build_vectorstore(chunks, embedding_fn)

    print(sep)
    print(f"✅ Ingestion complete!  {len(chunks)} chunks stored in '{CHROMA_DIR}'.")
    print("   Run  python main.py  to start the Q&A interface.")
    print(sep)


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    if force:
        print("⚠  --rebuild: existing index will be wiped.\n")
    run_ingestion(force_rebuild=force)

