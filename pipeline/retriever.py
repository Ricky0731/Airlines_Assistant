"""
pipeline/retriever.py — Hybrid retriever: BM25 (keyword) + ChromaDB (semantic).

EnsembleRetriever fuses both legs using Reciprocal Rank Fusion (RRF).
Optional source_filter restricts both legs to one PDF file.
"""

import warnings
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever           # ✅ verified
from config.settings import BM25_WEIGHT, SEMANTIC_WEIGHT, CANDIDATE_K


def build_hybrid_retriever(
    chunks: list[Document],
    vectorstore: Chroma,
    k: int = CANDIDATE_K,
    source_filter: Optional[str] = None,
) -> EnsembleRetriever:
    """
    Build an EnsembleRetriever combining:
        BM25      – sparse keyword matching   (weight = BM25_WEIGHT)
        Chroma    – dense semantic similarity  (weight = SEMANTIC_WEIGHT)

    Args:
        chunks        : all document chunks (used to build the in-memory BM25 index)
        vectorstore   : ChromaDB instance
        k             : candidates to fetch from each leg
        source_filter : restrict both legs to this PDF filename (or None for all docs)
    """
    # ── BM25 leg ──────────────────────────────────────────────────────────────
    bm25_corpus = (
        [c for c in chunks if c.metadata.get("source_file") == source_filter]
        if source_filter else chunks
    ) or chunks  # fallback to all if filter matched nothing

    bm25 = BM25Retriever.from_documents(bm25_corpus, k=k)

    # ── Semantic leg ──────────────────────────────────────────────────────────
    search_kwargs: dict = {"k": k}
    if source_filter:
        search_kwargs["filter"] = {"source_file": source_filter}

    semantic = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    # ── Ensemble (RRF fusion) ─────────────────────────────────────────────────
    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )

