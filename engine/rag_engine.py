"""
engine/rag_engine.py — AdvancedRAG orchestrator + pipeline factory functions.

AdvancedRAG ties together every stage:
    route → retrieve (hybrid) → rerank → grade → answer

Factories:
    build_pipeline()  – full build from scratch
    load_pipeline()   – load existing ChromaDB (skips re-embedding)
"""

from pathlib import Path

from langchain_groq import ChatGroq

from config.settings import (
    CHROMA_DIR, CANDIDATE_K, FINAL_TOP_K,
    BM25_WEIGHT, SEMANTIC_WEIGHT, GROQ_MODEL,
)
from ingestion.loader import load_pdfs
from ingestion.chunker import chunk_documents
from storage.vectorstore import get_embedding_fn, build_vectorstore, load_vectorstore
from pipeline.router import route_query
from pipeline.retriever import build_hybrid_retriever
from pipeline.reranker import rerank_chunks
from pipeline.grader import grade_chunks
from pipeline.summarizer import summarize_all_documents
from pipeline.answerer import answer_question


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm(api_key: str) -> ChatGroq:
    llm = ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0.2)
    print(f"✔ LLM ready: {GROQ_MODEL} via Groq.\n")
    return llm


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedRAG:
    """
    Orchestrates the full Advanced RAG pipeline:
        route → retrieve (hybrid) → rerank → grade → answer
    """

    def __init__(self, chunks, vectorstore, llm: ChatGroq):
        self.chunks         = chunks
        self.vectorstore    = vectorstore
        self.llm            = llm
        self.available_docs = sorted(
            {c.metadata.get("source_file", "") for c in chunks}
        )

    def query(self, question: str) -> dict:
        """
        Run one question through the entire pipeline.

        Returns:
            answer          – LLM response string
            mode            – routing mode used
            rewritten_query – clarified query
            sources         – de-duplicated list of source strings
        """
        divider = "─" * 55
        print(f"\n{divider}")
        print(f"  Q: {question}")
        print(divider)

        # ── [1] Route & rewrite ───────────────────────────────────────────────
        print("  [1/5] Routing query …")
        route           = route_query(question, self.llm, self.available_docs)
        mode            = route["mode"]
        rewritten_query = route["rewritten_query"]
        doc_filter      = route.get("document_name")

        print(f"        Mode    : {mode}")
        print(f"        Rewrite : {rewritten_query}")
        if doc_filter:
            print(f"        Filter  : {doc_filter}")

        # ── [2a] DOCUMENT_SUMMARY path ────────────────────────────────────────
        if mode == "DOCUMENT_SUMMARY":
            print("  [2/5] Summarizing documents (map-reduce) …")
            target = (
                [c for c in self.chunks if c.metadata.get("source_file") == doc_filter]
                if doc_filter else self.chunks
            ) or self.chunks
            answer = summarize_all_documents(rewritten_query, target, self.llm)
            return {
                "answer": answer, "mode": mode,
                "rewritten_query": rewritten_query, "sources": [],
            }

        # ── [2b] Hybrid retrieval ─────────────────────────────────────────────
        print("  [2/5] Hybrid retrieval (BM25 + Semantic, k=10 each) …")
        retriever  = build_hybrid_retriever(
            self.chunks, self.vectorstore, k=CANDIDATE_K,
            source_filter=doc_filter if mode == "DOCUMENT_SPECIFIC" else None,
        )
        candidates = retriever.invoke(rewritten_query)
        print(f"        {len(candidates)} candidates retrieved.")

        # ── [3] Rerank ────────────────────────────────────────────────────────
        print("  [3/5] Reranking …")
        reranked = rerank_chunks(rewritten_query, candidates, self.llm, top_k=FINAL_TOP_K)
        print(f"        {len(reranked)} chunks kept after reranking.")

        # ── [4] Relevance grading ─────────────────────────────────────────────
        print("  [4/5] Grading relevance …")
        relevant = grade_chunks(rewritten_query, reranked, self.llm)
        print(f"        {len(relevant)} chunks passed relevance check.")

        if not relevant:
            return {
                "answer": "I don't have enough information in the provided documents.",
                "mode": mode, "rewritten_query": rewritten_query, "sources": [],
            }

        # ── [5] Generate answer ───────────────────────────────────────────────
        print("  [5/5] Generating answer …")
        answer = answer_question(rewritten_query, relevant, self.llm)

        # Build de-duplicated source list
        seen, sources = set(), []
        for doc in relevant:
            m   = doc.metadata
            key = (m.get("source_file", "?"), m.get("page_number", "?"))
            if key not in seen:
                seen.add(key)
                sources.append(
                    f"  • {Path(key[0]).stem}  [page {key[1]}]"
                    f"  ({m.get('chunk_type', 'text')})"
                )

        return {
            "answer": answer, "mode": mode,
            "rewritten_query": rewritten_query, "sources": sources,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FACTORIES
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline(groq_api_key: str) -> AdvancedRAG:
    """
    Full build from scratch:
        load PDFs → chunk → embed → persist ChromaDB → init AdvancedRAG
    """
    sep = "=" * 58
    print(sep); print("STEP 1 — Loading PDFs"); print(sep)
    docs = load_pdfs()

    print(sep); print("STEP 2 — Chunking + metadata enrichment"); print(sep)
    chunks = chunk_documents(docs)

    print(sep); print("STEP 3 — Sentence-transformer embeddings"); print(sep)
    embedding_fn = get_embedding_fn()

    print(sep); print("STEP 4 — ChromaDB vector store"); print(sep)
    vectorstore = build_vectorstore(chunks, embedding_fn)

    print(sep); print("STEP 5 — Groq LLM"); print(sep)
    llm = _get_llm(groq_api_key)

    print("✅ Advanced RAG pipeline ready!\n")
    return AdvancedRAG(chunks=chunks, vectorstore=vectorstore, llm=llm)


def load_pipeline(groq_api_key: str) -> AdvancedRAG:
    """
    Load from an existing ChromaDB index (skips re-embedding).
    PDFs are re-parsed to rebuild the in-memory BM25 index.
    """
    print("  Re-parsing PDFs for BM25 (no re-embedding) …")
    docs   = load_pdfs()
    chunks = chunk_documents(docs)

    embedding_fn = get_embedding_fn()
    vectorstore  = load_vectorstore(embedding_fn)
    llm          = _get_llm(groq_api_key)

    print("✅ Pipeline loaded from existing index.\n")
    return AdvancedRAG(chunks=chunks, vectorstore=vectorstore, llm=llm)

