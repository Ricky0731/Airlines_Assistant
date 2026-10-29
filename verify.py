"""
Advanced RAG Pipeline
=====================
User Question
  → [1] Query Router (classify + rewrite)
  → [2] Metadata-filtered Hybrid Retrieval (BM25 + Semantic, k=10 each)
  → [3] LLM Reranker (keep top 3-5)
  → [4] Relevance Grader (drop irrelevant chunks)
  → [5] Groq LLM Answer with [Document, page X] citations

Modes:
  FACTUAL_QA       – hybrid retrieval → rerank → grade → answer
  DOCUMENT_SPECIFIC – same but filtered to one source document
  DOCUMENT_SUMMARY  – map-reduce summarization across all / one document
"""

import os
import re
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── SUPPRESS langchain-community deprecation (PyMuPDF / BM25 have no standalone pkg yet)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.retrievers import BM25Retriever

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR        = Path("data")
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL      = "openai/gpt-oss-120b"

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 120

BM25_WEIGHT     = 0.4
SEMANTIC_WEIGHT = 0.6
CANDIDATE_K     = 10   # chunks retrieved by each retrieval leg
FINAL_TOP_K     = 5    # chunks kept after reranking

# ══════════════════════════════════════════════════════════════════════════════
# METADATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_DOC_TYPE_MAP = {
    "aircraft_specs": ["airbus", "boeing", "a350", "a380", "aircraft", "plane"],
    "policy":         ["allowance", "discount", "program", "armed forces", "policy"],
    "process":        ["boarding", "seat", "change", "process", "procedure"],
    "awards":         ["award", "recognition"],
}

def _infer_document_type(filename: str) -> str:
    low = filename.lower()
    for doc_type, keywords in _DOC_TYPE_MAP.items():
        if any(kw in low for kw in keywords):
            return doc_type
    return "general"

def _extract_section_path(text: str) -> str:
    """Pull the first 1-3 Markdown headings as a breadcrumb path."""
    headings = re.findall(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)
    return " > ".join(h.strip() for h in headings[:3])

def _detect_chunk_type(text: str) -> str:
    """Heuristically classify a chunk as table / header / text."""
    lines = text.strip().splitlines()
    if not lines:
        return "text"
    pipe_ratio = sum(1 for ln in lines if "|" in ln) / len(lines)
    if pipe_ratio > 0.4:
        return "table"
    if re.match(r"^#{1,4}\s", lines[0]):
        return "header"
    return "text"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load every PDF in data_dir with PyMuPDFLoader.
    Attaches: source_file, document_title, document_type per page.
    """
    docs: list[Document] = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{data_dir}'.")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source_file"]    = pdf_path.name
            page.metadata["document_title"] = pdf_path.stem
            page.metadata["document_type"]  = _infer_document_type(pdf_path.name)
        docs.extend(pages)

    print(f"\n✔ Loaded {len(docs)} pages from {len(pdf_files)} PDFs.\n")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 2. CHUNK
# ══════════════════════════════════════════════════════════════════════════════
def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Two-stage splitting + metadata enrichment per chunk:
      chunk_id, chunk_length, page_number, section_path, chunk_type
    """
    # Stage 1 – Markdown structure aware
    md_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    md_chunks = md_splitter.split_documents(docs)
    print(f"  After MarkdownTextSplitter          : {len(md_chunks)} chunks")

    # Stage 2 – Recursive character fallback
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = rc_splitter.split_documents(md_chunks)
    print(f"  After RecursiveCharacterTextSplitter: {len(chunks)} chunks\n")

    # Enrich every chunk with derived metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]     = idx
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["section_path"] = _extract_section_path(chunk.page_content)
        chunk.metadata["chunk_type"]   = _detect_chunk_type(chunk.page_content)
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
def build_vectorstore(
    chunks: list[Document],
    embedding_fn: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """Embed all chunks and persist to a local ChromaDB collection."""
    print(f"  Embedding {len(chunks)} chunks → ChromaDB at '{persist_dir}' …")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    print(f"✔ Vector store ready  (collection: '{COLLECTION_NAME}').\n")
    return vs


def _load_vectorstore(
    embedding_fn: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════
def build_hybrid_retriever(
    chunks: list[Document],
    vectorstore: Chroma,
    k: int = CANDIDATE_K,
    source_filter: Optional[str] = None,
) -> EnsembleRetriever:
    """
    EnsembleRetriever = BM25 (sparse/keyword) + Chroma (dense/semantic).
    If source_filter is set, both legs restrict results to that source file.
    Fusion via Reciprocal Rank Fusion (RRF).
    """
    # ── BM25 leg ──────────────────────────────────────────────────────────────
    bm25_corpus = (
        [c for c in chunks if c.metadata.get("source_file") == source_filter]
        if source_filter else chunks
    ) or chunks   # fallback: use all if filter matched nothing

    bm25 = BM25Retriever.from_documents(bm25_corpus, k=k)

    # ── Semantic leg ──────────────────────────────────────────────────────────
    search_kwargs: dict = {"k": k}
    if source_filter:
        search_kwargs["filter"] = {"source_file": source_filter}

    semantic = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    return EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. QUERY ROUTER
# ══════════════════════════════════════════════════════════════════════════════
_ROUTER_SYS = """You are a query routing assistant for a document Q&A system.
Classify the user question and return a JSON object with these exact fields:

{{
  "mode": "<FACTUAL_QA | DOCUMENT_SPECIFIC | DOCUMENT_SUMMARY>",
  "rewritten_query": "<clearer, more specific version of the question>",
  "document_name": "<exact PDF filename if question targets one document, else null>",
  "document_type": "<policy | aircraft_specs | process | awards | null>"
}}

Mode rules:
  FACTUAL_QA       – general question that may span multiple documents
  DOCUMENT_SPECIFIC – question is clearly about ONE specific document/topic
  DOCUMENT_SUMMARY  – user wants a summary or overview

Available PDF files: {doc_list}

Return ONLY valid JSON. No markdown fences, no extra text."""


def route_query(
    question: str,
    llm: ChatGroq,
    available_docs: list[str],
) -> dict:
    """
    Classify and rewrite the user query using the Groq LLM.
    Returns: {mode, rewritten_query, document_name, document_type}
    """
    prompt = _ROUTER_SYS.format(doc_list=", ".join(available_docs))
    try:
        resp = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        text = resp.content.strip()
        # Strip any accidental markdown code fences
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            text = m.group(1).strip() if m else text
        result = json.loads(text)
    except Exception as exc:
        print(f"  [Router] Fallback to FACTUAL_QA  ({exc})")
        result = {}

    return {
        "mode":            result.get("mode", "FACTUAL_QA"),
        "rewritten_query": result.get("rewritten_query", question),
        "document_name":   result.get("document_name"),
        "document_type":   result.get("document_type"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. RERANKER
# ══════════════════════════════════════════════════════════════════════════════
_RERANK_SYS = """You are a document relevance ranker.
Given a question and numbered document chunks, return a JSON array of chunk indices
(0-based integers) ordered from MOST to LEAST relevant to the question.
Include only the top {top_k} indices.
Example output: [3, 0, 7, 2, 5]
Return ONLY the JSON array. No explanation."""


def rerank_chunks(
    question: str,
    chunks: list[Document],
    llm: ChatGroq,
    top_k: int = FINAL_TOP_K,
) -> list[Document]:
    """
    Ask the Groq LLM to rank the candidates by relevance.
    Returns the top_k most relevant Document objects.
    """
    if len(chunks) <= top_k:
        return chunks

    # Truncate each chunk to 400 chars to stay within token budget
    listing = "\n\n".join(
        f"[{i}] {c.metadata.get('source_file','?')} "
        f"(page {c.metadata.get('page_number','?')})\n"
        f"{c.page_content[:400]}"
        for i, c in enumerate(chunks)
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=_RERANK_SYS.format(top_k=top_k)),
            HumanMessage(content=f"Question: {question}\n\nChunks:\n{listing}"),
        ])
        text = resp.content.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            text = m.group(1).strip() if m else text
        indices = json.loads(text)
        valid   = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
        return [chunks[i] for i in valid[:top_k]]
    except Exception as exc:
        print(f"  [Reranker] Fallback ({exc})")
        return chunks[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 7. RELEVANCE GRADER
# ══════════════════════════════════════════════════════════════════════════════
_GRADE_SYS = """You are a binary relevance grader.
Does the document chunk help answer the question?
Reply with exactly one word: yes  or  no"""


def grade_chunks(
    question: str,
    chunks: list[Document],
    llm: ChatGroq,
) -> list[Document]:
    """
    Grade each chunk individually. Keep only chunks the LLM marks 'yes'.
    """
    relevant: list[Document] = []
    for chunk in chunks:
        try:
            resp = llm.invoke([
                SystemMessage(content=_GRADE_SYS),
                HumanMessage(
                    content=f"Question: {question}\n\nChunk:\n{chunk.page_content[:600]}"
                ),
            ])
            if resp.content.strip().lower().startswith("yes"):
                relevant.append(chunk)
        except Exception:
            relevant.append(chunk)   # keep on error to avoid empty context
    return relevant


# ══════════════════════════════════════════════════════════════════════════════
# 8. DOCUMENT SUMMARIZER (map-reduce)
# ══════════════════════════════════════════════════════════════════════════════
_MAP_TMPL = (
    "Summarize the following excerpt from '{doc_title}' in 3-5 sentences. "
    "Focus on key facts, policies, or information.\n\n"
    "Content:\n{content}\n\nSummary:"
)

_REDUCE_TMPL = (
    "You have per-document summaries below. "
    "Use them to answer the user's question comprehensively. "
    "Skip documents that are clearly unrelated. "
    "Cite each used document as [Document Name].\n\n"
    "User Question: {question}\n\n"
    "Document Summaries:\n{summaries}\n\n"
    "Answer:"
)


def summarize_all_documents(
    question: str,
    chunks: list[Document],
    llm: ChatGroq,
) -> str:
    """
    Map-Reduce summarization.
      Map    – summarize each source document independently.
      Reduce – combine summaries into a single coherent answer.
    """
    # Group chunks by source file
    doc_groups: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.metadata.get("source_file", "unknown")].append(chunk)

    # MAP step
    summaries: list[str] = []
    for filename, doc_chunks in doc_groups.items():
        content   = "\n\n".join(c.page_content for c in doc_chunks)[:4000]
        doc_title = Path(filename).stem
        try:
            resp    = llm.invoke(_MAP_TMPL.format(doc_title=doc_title, content=content))
            summary = resp.content.strip()
        except Exception as exc:
            summary = f"(Summarization failed: {exc})"
        summaries.append(f"### {doc_title}\n{summary}")
        print(f"    Summarized: {filename}")

    # REDUCE step
    combined = "\n\n".join(summaries)
    try:
        resp = llm.invoke(_REDUCE_TMPL.format(question=question, summaries=combined))
        return resp.content.strip()
    except Exception as exc:
        return f"Could not generate combined summary: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# 9. ANSWER QUESTION
# ══════════════════════════════════════════════════════════════════════════════
_FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise document assistant. Answer using ONLY the context below.\n\n"
            "Rules:\n"
            "1. Use only information directly present in the context.\n"
            "2. Ignore context sections that do not help answer the question.\n"
            "3. Do NOT mention documents that are unrelated to the answer.\n"
            "4. Cite every fact with its source using: [Document Name, page X]\n"
            "5. If the answer is not in the context, respond with exactly:\n"
            "   \"I don't have enough information in the provided documents.\"\n\n"
            "Context:\n{context}"
        ),
    ),
    ("human", "{input}"),
])


def answer_question(
    question: str,
    relevant_chunks: list[Document],
    llm: ChatGroq,
) -> str:
    """Stuff the relevant chunks into the final prompt and call the LLM."""
    chain = create_stuff_documents_chain(llm=llm, prompt=_FINAL_PROMPT)
    print("*************relevant chunks ********************")
    print(relevant_chunks)
    return chain.invoke({"input": question, "context": relevant_chunks})


# ══════════════════════════════════════════════════════════════════════════════
# 10. ADVANCED RAG ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
class AdvancedRAG:
    """
    Ties together all pipeline stages:
    route → retrieve → rerank → grade → answer
    """

    def __init__(
        self,
        chunks: list[Document],
        vectorstore: Chroma,
        llm: ChatGroq,
    ):
        self.chunks       = chunks
        self.vectorstore  = vectorstore
        self.llm          = llm
        self.available_docs = sorted(
            {c.metadata.get("source_file", "") for c in chunks}
        )

    def query(self, question: str) -> dict:
        """
        Full advanced RAG query pipeline.
        Returns: {answer, mode, rewritten_query, sources}
        """
        divider = "─" * 55
        print(f"\n{divider}")
        print(f"  Q: {question}")
        print(divider)

        # ── Step 1: Route & rewrite ──────────────────────────────────────────
        print("  [1/5] Routing query …")
        route = route_query(question, self.llm, self.available_docs)
        mode            = route["mode"]
        rewritten_query = route["rewritten_query"]
        doc_filter      = route.get("document_name")

        print(f"        Mode    : {mode}")
        print(f"        Rewrite : {rewritten_query}")
        if doc_filter:
            print(f"        Filter  : {doc_filter}")

        # ── Step 2a: DOCUMENT_SUMMARY path ───────────────────────────────────
        if mode == "DOCUMENT_SUMMARY":
            print("  [2/5] Summarizing (map-reduce) …")
            target = (
                [c for c in self.chunks if c.metadata.get("source_file") == doc_filter]
                if doc_filter else self.chunks
            ) or self.chunks
            answer = summarize_all_documents(rewritten_query, target, self.llm)
            return {
                "answer":         answer,
                "mode":           mode,
                "rewritten_query": rewritten_query,
                "sources":        [],
            }

        # ── Step 2b: Hybrid retrieval ─────────────────────────────────────────
        print("  [2/5] Hybrid retrieval (BM25 + Semantic, k=10 each) …")
        retriever  = build_hybrid_retriever(
            self.chunks,
            self.vectorstore,
            k=CANDIDATE_K,
            source_filter=doc_filter if mode == "DOCUMENT_SPECIFIC" else None,
        )
        candidates = retriever.invoke(rewritten_query)
        print(f"        {len(candidates)} candidates retrieved.")

        # ── Step 3: Rerank ───────────────────────────────────────────────────
        print("  [3/5] Reranking …")
        reranked = rerank_chunks(rewritten_query, candidates, self.llm, top_k=FINAL_TOP_K)
        print(f"        {len(reranked)} chunks kept after reranking.")

        # ── Step 4: Relevance grading ─────────────────────────────────────────
        print("  [4/5] Grading relevance …")
        relevant = grade_chunks(rewritten_query, reranked, self.llm)
        print(f"        {len(relevant)} chunks passed relevance check.")

        if not relevant:
            return {
                "answer":          "I don't have enough information in the provided documents.",
                "mode":            mode,
                "rewritten_query": rewritten_query,
                "sources":         [],
            }

        # ── Step 5: Generate answer ───────────────────────────────────────────
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
                    f"  • {Path(key[0]).stem}  [page {key[1]}]  "
                    f"({m.get('chunk_type','text')})"
                )

        return {
            "answer":          answer,
            "mode":            mode,
            "rewritten_query": rewritten_query,
            "sources":         sources,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_embedding_fn() -> HuggingFaceEmbeddings:
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✔ Embedding model loaded.\n")
    return emb


def _get_llm(api_key: str) -> ChatGroq:
    llm = ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0.2)
    print(f"✔ LLM ready: {GROQ_MODEL} via Groq.\n")
    return llm


def build_pipeline(groq_api_key: str) -> AdvancedRAG:
    """Build everything from scratch: load → chunk → embed → store → init."""
    sep = "=" * 58

    print(sep); print("STEP 1 — Loading PDFs"); print(sep)
    docs = load_pdfs()

    print(sep); print("STEP 2 — Chunking + metadata enrichment"); print(sep)
    chunks = chunk_documents(docs)

    print(sep); print("STEP 3 — Sentence-transformer embeddings"); print(sep)
    embedding_fn = _get_embedding_fn()

    print(sep); print("STEP 4 — ChromaDB vector store"); print(sep)
    vectorstore = build_vectorstore(chunks, embedding_fn)

    print(sep); print("STEP 5 — Groq LLM"); print(sep)
    llm = _get_llm(groq_api_key)

    print("✅ Advanced RAG pipeline ready!\n")
    return AdvancedRAG(chunks=chunks, vectorstore=vectorstore, llm=llm)


def load_pipeline(groq_api_key: str) -> AdvancedRAG:
    """
    Load from an existing ChromaDB index.
    PDFs are re-read for BM25 (in-memory only, no re-embedding).
    """
    print("  Re-parsing PDFs for BM25 (no re-embedding) …")
    docs   = load_pdfs()
    chunks = chunk_documents(docs)

    embedding_fn = _get_embedding_fn()
    vectorstore  = _load_vectorstore(embedding_fn)
    llm          = _get_llm(groq_api_key)

    print("✅ Pipeline loaded from disk.\n")
    return AdvancedRAG(chunks=chunks, vectorstore=vectorstore, llm=llm)


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        sys.exit("❌  GROQ_API_KEY not set. Add it to .env as: GROQ_API_KEY=gsk_...")

    # Build or load
    chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    if chroma_exists:
        print(f"\nFound existing index at '{CHROMA_DIR}'. Loading …\n")
        rag = load_pipeline(GROQ_API_KEY)
    else:
        print("\nNo existing index. Building from scratch …\n")
        rag = build_pipeline(GROQ_API_KEY)

    print("=" * 58)
    print(f"  Advanced RAG  |  BM25 {BM25_WEIGHT:.0%} + Semantic {SEMANTIC_WEIGHT:.0%}")
    print(f"  Candidates: {CANDIDATE_K} per leg  |  Final top-k: {FINAL_TOP_K}")
    print(f"  Modes: FACTUAL_QA / DOCUMENT_SPECIFIC / DOCUMENT_SUMMARY")
    print("  Type 'quit' to exit.")
    print("=" * 58)

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

        print(f"\n── Mode: {result['mode']}  |  Rewritten: \"{result['rewritten_query']}\"")

        if result["sources"]:
            print("\n── Sources ─────────────────────────────────────────────")
            print("\n".join(result["sources"]))
        else:
            print("\n── Sources: none")
