"""
ingestion/chunker.py — Two-stage document splitting + chunk-level metadata.

Stage 1: MarkdownTextSplitter       (preserves heading structure)
Stage 2: RecursiveCharacterTextSplitter  (breaks still-oversized chunks)

Metadata added per chunk:
    chunk_id      – sequential index
    chunk_length  – character count
    page_number   – 1-based page number
    section_path  – Markdown heading breadcrumb found inside the chunk
    chunk_type    – "text" | "table" | "header"
"""

import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _extract_section_path(text: str) -> str:
    headings = re.findall(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)
    return " > ".join(h.strip() for h in headings[:3])


def _detect_chunk_type(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "text"
    pipe_ratio = sum(1 for ln in lines if "|" in ln) / len(lines)
    if pipe_ratio > 0.4:
        return "table"
    if re.match(r"^#{1,4}\s", lines[0]):
        return "header"
    return "text"


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split *docs* into chunks and enrich each with derived metadata.
    Returns a flat list of Document chunks ready for embedding.
    """
    # Stage 1 — Markdown-aware
    md_splitter = MarkdownTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    md_chunks = md_splitter.split_documents(docs)
    print(f"  After MarkdownTextSplitter          : {len(md_chunks)} chunks")

    # Stage 2 — Recursive character fallback
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = rc_splitter.split_documents(md_chunks)
    print(f"  After RecursiveCharacterTextSplitter: {len(chunks)} chunks\n")

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"]     = idx
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["section_path"] = _extract_section_path(chunk.page_content)
        chunk.metadata["chunk_type"]   = _detect_chunk_type(chunk.page_content)
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    return chunks

