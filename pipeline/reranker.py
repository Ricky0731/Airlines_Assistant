"""
pipeline/reranker.py — LLM-based chunk reranker.

Asks the Groq LLM to rank retrieved chunks by relevance and returns
only the top-k most relevant ones.
"""

import json
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from config.settings import FINAL_TOP_K


_RERANK_SYSTEM = """You are a document relevance ranker.
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
    Ask the Groq LLM to rerank *chunks* by relevance to *question*.
    Returns the *top_k* most relevant Document objects.
    Falls back to first *top_k* chunks on any error.
    """
    if len(chunks) <= top_k:
        return chunks

    listing = "\n\n".join(
        f"[{i}] {c.metadata.get('source_file', '?')} "
        f"(page {c.metadata.get('page_number', '?')})\n"
        f"{c.page_content[:400]}"
        for i, c in enumerate(chunks)
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=_RERANK_SYSTEM.format(top_k=top_k)),
            HumanMessage(content=f"Question: {question}\n\nChunks:\n{listing}"),
        ])
        text = resp.content.strip()
        if "```" in text:
            m    = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            text = m.group(1).strip() if m else text
        indices = json.loads(text)
        valid   = [i for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
        return [chunks[i] for i in valid[:top_k]]
    except Exception as exc:
        print(f"  [Reranker] Fallback ({exc})")
        return chunks[:top_k]

