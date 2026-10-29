"""
pipeline/summarizer.py — Map-Reduce document summarization.

Used when the query router returns DOCUMENT_SUMMARY mode.

Map    – summarize each source document independently.
Reduce – combine per-document summaries into one coherent answer.
"""

from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_groq import ChatGroq


_MAP_TEMPLATE = (
    "Summarize the following excerpt from '{doc_title}' in 3-5 sentences. "
    "Focus on key facts, policies, or information.\n\n"
    "Content:\n{content}\n\nSummary:"
)

_REDUCE_TEMPLATE = (
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
    Map-Reduce summarization over *chunks* grouped by source document.

    Args:
        question : rewritten user question
        chunks   : chunks to summarize (may be pre-filtered to one document)
        llm      : Groq LLM instance
    """
    # Group chunks by source PDF
    doc_groups: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.metadata.get("source_file", "unknown")].append(chunk)

    # MAP — summarize each document
    summaries: list[str] = []
    for filename, doc_chunks in doc_groups.items():
        content   = "\n\n".join(c.page_content for c in doc_chunks)[:4000]
        doc_title = Path(filename).stem
        try:
            resp    = llm.invoke(_MAP_TEMPLATE.format(doc_title=doc_title, content=content))
            summary = resp.content.strip()
        except Exception as exc:
            summary = f"(Summarization failed: {exc})"
        summaries.append(f"### {doc_title}\n{summary}")
        print(f"    Summarized: {filename}")

    # REDUCE — combine into final answer
    combined = "\n\n".join(summaries)
    try:
        resp = llm.invoke(_REDUCE_TEMPLATE.format(question=question, summaries=combined))
        return resp.content.strip()
    except Exception as exc:
        return f"Could not generate combined summary: {exc}"

