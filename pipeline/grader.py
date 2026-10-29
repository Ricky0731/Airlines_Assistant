"""
pipeline/grader.py — LLM-based binary relevance grader.

For each retrieved chunk asks the LLM: "Does this chunk help answer the question?"
Only chunks marked 'yes' are kept. Fails open (keeps chunk) on any LLM error.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


_GRADE_SYSTEM = """You are a binary relevance grader.
Does the document chunk help answer the question?
Reply with exactly one word: yes  or  no"""


def grade_chunks(
    question: str,
    chunks: list[Document],
    llm: ChatGroq,
) -> list[Document]:
    """
    Grade each chunk in *chunks* for relevance to *question*.
    Returns only the chunks the LLM marks as 'yes'.
    """
    relevant: list[Document] = []
    for chunk in chunks:
        try:
            resp = llm.invoke([
                SystemMessage(content=_GRADE_SYSTEM),
                HumanMessage(
                    content=f"Question: {question}\n\nChunk:\n{chunk.page_content[:600]}"
                ),
            ])
            if resp.content.strip().lower().startswith("yes"):
                relevant.append(chunk)
        except Exception:
            relevant.append(chunk)  # keep on error to avoid empty context
    return relevant

