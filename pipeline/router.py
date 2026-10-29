"""
pipeline/router.py — LLM-based query router.

Classifies the user question into one of three modes and rewrites it
for better retrieval precision.

Modes:
    FACTUAL_QA        – general question spanning multiple documents
    DOCUMENT_SPECIFIC – question clearly about ONE document / topic
    DOCUMENT_SUMMARY  – user wants a summary or overview
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


_ROUTER_SYSTEM = """You are a query routing assistant for a document Q&A system.
Classify the user question and return a JSON object with these exact fields:

{{
  "mode": "<FACTUAL_QA | DOCUMENT_SPECIFIC | DOCUMENT_SUMMARY>",
  "rewritten_query": "<clearer, more specific version of the question>",
  "document_name": "<exact PDF filename if question targets one document, else null>",
  "document_type": "<policy | aircraft_specs | process | awards | null>"
}}

Mode rules:
  FACTUAL_QA        – general question that may span multiple documents
  DOCUMENT_SPECIFIC – question is clearly about ONE specific document / topic
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

    Returns:
        mode            – "FACTUAL_QA" | "DOCUMENT_SPECIFIC" | "DOCUMENT_SUMMARY"
        rewritten_query – improved query string
        document_name   – PDF filename if DOCUMENT_SPECIFIC, else None
        document_type   – inferred type or None
    """
    prompt = _ROUTER_SYSTEM.format(doc_list=", ".join(available_docs))
    try:
        resp = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ])
        text = resp.content.strip()
        if "```" in text:
            m    = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
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

