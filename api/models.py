"""api/models.py — Pydantic request/response models."""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the RAG pipeline.")


class QueryResponse(BaseModel):
    answer: str
    mode: str
    rewritten_query: str
    sources: list[str]
    success: bool = True


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    documents_loaded: int
    model: str

