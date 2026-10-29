"""api/routes/health.py — Health check endpoint."""

from fastapi import APIRouter
from api.models import HealthResponse
from config.settings import GROQ_MODEL

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API and RAG pipeline are ready."""
    from api.app import get_rag_pipeline
    pipeline = get_rag_pipeline()
    ready = pipeline is not None
    docs_count = len(pipeline.available_docs) if ready else 0

    return HealthResponse(
        status="ok" if ready else "initializing",
        pipeline_ready=ready,
        documents_loaded=docs_count,
        model=GROQ_MODEL,
    )
