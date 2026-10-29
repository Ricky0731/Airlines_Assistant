"""api/routes/query.py — Query endpoint for RAG."""

from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag_endpoint(payload: QueryRequest):
    """Ask a question to the Advanced RAG pipeline."""
    from api.app import get_rag_pipeline

    pipeline = get_rag_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Please check API key and server logs."
        )

    try:
        result = pipeline.query(payload.question)
        return QueryResponse(
            answer=result["answer"],
            mode=result["mode"],
            rewritten_query=result["rewritten_query"],
            sources=result["sources"],
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

