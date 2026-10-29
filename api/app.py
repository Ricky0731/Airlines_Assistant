"""
api/app.py — FastAPI application setup with CORS and lifespan event for RAG pipeline.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from config.settings import CHROMA_DIR
from engine.rag_engine import AdvancedRAG, load_pipeline, build_pipeline
from api.routes import health_router, query_router

rag_pipeline: AdvancedRAG | None = None


def get_rag_pipeline() -> AdvancedRAG | None:
    return rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        print("⚠️ Warning: GROQ_API_KEY is not set in environment or .env file.")
    else:
        try:
            chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
            if chroma_exists:
                print(f"\n[FastAPI] Loading existing ChromaDB from '{CHROMA_DIR}'...")
                rag_pipeline = load_pipeline(groq_api_key)
            else:
                print("\n[FastAPI] Building index from scratch...")
                rag_pipeline = build_pipeline(groq_api_key)
            print("🚀 [FastAPI] Advanced RAG pipeline initialized successfully!\n")
        except Exception as e:
            print(f"❌ [FastAPI] Failed to initialize RAG pipeline: {e}")

    yield


app = FastAPI(
    title="Advanced RAG API",
    description="Portfolio Advanced RAG Assistant with Hybrid Search (BM25 + Semantic), Groq LLM Reranking & Verification",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev and Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel deployment and localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(query_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": "Advanced RAG API is running.",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }

