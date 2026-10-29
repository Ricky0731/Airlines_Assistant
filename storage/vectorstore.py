"""
storage/vectorstore.py — ChromaDB vector store: build, persist, and load.
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (CPU, normalised).
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def get_embedding_fn() -> HuggingFaceEmbeddings:
    """Load and return the sentence-transformer embedding function."""
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✔ Embedding model loaded.\n")
    return emb


def build_vectorstore(
    chunks: list[Document],
    embedding_fn: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """Embed *chunks* and persist to a local ChromaDB collection."""
    print(f"  Embedding {len(chunks)} chunks → ChromaDB at '{persist_dir}' …")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    print(f"✔ Vector store ready  (collection: '{COLLECTION_NAME}').\n")
    return vs


def load_vectorstore(
    embedding_fn: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_DIR,
) -> Chroma:
    """Load an already-persisted ChromaDB collection from disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )

