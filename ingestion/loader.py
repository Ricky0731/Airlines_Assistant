"""
ingestion/loader.py — PDF loading with document-level metadata.
Uses PyMuPDFLoader (langchain-community; no standalone package yet).
"""

import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import PyMuPDFLoader

from langchain_core.documents import Document
from config.settings import DATA_DIR, DOC_TYPE_MAP


def infer_document_type(filename: str) -> str:
    """Map a PDF filename to a semantic document-type label."""
    low = filename.lower()
    for doc_type, keywords in DOC_TYPE_MAP.items():
        if any(kw in low for kw in keywords):
            return doc_type
    return "general"


def load_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load every PDF in *data_dir* with PyMuPDFLoader.

    Metadata attached per page:
        source_file    – original PDF filename  (e.g. "Baggage Allowance.pdf")
        document_title – filename without extension
        document_type  – inferred label  (policy / aircraft_specs / …)
    """
    docs: list[Document] = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{data_dir}'.")

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source_file"]    = pdf_path.name
            page.metadata["document_title"] = pdf_path.stem
            page.metadata["document_type"]  = infer_document_type(pdf_path.name)
        docs.extend(pages)

    print(f"\n✔ Loaded {len(docs)} pages from {len(pdf_files)} PDFs.\n")
    return docs

