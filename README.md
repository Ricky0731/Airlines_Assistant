# ✈️ Airlines Assistant — Advanced RAG System

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6600?style=flat)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logo=groq&logoColor=white)](https://groq.com)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white)](https://vercel.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, full-stack **Advanced Retrieval-Augmented Generation (RAG)** assistant built to query complex airline documents, aircraft technical specifications, customer policies, and operational procedures with zero hallucinations and verifiable citations.

---

## 🌟 Why Advanced RAG?

Standard (naive) RAG pipelines rely strictly on vector embeddings and top-$k$ nearest neighbor search. In production, this fails when handling:
- **Exact technical keywords & model numbers** (e.g., *Airbus A350 vs. A380*, specific baggage allowance numbers, military discount codes).
- **Ambiguous or poorly phrased user queries**.
- **Context clutter and noise** leading to hallucinations.

**Airlines Assistant** addresses these bottlenecks with a 5-stage retrieval & self-correction pipeline:
1. **Dynamic Query Routing & Rewriting:** Detects user intent (`FACTUAL_QA`, `DOCUMENT_SPECIFIC`, `DOCUMENT_SUMMARY`) and optimizes query phrasing.
2. **Hybrid Search (Sparse + Dense):** Ensembles **BM25 keyword search** (40%) and **Dense Semantic Retrieval** (60%) via `all-MiniLM-L6-v2` in ChromaDB.
3. **Cross-Encoder / LLM Reranking:** Filters down candidate chunks ($K=10$ per leg) to the top 5 highest-relevance passages.
4. **Self-Reflective Relevance Grading:** Evaluates whether retrieved chunks actually contain the answer before generation.
5. **Grounded Generation with Strict Citations:** Answers are produced by high-speed Groq Cloud LLM with exact source file and page-level attribution.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[ 1. Query Router & Rewriter ] ───► Intent: FACTUAL_QA / DOC_SPECIFIC / SUMMARY
    │
    ▼
[ 2. Hybrid Retriever ]
    ├── Sparse Retrieval (BM25 - 40%) ───► Exact codes, numbers, keywords
    └── Dense Retrieval (ChromaDB - 60%) ─► Semantic meaning (all-MiniLM-L6-v2)
    │
    ▼
[ 3. LLM Reranker ] ─────────────────► 10 Candidates per leg ➔ Top 5 chunks
    │
    ▼
[ 4. Relevance Grader ] ─────────────► Filters out irrelevant / noisy chunks
    │
    ▼
[ 5. Grounded Generator (Groq) ] ────► Synthesizes response + File & Page Citations
    │
    ▼
Verified Response to User
```

---

## 📁 Repository Structure

```
Airlines_Assistant/
├── api/                   # FastAPI Backend Application
│   ├── app.py             # App setup, CORS middleware & lifespan loader
│   ├── models.py          # Pydantic request & response schemas
│   └── routes/
│       ├── health.py      # GET /api/health (status & document stats)
│       └── query.py       # POST /api/query (RAG query processing)
├── config/
│   └── settings.py        # Centralized configurations, weights & constants
├── data/                  # Airline policy & aircraft spec PDFs
│   ├── Airbus A350.pdf
│   ├── Airbus A380.pdf
│   ├── Armed Forces Discount Program.pdf
│   ├── Awards & Recognitions.pdf
│   ├── Baggage Allowance.pdf
│   ├── Boarding Process.pdf
│   └── Change Your Seat.pdf
├── engine/
│   └── rag_engine.py      # Core AdvancedRAG orchestrator & pipeline builder
├── frontend/              # Web Client (Vercel-ready)
│   ├── app.js             # API connector, dynamic typing & status indicator
│   ├── index.html         # Responsive Chat UI
│   ├── style.css          # Dark-mode Glassmorphism styling
│   └── vercel.json        # Vercel deployment configuration
├── ingestion/
│   ├── chunker.py         # Recursive character text splitting
│   ├── ingest.py          # Standalone ingestion CLI script
│   └── loader.py          # PyMuPDF fast PDF document parser
├── pipeline/
│   ├── answerer.py        # Citation-enforced answer generator
│   ├── grader.py          # Chunk relevance verification
│   ├── reranker.py        # LLM candidate reranking
│   ├── retriever.py       # Hybrid BM25 + Vector ensemble retriever
│   ├── router.py          # Query routing & rewriting
│   └── summarizer.py      # Map-reduce document summarization
├── storage/
│   └── vectorstore.py     # ChromaDB vector store wrapper & embeddings
├── server.py              # Backend entry point for Uvicorn
├── main.py                # Interactive CLI runner
├── requirements.txt       # Python dependencies
└── DEPLOYMENT.md          # Cloud deployment documentation
```

---

## 🛠️ Tech Stack

- **Large Language Model:** Groq Cloud (`openai/gpt-oss-120b` or LLaMA-3) for ultra-low latency inference.
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database:** [ChromaDB](https://www.trychroma.com) (local persistent storage).
- **Hybrid Search:** `rank-bm25` (Sparse) + ChromaDB (Dense) via LangChain's `EnsembleRetriever`.
- **PDF Parser:** [PyMuPDF](https://pymupdf.readthedocs.io/) (`fitz`) for fast, structure-preserving extraction.
- **Backend API:** [FastAPI](https://fastapi.tiangolo.com/) with asynchronous lifespan management.
- **Frontend UI:** Modern responsive Glassmorphism interface (HTML5, CSS3, Vanilla JS).

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10 or higher
- A free API key from [Groq Console](https://console.groq.com)

### 2. Clone the Repository
```bash
git clone https://github.com/Ricky0731/Airlines_Assistant.git
cd Airlines_Assistant
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 5. Run the Application

#### Option A: Interactive CLI Mode
```bash
python main.py
```
*(On first run, it will automatically parse documents in `data/` and build the ChromaDB index).*

#### Option B: Full-Stack Web App (Backend + Frontend)
1. **Start the FastAPI Backend:**
   ```bash
   python server.py
   ```
   - API runs at: `http://localhost:8000`
   - Interactive Swagger API docs: `http://localhost:8000/docs`

2. **Launch the Frontend:**
   Open `frontend/index.html` in your browser, or start a local server:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   Visit `http://localhost:3000` in your browser.

---

## 🌐 Deployment

### Deploy Backend to Render (Free)
1. Push this repository to GitHub.
2. Sign in to [Render.com](https://render.com) and create a **New Web Service**.
3. Select your repository and specify:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py`
4. Add environment variable: `GROQ_API_KEY`.
5. Deploy and copy your backend URL (e.g., `https://my-rag-backend.onrender.com`).

### Deploy Frontend to Vercel (Free)
1. Go to [Vercel.com](https://vercel.com) and click **Add New Project**.
2. Select your repository.
3. In **Root Directory**, choose `frontend`.
4. Click **Deploy**.
5. Update `API_URL` in `frontend/app.js` with your live Render backend URL.

---

## 📡 API Endpoints

### `GET /api/health`
Checks the server status, active LLM model, and vector store readiness.

**Response:**
```json
{
  "status": "healthy",
  "pipeline_ready": true,
  "documents_loaded": 7,
  "model": "openai/gpt-oss-120b"
}
```

### `POST /api/query`
Processes a question through the full 5-stage Advanced RAG pipeline.

**Request:**
```json
{
  "question": "What is the checked baggage allowance for economy passengers?"
}
```

**Response:**
```json
{
  "answer": "Economy Class passengers are entitled to one piece of checked baggage up to 23 kg (50 lbs)...",
  "mode": "DOCUMENT_SPECIFIC",
  "rewritten_query": "checked baggage allowance economy class limits",
  "sources": [
    "Baggage Allowance.pdf (Page 2)",
    "Baggage Allowance.pdf (Page 3)"
  ],
  "success": true
}
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

