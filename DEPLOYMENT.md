# Advanced RAG — Full-Stack Portfolio Application

Production-grade Advanced RAG system with Hybrid Search, LLM Reranking, Relevance Verification, and Citations.

---

## 🏗️ Architecture

- **Backend:** FastAPI (Python), ChromaDB, BM25, Sentence-Transformers, Groq Cloud LLM
- **Frontend:** Modern Responsive Glassmorphism Web App (HTML5, CSS3, Vanilla JS)
- **Deployment Strategy:**
  - **Frontend:** [Vercel](https://vercel.com) (Static CDN, super fast, 100% free)
  - **Backend:** [Render](https://render.com) (Free tier web service with persistent environment)

---

## 📁 Project Structure

```
basic_rag/
├── api/                   # FastAPI Backend
│   ├── app.py             # FastAPI App instance with CORS & lifespan
│   ├── models.py          # Pydantic Request/Response models
│   └── routes/
│       ├── health.py      # GET /api/health
│       └── query.py       # POST /api/query
├── frontend/              # Ready for Vercel deployment
│   ├── index.html         # Modern Chat UI
│   ├── style.css          # Dark Mode GitHub-styled UI
│   ├── app.js             # API connector & typing indicators
│   └── vercel.json        # Vercel configuration
├── config/                # Pipeline settings
├── ingestion/             # Document loader & chunker
├── pipeline/              # Router, Retriever, Reranker, Grader, Answerer
├── storage/               # ChromaDB vector store
├── engine/                # Advanced RAG orchestrator
├── server.py              # Backend start script
└── requirements.txt
```

---

## 🚀 Running Locally

### 1. Start the FastAPI Backend
```bash
python server.py
# Server runs on: http://localhost:8000
# Interactive API Docs: http://localhost:8000/docs
```

### 2. Launch the Frontend
You can open `frontend/index.html` directly in your browser or run a quick local server:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000 in your browser
```

---

## 🌐 Deploying to Portfolio

### Step 1: Deploy Backend to Render (Free)
1. Push this project to a GitHub repository.
2. Sign in to [Render.com](https://render.com) and click **New Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python server.py`
5. Under **Environment Variables**, add:
   - `GROQ_API_KEY`: `your_groq_api_key_here`
6. Click **Deploy**. Render will generate a live URL (e.g., `https://my-rag-backend.onrender.com`).

### Step 2: Deploy Frontend to Vercel (Free)
1. Go to [Vercel.com](https://vercel.com) and click **Add New Project**.
2. Select your repository.
3. In **Root Directory**, choose `frontend`.
4. Click **Deploy**!
5. In your live Vercel app, simply paste your Render backend URL into the **Backend API URL** box in the left sidebar, and it will immediately connect!

