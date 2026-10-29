"""
server.py — FastAPI server entrypoint for production (e.g. Render/Railway) and local testing.

Run locally:
    python server.py
    # OR:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)

