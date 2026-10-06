"""
FastAPI Server Entry Point for Mini-X Campus Social Recommender System.
Usage:
    python run_server.py --port 8000
Serves the REST API and the modern interactive web interface at http://localhost:8000.
"""

import argparse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from api.routes import initialize_app_state, router
from config import FRONTEND_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load datasets, models, and build graph in memory
    print("Starting Mini-X Recommendation Server...")
    initialize_app_state()
    yield
    print("Shutting down Mini-X Server.")


app = FastAPI(
    title="Mini-X Campus Social Recommender API",
    description="Multi-Stage Hybrid RecSys Pipeline inspired by xai-org/x-algorithm",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router)

# Mount frontend directory for static assets
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serves the main interactive social UI dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Mini-X API is running. UI index.html not yet found."}
    return FileResponse(index_path)


def main():
    parser = argparse.ArgumentParser(description="Run Mini-X Recommendation Demo Server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")

    args = parser.parse_args()
    print(f"Launching Mini-X Server on http://{args.host}:{args.port}")
    uvicorn.run("run_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
