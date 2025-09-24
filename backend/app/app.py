"""
Recall Study App Backend
FastAPI application for managing workspaces, files, and quiz generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from app.routes.workspaces import router as workspaces_router
from app.routes.files import router as files_router
from app.routes.quiz import router as quiz_router
from app.routes.progress import router as progress_router
from app.routes.search import router as search_router
from app.routes.knowledge_graph import router as knowledge_graph_router
from app.routes.quiz_improvements import router as quiz_improvements_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info("Starting up Recall API...")
    yield
    logger.info("Shutting down Recall API...")
    # Cancel all pending tasks to ensure clean shutdown
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} pending tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="Recall API",
    description="Backend API for the Recall study application",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for Electron app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Electron app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for file serving
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Running in a PyInstaller bundle
    static_dir = Path(sys._MEIPASS) / "static"
else:
    # Running in normal Python environment
    static_dir = Path(__file__).parent / "static"

static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(workspaces_router, prefix="/api/workspaces", tags=["workspaces"])
app.include_router(files_router, prefix="/api/files", tags=["files"])
app.include_router(quiz_router, prefix="/api/quiz", tags=["quiz"])
app.include_router(progress_router, prefix="/api/progress", tags=["progress"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(knowledge_graph_router, tags=["knowledge-graph"])
app.include_router(quiz_improvements_router, tags=["quiz-improvements"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Recall API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
