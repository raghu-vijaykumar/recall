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
from pathlib import Path

from app.routes.workspaces import router as workspaces_router
from app.routes.files import router as files_router
from app.routes.quiz import router as quiz_router
from app.routes.progress import router as progress_router

# Create FastAPI app
app = FastAPI(
    title="Recall API",
    description="Backend API for the Recall study application",
    version="1.0.0",
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
    static_dir = Path(sys._MEIPASS) / "app" / "static"
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


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Recall API is running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
