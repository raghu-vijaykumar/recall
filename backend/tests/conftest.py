import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import uuid
import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import routes
from backend.app.routes.workspaces import router as workspaces_router
from backend.app.routes.files import router as files_router
from backend.app.routes.quiz import router as quiz_router
from backend.app.routes.progress import router as progress_router
from backend.app.routes.search import router as search_router
from backend.app.routes.knowledge_graph import router as knowledge_graph_router

# Import migration service for test database setup
from backend.app.services.migration_service import MigrationService


@pytest.fixture(scope="function", autouse=True)
def setup_test_database(tmp_path):
    """Set up isolated test database for each test."""
    # Create a unique database file for each test
    test_db_path = tmp_path / f"test_recall_{uuid.uuid4().hex}.db"

    # Set environment variable to use test database
    original_db_path = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = str(test_db_path)

    yield

    # Clean up
    if original_db_path:
        os.environ["DATABASE_PATH"] = original_db_path
    else:
        os.environ.pop("DATABASE_PATH", None)

    # Database file will be automatically cleaned up by tmp_path fixture


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    test_app = FastAPI(title="Test Recall API")

    # Configure CORS
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    static_dir = Path(__file__).parent.parent / "app" / "static"
    static_dir.mkdir(exist_ok=True)
    test_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    test_app.include_router(
        workspaces_router, prefix="/api/workspaces", tags=["workspaces"]
    )
    test_app.include_router(files_router, prefix="/api/files", tags=["files"])
    test_app.include_router(quiz_router, prefix="/api/quiz", tags=["quiz"])
    test_app.include_router(progress_router, prefix="/api/progress", tags=["progress"])
    test_app.include_router(search_router, prefix="/api/search", tags=["search"])
    test_app.include_router(
        knowledge_graph_router, prefix="/api/knowledge-graph", tags=["knowledge-graph"]
    )

    return test_app


@pytest.fixture
def client(app):
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    """Create a test database session factory with initialized schema."""
    # Get the test database path from environment
    db_path = os.environ.get("DATABASE_PATH", ":memory:")

    # Initialize database with migrations
    migrations_dir = Path(__file__).parent.parent / "migrations"
    migration_service = MigrationService(str(db_path), str(migrations_dir))

    if not migration_service.apply_pending_migrations():
        raise RuntimeError("Failed to apply database migrations for tests")

    # Create async engine for test database
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)

    # Create session factory
    async_session_factory = sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    yield async_session_factory

    # Properly close the async engine to avoid ResourceWarning
    asyncio.run(engine.dispose())
