import pytest
from fastapi.testclient import TestClient
from backend.app.app import app


def test_root_endpoint():
    """Test the root endpoint returns correct response."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_check_endpoint():
    """Test the health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_cors_headers():
    """Test that CORS headers are properly set."""
    client = TestClient(app)
    response = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers


def test_static_files_mount():
    """Test that static files are properly mounted."""
    # This is more of an integration test, but we can check the app has the mount
    from backend.app.app import app

    # Check that static files are mounted
    mounts = [route.path for route in app.routes if hasattr(route, "path")]
    assert "/static" in str(mounts) or any(
        "/static" in str(route) for route in app.routes
    )
