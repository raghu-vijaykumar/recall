import pytest
from unittest.mock import patch


def test_get_workspaces_empty(client):
    """Test getting workspaces when none exist."""
    response = client.get("/api/workspaces/")
    assert response.status_code == 200
    assert response.json() == []


def test_create_workspace(client):
    """Test creating a new workspace."""
    workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
    response = client.post("/api/workspaces/", json=workspace_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == workspace_data["name"]
    assert data["folder_path"] == workspace_data["folder_path"]
    assert "id" in data


def test_get_workspace(client):
    """Test getting a specific workspace."""
    # First create a workspace
    workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
    create_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_response.json()["id"]

    # Then get it
    response = client.get(f"/api/workspaces/{workspace_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == workspace_id
    assert data["name"] == workspace_data["name"]


def test_get_workspace_not_found(client):
    """Test getting a non-existent workspace."""
    response = client.get("/api/workspaces/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Workspace not found"


def test_update_workspace(client):
    """Test updating a workspace."""
    # Create workspace
    workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
    create_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_response.json()["id"]

    # Update it
    update_data = {"name": "Updated Workspace", "folder_path": "/updated/path"}
    response = client.put(f"/api/workspaces/{workspace_id}", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["folder_path"] == update_data["folder_path"]


def test_delete_workspace(client):
    """Test deleting a workspace."""
    # Create workspace
    workspace_data = {"name": "Test Workspace", "path": "/test/path"}
    create_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_response.json()["id"]

    # Delete it
    response = client.delete(f"/api/workspaces/{workspace_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Workspace deleted successfully"

    # Verify it's gone
    get_response = client.get(f"/api/workspaces/{workspace_id}")
    assert get_response.status_code == 404


def test_get_workspace_stats(client):
    """Test getting workspace statistics."""
    # Create workspace
    workspace_data = {"name": "Test Workspace", "path": "/test/path"}
    create_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_response.json()["id"]

    # Get stats
    response = client.get(f"/api/workspaces/{workspace_id}/stats")
    assert response.status_code == 200
    data = response.json()
    # Stats structure depends on your WorkspaceStats model
    assert isinstance(data, dict)
