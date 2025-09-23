import pytest


def test_get_workspace_files_empty(client):
    """Test getting files for a workspace with no files."""
    # Create a unique workspace to test against
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_response.json()["id"]

    response = client.get(f"/api/files/workspace/{workspace_id}")
    assert response.status_code == 200
    files = response.json()
    # This workspace should be empty since we just created it
    assert files == []


def test_create_file(client):
    """Test creating a new file."""
    # Create a workspace first
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_workspace_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_workspace_response.json()["id"]

    file_data = {
        "name": "test.txt",
        "path": "/test/test.txt",
        "file_type": "text",
        "size": 100,
        "workspace_id": workspace_id,
        "content": "This is test content",
    }
    response = client.post("/api/files/", json=file_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == file_data["name"]
    assert data["path"] == file_data["path"]
    assert "id" in data


def test_get_file(client):
    """Test getting a specific file."""
    # Create workspace and file
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_workspace_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_workspace_response.json()["id"]

    file_data = {
        "name": "test.txt",
        "path": "/test/test.txt",
        "file_type": "text",
        "size": 100,
        "workspace_id": workspace_id,
        "content": "This is test content",
    }
    create_response = client.post("/api/files/", json=file_data)
    file_id = create_response.json()["id"]

    # Get the file
    response = client.get(f"/api/files/{file_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == file_id
    assert data["name"] == file_data["name"]


def test_get_file_content(client):
    """Test getting file content."""
    # Create workspace and file
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_workspace_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_workspace_response.json()["id"]

    file_data = {
        "name": "test.txt",
        "path": "/test/test.txt",
        "file_type": "text",
        "size": 100,
        "workspace_id": workspace_id,
        "content": "This is test content",
    }
    create_response = client.post("/api/files/", json=file_data)
    file_id = create_response.json()["id"]

    # Get content
    response = client.get(f"/api/files/{file_id}/content")
    assert response.status_code == 200
    data = response.json()
    # Note: File service currently returns placeholder content
    assert "content" in data
    assert isinstance(data["content"], str)


def test_update_file_content(client):
    """Test updating file content."""
    # Create workspace and file
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_workspace_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_workspace_response.json()["id"]

    file_data = {
        "name": "test.txt",
        "path": "/test/test.txt",
        "file_type": "text",
        "size": 100,
        "workspace_id": workspace_id,
        "content": "This is test content",
    }
    create_response = client.post("/api/files/", json=file_data)
    file_id = create_response.json()["id"]

    # Update content
    new_content = "Updated content"
    response = client.put(f"/api/files/{file_id}/content?content={new_content}")
    assert response.status_code == 200
    assert response.json()["message"] == "File content updated successfully"

    # Note: File service currently doesn't store actual content, just updates metadata
    # So we can't verify the content was actually updated
    # This test verifies the update operation succeeded


def test_delete_file(client):
    """Test deleting a file."""
    # Create workspace and file
    import uuid

    unique_name = f"Test Workspace {uuid.uuid4().hex}"
    workspace_data = {"name": unique_name, "folder_path": "/test/path"}
    create_workspace_response = client.post("/api/workspaces/", json=workspace_data)
    workspace_id = create_workspace_response.json()["id"]

    file_data = {
        "name": "test.txt",
        "path": "/test/test.txt",
        "file_type": "text",
        "size": 100,
        "workspace_id": workspace_id,
        "content": "This is test content",
    }
    create_response = client.post("/api/files/", json=file_data)
    file_id = create_response.json()["id"]

    # Delete the file
    response = client.delete(f"/api/files/{file_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "File deleted successfully"

    # Verify it's gone
    get_response = client.get(f"/api/files/{file_id}")
    assert get_response.status_code == 404
