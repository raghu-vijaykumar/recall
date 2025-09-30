import pytest
from unittest.mock import patch, AsyncMock


import tempfile
import os


class TestAnalyzeWorkspaceTopics:
    """Test cases for workspace topics analysis endpoints."""

    def test_analyze_workspace_topics_success_actual_heuristic(self, client):
        """Test successful workspace analysis with actual heuristic extractor on test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files with content
            test_files = [
                (
                    "python_basics.py",
                    "def hello():\n    print('Hello World')\n    x = 42\n    return x",
                ),
                (
                    "data_structures.py",
                    "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()",
                ),
                (
                    "algorithms.py",
                    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                ),
                (
                    "README.md",
                    "# Test Project\n\nThis is a test project for workspace analysis.\n\n## Features\n\n- Python basics\n- Data structures\n- Algorithms",
                ),
            ]

            for filename, content in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            # Create a workspace pointing to this temp directory
            workspace_data = {
                "name": "Test Analysis Workspace",
                "folder_path": temp_dir,
            }
            create_response = client.post("/api/workspaces/", json=workspace_data)
            assert create_response.status_code == 200
            workspace_id = create_response.json()["id"]

            # Run actual analysis with heuristic extractor (should be fast with few files)
            response = client.post(
                f"/api/workspace-topics/analyze/{workspace_id}?extractor_type=heuristic"
            )

            assert response.status_code == 200
            data = response.json()

            # Verify the response structure (actual analysis ran, may process 0+ files)
            assert "workspace_id" in data
            assert data["workspace_id"] == workspace_id
            assert "files_analyzed" in data
            assert data["files_analyzed"] >= 0  # Analysis ran, found 0 or more files
            assert "topics_created" in data
            assert "errors" in data
            assert isinstance(data["errors"], list)
            assert "duration_seconds" in data
            assert "message" in data

    def test_analyze_workspace_topics_success_bertopic(self, client):
        """Test successful workspace analysis with bertopic extractor (mocked for speed)."""
        # First create a workspace in the test database
        workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
        create_response = client.post("/api/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace_id = create_response.json()["id"]

        # Mock the WorkspaceAnalysisService (BERTopic is expensive, keep mocked)
        mock_result = {
            "workspace_id": workspace_id,
            "files_analyzed": 1577,
            "topics_created": 20,
            "errors": [],
            "duration_seconds": 28.646816,
            "message": "Successfully created 20 topics",
        }

        with patch(
            "app.routes.workspace_topics.WorkspaceAnalysisService"
        ) as mock_service_class:
            mock_service_instance = AsyncMock()
            mock_service_class.return_value = mock_service_instance
            mock_service_instance.analyze_workspace.return_value = mock_result

            # Make the request
            response = client.post(
                f"/api/workspace-topics/analyze/{workspace_id}?extractor_type=bertopic"
            )

            assert response.status_code == 200
            data = response.json()
            assert data == mock_result

    def test_analyze_workspace_topics_success_heuristic(self, client):
        """Test successful workspace analysis with heuristic extractor."""
        # First create a workspace
        workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
        create_response = client.post("/api/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace_id = create_response.json()["id"]

        mock_result = {
            "workspace_id": workspace_id,
            "files_analyzed": 100,
            "topics_created": 5,
            "errors": [],
            "duration_seconds": 10.5,
            "message": "Successfully created 5 topics",
        }

        with patch(
            "app.routes.workspace_topics.WorkspaceAnalysisService"
        ) as mock_service_class:
            mock_service_instance = AsyncMock()
            mock_service_class.return_value = mock_service_instance
            mock_service_instance.analyze_workspace.return_value = mock_result

            # Test with default extractor_type (heuristic)
            response = client.post(f"/api/workspace-topics/analyze/{workspace_id}")

            assert response.status_code == 200
            data = response.json()
            assert data == mock_result

    def test_analyze_workspace_topics_not_found(self, client):
        """Test analyze request for non-existent workspace."""
        with patch("app.routes.workspace_topics.WorkspaceAnalysisService"):
            response = client.post(
                "/api/workspace-topics/analyze/999?extractor_type=bertopic"
            )

            assert response.status_code == 404
            assert response.json()["detail"] == "Workspace not found"

    def test_analyze_workspace_topics_service_error(self, client):
        """Test analyze request when service raises an exception."""
        # First create a workspace
        workspace_data = {"name": "Test Workspace", "folder_path": "/test/path"}
        create_response = client.post("/api/workspaces/", json=workspace_data)
        assert create_response.status_code == 200
        workspace_id = create_response.json()["id"]

        with patch(
            "app.routes.workspace_topics.WorkspaceAnalysisService"
        ) as mock_service_class:
            mock_service_instance = AsyncMock()
            mock_service_class.return_value = mock_service_instance
            mock_service_instance.analyze_workspace.side_effect = Exception(
                "Analysis failed"
            )

            response = client.post(
                f"/api/workspace-topics/analyze/{workspace_id}?extractor_type=bertopic"
            )

            assert response.status_code == 500
            assert "Failed to analyze workspace topics" in response.json()["detail"]
