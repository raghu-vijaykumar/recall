import pytest
import os
import tempfile


def test_search_content_empty_query(client):
    """Test search with empty query."""
    request_data = {"workspace_id": 1, "query": "", "folder_path": "/tmp"}
    response = client.post("/api/search/content", json=request_data)
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_search_content_invalid_path(client):
    """Test search with invalid folder path."""
    request_data = {
        "workspace_id": 1,
        "query": "test",
        "folder_path": "/nonexistent/path",
    }
    response = client.post("/api/search/content", json=request_data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_search_content_no_matches(client):
    """Test search that returns no matches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with content that won't match
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("This is some test content that won't match our search.")

        request_data = {
            "workspace_id": 1,
            "query": "nonexistentword",
            "folder_path": temp_dir,
        }
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        assert response.json() == []


def test_search_content_with_matches(client):
    """Test search that finds matches."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with content that will match
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write(
                "This is some test content.\nAnother line with test in it.\nNo match here."
            )

        request_data = {"workspace_id": 1, "query": "test", "folder_path": temp_dir}
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["name"] == "test.txt"
        assert len(results[0]["matches"]) == 2  # Two lines contain "test"


def test_search_content_case_insensitive(client):
    """Test that search is case insensitive."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("This is some TEST content.")

        request_data = {
            "workspace_id": 1,
            "query": "test",  # lowercase
            "folder_path": temp_dir,
        }
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1


def test_search_content_multiple_files(client):
    """Test search across multiple files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test files
        file1 = os.path.join(temp_dir, "file1.txt")
        file2 = os.path.join(temp_dir, "file2.txt")
        file3 = os.path.join(temp_dir, "file3.txt")

        with open(file1, "w") as f:
            f.write("This has the search term.")
        with open(file2, "w") as f:
            f.write("This also has the search term.")
        with open(file3, "w") as f:
            f.write("This does not have it.")

        request_data = {
            "workspace_id": 1,
            "query": "search term",
            "folder_path": temp_dir,
        }
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2  # Two files match
        file_names = [r["name"] for r in results]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "file3.txt" not in file_names


def test_search_content_skips_binary_files(client):
    """Test that binary files are skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a binary file (containing null bytes)
        binary_file = os.path.join(temp_dir, "binary.dat")
        with open(binary_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03text\x00\x00")

        request_data = {"workspace_id": 1, "query": "text", "folder_path": temp_dir}
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        # Binary file should be skipped, so no results
        assert response.json() == []


def test_search_content_skips_node_modules(client):
    """Test that node_modules directory is skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create node_modules directory with a file
        node_modules_dir = os.path.join(temp_dir, "node_modules")
        os.makedirs(node_modules_dir)
        test_file = os.path.join(node_modules_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("This should not be found because it's in node_modules.")

        request_data = {"workspace_id": 1, "query": "found", "folder_path": temp_dir}
        response = client.post("/api/search/content", json=request_data)
        assert response.status_code == 200
        # Should not find the file in node_modules
        assert response.json() == []
