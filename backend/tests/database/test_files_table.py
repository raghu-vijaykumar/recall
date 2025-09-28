import pytest
from datetime import datetime
from backend.app.database.files import FileDatabase
from backend.app.services.database import DatabaseService


class TestFileDatabase:
    """Test cases for FileDatabase class."""

    @pytest.fixture
    def file_db(self):
        """Create FileDatabase instance."""
        db_service = DatabaseService()
        return FileDatabase(db_service)

    def test_create_and_get_file(self, file_db):
        """Test creating and retrieving a file."""
        # First create a workspace (required for foreign key)
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Test Workspace", "type": "study"}
        )

        # Create test file
        file_data = {
            "workspace_id": workspace_id,
            "name": "test.txt",
            "path": "/path/test.txt",
            "file_type": "text",
            "size": 100,
            "content_hash": "abc123",
            "question_count": 5,
        }

        # Create file
        file_id = file_db.create_file(file_data)
        assert file_id > 0

        # Get the file back
        file = file_db.get_file(file_id)
        assert file is not None
        assert file["id"] == file_id
        assert file["workspace_id"] == workspace_id
        assert file["name"] == "test.txt"
        assert file["path"] == "/path/test.txt"
        assert file["file_type"] == "text"
        assert file["size"] == 100
        assert file["content_hash"] == "abc123"
        assert file["question_count"] == 5

    def test_get_file_not_found(self, file_db):
        """Test getting a non-existent file."""
        file = file_db.get_file(99999)
        assert file is None

    def test_get_files_by_workspace_empty(self, file_db):
        """Test getting files for workspace with no files."""
        # Create workspace
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Workspace", "type": "study"}
        )

        # Get files
        files = file_db.get_files_by_workspace(workspace_id)
        assert isinstance(files, list)
        assert len(files) == 0

    def test_get_files_by_workspace_with_data(self, file_db):
        """Test getting files for workspace with files."""
        # Create workspace
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Files Workspace", "type": "study"}
        )

        # Create multiple files
        file_data = [
            {
                "workspace_id": workspace_id,
                "name": "file1.txt",
                "path": "/path/file1.txt",
                "file_type": "text",
                "size": 100,
            },
            {
                "workspace_id": workspace_id,
                "name": "file2.txt",
                "path": "/path/file2.txt",
                "file_type": "text",
                "size": 200,
            },
            {
                "workspace_id": workspace_id,
                "name": "file3.txt",
                "path": "/path/file3.txt",
                "file_type": "text",
                "size": 150,
            },
        ]

        for data in file_data:
            file_db.create_file(data)

        # Get files
        files = file_db.get_files_by_workspace(workspace_id)
        assert len(files) == 3

        # Should be ordered by path
        assert files[0]["name"] == "file1.txt"
        assert files[1]["name"] == "file2.txt"
        assert files[2]["name"] == "file3.txt"

    def test_update_file(self, file_db):
        """Test updating a file."""
        # Create workspace and file
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Update Workspace", "type": "study"}
        )

        initial_data = {
            "workspace_id": workspace_id,
            "name": "old_name.txt",
            "path": "/path/old_name.txt",
            "file_type": "text",
            "size": 100,
            "question_count": 0,
        }
        file_id = file_db.create_file(initial_data)

        # Update file
        update_data = {
            "name": "new_name.txt",
            "path": "/path/new_name.txt",
            "size": 150,
            "question_count": 10,
        }
        result = file_db.update_file(file_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        file = file_db.get_file(file_id)
        assert file["name"] == "new_name.txt"
        assert file["path"] == "/path/new_name.txt"
        assert file["size"] == 150
        assert file["question_count"] == 10

    def test_delete_file(self, file_db):
        """Test deleting a file."""
        # Create workspace and file
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Delete Workspace", "type": "study"}
        )

        file_data = {
            "workspace_id": workspace_id,
            "name": "to_delete.txt",
            "path": "/path/to_delete.txt",
            "file_type": "text",
            "size": 100,
        }
        file_id = file_db.create_file(file_data)

        # Verify it exists
        file = file_db.get_file(file_id)
        assert file is not None

        # Delete file
        result = file_db.delete_file(file_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        file = file_db.get_file(file_id)
        assert file is None

    def test_check_path_exists(self, file_db):
        """Test checking if file path exists."""
        # Create workspace
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Path Check Workspace", "type": "study"}
        )

        # Create a file
        file_data = {
            "workspace_id": workspace_id,
            "name": "existing.txt",
            "path": "/path/existing.txt",
            "file_type": "text",
            "size": 100,
        }
        file_id = file_db.create_file(file_data)

        # Check path exists
        assert file_db.check_path_exists(workspace_id, "/path/existing.txt") is True
        assert file_db.check_path_exists(workspace_id, "/path/nonexistent.txt") is False

        # Check with exclude_id
        assert (
            file_db.check_path_exists(
                workspace_id, "/path/existing.txt", exclude_id=file_id
            )
            is False
        )

    def test_get_workspace_folder_path(self, file_db):
        """Test getting workspace folder path."""
        # Create workspace with folder path
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces",
            {
                "name": "Folder Path Workspace",
                "type": "study",
                "folder_path": "/workspace/folder",
            },
        )

        # Get folder path
        folder_path = file_db.get_workspace_folder_path(workspace_id)
        assert folder_path == "/workspace/folder"

    def test_get_workspace_folder_path_not_found(self, file_db):
        """Test getting folder path for non-existent workspace."""
        folder_path = file_db.get_workspace_folder_path(99999)
        assert folder_path is None

    def test_update_file_content_hash(self, file_db):
        """Test updating file content hash and size."""
        # Create workspace and file
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Hash Update Workspace", "type": "study"}
        )

        file_data = {
            "workspace_id": workspace_id,
            "name": "hash_test.txt",
            "path": "/path/hash_test.txt",
            "file_type": "text",
            "size": 100,
            "content_hash": "old_hash",
        }
        file_id = file_db.create_file(file_data)

        # Update content hash
        result = file_db.update_file_content_hash(file_id, "new_hash", 150)
        assert result == 1  # Should affect 1 row

        # Verify update
        file = file_db.get_file(file_id)
        assert file["content_hash"] == "new_hash"
        assert file["size"] == 150

    def test_get_file_count_by_workspace(self, file_db):
        """Test getting file count for workspace."""
        # Create workspace
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Count Workspace", "type": "study"}
        )

        # Initially should be 0
        count = file_db.get_file_count_by_workspace(workspace_id)
        assert count == 0

        # Add files
        file_data = [
            {
                "workspace_id": workspace_id,
                "name": "file1.txt",
                "path": "/path/file1.txt",
                "file_type": "text",
                "size": 100,
            },
            {
                "workspace_id": workspace_id,
                "name": "file2.txt",
                "path": "/path/file2.txt",
                "file_type": "text",
                "size": 200,
            },
            {
                "workspace_id": workspace_id,
                "name": "file3.txt",
                "path": "/path/file3.txt",
                "file_type": "text",
                "size": 150,
            },
        ]

        for data in file_data:
            file_db.create_file(data)

        # Check count
        count = file_db.get_file_count_by_workspace(workspace_id)
        assert count == 3

    def test_get_total_size_by_workspace(self, file_db):
        """Test getting total size of files in workspace."""
        # Create workspace
        db_service = file_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Size Workspace", "type": "study"}
        )

        # Initially should be 0
        total_size = file_db.get_total_size_by_workspace(workspace_id)
        assert total_size == 0

        # Add files
        file_data = [
            {
                "workspace_id": workspace_id,
                "name": "file1.txt",
                "path": "/path/file1.txt",
                "file_type": "text",
                "size": 100,
            },
            {
                "workspace_id": workspace_id,
                "name": "file2.txt",
                "path": "/path/file2.txt",
                "file_type": "text",
                "size": 200,
            },
            {
                "workspace_id": workspace_id,
                "name": "file3.txt",
                "path": "/path/file3.txt",
                "file_type": "text",
                "size": 150,
            },
        ]

        for data in file_data:
            file_db.create_file(data)

        # Check total size
        total_size = file_db.get_total_size_by_workspace(workspace_id)
        assert total_size == 450  # 100 + 200 + 150
