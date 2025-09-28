import pytest
from datetime import datetime
from backend.app.database.workspaces import WorkspaceDatabase
from backend.app.services.database import DatabaseService


class TestWorkspaceDatabase:
    """Test cases for WorkspaceDatabase class."""

    @pytest.fixture
    def workspace_db(self):
        """Create WorkspaceDatabase instance."""
        db_service = DatabaseService()
        return WorkspaceDatabase(db_service)

    def test_create_and_get_workspace(self, workspace_db):
        """Test creating and retrieving a workspace."""
        # Create test workspace
        workspace_data = {
            "name": "Test Workspace",
            "description": "A test workspace",
            "type": "study",
            "color": "#ff0000",
            "folder_path": "/test/path",
        }

        # Create workspace
        workspace_id = workspace_db.create_workspace(workspace_data)
        assert workspace_id > 0

        # Get the workspace back
        workspace = workspace_db.get_workspace(workspace_id)
        assert workspace is not None
        assert workspace["id"] == workspace_id
        assert workspace["name"] == "Test Workspace"
        assert workspace["description"] == "A test workspace"
        assert workspace["type"] == "study"
        assert workspace["color"] == "#ff0000"
        assert workspace["folder_path"] == "/test/path"

    def test_get_workspace_not_found(self, workspace_db):
        """Test getting a non-existent workspace."""
        workspace = workspace_db.get_workspace(99999)
        assert workspace is None

    def test_get_all_workspaces_empty(self, workspace_db):
        """Test getting all workspaces when none exist."""
        workspaces = workspace_db.get_all_workspaces()
        assert isinstance(workspaces, list)
        assert len(workspaces) == 0

    def test_get_all_workspaces_with_data(self, workspace_db):
        """Test getting all workspaces when data exists."""
        # Create multiple workspaces
        workspaces_data = [
            {"name": "Workspace 1", "type": "study"},
            {"name": "Workspace 2", "type": "project"},
            {"name": "Workspace 3", "type": "study"},
        ]

        created_ids = []
        for data in workspaces_data:
            workspace_id = workspace_db.create_workspace(data)
            created_ids.append(workspace_id)

        # Get all workspaces
        workspaces = workspace_db.get_all_workspaces()
        assert len(workspaces) == 3

        # Check that all workspaces are present
        names = [w["name"] for w in workspaces]
        assert "Workspace 1" in names
        assert "Workspace 2" in names
        assert "Workspace 3" in names

    def test_update_workspace(self, workspace_db):
        """Test updating a workspace."""
        # Create initial workspace
        initial_data = {
            "name": "Initial Name",
            "description": "Initial description",
            "type": "study",
        }
        workspace_id = workspace_db.create_workspace(initial_data)

        # Update workspace
        update_data = {
            "name": "Updated Name",
            "description": "Updated description",
            "color": "#00ff00",
        }
        result = workspace_db.update_workspace(workspace_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        workspace = workspace_db.get_workspace(workspace_id)
        assert workspace["name"] == "Updated Name"
        assert workspace["description"] == "Updated description"
        assert workspace["color"] == "#00ff00"
        assert workspace["type"] == "study"  # Should remain unchanged

    def test_delete_workspace(self, workspace_db):
        """Test deleting a workspace."""
        # Create workspace
        workspace_data = {"name": "To Delete", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Verify it exists
        workspace = workspace_db.get_workspace(workspace_id)
        assert workspace is not None

        # Delete workspace
        result = workspace_db.delete_workspace(workspace_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        workspace = workspace_db.get_workspace(workspace_id)
        assert workspace is None

    def test_get_file_count_empty_workspace(self, workspace_db):
        """Test getting file count for workspace with no files."""
        # Create workspace
        workspace_data = {"name": "Empty Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get file count
        count = workspace_db.get_file_count(workspace_id)
        assert count == 0

    def test_get_file_count_with_files(self, workspace_db):
        """Test getting file count for workspace with files."""
        # Create workspace
        workspace_data = {"name": "Workspace with Files", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add files to workspace
        db_service = workspace_db.db
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
        ]

        for data in file_data:
            db_service.insert("files", data)

        # Get file count
        count = workspace_db.get_file_count(workspace_id)
        assert count == 2

    def test_get_question_stats_empty_workspace(self, workspace_db):
        """Test getting question stats for workspace with no questions."""
        # Create workspace
        workspace_data = {"name": "Empty Stats Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get question stats
        stats = workspace_db.get_question_stats(workspace_id)
        assert stats["total_questions"] == 0
        assert stats["completed_questions"] == 0

    def test_get_question_stats_with_data(self, workspace_db):
        """Test getting question stats for workspace with questions."""
        # Create workspace
        workspace_data = {"name": "Stats Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add file to workspace
        db_service = workspace_db.db
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "test.txt",
                "path": "/path/test.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Add questions
        question_data = [
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q1",
                "correct_answer": "A",
                "times_correct": 1,
            },
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q2",
                "correct_answer": "B",
                "times_correct": 0,
            },
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q3",
                "correct_answer": "C",
                "times_correct": 2,
            },
        ]

        for data in question_data:
            db_service.insert("questions", data)

        # Get question stats
        stats = workspace_db.get_question_stats(workspace_id)
        assert stats["total_questions"] == 3
        assert stats["completed_questions"] == 2  # Questions with times_correct > 0

    def test_get_last_studied_no_progress(self, workspace_db):
        """Test getting last studied date when no progress exists."""
        # Create workspace
        workspace_data = {"name": "No Progress Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get last studied
        last_studied = workspace_db.get_last_studied(workspace_id)
        assert last_studied is None

    def test_get_last_studied_with_progress(self, workspace_db):
        """Test getting last studied date with progress data."""
        # Create workspace
        workspace_data = {"name": "Progress Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add progress entries
        db_service = workspace_db.db
        progress_data = [
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-01T10:00:00",
            },
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-03T10:00:00",
            },
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-02T10:00:00",
            },
        ]

        for data in progress_data:
            db_service.insert("progress", data)

        # Get last studied
        last_studied = workspace_db.get_last_studied(workspace_id)
        assert last_studied is not None
        assert last_studied.strftime("%Y-%m-%d") == "2023-01-03"

    def test_get_file_stats_empty_workspace(self, workspace_db):
        """Test getting file stats for workspace with no files."""
        # Create workspace
        workspace_data = {"name": "Empty File Stats", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get file stats
        stats = workspace_db.get_file_stats(workspace_id)
        assert stats["total_files"] == 0
        assert stats["total_size"] == 0

    def test_get_file_stats_with_files(self, workspace_db):
        """Test getting file stats for workspace with files."""
        # Create workspace
        workspace_data = {"name": "File Stats Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add files
        db_service = workspace_db.db
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
                "size": 250,
            },
        ]

        for data in file_data:
            db_service.insert("files", data)

        # Get file stats
        stats = workspace_db.get_file_stats(workspace_id)
        assert stats["total_files"] == 2
        assert stats["total_size"] == 350

    def test_get_question_accuracy_stats_empty(self, workspace_db):
        """Test getting question accuracy stats for empty workspace."""
        # Create workspace
        workspace_data = {"name": "Empty Accuracy", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get accuracy stats
        stats = workspace_db.get_question_accuracy_stats(workspace_id)
        assert stats["total_questions"] == 0
        assert stats["avg_accuracy"] == 0
        assert stats["total_attempts"] == 0

    def test_get_question_accuracy_stats_with_data(self, workspace_db):
        """Test getting question accuracy stats with data."""
        # Create workspace
        workspace_data = {"name": "Accuracy Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add file and questions
        db_service = workspace_db.db
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "test.txt",
                "path": "/path/test.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        question_data = [
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q1",
                "correct_answer": "A",
                "times_asked": 5,
                "times_correct": 3,
            },
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q2",
                "correct_answer": "B",
                "times_asked": 0,
                "times_correct": 0,
            },
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Q3",
                "correct_answer": "C",
                "times_asked": 10,
                "times_correct": 8,
            },
        ]

        for data in question_data:
            db_service.insert("questions", data)

        # Get accuracy stats
        stats = workspace_db.get_question_accuracy_stats(workspace_id)
        assert stats["total_questions"] == 3
        assert stats["total_attempts"] == 15  # 5 + 0 + 10
        # Average accuracy: (3/5 + 0/0 + 8/10) / 3 = (0.6 + 0 + 0.8) / 3 = 1.4 / 3 = 0.466...
        assert abs(stats["avg_accuracy"] - 0.466666) < 0.01

    def test_get_session_stats_empty(self, workspace_db):
        """Test getting session stats for workspace with no sessions."""
        # Create workspace
        workspace_data = {"name": "Empty Sessions", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get session stats
        stats = workspace_db.get_session_stats(workspace_id)
        assert stats["total_sessions"] == 0
        assert stats["avg_score"] == 0
        assert stats["total_time"] == 0

    def test_get_session_stats_with_data(self, workspace_db):
        """Test getting session stats with completed sessions."""
        # Create workspace
        workspace_data = {"name": "Session Stats Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add completed quiz sessions
        db_service = workspace_db.db
        session_data = [
            {
                "workspace_id": workspace_id,
                "total_questions": 10,
                "correct_answers": 8,
                "total_time": 300,
                "status": "completed",
            },
            {
                "workspace_id": workspace_id,
                "total_questions": 10,
                "correct_answers": 6,
                "total_time": 250,
                "status": "completed",
            },
            {
                "workspace_id": workspace_id,
                "total_questions": 10,
                "correct_answers": 5,
                "total_time": 200,
                "status": "in_progress",
            },  # Not completed
        ]

        for data in session_data:
            db_service.insert("quiz_sessions", data)

        # Get session stats
        stats = workspace_db.get_session_stats(workspace_id)
        assert stats["total_sessions"] == 2  # Only completed sessions
        assert abs(stats["avg_score"] - 0.7) < 0.01  # (8/10 + 6/10) / 2 = 0.7
        assert stats["total_time"] == 550  # 300 + 250

    def test_get_progress_stats_empty(self, workspace_db):
        """Test getting progress stats for workspace with no progress."""
        # Create workspace
        workspace_data = {"name": "Empty Progress", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get progress stats
        stats = workspace_db.get_progress_stats(workspace_id)
        assert stats["study_sessions"] == 0
        assert stats["last_study"] is None

    def test_get_progress_stats_with_data(self, workspace_db):
        """Test getting progress stats with progress data."""
        # Create workspace
        workspace_data = {"name": "Progress Stats Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add progress entries
        db_service = workspace_db.db
        progress_data = [
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-01T10:00:00",
            },
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-01T10:30:00",
            },
            {
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-02T11:00:00",
            },
            {
                "workspace_id": workspace_id,
                "action_type": "file_viewed",
                "timestamp": "2023-01-02T11:15:00",
            },  # Not counted
        ]

        for data in progress_data:
            db_service.insert("progress", data)

        # Get progress stats
        stats = workspace_db.get_progress_stats(workspace_id)
        assert stats["study_sessions"] == 2  # quiz_started and quiz_completed count
        assert stats["last_study"] == "2023-01-02T11:00:00"

    def test_get_recent_sessions_for_streak_empty(self, workspace_db):
        """Test getting recent sessions for streak with no data."""
        # Create workspace
        workspace_data = {"name": "Empty Streak", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Get recent sessions
        sessions = workspace_db.get_recent_sessions_for_streak(workspace_id)
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_get_recent_sessions_for_streak_with_data(self, workspace_db):
        """Test getting recent sessions for streak calculation."""
        # Create workspace
        workspace_data = {"name": "Streak Workspace", "type": "study"}
        workspace_id = workspace_db.create_workspace(workspace_data)

        # Add progress entries for different dates
        db_service = workspace_db.db
        dates = ["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-05"]
        for i, date in enumerate(dates):
            db_service.insert(
                "progress",
                {
                    "workspace_id": workspace_id,
                    "action_type": "quiz_completed",
                    "timestamp": f"{date}T10:00:{i:02d}",
                },
            )

        # Get recent sessions
        sessions = workspace_db.get_recent_sessions_for_streak(workspace_id)
        assert len(sessions) == 4  # Unique dates, limited to 30
        # Should be ordered by date descending
        assert sessions[0]["date"] == "2023-01-05"
        assert sessions[1]["date"] == "2023-01-03"
        assert sessions[2]["date"] == "2023-01-02"
        assert sessions[3]["date"] == "2023-01-01"
