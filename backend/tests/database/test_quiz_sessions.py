import pytest
from datetime import datetime
from backend.app.database.quiz_sessions import QuizSessionDatabase
from backend.app.services.database import DatabaseService


class TestQuizSessionDatabase:
    """Test cases for QuizSessionDatabase class."""

    @pytest.fixture
    def quiz_session_db(self):
        """Create QuizSessionDatabase instance."""
        db_service = DatabaseService()
        return QuizSessionDatabase(db_service)

    def test_create_and_get_quiz_session(self, quiz_session_db):
        """Test creating and retrieving a quiz session."""
        # Create workspace first
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Session Test Workspace", "type": "study"}
        )

        # Create test session
        session_data = {
            "workspace_id": workspace_id,
            "question_count": 10,
            "difficulty_filter": "medium",
            "status": "created",
        }

        # Create session
        session_id = quiz_session_db.create_quiz_session(session_data)
        assert session_id > 0

        # Get the session back
        session = quiz_session_db.get_quiz_session(session_id)
        assert session is not None
        assert session["id"] == session_id
        assert session["workspace_id"] == workspace_id
        assert session["question_count"] == 10
        assert session["difficulty_filter"] == "medium"
        assert session["status"] == "created"

    def test_get_quiz_session_not_found(self, quiz_session_db):
        """Test getting a non-existent quiz session."""
        session = quiz_session_db.get_quiz_session(99999)
        assert session is None

    def test_get_sessions_by_workspace_empty(self, quiz_session_db):
        """Test getting sessions for workspace with no sessions."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Sessions Workspace", "type": "study"}
        )

        # Get sessions
        sessions = quiz_session_db.get_sessions_by_workspace(workspace_id)
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_get_sessions_by_workspace_with_data(self, quiz_session_db):
        """Test getting sessions for workspace with sessions."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Sessions Workspace", "type": "study"}
        )

        # Create multiple sessions
        session_data = [
            {"workspace_id": workspace_id, "question_count": 10, "status": "created"},
            {
                "workspace_id": workspace_id,
                "question_count": 15,
                "status": "in_progress",
            },
            {"workspace_id": workspace_id, "question_count": 20, "status": "completed"},
        ]

        for data in session_data:
            quiz_session_db.create_quiz_session(data)

        # Get sessions
        sessions = quiz_session_db.get_sessions_by_workspace(workspace_id)
        assert len(sessions) == 3

        # Should be ordered by created_at DESC
        assert sessions[0]["question_count"] == 20  # Most recent first
        assert sessions[1]["question_count"] == 15
        assert sessions[2]["question_count"] == 10

    def test_update_quiz_session(self, quiz_session_db):
        """Test updating a quiz session."""
        # Create workspace and session
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Update Session Workspace", "type": "study"}
        )

        initial_data = {
            "workspace_id": workspace_id,
            "question_count": 10,
            "status": "created",
        }
        session_id = quiz_session_db.create_quiz_session(initial_data)

        # Update session
        update_data = {
            "question_count": 15,
            "status": "in_progress",
            "correct_answers": 8,
        }
        result = quiz_session_db.update_quiz_session(session_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        session = quiz_session_db.get_quiz_session(session_id)
        assert session["question_count"] == 15
        assert session["status"] == "in_progress"
        assert session["correct_answers"] == 8

    def test_delete_quiz_session(self, quiz_session_db):
        """Test deleting a quiz session."""
        # Create workspace and session
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Delete Session Workspace", "type": "study"}
        )

        session_data = {
            "workspace_id": workspace_id,
            "question_count": 10,
            "status": "created",
        }
        session_id = quiz_session_db.create_quiz_session(session_data)

        # Verify it exists
        session = quiz_session_db.get_quiz_session(session_id)
        assert session is not None

        # Delete session
        result = quiz_session_db.delete_quiz_session(session_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        session = quiz_session_db.get_quiz_session(session_id)
        assert session is None

    def test_start_session(self, quiz_session_db):
        """Test starting a quiz session."""
        # Create workspace and session
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Start Session Workspace", "type": "study"}
        )

        session_data = {
            "workspace_id": workspace_id,
            "question_count": 10,
            "status": "created",
        }
        session_id = quiz_session_db.create_quiz_session(session_data)

        # Start session
        result = quiz_session_db.start_session(session_id)
        assert result == 1  # Should affect 1 row

        # Verify session started
        session = quiz_session_db.get_quiz_session(session_id)
        assert session["status"] == "in_progress"
        assert session["started_at"] is not None

    def test_complete_session(self, quiz_session_db):
        """Test completing a quiz session."""
        # Create workspace and session
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Complete Session Workspace", "type": "study"}
        )

        session_data = {
            "workspace_id": workspace_id,
            "question_count": 10,
            "status": "in_progress",
        }
        session_id = quiz_session_db.create_quiz_session(session_data)

        # Complete session
        result = quiz_session_db.complete_session(session_id, 8, 300)
        assert result == 1  # Should affect 1 row

        # Verify session completed
        session = quiz_session_db.get_quiz_session(session_id)
        assert session["status"] == "completed"
        assert session["correct_answers"] == 8
        assert session["total_time"] == 300
        assert session["completed_at"] is not None

    def test_get_session_stats_empty(self, quiz_session_db):
        """Test getting session stats for workspace with no completed sessions."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Stats Workspace", "type": "study"}
        )

        # Get stats
        stats = quiz_session_db.get_session_stats(workspace_id)
        assert stats["total_sessions"] == 0
        assert stats["avg_score"] == 0
        assert stats["total_time"] == 0

    def test_get_session_stats_with_data(self, quiz_session_db):
        """Test getting session stats for workspace with completed sessions."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Stats Workspace", "type": "study"}
        )

        # Create sessions with different statuses
        session_data = [
            {
                "workspace_id": workspace_id,
                "question_count": 10,
                "total_questions": 10,
                "correct_answers": 8,
                "total_time": 300,
                "status": "completed",
            },
            {
                "workspace_id": workspace_id,
                "question_count": 10,
                "total_questions": 10,
                "correct_answers": 6,
                "total_time": 250,
                "status": "completed",
            },
            {
                "workspace_id": workspace_id,
                "question_count": 10,
                "total_questions": 10,
                "correct_answers": 5,
                "total_time": 200,
                "status": "in_progress",  # Not completed, should not be counted
            },
        ]

        for data in session_data:
            quiz_session_db.create_quiz_session(data)

        # Get stats
        stats = quiz_session_db.get_session_stats(workspace_id)
        assert stats["total_sessions"] == 2  # Only completed sessions
        assert abs(stats["avg_score"] - 0.7) < 0.01  # (8/10 + 6/10) / 2 = 0.7
        assert stats["total_time"] == 550  # 300 + 250

    def test_get_recent_sessions_empty(self, quiz_session_db):
        """Test getting recent sessions for workspace with no sessions."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Recent Workspace", "type": "study"}
        )

        # Get recent sessions
        sessions = quiz_session_db.get_recent_sessions(workspace_id)
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_get_recent_sessions_with_data(self, quiz_session_db):
        """Test getting recent sessions for workspace."""
        # Create workspace
        db_service = quiz_session_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Recent Sessions Workspace", "type": "study"}
        )

        # Create multiple sessions
        session_data = [
            {"workspace_id": workspace_id, "question_count": 10, "status": "created"},
            {
                "workspace_id": workspace_id,
                "question_count": 15,
                "status": "in_progress",
            },
            {"workspace_id": workspace_id, "question_count": 20, "status": "completed"},
            {"workspace_id": workspace_id, "question_count": 25, "status": "created"},
        ]

        for data in session_data:
            quiz_session_db.create_quiz_session(data)

        # Get recent sessions
        sessions = quiz_session_db.get_recent_sessions(workspace_id, limit=3)
        assert len(sessions) == 3

        # Should be ordered by created_at DESC
        assert sessions[0]["question_count"] == 25  # Most recent first
        assert sessions[1]["question_count"] == 20
        assert sessions[2]["question_count"] == 15
