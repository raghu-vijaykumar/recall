import pytest
from datetime import datetime, timedelta
from backend.app.database.progress import ProgressDatabase
from backend.app.services.database import DatabaseService


class TestProgressDatabase:
    """Test cases for ProgressDatabase class."""

    @pytest.fixture
    def progress_db(self):
        """Create ProgressDatabase instance."""
        db_service = DatabaseService()
        return ProgressDatabase(db_service)

    def test_log_progress(self, progress_db):
        """Test logging a progress event."""
        # Create workspace for foreign key
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Progress Test Workspace", "type": "study"}
        )

        # Log progress event
        progress_data = {
            "user_id": "user123",
            "workspace_id": workspace_id,
            "action_type": "quiz_started",
            "value": 1.0,
            "metadata": '{"question_count": 10}',
        }

        progress_id = progress_db.log_progress(progress_data)
        assert progress_id > 0

    def test_get_progress_by_workspace_empty(self, progress_db):
        """Test getting progress for workspace with no events."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Progress Workspace", "type": "study"}
        )

        # Get progress
        progress_events = progress_db.get_progress_by_workspace(workspace_id)
        assert isinstance(progress_events, list)
        assert len(progress_events) == 0

    def test_get_progress_by_workspace_with_data(self, progress_db):
        """Test getting progress events for workspace."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Progress Events Workspace", "type": "study"}
        )

        # Log multiple progress events
        progress_data = [
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-01T10:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-01T10:30:00",
                "value": 85.0,
            },
            {
                "user_id": "user456",
                "workspace_id": workspace_id,
                "action_type": "file_viewed",
                "timestamp": "2023-01-01T11:00:00",
                "value": 1.0,
            },
        ]

        for data in progress_data:
            progress_db.log_progress(data)

        # Get progress events
        events = progress_db.get_progress_by_workspace(workspace_id, limit=10)
        assert len(events) == 3

        # Should be ordered by timestamp DESC
        assert events[0]["action_type"] == "file_viewed"
        assert events[1]["action_type"] == "quiz_completed"
        assert events[2]["action_type"] == "quiz_started"

    def test_get_progress_by_user_empty(self, progress_db):
        """Test getting progress for user with no events."""
        progress_events = progress_db.get_progress_by_user("nonexistent_user")
        assert isinstance(progress_events, list)
        assert len(progress_events) == 0

    def test_get_progress_by_user_with_data(self, progress_db):
        """Test getting progress events for user."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "User Progress Workspace", "type": "study"}
        )

        # Log progress events for different users
        progress_data = [
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-01T10:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user456",
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-01T10:30:00",
                "value": 85.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "file_viewed",
                "timestamp": "2023-01-01T11:00:00",
                "value": 1.0,
            },
        ]

        for data in progress_data:
            progress_db.log_progress(data)

        # Get progress for user123
        events = progress_db.get_progress_by_user("user123", limit=10)
        assert len(events) == 2

        # Should be ordered by timestamp DESC
        assert events[0]["action_type"] == "file_viewed"
        assert events[1]["action_type"] == "quiz_started"

    def test_get_study_sessions_count_empty(self, progress_db):
        """Test counting study sessions for workspace with no sessions."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Sessions Workspace", "type": "study"}
        )

        count = progress_db.get_study_sessions_count(workspace_id)
        assert count == 0

    def test_get_study_sessions_count_with_data(self, progress_db):
        """Test counting study sessions for workspace."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Sessions Count Workspace", "type": "study"}
        )

        # Log various progress events
        progress_data = [
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-01T10:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-01T10:30:00",
                "value": 85.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "file_viewed",  # Not counted
                "timestamp": "2023-01-01T11:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",  # Another session
                "timestamp": "2023-01-02T10:00:00",
                "value": 1.0,
            },
        ]

        for data in progress_data:
            progress_db.log_progress(data)

        # Count study sessions
        count = progress_db.get_study_sessions_count(workspace_id)
        assert count == 2  # Two quiz_started/quiz_completed events

    def test_get_last_study_date_no_sessions(self, progress_db):
        """Test getting last study date for workspace with no sessions."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "No Study Workspace", "type": "study"}
        )

        last_date = progress_db.get_last_study_date(workspace_id)
        assert last_date is None

    def test_get_last_study_date_with_sessions(self, progress_db):
        """Test getting last study date for workspace with sessions."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Last Study Workspace", "type": "study"}
        )

        # Log progress events with different dates
        progress_data = [
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-01T10:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": "2023-01-03T10:30:00",  # Latest
                "value": 85.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "file_viewed",  # Not counted
                "timestamp": "2023-01-04T11:00:00",
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": "2023-01-02T10:00:00",
                "value": 1.0,
            },
        ]

        for data in progress_data:
            progress_db.log_progress(data)

        # Get last study date
        last_date = progress_db.get_last_study_date(workspace_id)
        assert last_date is not None
        assert last_date.strftime("%Y-%m-%d") == "2023-01-03"

    def test_get_recent_progress_empty(self, progress_db):
        """Test getting recent progress for workspace with no events."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Recent Workspace", "type": "study"}
        )

        recent_events = progress_db.get_recent_progress(workspace_id, days=30)
        assert isinstance(recent_events, list)
        assert len(recent_events) == 0

    def test_get_recent_progress_with_data(self, progress_db):
        """Test getting recent progress events."""
        # Create workspace
        db_service = progress_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Recent Progress Workspace", "type": "study"}
        )

        # Calculate dates
        now = datetime.now()
        recent_date = (now - timedelta(days=5)).isoformat()
        old_date = (now - timedelta(days=35)).isoformat()

        # Log progress events with different dates
        progress_data = [
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_started",
                "timestamp": (now - timedelta(days=5)).isoformat(),
                "value": 1.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "quiz_completed",
                "timestamp": old_date,  # Too old, should be excluded
                "value": 85.0,
            },
            {
                "user_id": "user123",
                "workspace_id": workspace_id,
                "action_type": "file_viewed",
                "timestamp": (now - timedelta(days=4)).isoformat(),
                "value": 1.0,
            },
        ]

        for data in progress_data:
            progress_db.log_progress(data)

        # Get recent progress (last 30 days)
        recent_events = progress_db.get_recent_progress(workspace_id, days=30)
        assert len(recent_events) == 2  # Only recent events

        # Should be ordered by timestamp DESC
        assert recent_events[0]["action_type"] == "file_viewed"
        assert recent_events[1]["action_type"] == "quiz_started"
