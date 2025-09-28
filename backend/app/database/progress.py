"""
Database operations for progress table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class ProgressDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def log_progress(self, data: Dict[str, Any]) -> int:
        """Log a progress event"""
        return self.db.insert("progress", data)

    def get_progress_by_workspace(
        self, workspace_id: int, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get progress events for a workspace"""
        return self.db.execute_query(
            "SELECT * FROM progress WHERE workspace_id = ? ORDER BY timestamp DESC LIMIT ?",
            (workspace_id, limit),
        )

    def get_progress_by_user(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get progress events for a user"""
        return self.db.execute_query(
            "SELECT * FROM progress WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        )

    def get_study_sessions_count(self, workspace_id: int) -> int:
        """Count study sessions for workspace"""
        result = self.db.execute_query(
            """
            SELECT COUNT(*) as count FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_started'
        """,
            (workspace_id,),
        )
        return result[0]["count"] if result else 0

    def get_last_study_date(self, workspace_id: int) -> Optional[datetime]:
        """Get last study date for workspace"""
        result = self.db.execute_query(
            """
            SELECT MAX(timestamp) as last_study FROM progress
            WHERE workspace_id = ? AND action_type IN ('quiz_started', 'quiz_completed')
        """,
            (workspace_id,),
        )
        if result and result[0]["last_study"]:
            return datetime.fromisoformat(result[0]["last_study"])
        return None

    def get_recent_progress(
        self, workspace_id: int, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent progress events"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return self.db.execute_query(
            "SELECT * FROM progress WHERE workspace_id = ? AND timestamp >= ? ORDER BY timestamp DESC",
            (workspace_id, cutoff_date.isoformat()),
        )
