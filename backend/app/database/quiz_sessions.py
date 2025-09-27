"""
Database operations for quiz_sessions table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class QuizSessionDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_quiz_session(self, data: Dict[str, Any]) -> int:
        """Create a new quiz session"""
        return self.db.insert("quiz_sessions", data)

    def get_quiz_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get quiz session by ID"""
        return self.db.get_by_id("quiz_sessions", session_id)

    def get_sessions_by_workspace(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get all quiz sessions for a workspace"""
        return self.db.execute_query(
            "SELECT * FROM quiz_sessions WHERE workspace_id = ? ORDER BY created_at DESC",
            (workspace_id,),
        )

    def update_quiz_session(self, session_id: int, data: Dict[str, Any]) -> int:
        """Update quiz session information"""
        return self.db.update("quiz_sessions", session_id, data)

    def delete_quiz_session(self, session_id: int) -> int:
        """Delete a quiz session"""
        return self.db.delete("quiz_sessions", session_id)

    def start_session(self, session_id: int) -> int:
        """Mark session as started"""
        return self.update_quiz_session(
            session_id, {"started_at": datetime.now(), "status": "in_progress"}
        )

    def complete_session(
        self, session_id: int, correct_answers: int, total_time: int
    ) -> int:
        """Mark session as completed"""
        return self.update_quiz_session(
            session_id,
            {
                "completed_at": datetime.now(),
                "correct_answers": correct_answers,
                "total_time": total_time,
                "status": "completed",
            },
        )

    def get_session_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get session statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_sessions,
                AVG(correct_answers * 1.0 / total_questions) as avg_score,
                SUM(total_time) as total_time
            FROM quiz_sessions
            WHERE workspace_id = ? AND status = 'completed'
        """,
            (workspace_id,),
        )
        return (
            result[0]
            if result
            else {"total_sessions": 0, "avg_score": 0, "total_time": 0}
        )

    def get_recent_sessions(
        self, workspace_id: int, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent quiz sessions for workspace"""
        return self.db.execute_query(
            "SELECT * FROM quiz_sessions WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ?",
            (workspace_id, limit),
        )
