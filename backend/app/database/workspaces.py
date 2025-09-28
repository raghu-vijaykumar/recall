"""
Database operations for workspaces table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class WorkspaceDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_workspace(self, data: Dict[str, Any]) -> int:
        """Create a new workspace"""
        return self.db.insert("workspaces", data)

    def get_workspace(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        """Get a workspace by ID"""
        return self.db.get_by_id("workspaces", workspace_id)

    def get_all_workspaces(self) -> List[Dict[str, Any]]:
        """Get all workspaces"""
        return self.db.get_all("workspaces")

    def update_workspace(self, workspace_id: int, data: Dict[str, Any]) -> int:
        """Update a workspace"""
        return self.db.update("workspaces", workspace_id, data)

    def delete_workspace(self, workspace_id: int) -> int:
        """Delete a workspace"""
        return self.db.delete("workspaces", workspace_id)

    def get_file_count(self, workspace_id: int) -> int:
        """Get file count for workspace"""
        return self.db.count("files", "workspace_id = ?", (workspace_id,))

    def get_question_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get question statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_questions,
                SUM(CASE WHEN times_correct > 0 THEN 1 ELSE 0 END) as completed_questions
            FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = ?
        """,
            (workspace_id,),
        )
        if result:
            stats = result[0]
            stats["completed_questions"] = stats["completed_questions"] or 0
            return stats
        else:
            return {"total_questions": 0, "completed_questions": 0}

    def get_last_studied(self, workspace_id: int) -> Optional[datetime]:
        """Get last studied date for workspace"""
        result = self.db.execute_query(
            """
            SELECT MAX(timestamp) as last_studied
            FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_completed'
        """,
            (workspace_id,),
        )
        if result and result[0]["last_studied"]:
            return datetime.fromisoformat(result[0]["last_studied"])
        return None

    def get_file_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get file statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_files,
                COALESCE(SUM(size), 0) as total_size
            FROM files
            WHERE workspace_id = ?
        """,
            (workspace_id,),
        )
        return result[0] if result else {"total_files": 0, "total_size": 0}

    def get_question_accuracy_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get question accuracy statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_questions,
                COALESCE(AVG(CASE WHEN times_asked > 0 THEN times_correct * 1.0 / times_asked ELSE 0 END), 0) as avg_accuracy,
                COALESCE(SUM(times_asked), 0) as total_attempts
            FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = ?
        """,
            (workspace_id,),
        )
        return (
            result[0]
            if result
            else {"total_questions": 0, "avg_accuracy": 0, "total_attempts": 0}
        )

    def get_session_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get quiz session statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_sessions,
                COALESCE(AVG(correct_answers * 1.0 / total_questions), 0) as avg_score,
                COALESCE(SUM(total_time), 0) as total_time
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

    def get_progress_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get progress statistics for workspace"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as study_sessions,
                MAX(timestamp) as last_study
            FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_started'
        """,
            (workspace_id,),
        )
        return result[0] if result else {"study_sessions": 0, "last_study": None}

    def get_recent_sessions_for_streak(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get recent quiz sessions for streak calculation"""
        return self.db.execute_query(
            """
            SELECT DISTINCT DATE(timestamp) as date
            FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_completed'
            ORDER BY date DESC
            LIMIT 30
        """,
            (workspace_id,),
        )
