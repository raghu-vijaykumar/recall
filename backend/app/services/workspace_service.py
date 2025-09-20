"""
Workspace service for managing study workspaces
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from app.models.workspace import (
    Workspace,
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceStats,
)
from app.services.database import DatabaseService


class WorkspaceService:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_workspace(self, workspace_data: WorkspaceCreate) -> Workspace:
        """Create a new workspace"""
        data = {
            "name": workspace_data.name,
            "description": workspace_data.description,
            "type": workspace_data.type.value,
            "color": workspace_data.color,
            "folder_path": workspace_data.folder_path,
        }

        workspace_id = self.db.insert("workspaces", data)
        return self.get_workspace(workspace_id)

    def get_workspace(self, workspace_id: int) -> Optional[Workspace]:
        """Get a workspace by ID with stats"""
        workspace_data = self.db.get_by_id("workspaces", workspace_id)
        if not workspace_data:
            return None

        # Get file count and other stats
        file_count = self.db.count("files", "workspace_id = ?", (workspace_id,))

        # Get question stats
        question_stats = self.db.execute_query(
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

        stats = (
            question_stats[0]
            if question_stats
            else {"total_questions": 0, "completed_questions": None}
        )

        # Get last studied date
        last_studied_result = self.db.execute_query(
            """
            SELECT MAX(timestamp) as last_studied
            FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_completed'
        """,
            (workspace_id,),
        )

        last_studied = None
        if last_studied_result and last_studied_result[0]["last_studied"]:
            last_studied = datetime.fromisoformat(
                last_studied_result[0]["last_studied"]
            )

        return Workspace(
            id=workspace_data["id"],
            name=workspace_data["name"],
            description=workspace_data["description"],
            type=workspace_data["type"],
            color=workspace_data["color"],
            folder_path=workspace_data.get("folder_path"),
            created_at=datetime.fromisoformat(workspace_data["created_at"]),
            updated_at=datetime.fromisoformat(workspace_data["updated_at"]),
            file_count=file_count,
            total_questions=stats["total_questions"] or 0,
            completed_questions=stats["completed_questions"] or 0,
            last_studied=last_studied,
        )

    def get_all_workspaces(self) -> List[Workspace]:
        """Get all workspaces with stats"""
        workspaces_data = self.db.get_all("workspaces")
        return [self.get_workspace(w["id"]) for w in workspaces_data]

    def update_workspace(
        self, workspace_id: int, update_data: WorkspaceUpdate
    ) -> Optional[Workspace]:
        """Update a workspace"""
        data = {}
        if update_data.name is not None:
            data["name"] = update_data.name
        if update_data.description is not None:
            data["description"] = update_data.description
        if update_data.type is not None:
            data["type"] = update_data.type.value
        if update_data.color is not None:
            data["color"] = update_data.color

        if data:
            self.db.update("workspaces", workspace_id, data)

        return self.get_workspace(workspace_id)

    def delete_workspace(self, workspace_id: int) -> bool:
        """Delete a workspace and all associated data"""
        # This will cascade delete files, questions, etc. due to foreign keys
        result = self.db.delete("workspaces", workspace_id)
        return result > 0

    def get_workspace_stats(self, workspace_id: int) -> WorkspaceStats:
        """Get detailed statistics for a workspace"""
        # File stats
        file_stats = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_files,
                SUM(size) as total_size
            FROM files
            WHERE workspace_id = ?
        """,
            (workspace_id,),
        )

        # Question stats
        question_stats = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_questions,
                AVG(CASE WHEN times_asked > 0 THEN times_correct * 1.0 / times_asked ELSE 0 END) as avg_accuracy,
                SUM(times_asked) as total_attempts
            FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = ?
        """,
            (workspace_id,),
        )

        # Quiz session stats
        session_stats = self.db.execute_query(
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

        # Progress stats
        progress_stats = self.db.execute_query(
            """
            SELECT
                COUNT(*) as study_sessions,
                MAX(timestamp) as last_study
            FROM progress
            WHERE workspace_id = ? AND action_type IN ('quiz_started', 'quiz_completed')
        """,
            (workspace_id,),
        )

        f_stats = file_stats[0] if file_stats else {"total_files": 0, "total_size": 0}
        q_stats = (
            question_stats[0]
            if question_stats
            else {"total_questions": 0, "avg_accuracy": 0, "total_attempts": 0}
        )
        s_stats = (
            session_stats[0]
            if session_stats
            else {"total_sessions": 0, "avg_score": 0, "total_time": 0}
        )
        p_stats = (
            progress_stats[0]
            if progress_stats
            else {"study_sessions": 0, "last_study": None}
        )

        return WorkspaceStats(
            workspace_id=workspace_id,
            total_files=f_stats["total_files"],
            total_questions=q_stats["total_questions"] or 0,
            correct_answers=int(
                (q_stats["avg_accuracy"] or 0) * (q_stats["total_attempts"] or 0)
            ),
            incorrect_answers=(q_stats["total_attempts"] or 0)
            - int((q_stats["avg_accuracy"] or 0) * (q_stats["total_attempts"] or 0)),
            study_streak=self._calculate_streak(workspace_id),
            average_score=s_stats["avg_score"] or 0,
            last_study_date=(
                datetime.fromisoformat(p_stats["last_study"])
                if p_stats["last_study"]
                else None
            ),
        )

    def _calculate_streak(self, workspace_id: int) -> int:
        """Calculate current study streak for workspace"""
        # Get recent quiz completions
        recent_sessions = self.db.execute_query(
            """
            SELECT DATE(timestamp) as date
            FROM progress
            WHERE workspace_id = ? AND action_type = 'quiz_completed'
            ORDER BY timestamp DESC
            LIMIT 30
        """,
            (workspace_id,),
        )

        if not recent_sessions:
            return 0

        # Calculate streak (consecutive days)
        streak = 0
        current_date = datetime.now().date()

        for session in recent_sessions:
            session_date = datetime.fromisoformat(session["date"]).date()
            if session_date == current_date:
                streak += 1
                current_date = current_date.replace(day=current_date.day - 1)
            elif session_date == current_date.replace(day=current_date.day - 1):
                streak += 1
                current_date = session_date.replace(day=session_date.day - 1)
            else:
                break

        return streak
