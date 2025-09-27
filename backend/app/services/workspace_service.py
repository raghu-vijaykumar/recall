"""
Workspace service for managing study workspaces
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import logging

from app.models.workspace import (
    Workspace,
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceStats,
)
from app.database.workspaces import WorkspaceDatabase

logger = logging.getLogger(__name__)


class WorkspaceService:
    def __init__(self, db_service):
        self.workspace_db = WorkspaceDatabase(db_service)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def create_workspace(self, workspace_data: WorkspaceCreate) -> Workspace:
        """Create a new workspace"""
        data = {
            "name": workspace_data.name,
            "description": workspace_data.description,
            "type": workspace_data.type.value,
            "color": workspace_data.color,
            "folder_path": workspace_data.folder_path,
        }

        workspace_id = self.workspace_db.create_workspace(data)
        return self.get_workspace(workspace_id)

    def get_workspace(self, workspace_id: int) -> Optional[Workspace]:
        """Get a workspace by ID with stats"""
        workspace_data = self.workspace_db.get_workspace(workspace_id)
        if not workspace_data:
            return None

        # Get file count and other stats
        file_count = self.workspace_db.get_file_count(workspace_id)
        question_stats = self.workspace_db.get_question_stats(workspace_id)
        last_studied = self.workspace_db.get_last_studied(workspace_id)

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
            total_questions=question_stats["total_questions"] or 0,
            completed_questions=question_stats["completed_questions"] or 0,
            last_studied=last_studied,
        )

    def get_all_workspaces(self) -> List[Workspace]:
        """Get all workspaces with stats"""
        logger.debug("Attempting to retrieve all workspaces.")
        try:
            workspaces_data = self.workspace_db.get_all_workspaces()
            logger.debug(f"Retrieved {len(workspaces_data)} raw workspaces.")
            workspaces = []
            for w in workspaces_data:
                try:
                    workspace = self.get_workspace(w["id"])
                    if workspace:
                        workspaces.append(workspace)
                except Exception as e:
                    logger.error(
                        f"Error getting detailed workspace for ID {w['id']}: {e}"
                    )
            return workspaces
        except Exception as e:
            logger.error(f"Error retrieving all workspaces: {e}")
            raise

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
        if update_data.folder_path is not None:
            data["folder_path"] = update_data.folder_path

        if data:
            self.workspace_db.update_workspace(workspace_id, data)

        return self.get_workspace(workspace_id)

    def delete_workspace(self, workspace_id: int) -> bool:
        """Delete a workspace and all associated data"""
        # This will cascade delete files, questions, etc. due to foreign keys
        result = self.workspace_db.delete_workspace(workspace_id)
        return result > 0

    def get_workspace_stats(self, workspace_id: int) -> WorkspaceStats:
        """Get detailed statistics for a workspace"""
        # File stats
        file_stats = self.workspace_db.get_file_stats(workspace_id)

        # Question stats
        question_stats = self.workspace_db.get_question_accuracy_stats(workspace_id)

        # Quiz session stats
        session_stats = self.workspace_db.get_session_stats(workspace_id)

        # Progress stats
        progress_stats = self.workspace_db.get_progress_stats(workspace_id)

        return WorkspaceStats(
            workspace_id=workspace_id,
            total_files=file_stats["total_files"],
            total_questions=question_stats["total_questions"] or 0,
            correct_answers=int(
                (question_stats["avg_accuracy"] or 0)
                * (question_stats["total_attempts"] or 0)
            ),
            incorrect_answers=(question_stats["total_attempts"] or 0)
            - int(
                (question_stats["avg_accuracy"] or 0)
                * (question_stats["total_attempts"] or 0)
            ),
            study_streak=self._calculate_streak(workspace_id),
            average_score=session_stats["avg_score"] or 0,
            last_study_date=(
                datetime.fromisoformat(progress_stats["last_study"])
                if progress_stats["last_study"]
                else None
            ),
        )

    def _calculate_streak(self, workspace_id: int) -> int:
        """Calculate current study streak for workspace"""
        # Get recent quiz completions
        recent_sessions = self.workspace_db.get_recent_sessions_for_streak(workspace_id)

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
