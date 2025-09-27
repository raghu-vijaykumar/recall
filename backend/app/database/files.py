"""
Database operations for files table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class FileDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_file(self, data: Dict[str, Any]) -> int:
        """Create a new file"""
        return self.db.insert("files", data)

    def get_file(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get file by ID"""
        return self.db.get_by_id("files", file_id)

    def get_files_by_workspace(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get all files in a workspace"""
        return self.db.execute_query(
            """
            SELECT id, workspace_id, name, path, file_type, size,
                   content_hash, question_count, last_processed,
                   created_at, updated_at
            FROM files WHERE workspace_id = ?
            ORDER BY path
        """,
            (workspace_id,),
        )

    def update_file(self, file_id: int, data: Dict[str, Any]) -> int:
        """Update file information"""
        return self.db.update("files", file_id, data)

    def delete_file(self, file_id: int) -> int:
        """Delete a file"""
        return self.db.delete("files", file_id)

    def check_path_exists(
        self, workspace_id: int, path: str, exclude_id: Optional[int] = None
    ) -> bool:
        """Check if file path already exists in workspace"""
        query = "SELECT id FROM files WHERE workspace_id = ? AND path = ?"
        params = [workspace_id, path]
        if exclude_id:
            query += " AND id != ?"
            params.append(exclude_id)
        result = self.db.execute_query(query, tuple(params))
        return len(result) > 0

    def get_workspace_folder_path(self, workspace_id: int) -> Optional[str]:
        """Get folder path for workspace"""
        result = self.db.execute_query(
            "SELECT folder_path FROM workspaces WHERE id = ?",
            (workspace_id,),
        )
        return result[0]["folder_path"] if result else None

    def update_file_content_hash(
        self, file_id: int, content_hash: str, size: int
    ) -> int:
        """Update file content hash and size"""
        return self.db.update(
            "files",
            file_id,
            {"content_hash": content_hash, "size": size, "updated_at": datetime.now()},
        )

    def get_file_count_by_workspace(self, workspace_id: int) -> int:
        """Get file count for workspace"""
        return self.db.count("files", "workspace_id = ?", (workspace_id,))

    def get_total_size_by_workspace(self, workspace_id: int) -> int:
        """Get total size of files in workspace"""
        result = self.db.execute_query(
            "SELECT SUM(size) as total_size FROM files WHERE workspace_id = ?",
            (workspace_id,),
        )
        return result[0]["total_size"] if result and result[0]["total_size"] else 0
