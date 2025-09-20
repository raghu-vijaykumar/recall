"""
File service for managing files in workspaces
"""

import os
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from app.models.file import (
    FileItem,
    FileCreate,
    FileUpdate,
    FileType,
    FileTreeNode,
)
from app.services.database import DatabaseService


class FileService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def _get_file_type(self, filename: str) -> FileType:
        """Determine file type based on extension"""
        ext = Path(filename).suffix.lower()
        type_map = {
            ".txt": FileType.TEXT,
            ".md": FileType.MARKDOWN,
            ".markdown": FileType.MARKDOWN,
            ".py": FileType.CODE,
            ".js": FileType.CODE,
            ".ts": FileType.CODE,
            ".html": FileType.CODE,
            ".css": FileType.CODE,
            ".json": FileType.CODE,
            ".xml": FileType.CODE,
            ".pdf": FileType.PDF,
            ".jpg": FileType.IMAGE,
            ".jpeg": FileType.IMAGE,
            ".png": FileType.IMAGE,
            ".gif": FileType.IMAGE,
        }
        return type_map.get(ext, FileType.OTHER)

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def create_file(self, file_data: FileCreate) -> FileItem:
        """Create a new file"""
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()

            # Check if file path already exists in workspace
            cursor.execute(
                "SELECT id FROM files WHERE workspace_id = ? AND path = ?",
                (file_data.workspace_id, file_data.path),
            )
            if cursor.fetchone():
                raise ValueError(
                    f"File path '{file_data.path}' already exists in workspace"
                )

            # Calculate content hash if content provided
            content_hash = None
            if file_data.content:
                content_hash = self._calculate_content_hash(file_data.content)

            # Insert file record
            cursor.execute(
                """
                INSERT INTO files (
                    workspace_id, name, path, file_type, size,
                    content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_data.workspace_id,
                    file_data.name,
                    file_data.path,
                    file_data.file_type.value,
                    file_data.size,
                    content_hash,
                    datetime.now(),
                    datetime.now(),
                ),
            )

            file_id = cursor.lastrowid

            # If content provided, we could store it in a separate content table
            # For now, we'll assume content is stored elsewhere or handled by frontend

            conn.commit()

            # Return created file
            return self.get_file(file_id)

    def get_file(self, file_id: int) -> Optional[FileItem]:
        """Get file by ID"""
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, workspace_id, name, path, file_type, size,
                       content_hash, question_count, last_processed,
                       created_at, updated_at
                FROM files WHERE id = ?
                """,
                (file_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return FileItem(
                id=row[0],
                workspace_id=row[1],
                name=row[2],
                path=row[3],
                file_type=FileType(row[4]),
                size=row[5],
                content_hash=row[6],
                question_count=row[7] or 0,
                last_processed=row[8],
                created_at=row[9],
                updated_at=row[10],
            )

    def get_files_by_workspace(self, workspace_id: int) -> List[FileItem]:
        """Get all files in a workspace"""
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, workspace_id, name, path, file_type, size,
                       content_hash, question_count, last_processed,
                       created_at, updated_at
                FROM files WHERE workspace_id = ?
                ORDER BY path
                """,
                (workspace_id,),
            )
            rows = cursor.fetchall()

            files = []
            for row in rows:
                files.append(
                    FileItem(
                        id=row[0],
                        workspace_id=row[1],
                        name=row[2],
                        path=row[3],
                        file_type=FileType(row[4]),
                        size=row[5],
                        content_hash=row[6],
                        question_count=row[7] or 0,
                        last_processed=row[8],
                        created_at=row[9],
                        updated_at=row[10],
                    )
                )

            return files

    def update_file(self, file_id: int, update_data: FileUpdate) -> Optional[FileItem]:
        """Update file information"""
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()

            # Check if file exists
            cursor.execute("SELECT id FROM files WHERE id = ?", (file_id,))
            if not cursor.fetchone():
                return None

            # Build update query dynamically
            update_fields = []
            values = []

            if update_data.name is not None:
                update_fields.append("name = ?")
                values.append(update_data.name)

            if update_data.path is not None:
                # Check if new path conflicts
                cursor.execute(
                    "SELECT id FROM files WHERE workspace_id = (SELECT workspace_id FROM files WHERE id = ?) AND path = ? AND id != ?",
                    (file_id, update_data.path, file_id),
                )
                if cursor.fetchone():
                    raise ValueError(f"File path '{update_data.path}' already exists")
                update_fields.append("path = ?")
                values.append(update_data.path)

            if update_data.content is not None:
                content_hash = self._calculate_content_hash(update_data.content)
                update_fields.append("content_hash = ?")
                values.append(content_hash)

            if not update_fields:
                return self.get_file(file_id)

            # Add updated_at
            update_fields.append("updated_at = ?")
            values.append(datetime.now())
            values.append(file_id)

            query = f"UPDATE files SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()

            return self.get_file(file_id)

    def delete_file(self, file_id: int) -> bool:
        """Delete a file"""
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_file_tree(self, workspace_id: int) -> List[FileTreeNode]:
        """Get file tree structure for workspace"""
        files = self.get_files_by_workspace(workspace_id)

        # Build tree structure
        tree_dict: Dict[str, FileTreeNode] = {}

        for file in files:
            path_parts = file.path.split("/")
            current_path = ""

            for i, part in enumerate(path_parts):
                current_path = f"{current_path}/{part}" if current_path else part
                is_file = i == len(path_parts) - 1

                if current_path not in tree_dict:
                    tree_dict[current_path] = FileTreeNode(
                        name=part,
                        path=current_path,
                        type="file" if is_file else "directory",
                        children=[] if not is_file else None,
                        file_info=file if is_file else None,
                    )

                # Add to parent
                if i > 0:
                    parent_path = "/".join(path_parts[:i])
                    if parent_path in tree_dict:
                        if tree_dict[parent_path].children is None:
                            tree_dict[parent_path].children = []
                        if (
                            tree_dict[current_path]
                            not in tree_dict[parent_path].children
                        ):
                            tree_dict[parent_path].children.append(
                                tree_dict[current_path]
                            )

        # Return root level items
        root_items = []
        for path, node in tree_dict.items():
            if "/" not in path:
                root_items.append(node)

        return root_items

    def get_file_content(self, file_id: int) -> Optional[str]:
        """Get file content (placeholder - would need content storage)"""
        # In a real implementation, you'd store file content in the database
        # or file system and retrieve it here
        # For now, return a placeholder
        file = self.get_file(file_id)
        if file:
            return f"// Content for {file.name}\n// This is a placeholder\n"
        return None

    def save_file_content(self, file_id: int, content: str) -> bool:
        """Save file content (placeholder)"""
        # In a real implementation, you'd save the content
        # and update the hash and size
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()

            content_hash = self._calculate_content_hash(content)
            size = len(content.encode("utf-8"))

            cursor.execute(
                "UPDATE files SET content_hash = ?, size = ?, updated_at = ? WHERE id = ?",
                (content_hash, size, datetime.now(), file_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def scan_workspace_folder(self, workspace_id: int) -> List[FileItem]:
        """Scan folder associated with workspace and create files"""
        # Get workspace to find folder path
        with self.db_service.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT folder_path FROM workspaces WHERE id = ?", (workspace_id,)
            )
            row = cursor.fetchone()

            if not row or not row[0]:
                raise ValueError("Workspace does not have an associated folder")

            folder_path = row[0]

        # Check if folder exists
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Scan folder for files
        created_files = []
        for root, dirs, files in os.walk(folder_path):
            # Skip common directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", ".git"]
            ]

            for file in files:
                file_path = os.path.join(root, file)

                # Get relative path from workspace folder
                rel_path = os.path.relpath(file_path, folder_path)

                # Check file type
                file_type = self._get_file_type(file)
                if file_type == FileType.OTHER:
                    continue  # Skip unsupported file types

                try:
                    # Read file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create file in database
                    file_data = FileCreate(
                        name=file,
                        path=rel_path,
                        file_type=file_type,
                        size=len(content.encode("utf-8")),
                        workspace_id=workspace_id,
                        content=content,
                    )

                    created_file = self.create_file(file_data)
                    created_files.append(created_file)

                except (UnicodeDecodeError, IOError) as e:
                    # Skip files that can't be read as text
                    print(f"Skipping file {file_path}: {e}")
                    continue
                except ValueError as e:
                    # Skip files that already exist
                    if "already exists" in str(e):
                        print(f"File already exists: {rel_path}")
                        continue
                    else:
                        raise

        return created_files
