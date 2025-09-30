"""
Streamlined Workspace Analysis Service for topic extraction from workspace files.
Analyzes workspace folder content and creates topic areas in the database.
Uses pluggable topic extractors for flexibility.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import TopicArea
from .topicExtractor.service import TopicExtractionService
from .document_preprocessing import create_document_preprocessing_service
from ..database.workspace_topics import WorkspaceTopicsDatabase


class WorkspaceAnalysisService:
    """
    Streamlined service for analyzing workspace folders to extract topic areas.
    Reads workspace files, processes them, and creates topic areas in the database.
    Uses pluggable topic extractors (HeuristicExtractor or BERTopicExtractor).
    """

    def __init__(
        self,
        db: AsyncSession,
        extractor_type: str = "heuristic",  # "heuristic" or "bertopic"
        extractor_config: Optional[Dict[str, Any]] = None,
    ):
        self.db = db
        self.extractor_type = extractor_type
        self.extractor_config = extractor_config or {}

        self.topic_service = TopicExtractionService(
            db=db, extractor_type=extractor_type, extractor_config=self.extractor_config
        )
        self.workspace_topics_db = WorkspaceTopicsDatabase(db)
        self.preprocessing_service = create_document_preprocessing_service()

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding handling"""
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Try with errors='ignore'
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        except Exception:
            return ""

    async def analyze_workspace(
        self,
        workspace_id: int,
        workspace_path: str,
    ) -> Dict[str, Any]:
        """
        Analyze a workspace folder to extract topic areas.

        Args:
            workspace_id: ID of the workspace to analyze
            workspace_path: Path to the workspace directory

        Returns:
            Analysis results with topic statistics
        """
        start_time = datetime.utcnow()
        logging.info(
            f"[TOPIC_ANALYSIS] Starting topic analysis for workspace {workspace_id} at {workspace_path}"
        )

        try:
            # Preprocess workspace files using the generic preprocessing service
            processed_docs, file_metadata = (
                await self.preprocessing_service.preprocess_workspace_files(
                    workspace_path
                )
            )

            if not processed_docs or not file_metadata:
                return {
                    "workspace_id": workspace_id,
                    "files_analyzed": 0,
                    "topics_created": 0,
                    "errors": ["No files found or processed in workspace"],
                    "duration_seconds": 0.0,
                    "message": "No files found",
                }

            # Extract topics using the topic extractor
            logging.info(
                f"[TOPIC_ANALYSIS] Extracting topics from {len(processed_docs)} processed documents"
            )
            topic_areas = await self.topic_service.extract_topics(
                workspace_id, file_metadata
            )

            # Store topics in database
            stored_topics = []
            for topic_area in topic_areas:
                try:
                    await self.workspace_topics_db.store_topic_area(topic_area)
                    stored_topics.append(topic_area)
                    logging.info(f"[TOPIC_ANALYSIS] Stored topic: {topic_area.name}")
                except Exception as e:
                    logging.error(
                        f"[TOPIC_ANALYSIS] Failed to store topic {topic_area.name}: {e}"
                    )

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            logging.info(
                f"[TOPIC_ANALYSIS] Topic analysis completed for workspace {workspace_id} in {duration:.2f} seconds. "
                f"Created {len(stored_topics)} topics from {len(processed_docs)} processed documents."
            )

            return {
                "workspace_id": workspace_id,
                "files_analyzed": len(processed_docs),
                "topics_created": len(stored_topics),
                "errors": [],
                "duration_seconds": duration,
                "message": f"Successfully created {len(stored_topics)} topics",
            }

        except Exception as e:
            logging.error(
                f"[TOPIC_ANALYSIS] Error analyzing workspace {workspace_id}: {e}"
            )
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "topics_created": 0,
                "errors": [str(e)],
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "message": "Analysis failed",
            }

    async def _get_workspace_file_data(
        self, workspace_id: int, workspace_path: str
    ) -> List[Dict[str, Any]]:
        """
        Get all relevant file data from the workspace for topic extraction.

        Args:
            workspace_id: ID of the workspace
            workspace_path: Path to the workspace directory

        Returns:
            List of file data dictionaries
        """
        workspace_dir = Path(workspace_path)
        if not workspace_dir.exists() or not workspace_dir.is_dir():
            logging.warning(f"Workspace directory does not exist: {workspace_path}")
            return []

        file_data = []

        # Get all relevant files (exclude common non-text files)
        exclude_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".svg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".pyc",
            "__pycache__",
            ".git",
        }
        exclude_names = {"node_modules", ".git", "__pycache__", ".vscode"}

        for file_path in workspace_dir.rglob("*"):
            if file_path.is_file():
                # Skip excluded files
                if (
                    file_path.name in exclude_names
                    or file_path.name.startswith(".")
                    or any(part in exclude_names for part in file_path.parts)
                ):
                    continue

                ext = file_path.suffix.lower()
                if ext in exclude_extensions:
                    continue

                try:
                    # Read file content
                    content = self._read_file_content(file_path)
                    if content and len(content.split()) > 5:  # At least 5 words
                        file_info = {
                            "id": str(file_path.relative_to(workspace_dir)),
                            "file_path": str(file_path.relative_to(workspace_dir)),
                            "name": file_path.name,
                            "content": content,
                            "file_type": file_path.suffix.lstrip("."),
                            "size": file_path.stat().st_size,
                        }
                        file_data.append(file_info)
                except Exception as e:
                    logging.debug(f"Skipping file {file_path}: {e}")

        logging.info(f"Collected {len(file_data)} files from workspace {workspace_id}")
        return file_data
