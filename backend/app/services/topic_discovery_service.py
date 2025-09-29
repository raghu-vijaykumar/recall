"""
Simplified Topics-Only Topic Discovery Service.
Implements the simplified architecture: Files → Preprocessing → TopicExtractor → Topics → Knowledge Graph
"""

import hashlib
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .file_service import FileService
from .topicExtractor.service import TopicExtractionService
from .knowledge_graph_builder import LLMKnowledgeGraphBuilder
from ..models import TopicArea, WorkspaceTopicGraph, WorkspaceTopicAnalysis


class TopicDiscoveryService:
    """
    Simplified topic discovery service that implements topics-only architecture.
    Detects file changes and runs full topic discovery pipeline.
    """

    def __init__(
        self,
        db: AsyncSession,
        topic_extractor_service: Optional[TopicExtractionService] = None,
        kg_builder: Optional[LLMKnowledgeGraphBuilder] = None,
    ):
        self.db = db
        self.file_service = FileService(db)

        # Initialize topic extraction service
        self.topic_extractor_service = (
            topic_extractor_service
            or TopicExtractionService(
                db=self.db,
                extractor_type="bertopic",  # Use BERTopic by default
                extractor_config={
                    "model_name": "all-MiniLM-L6-v2",
                    "min_topic_size": 5,
                    "diversity": 0.3,
                    "coherence_threshold": 0.1,
                },
            )
        )

        # Initialize knowledge graph builder
        self.kg_builder = kg_builder or LLMKnowledgeGraphBuilder(
            db=self.db, llm_config={"provider": "gemini"}
        )

    async def discover_topics_for_workspace(
        self, workspace_id: int, force_refresh: bool = False
    ) -> WorkspaceTopicAnalysis:
        """
        Discover topics for a workspace using the simplified pipeline.

        Args:
            workspace_id: Workspace ID to process
            force_refresh: Force full reprocessing regardless of file changes

        Returns:
            Analysis results with discovered topics
        """
        logging.info(f"Starting topic discovery for workspace {workspace_id}")

        # Check if workspace analysis is needed
        if not force_refresh and not await self._has_workspace_changes(workspace_id):
            logging.info(
                f"No changes detected for workspace {workspace_id}, returning existing analysis"
            )
            return await self._get_existing_analysis(workspace_id)

        # Get all files for the workspace
        workspace_files = await self._get_workspace_files(workspace_id)
        if not workspace_files:
            logging.info(f"No files found for workspace {workspace_id}")
            return self._create_empty_analysis(workspace_id)

        logging.info(
            f"Processing {len(workspace_files)} files for workspace {workspace_id}"
        )

        # Extract file content and metadata
        file_data = await self._prepare_file_data(workspace_files)

        # Run topic extraction directly on file data
        topics = await self.topic_extractor_service.extract_topics(
            workspace_id, file_data
        )

        logging.info(f"Discovered {len(topics)} topics for workspace {workspace_id}")

        # Store topics in database
        stored_topics = await self._store_topics(topics)

        # Build knowledge graph relationships
        if len(stored_topics) > 1:
            relationships = await self.kg_builder.build_topic_relationships(
                stored_topics, workspace_id
            )
            logging.info(f"Built {len(relationships)} topic relationships")
        else:
            relationships = []

        # Mark files as processed
        await self._mark_files_processed(workspace_files)

        # Create analysis result
        analysis = WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=stored_topics,
            total_concepts=0,  # No concepts in simplified architecture
            total_files=len(workspace_files),
            coverage_distribution={
                t.topic_area_id: t.coverage_score for t in stored_topics
            },
            learning_paths=[],  # Will be generated separately if needed
            recommendations=[],  # Will be generated separately if needed
            analysis_timestamp=datetime.now(UTC),
            next_analysis_suggested=datetime.now(UTC) + timedelta(days=7),
        )

        # Store analysis metadata
        await self._store_analysis_result(analysis)

        logging.info(f"Completed topic discovery for workspace {workspace_id}")
        return analysis

    async def _has_workspace_changes(self, workspace_id: int) -> bool:
        """Check if any files in the workspace have changed since last processing."""
        query = text(
            """
            SELECT COUNT(*) as unprocessed_count
            FROM files
            WHERE workspace_id = :workspace_id
              AND (topic_discovery_processed IS NULL OR topic_discovery_processed = FALSE)
        """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
        row = result.fetchone()

        return (row.unprocessed_count if row else 0) > 0

    async def _get_workspace_files(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get all files for a workspace."""
        query = text(
            """
            SELECT id, workspace_id, name, path, file_type, content_hash, size, created_at, updated_at
            FROM files
            WHERE workspace_id = :workspace_id
            ORDER BY updated_at DESC
        """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
        rows = result.fetchall()

        files = []
        for row in rows:
            files.append(
                {
                    "id": row.id,
                    "workspace_id": row.workspace_id,
                    "name": row.name,
                    "path": row.path,
                    "file_type": row.file_type,
                    "content_hash": row.content_hash,
                    "size": row.size,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )

        return files

    async def _prepare_file_data(
        self, files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare file data for topic extraction."""
        file_data = []

        for file_info in files:
            try:
                # Read file content
                content = await self._read_file_content(file_info["path"])

                if content:
                    file_data.append(
                        {
                            "id": file_info["id"],
                            "name": file_info["name"],
                            "path": file_info["path"],
                            "content": content,
                            "file_type": file_info["file_type"],
                            "size": file_info["size"],
                            "hash": file_info["content_hash"],
                        }
                    )
            except Exception as e:
                logging.warning(f"Failed to read file {file_info['path']}: {e}")

        return file_data

    async def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content from disk."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Limit content size for performance
            max_content_length = 50000  # ~50KB per file
            if len(content) > max_content_length:
                content = content[:max_content_length]

            return content

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None

    async def _store_topics(self, topics: List[TopicArea]) -> List[TopicArea]:
        """Store topic areas in database."""
        stored_topics = []

        for topic in topics:
            query = text(
                """
                INSERT OR REPLACE INTO topic_areas
                (topic_area_id, workspace_id, name, description, coverage_score,
                 concept_count, file_count, explored_percentage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            )

            await self.db.execute(
                query,
                (
                    topic.topic_area_id,
                    topic.workspace_id,
                    topic.name,
                    topic.description,
                    topic.coverage_score,
                    topic.concept_count,  # Will be 0 in simplified architecture
                    topic.file_count,
                    topic.explored_percentage,
                    topic.created_at,
                    topic.updated_at,
                ),
            )

            stored_topics.append(topic)

        await self.db.commit()
        return stored_topics

    async def _mark_files_processed(self, files: List[Dict[str, Any]]) -> None:
        """Mark files as processed for topic discovery."""
        now = datetime.now(UTC)

        for file_info in files:
            query = text(
                """
                UPDATE files
                SET topic_discovery_processed = TRUE, topic_discovery_processed_at = ?
                WHERE id = ?
            """
            )

            await self.db.execute(query, (now, file_info["id"]))

        await self.db.commit()

    async def _store_analysis_result(self, analysis: WorkspaceTopicAnalysis) -> None:
        """Store analysis result metadata."""
        # For now, just ensure the data is committed
        # Can be extended to store analysis metadata if needed
        await self.db.commit()

    async def _get_existing_analysis(
        self, workspace_id: int
    ) -> Optional[WorkspaceTopicAnalysis]:
        """Get existing analysis if available."""
        # Get topic areas
        query = text("SELECT * FROM topic_areas WHERE workspace_id = ?")
        result = await self.db.execute(query, (workspace_id,))
        rows = result.fetchall()

        if not rows:
            return None

        topics = []
        for row in rows:
            topics.append(
                TopicArea(
                    topic_area_id=row.topic_area_id,
                    workspace_id=row.workspace_id,
                    name=row.name,
                    description=row.description,
                    coverage_score=row.coverage_score,
                    concept_count=row.concept_count,
                    file_count=row.file_count,
                    explored_percentage=row.explored_percentage,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

        return WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=topics,
            total_concepts=0,
            total_files=0,  # Would need to count if needed
            coverage_distribution={t.topic_area_id: t.coverage_score for t in topics},
            learning_paths=[],
            recommendations=[],
            analysis_timestamp=datetime.now(UTC),
            next_analysis_suggested=datetime.now(UTC) + timedelta(days=7),
        )

    def _create_empty_analysis(self, workspace_id: int) -> WorkspaceTopicAnalysis:
        """Create empty analysis for workspaces with no content."""
        return WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=[],
            total_concepts=0,
            total_files=0,
            coverage_distribution={},
            learning_paths=[],
            recommendations=[],
            analysis_timestamp=datetime.now(UTC),
            next_analysis_suggested=datetime.now(UTC) + timedelta(days=1),
        )

    async def get_workspace_topic_graph(self, workspace_id: int) -> WorkspaceTopicGraph:
        """Get the complete topic graph for a workspace."""
        return await self.kg_builder.get_workspace_topic_graph(workspace_id)
