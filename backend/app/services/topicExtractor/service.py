"""
Topic Extraction Service for discovering topic areas in workspaces.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .base import BaseTopicExtractor
from .embedding_cluster_extractor import EmbeddingClusterExtractor
from ...models import TopicArea
from ..embedding_service import EmbeddingService


class TopicExtractionService:
    """
    Service for extracting topic areas from workspace concepts.
    Orchestrates different topic extraction strategies.
    """

    def __init__(
        self,
        db: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
        topic_extractor: Optional[BaseTopicExtractor] = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.topic_extractor = topic_extractor or EmbeddingClusterExtractor(
            embedding_service=self.embedding_service
        )

    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas from concept data using the configured extractor.

        Args:
            workspace_id: ID of the workspace
            concepts_data: List of concept dictionaries

        Returns:
            List of extracted TopicArea objects
        """
        logging.info(
            f"Extracting topics for workspace {workspace_id} with {len(concepts_data)} concepts"
        )

        # Use the configured topic extractor
        topic_areas = await self.topic_extractor.extract_topics(
            workspace_id, concepts_data
        )

        # Calculate file counts for each topic area
        for topic_area in topic_areas:
            topic_area.file_count = await self._get_topic_file_count(
                workspace_id, topic_area.topic_area_id
            )

        logging.info(
            f"Extracted {len(topic_areas)} topic areas for workspace {workspace_id}"
        )
        return topic_areas

    async def _get_topic_file_count(self, workspace_id: int, topic_area_id: str) -> int:
        """
        Get the number of unique files associated with concepts in a topic area.
        Note: This is a simplified implementation. In practice, you'd need to
        link topic areas to concepts and then to files.
        """
        # For now, return a placeholder. In the full implementation,
        # this would query the database for files associated with the topic area's concepts
        return 0

    async def calculate_topic_metrics(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> None:
        """
        Calculate coverage scores and exploration percentages for topic areas.
        """
        for topic_area in topic_areas:
            # Get concepts for this topic area
            topic_concepts = await self._get_topic_concepts(
                workspace_id, topic_area.topic_area_id
            )

            # Calculate exploration percentage based on quiz performance
            explored_count = 0
            for concept in topic_concepts:
                # Check if concept has been quizzed recently or has high performance
                if await self._is_concept_explored(workspace_id, concept["id"]):
                    explored_count += 1

            topic_area.explored_percentage = (
                explored_count / len(topic_concepts) if topic_concepts else 0.0
            )

            # Update coverage score based on multiple factors
            base_coverage = topic_area.coverage_score
            exploration_bonus = topic_area.explored_percentage * 0.2
            topic_area.coverage_score = min(1.0, base_coverage + exploration_bonus)

    async def _get_topic_concepts(
        self, workspace_id: int, topic_area_id: str
    ) -> List[Dict[str, Any]]:
        """Get concepts associated with a topic area"""
        query = text(
            """
            SELECT c.*, tcl.relevance_score
            FROM concepts c
            JOIN topic_concept_links tcl ON c.concept_id = tcl.concept_id
            WHERE tcl.topic_area_id = :topic_area_id
            """
        )

        result = await self.db.execute(query, {"topic_area_id": topic_area_id})
        rows = result.fetchall()

        concepts = []
        for row in rows:
            concepts.append(
                {
                    "id": row.concept_id,
                    "name": row.name,
                    "description": row.description,
                    "relevance_score": row.relevance_score,
                }
            )

        return concepts

    async def _is_concept_explored(self, workspace_id: int, concept_id: str) -> bool:
        """Check if a concept has been sufficiently explored"""
        # Check quiz performance for this concept
        query = text(
            """
            SELECT COUNT(*) as quiz_count,
                   AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as avg_score
            FROM questions q
            JOIN answers a ON q.id = a.question_id
            WHERE q.kg_concept_ids LIKE :concept_pattern
            AND q.file_id IN (
                SELECT id FROM files WHERE workspace_id = :workspace_id
            )
            """
        )

        # SQLite doesn't support complex LIKE patterns easily, so we'll check differently
        concept_pattern = f"%{concept_id}%"

        result = await self.db.execute(
            query, {"concept_pattern": concept_pattern, "workspace_id": workspace_id}
        )

        row = result.fetchone()
        if row and row.quiz_count and row.quiz_count > 0:
            # Consider explored if quizzed at least 3 times with >70% accuracy
            return row.quiz_count >= 3 and (row.avg_score or 0) > 0.7

        return False
