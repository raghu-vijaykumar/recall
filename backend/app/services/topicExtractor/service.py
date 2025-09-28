"""
Topic Extraction Service for discovering topic areas in workspaces.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .base import BaseTopicExtractor
from .embedding_cluster_extractor import EmbeddingClusterExtractor
from .bertopic_extractor import BERTopicExtractor
from ...models import TopicArea, TopicConceptLink
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
        extractor_type: str = "embedding_cluster",  # "bertopic", "embedding_cluster", "hybrid"
        extractor_config: Optional[Dict[str, Any]] = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.extractor_type = extractor_type
        self.extractor_config = extractor_config or {}

        # Create topic extractor based on type
        if topic_extractor:
            self.topic_extractor = topic_extractor
        else:
            self.topic_extractor = self._create_extractor()

    def _create_extractor(self) -> BaseTopicExtractor:
        """Create topic extractor based on configured type"""
        logging.info(f"Creating {self.extractor_type} topic extractor")

        if self.extractor_type == "bertopic":
            return BERTopicExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )
        elif self.extractor_type == "embedding_cluster":
            return EmbeddingClusterExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )
        elif self.extractor_type == "hybrid":
            # For hybrid, we'll use BERTopic as primary with embedding_cluster as fallback
            return BERTopicExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )
        else:
            logging.warning(
                f"Unknown extractor type: {self.extractor_type}, falling back to embedding_cluster"
            )
            return EmbeddingClusterExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )

    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Extract topic areas from concept data using the configured extractor.

        Args:
            workspace_id: ID of the workspace
            concepts_data: List of concept dictionaries

        Returns:
            Tuple of (topic_areas, concept_links)
        """
        logging.info(
            f"Extracting topics for workspace {workspace_id} with {len(concepts_data)} concepts"
        )

        # Use the configured topic extractor
        topic_areas, concept_links = await self.topic_extractor.extract_topics(
            workspace_id, concepts_data
        )

        # Calculate file counts for each topic area
        for topic_area in topic_areas:
            topic_area.file_count = await self._get_topic_file_count(
                workspace_id, topic_area.topic_area_id
            )

        logging.info(
            f"Extracted {len(topic_areas)} topic areas and {len(concept_links)} concept links for workspace {workspace_id}"
        )
        return topic_areas, concept_links

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

    async def _get_workspace_concept_count(self, workspace_id: int) -> int:
        """Get total number of concepts in workspace"""
        query = text(
            """
            SELECT COUNT(*) FROM concepts c
            JOIN concept_files cf ON c.concept_id = cf.concept_id
            WHERE cf.workspace_id = :workspace_id
            """
        )
        result = await self.db.execute(query, {"workspace_id": workspace_id})
        return result.scalar() or 0

    async def _get_workspace_file_count(self, workspace_id: int) -> int:
        """Get total number of files in workspace"""
        query = text("SELECT COUNT(*) FROM files WHERE workspace_id = :workspace_id")
        result = await self.db.execute(query, {"workspace_id": workspace_id})
        return result.scalar() or 0

    async def calculate_workspace_topic_metrics(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> None:
        """
        Calculate comprehensive metrics for topic areas including coverage and exploration.
        This consolidates the metric calculation logic from workspace_topic_discovery_service.
        """
        logging.info(
            f"Calculating comprehensive metrics for {len(topic_areas)} topic areas"
        )

        for topic_area in topic_areas:
            # Get concepts for this topic area
            topic_concepts = await self._get_topic_concepts(
                workspace_id, topic_area.topic_area_id
            )
            topic_area.concept_count = len(topic_concepts)

            # Get file count for this topic area
            topic_area.file_count = await self._get_topic_file_count_by_concepts(
                workspace_id, topic_concepts
            )

            # Calculate coverage score (based on concept count and file coverage)
            # Normalize by total workspace concepts and files
            total_workspace_concepts = await self._get_workspace_concept_count(
                workspace_id
            )
            total_workspace_files = await self._get_workspace_file_count(workspace_id)

            concept_coverage = (
                topic_area.concept_count / total_workspace_concepts
                if total_workspace_concepts > 0
                else 0
            )
            file_coverage = (
                topic_area.file_count / total_workspace_files
                if total_workspace_files > 0
                else 0
            )

            # Weighted coverage score (using same weights as workspace_topic_discovery_service)
            coverage_weight = 0.6  # Weight for coverage in scoring
            explored_weight = 0.4  # Weight for exploration in scoring
            topic_area.coverage_score = (
                coverage_weight * concept_coverage
                + (1 - coverage_weight) * file_coverage
            )

            # Calculate exploration percentage
            explored_concepts = 0
            for concept in topic_concepts:
                if await self._is_concept_explored(workspace_id, concept["id"]):
                    explored_concepts += 1

            topic_area.explored_percentage = (
                explored_concepts / topic_area.concept_count
                if topic_area.concept_count > 0
                else 0
            )

            logging.info(
                f"Topic area '{topic_area.name}': {topic_area.concept_count} concepts, "
                f"{topic_area.file_count} files, coverage={topic_area.coverage_score:.3f}, "
                f"explored={topic_area.explored_percentage:.1%}"
            )

    async def _get_topic_file_count_by_concepts(
        self, workspace_id: int, concepts: List[Dict[str, Any]]
    ) -> int:
        """Get number of unique files covering the given concepts"""
        if not concepts:
            return 0

        concept_ids = [c["id"] for c in concepts]

        # Use proper parameter binding to avoid SQL injection and formatting issues
        placeholders = ",".join(f":concept_{i}" for i in range(len(concept_ids)))

        query_str = f"""
            SELECT COUNT(DISTINCT cf.file_id)
            FROM concept_files cf
            WHERE cf.concept_id IN ({placeholders})
            AND cf.workspace_id = :workspace_id
        """

        # Create parameter dictionary
        params = {
            f"concept_{i}": concept_id for i, concept_id in enumerate(concept_ids)
        }
        params["workspace_id"] = workspace_id

        query = text(query_str)
        result = await self.db.execute(query, params)
        return result.scalar() or 0
