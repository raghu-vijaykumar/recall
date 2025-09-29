"""
Topic Extraction Service for discovering topic areas in workspaces.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from .base import BaseTopicExtractor
from .heuristic_extractor import HeuristicExtractor
from .bertopic_extractor import BERTopicExtractor
from ...models import TopicArea
from ..embedding_service import EmbeddingService


class TopicExtractionService:
    """
    Service for extracting topic areas directly from workspace files.
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
            return HeuristicExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )
        elif self.extractor_type == "hybrid":
            # For hybrid, we'll use BERTopic as primary with embedding_cluster as fallback
            return BERTopicExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )
        else:
            logging.warning(
                f"Unknown extractor type: {self.extractor_type}, falling back to heuristic"
            )
            return HeuristicExtractor(
                embedding_service=self.embedding_service, **self.extractor_config
            )

    async def extract_topics(
        self, workspace_id: int, file_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas directly from file data using the configured extractor.

        Args:
            workspace_id: ID of the workspace
            file_data: List of file dictionaries

        Returns:
            List of discovered topic areas
        """
        logging.info(
            f"Extracting topics for workspace {workspace_id} with {len(file_data)} documents"
        )

        # Use the configured topic extractor
        result = await self.topic_extractor.extract_topics(workspace_id, file_data)

        # Handle both tuple and list returns
        if isinstance(result, tuple):
            topic_areas, _ = result
        else:
            topic_areas = result

        # Calculate file counts and metrics for each topic area
        for topic_area in topic_areas:
            # For topics-only architecture, file count is parsed from topic description or set to 0
            topic_area.file_count = (
                0  # Will be calculated by higher-level service if needed
            )
            topic_area.explored_percentage = 0.0  # Topics start unexplored

        logging.info(
            f"Extracted {len(topic_areas)} topic areas for workspace {workspace_id}"
        )
        return topic_areas
