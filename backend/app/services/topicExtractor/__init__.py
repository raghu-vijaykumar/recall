"""
Topic Extractor module for discovering topic areas in workspaces.
"""

from .base import BaseTopicExtractor
from .embedding_cluster_extractor import EmbeddingClusterExtractor
from .service import TopicExtractionService

__all__ = [
    "BaseTopicExtractor",
    "EmbeddingClusterExtractor",
    "TopicExtractionService",
]
