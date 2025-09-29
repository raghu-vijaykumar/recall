"""
Topic Extractor module for discovering topic areas in workspaces.
"""

from .base import BaseTopicExtractor
from .bertopic_extractor import BertTopicExtractor
from .heuristic_extractor import HeuristicTopicExtractor
from .service import TopicExtractionService

__all__ = [
    "BaseTopicExtractor",
    "BertTopicExtractor",
    "HeuristicTopicExtractor",
    "TopicExtractionService",
]
