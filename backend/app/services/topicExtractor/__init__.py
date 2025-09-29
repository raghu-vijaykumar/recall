"""
Topic Extractor module for discovering topic areas in workspaces.
"""

from .base import BaseTopicExtractor
from .bertopic_extractor import BERTopicExtractor
from .heuristic_extractor import HeuristicExtractor
from .service import TopicExtractionService

__all__ = [
    "BaseTopicExtractor",
    "BERTopicExtractor",
    "HeuristicExtractor",
    "TopicExtractionService",
]
