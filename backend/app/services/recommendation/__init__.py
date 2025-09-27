"""
Recommendation module for generating learning paths and recommendations.
"""

from .learning_path_generator import LearningPathGenerator
from .recommendation_engine import RecommendationEngine
from .service import RecommendationService

__all__ = [
    "LearningPathGenerator",
    "RecommendationEngine",
    "RecommendationService",
]
