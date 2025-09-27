"""
Recommendation Service for generating learning paths and recommendations.
"""

from typing import List

from ...models import TopicArea, LearningPath, LearningRecommendation
from .learning_path_generator import LearningPathGenerator
from .recommendation_engine import RecommendationEngine


class RecommendationService:
    """
    Service for generating learning recommendations and paths.
    Orchestrates learning path generation and recommendation engines.
    """

    def __init__(
        self,
        learning_path_generator: LearningPathGenerator = None,
        recommendation_engine: RecommendationEngine = None,
    ):
        self.learning_path_generator = (
            learning_path_generator or LearningPathGenerator()
        )
        self.recommendation_engine = recommendation_engine or RecommendationEngine()

    def generate_learning_paths(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningPath]:
        """
        Generate learning paths for the workspace.

        Args:
            workspace_id: ID of the workspace
            topic_areas: List of topic areas

        Returns:
            List of generated learning paths
        """
        return self.learning_path_generator.generate_learning_paths(
            workspace_id, topic_areas
        )

    def generate_recommendations(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningRecommendation]:
        """
        Generate learning recommendations for the workspace.

        Args:
            workspace_id: ID of the workspace
            topic_areas: List of topic areas

        Returns:
            List of learning recommendations
        """
        return self.recommendation_engine.generate_recommendations(
            workspace_id, topic_areas
        )
