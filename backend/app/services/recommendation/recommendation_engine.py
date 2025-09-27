"""
Recommendation Engine for generating personalized learning recommendations.
"""

from typing import List
from datetime import datetime

from ...models import TopicArea, LearningRecommendation


class RecommendationEngine:
    """
    Generates personalized learning recommendations based on topic analysis.
    """

    def __init__(self, max_recommendations: int = 10):
        self.max_recommendations = max_recommendations

    def generate_recommendations(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningRecommendation]:
        """
        Generate personalized learning recommendations.

        Args:
            workspace_id: ID of the workspace
            topic_areas: List of topic areas to analyze

        Returns:
            List of learning recommendations
        """
        recommendations = []

        for topic_area in topic_areas:
            # Recommendation based on low exploration
            if topic_area.explored_percentage < 0.3:
                recommendation = self._create_exploration_recommendation(
                    workspace_id, topic_area
                )
                recommendations.append(recommendation)

            # Recommendation based on high coverage but low exploration
            elif (
                topic_area.coverage_score > 0.8 and topic_area.explored_percentage < 0.5
            ):
                recommendation = self._create_practice_recommendation(
                    workspace_id, topic_area
                )
                recommendations.append(recommendation)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations[: self.max_recommendations]

    def _create_exploration_recommendation(
        self, workspace_id: int, topic_area: TopicArea
    ) -> LearningRecommendation:
        """Create a recommendation for exploring a topic area"""
        return LearningRecommendation(
            recommendation_id=f"rec_{workspace_id}_{topic_area.topic_area_id}_explore",
            workspace_id=workspace_id,
            recommendation_type="concept_gap",
            topic_area_id=topic_area.topic_area_id,
            priority_score=min(1.0, (1.0 - topic_area.explored_percentage) * 0.8),
            reason=f"You've only explored {topic_area.explored_percentage:.1%} of {topic_area.name}",
            suggested_action=f"Take a quiz on {topic_area.name} to improve your understanding",
            created_at=datetime.utcnow(),
        )

    def _create_practice_recommendation(
        self, workspace_id: int, topic_area: TopicArea
    ) -> LearningRecommendation:
        """Create a recommendation for practicing a well-covered topic"""
        return LearningRecommendation(
            recommendation_id=f"rec_{workspace_id}_{topic_area.topic_area_id}_practice",
            workspace_id=workspace_id,
            recommendation_type="quiz_performance",
            topic_area_id=topic_area.topic_area_id,
            priority_score=0.7,
            reason=f"You have good coverage of {topic_area.name} but need more practice",
            suggested_action=f"Practice quizzes on {topic_area.name} concepts",
            created_at=datetime.utcnow(),
        )
