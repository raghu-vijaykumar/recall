"""
Learning Path Generator for creating recommended learning trajectories.
"""

from typing import List, Optional
from datetime import datetime

from ...models import TopicArea, LearningPath, LearningPathCreate


class LearningPathGenerator:
    """
    Generates learning paths based on topic areas and user progress.
    """

    def __init__(self, max_paths: int = 5):
        self.max_paths = max_paths

    def generate_learning_paths(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningPath]:
        """
        Generate recommended learning paths based on topic areas.

        Args:
            workspace_id: ID of the workspace
            topic_areas: List of discovered topic areas

        Returns:
            List of generated learning paths
        """
        learning_paths = []

        # Sort topic areas by coverage and exploration
        sorted_topics = sorted(
            topic_areas,
            key=lambda ta: ta.coverage_score + ta.explored_percentage,
            reverse=True,
        )

        # Create comprehensive learning path
        if len(sorted_topics) >= 3:
            comprehensive_path = self._create_comprehensive_path(
                workspace_id, sorted_topics
            )
            learning_paths.append(comprehensive_path)

        # Create focused learning paths for high-priority topics
        high_priority_topics = [ta for ta in sorted_topics if ta.coverage_score > 0.7][
            : self.max_paths
        ]

        for i, topic in enumerate(high_priority_topics):
            focused_path = self._create_focused_path(workspace_id, topic, i)
            learning_paths.append(focused_path)

        return learning_paths

    def _create_comprehensive_path(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> LearningPath:
        """Create a comprehensive learning path covering all major topics"""
        return LearningPath(
            learning_path_id=f"path_{workspace_id}_comprehensive",
            workspace_id=workspace_id,
            name="Comprehensive Workspace Mastery",
            description="Complete learning path covering all major topics in your workspace",
            topic_areas=[ta.topic_area_id for ta in topic_areas],
            estimated_hours=sum(
                ta.concept_count * 2 for ta in topic_areas
            ),  # Rough estimate
            difficulty_level=self._calculate_path_difficulty(topic_areas),
            prerequisites=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def _create_focused_path(
        self, workspace_id: int, topic_area: TopicArea, index: int
    ) -> LearningPath:
        """Create a focused learning path for a specific topic"""
        return LearningPath(
            learning_path_id=f"path_{workspace_id}_focused_{index}",
            workspace_id=workspace_id,
            name=f"Deep Dive: {topic_area.name}",
            description=f"Focused learning path for mastering {topic_area.name}",
            topic_areas=[topic_area.topic_area_id],
            estimated_hours=topic_area.concept_count * 3,  # More intensive
            difficulty_level="intermediate",
            prerequisites=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    def _calculate_path_difficulty(self, topic_areas: List[TopicArea]) -> str:
        """Calculate overall difficulty of a learning path"""
        if not topic_areas:
            return "beginner"

        avg_coverage = sum(ta.coverage_score for ta in topic_areas) / len(topic_areas)

        if avg_coverage < 0.4:
            return "beginner"
        elif avg_coverage < 0.7:
            return "intermediate"
        else:
            return "advanced"
