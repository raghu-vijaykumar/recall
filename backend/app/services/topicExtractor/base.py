"""
Base classes and interfaces for topic extraction.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...models import TopicArea


class BaseTopicExtractor(ABC):
    """
    Abstract base class for topic extraction strategies.
    """

    def __init__(self, min_topic_concepts: int = 3, max_topic_areas: int = 20):
        self.min_topic_concepts = min_topic_concepts
        self.max_topic_areas = max_topic_areas

    @abstractmethod
    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas from concept data.

        Args:
            workspace_id: ID of the workspace
            concepts_data: List of concept dictionaries with id, name, description, relevance_score

        Returns:
            List of discovered TopicArea objects
        """
        pass

    def _generate_topic_name(self, concepts: List[Dict[str, Any]]) -> str:
        """Generate a descriptive name for a topic cluster"""
        # Use the most common words across concept names
        all_words = []
        for concept in concepts:
            words = concept["name"].split()
            all_words.extend(words)

        from collections import Counter

        word_counts = Counter(all_words)
        top_words = [word for word, count in word_counts.most_common(3)]

        # Create a title-case name
        if len(top_words) >= 2:
            return " ".join(top_words[:2]).title()
        elif top_words:
            return top_words[0].title()
        else:
            return "General Concepts"

    def _generate_topic_description(self, concepts: List[Dict[str, Any]]) -> str:
        """Generate a description for a topic cluster"""
        concept_names = [c["name"] for c in concepts[:5]]  # Use first 5 concepts
        names_str = ", ".join(concept_names)

        if len(concepts) > 5:
            names_str += f", and {len(concepts) - 5} more"

        return f"Topic area covering concepts like {names_str}"
