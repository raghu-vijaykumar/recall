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
        self, workspace_id: int, file_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas directly from file/workspace data.

        Args:
            workspace_id: ID of the workspace
            file_data: List of file dictionaries with content, metadata, etc.

        Returns:
            List of discovered TopicArea objects
        """
        pass

    def _generate_topic_name(self, content_samples: List[str]) -> str:
        """Generate a descriptive name for a topic from content samples"""
        # Use the most common nouns/keywords from content samples
        from collections import Counter
        import re

        all_words = []
        for sample in content_samples:
            # Extract word-like tokens, focusing on potential topic keywords
            words = re.findall(r"\b[a-zA-Z]{3,}\b", sample.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)

        # Filter out common stop words
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "an",
            "a",
        }
        meaningful_words = [
            (word, count)
            for word, count in word_counts.most_common(10)
            if word not in stop_words
        ]

        top_words = [word for word, count in meaningful_words[:3]]

        # Create a title-case name
        if len(top_words) >= 2:
            return " ".join(top_words[:2]).title()
        elif top_words:
            return top_words[0].title()
        else:
            return "General Topic"

    def _generate_topic_description(
        self, content_samples: List[str], file_count: int = 0
    ) -> str:
        """Generate a description for a topic cluster"""
        # Sample some content snippets for description
        samples = content_samples[:3] if content_samples else []
        if samples:
            # Take first 50 characters from each sample
            snippets = []
            for sample in samples:
                snippet = sample[:100].strip()
                if len(snippet) > 50:
                    snippet = snippet[:47] + "..."
                snippets.append(f'"{snippet}"')

            if file_count > 0:
                return f"Topic covering content like {', '.join(snippets[:2])} across {file_count} files"
            else:
                return f"Topic covering content like {', '.join(snippets[:2])}"
        else:
            return f"Topic area identified from workspace content"
