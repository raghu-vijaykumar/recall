"""
Concrete implementations of concept ranking algorithms.
"""

import math
import re
from typing import List, Dict, Any

from .base import RankingAlgorithm


class FrequencyBasedRanking(RankingAlgorithm):
    """Rank concepts based on frequency and other factors"""

    def rank_concepts(
        self, concepts: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """Rank concepts using frequency-based scoring"""
        content_lower = content.lower()

        for concept in concepts:
            term = concept["name"].lower()
            score = 0.5  # Base score

            # Frequency boost
            frequency = content_lower.count(term)
            score += min(0.3, frequency * 0.05)

            # Length boost for longer terms
            if len(concept["name"]) > 6:
                score += 0.1

            # Capitalization boost (likely proper nouns)
            if concept["name"][0].isupper():
                score += 0.15

            # Technical term indicators
            if re.search(r"[0-9_]", concept["name"]):
                score += 0.1

            concept["score"] = min(1.0, score)

        return sorted(concepts, key=lambda x: x["score"], reverse=True)


class TFIDFRanking(RankingAlgorithm):
    """Rank concepts using TF-IDF like scoring"""

    def rank_concepts(
        self, concepts: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """Rank concepts using TF-IDF inspired scoring"""
        words = re.findall(r"\b\w+\b", content.lower())
        total_words = len(words)

        # Calculate term frequencies
        term_freq = {}
        for word in words:
            term_freq[word] = term_freq.get(word, 0) + 1

        # Simple IDF approximation (rarer terms get higher scores)
        for concept in concepts:
            term = concept["name"].lower()
            tf = term_freq.get(term, 0) / total_words if total_words > 0 else 0

            # Approximate IDF (inverse of frequency)
            idf = 1.0 / (1.0 + math.log(1.0 + term_freq.get(term, 1)))

            score = tf * idf

            # Boost for longer terms and proper nouns
            if len(concept["name"]) > 6:
                score += 0.1
            if concept["name"][0].isupper():
                score += 0.1

            concept["score"] = min(1.0, score)

        return sorted(concepts, key=lambda x: x["score"], reverse=True)
