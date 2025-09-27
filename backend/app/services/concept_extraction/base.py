"""
Base classes for concept extraction and ranking algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Optional


class ConceptExtractor(ABC):
    """Abstract base class for concept extraction methods"""

    def __init__(self, stop_words: Optional[Set[str]] = None):
        self.stop_words = stop_words or set()

    @abstractmethod
    def extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts from text content"""
        pass

    def _is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word"""
        return word.lower() in self.stop_words

    def _is_gibberish_term(self, term: str) -> bool:
        """Generic term gibberish detection"""
        if not term or len(term) < 3:
            return False

        # Simple heuristics for gibberish detection
        if len(term) > 50:  # Too long
            return True

        # Too many non-alphabetic characters
        alpha_ratio = sum(1 for c in term if c.isalpha()) / len(term)
        if alpha_ratio < 0.5:
            return True

        # All caps or all lowercase for long terms might be gibberish
        if len(term) > 10 and (term.isupper() or term.islower()):
            return True

        return False


class RankingAlgorithm(ABC):
    """Abstract base class for concept ranking algorithms"""

    @abstractmethod
    def rank_concepts(
        self, concepts: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """Rank concepts based on specific algorithm"""
        pass
