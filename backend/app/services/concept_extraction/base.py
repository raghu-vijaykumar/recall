"""
Base classes for concept extraction and ranking algorithms.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Optional
from pathlib import Path


class ConceptExtractor(ABC):
    """Abstract base class for concept extraction methods"""

    def __init__(self, stop_words: Optional[Set[str]] = None):
        self.stop_words = stop_words or set()
        self.non_concept_words = self._load_non_concept_words()

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

    def _load_non_concept_words(self) -> Set[str]:
        """Load non-concept words from external file"""
        non_concept_words = set()

        # Load from file
        non_concept_file = (
            Path(__file__).parent.parent.parent / "resources" / "non_concept_words.txt"
        )
        if non_concept_file.exists():
            try:
                with open(non_concept_file, "r", encoding="utf-8") as f:
                    non_concept_words = {
                        line.strip().lower() for line in f if line.strip()
                    }
            except Exception as e:
                logging.warning(f"Failed to load non-concept words from file: {e}")
                # Fallback to empty set if file loading fails
                non_concept_words = set()

        return non_concept_words

    def _is_non_concept_word(self, word: str) -> bool:
        """Check if word is a non-concept word"""
        return word.lower() in self.non_concept_words


class RankingAlgorithm(ABC):
    """Abstract base class for concept ranking algorithms"""

    @abstractmethod
    def rank_concepts(
        self, concepts: List[Dict[str, Any]], content: str
    ) -> List[Dict[str, Any]]:
        """Rank concepts based on specific algorithm"""
        pass
