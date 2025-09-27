"""
Concrete implementations of concept extractors.
"""

import re
from typing import List, Dict, Any, Optional

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .base import ConceptExtractor


class EntityRecognitionExtractor(ConceptExtractor):
    """Entity recognition based concept extraction using patterns and NLP"""

    def __init__(self, stop_words: Optional[set] = None, use_spacy: bool = False):
        super().__init__(stop_words)
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Using pattern-based extraction.")
                self.use_spacy = False

    def extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts using entity recognition patterns"""
        concepts = []

        if self.use_spacy:
            concepts.extend(self._extract_with_spacy(content))
        else:
            concepts.extend(self._extract_with_patterns(content))

        # Filter out stop words and gibberish
        filtered_concepts = []
        for concept in concepts:
            if not self._is_stop_word(concept["name"]) and not self._is_gibberish_term(
                concept["name"]
            ):
                filtered_concepts.append(concept)

        return filtered_concepts

    def _extract_with_spacy(self, content: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy"""
        concepts = []
        doc = self.nlp(content)

        for ent in doc.ents:
            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
            ]:
                concepts.append(
                    {
                        "name": ent.text,
                        "type": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "score": 0.8,  # Base score for named entities
                    }
                )

        return concepts

    def _extract_with_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts using regex patterns"""
        concepts = []
        lines = content.split("\n")

        # Multi-word concept patterns
        multi_word_patterns = [
            # Title case phrases
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            # Technical terms with patterns
            r"\b[A-Z][a-z]+\s+(?:Pattern|Algorithm|Method|Class|Interface|Framework|Library|Protocol|Database|System|Network|Security|Authentication|Authorization)\b",
            # Design patterns
            r"\b(?:Design|Creational|Structural|Behavioral|Factory|Singleton|Observer|Strategy|Command|Adapter|Bridge|Composite|Decorator|Facade|Proxy|Template|Iterator|State|Memento|Visitor)\s+[A-Z][a-z]+\b",
            # Programming concepts
            r"\b(?:Object|Class|Method|Function|Variable|Constant|Interface|Abstract|Concrete|Static|Dynamic|Virtual|Override|Implement|Inherit|Constructor|Destructor|Exception|Thread|Process|Memory|Cache|Database|Query|Transaction)\s+[A-Z][a-z]+\b",
        ]

        # Single word patterns
        single_word_patterns = [
            # Long capitalized words
            r"\b[A-Z][a-z]{6,}\b",
            # Long lowercase technical terms
            r"\b[a-z]{7,}\b",
        ]

        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            # Clean line
            cleaned_line = self._clean_line(line)
            if len(cleaned_line) < 5:
                continue

            # Extract multi-word concepts
            for pattern in multi_word_patterns:
                matches = re.findall(pattern, cleaned_line)
                for match in matches:
                    if len(match) > 3:
                        context = self._get_context(lines, line_num)
                        concepts.append(
                            {
                                "name": match,
                                "type": "multi_word",
                                "line": line_num,
                                "context": context,
                                "score": 0.7,
                            }
                        )

            # Extract single word concepts
            for pattern in single_word_patterns:
                matches = re.findall(pattern, cleaned_line)
                for match in matches:
                    if len(match) > 3:
                        context = self._get_context(lines, line_num)
                        concepts.append(
                            {
                                "name": match,
                                "type": "single_word",
                                "line": line_num,
                                "context": context,
                                "score": 0.6,
                            }
                        )

        return concepts

    def _clean_line(self, line: str) -> str:
        """Clean markdown and formatting from line"""
        # Remove markdown headers, bold, italic, code, links
        line = re.sub(r"#+\s*", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", line)
        return line

    def _get_context(
        self, lines: List[str], line_num: int, context_size: int = 2
    ) -> str:
        """Get surrounding context for a line"""
        start = max(0, line_num - context_size)
        end = min(len(lines), line_num + context_size + 1)
        return "\n".join(lines[start:end])
