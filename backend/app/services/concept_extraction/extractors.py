"""
Concrete implementations of concept extractors.
"""

import re
import math
from typing import List, Dict, Any, Optional

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .base import ConceptExtractor


class EntityRecognitionExtractor(ConceptExtractor):
    """Entity recognition based concept extraction using spaCy NLP"""

    def __init__(self, stop_words: Optional[set] = None, use_spacy: bool = False):
        super().__init__(stop_words)
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print(
                    "Warning: spaCy model not found. Concept extraction will not work."
                )
                self.use_spacy = False

    def extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts using spaCy entity recognition"""
        if not self.use_spacy:
            raise ValueError("spaCy is required for concept extraction")

        concepts = self._extract_with_spacy(content)

        # Filter out stop words, non-concept words and gibberish
        filtered_concepts = []
        for concept in concepts:
            if (
                not self._is_stop_word(concept["name"])
                and not self._is_gibberish_term(concept["name"])
                and not self._is_non_concept_word(concept["name"])
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


class HeuristicExtractor(ConceptExtractor):
    """Heuristic-based concept extraction using pattern matching and scoring"""

    def __init__(self, stop_words: Optional[set] = None):
        super().__init__(stop_words)

    def extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts using heuristic pattern matching"""
        concepts = []

        # Split content into lines for line tracking
        lines = content.split("\n")

        # Process each line to find concepts
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            # Skip file header lines (they contain file paths and are not content)
            if (
                line.startswith("=" * 20)
                or "FILE:" in line
                or "Generated from:" in line
            ):
                continue

            # Clean the line for concept extraction
            cleaned_line = self._preprocess_text(line)
            if len(cleaned_line) < 5:  # Skip very short lines
                continue

            # Extract potential concepts from this line
            line_concepts = self._extract_concepts_from_line(
                cleaned_line, line_num, lines
            )

            concepts.extend(line_concepts)

        # Remove duplicates and apply quality threshold
        seen = set()
        unique_concepts = []
        for concept in concepts:
            # Lower threshold for heuristic extraction
            if concept["score"] > 0.7 and concept["name"].lower() not in seen:
                seen.add(concept["name"].lower())
                unique_concepts.append(concept)

        return unique_concepts

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not text:
            return ""

        # Remove markdown formatting
        text = re.sub(r"#+\s*", "", text)  # Headers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Links

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove extra whitespace
        text = text.strip()

        return text

    def _extract_concepts_from_line(
        self, line: str, line_num: int, all_lines: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from a single line with line number tracking
        Improved filtering to avoid gibberish and lorem ipsum content
        """
        concepts = []

        # Skip lines that look like lorem ipsum or template content
        if self._is_lorem_ipsum_or_gibberish(line):
            return concepts

        # Extract potential multi-word concepts first
        # Look for patterns like: "Machine Learning", "Design Pattern", "Abstract Factory"
        multi_word_patterns = [
            # Technical terms with multiple words
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Title Case phrases
            r"\b[a-z]+(?:\s+[a-z]+){1,3}\b",  # lowercase phrases (2-4 words)
        ]

        for pattern in multi_word_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if self._is_meaningful_concept(match) and not self._is_gibberish_term(
                    match
                ):
                    # Get broader context (surrounding lines)
                    context_lines = self._get_context_lines(
                        all_lines, line_num, context_size=2
                    )
                    context_text = "\n".join(context_lines)

                    score = self._calculate_concept_score(
                        match, "\n".join(all_lines), context_text
                    )

                    if score > 0.85:  # Higher threshold for multi-word concepts
                        concepts.append(
                            {
                                "name": match,
                                "description": f"Concept mentioned in context: {context_text[:120]}...",
                                "snippet": context_text,
                                "score": score,
                                "start_line": line_num,
                                "end_line": line_num,
                            }
                        )

        # Also extract high-quality single words (technical terms)
        single_word_patterns = [
            r"\b[A-Z][a-z]{6,}\b",  # Long capitalized words (likely specific terms)
            r"\b[a-z]{7,}\b",  # Long lowercase words (technical terms)
        ]

        for pattern in single_word_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if self._is_meaningful_concept(match) and not self._is_gibberish_term(
                    match
                ):
                    # Get broader context
                    context_lines = self._get_context_lines(
                        all_lines, line_num, context_size=1
                    )
                    context_text = "\n".join(context_lines)

                    score = self._calculate_concept_score(
                        match, "\n".join(all_lines), context_text
                    )

                    if score > 0.85:  # Higher threshold for single words
                        concepts.append(
                            {
                                "name": match,
                                "description": f"Concept mentioned in context: {context_text[:120]}...",
                                "snippet": context_text,
                                "score": score,
                                "start_line": line_num,
                                "end_line": line_num,
                            }
                        )

        return concepts

    def _get_context_lines(
        self, all_lines: List[str], center_line: int, context_size: int = 2
    ) -> List[str]:
        """
        Get context lines around a center line
        """
        start = max(0, center_line - context_size)
        end = min(len(all_lines), center_line + context_size + 1)
        return all_lines[start:end]

    def _is_meaningful_concept(self, term: str) -> bool:
        """
        Check if a term is likely to be a meaningful concept
        """
        if not term or len(term) < 4:
            return False

        term_lower = term.lower()

        # Skip common non-concept words
        if self._is_stop_word(term):
            return False

        # Must contain at least one letter
        if not re.search(r"[a-zA-Z]", term):
            return False

        # Skip if it's just numbers or symbols
        if re.match(r"^[^a-zA-Z]*$", term):
            return False

        return True

    def _is_lorem_ipsum_or_gibberish(self, line: str) -> bool:
        """
        Generic gibberish detection using statistical analysis
        """
        if not line or len(line.strip()) < 10:
            return False

        # Use combined quality assessment
        quality = self._assess_content_quality(line)
        return quality["is_gibberish"]

    def _assess_content_quality(self, text: str) -> dict:
        """
        Combined quality assessment using multiple statistical measures
        """
        if not text or len(text.strip()) < 5:
            return {"quality_score": 0.0, "is_gibberish": True}

        # Run all statistical analyses
        char_entropy = self._calculate_text_entropy(text)
        word_entropy = self._calculate_word_entropy(text)
        repetition_score = self._detect_repetitive_patterns(text)
        word_dist = self._analyze_word_distribution(text)
        word_len = self._analyze_word_lengths(text)
        gini_coefficient = self._analyze_stop_word_ratio(text)
        template_score = self._detect_template_patterns(text)

        # Weighted quality score (0.0 = gibberish, 1.0 = high quality)
        quality_score = (
            (char_entropy / 5.0) * 0.2  # Character entropy
            + (word_entropy / 8.0) * 0.2  # Word entropy
            + (1.0 - repetition_score) * 0.2  # Low repetition = good
            + (1.0 - word_dist["gibberish_score"]) * 0.15  # Word distribution
            + word_len["coherence_score"] * 0.15  # Word length coherence
            + (1.0 - gini_coefficient) * 0.05  # Even word distribution
            + (1.0 - template_score) * 0.05  # Low template similarity
        )

        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "quality_score": quality_score,
            "is_gibberish": quality_score < 0.3,  # Threshold for gibberish
            "char_entropy": char_entropy,
            "word_entropy": word_entropy,
            "repetition_score": repetition_score,
            "gini_coefficient": gini_coefficient,
            "template_score": template_score,
        }

    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text characters"""
        if not text:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_word_entropy(self, text: str) -> float:
        """Calculate entropy based on word distribution"""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 3:
            return 0.0

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        entropy = 0.0
        total_words = len(words)
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)

        return entropy

    def _detect_repetitive_patterns(self, text: str, n: int = 3) -> float:
        """Detect repetitive n-gram patterns"""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < n * 2:
            return 0.0

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))

        # Count n-gram frequencies
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Calculate repetition ratio
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)

        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def _analyze_word_distribution(self, text: str) -> dict:
        """Analyze word frequency distribution for gibberish detection"""
        words = re.findall(r"\b\w+\b", text.lower())

        if len(words) < 5:
            return {"gibberish_score": 1.0}  # Too short to analyze

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Calculate distribution metrics
        unique_words = len(word_counts)
        total_words = len(words)
        uniqueness_ratio = unique_words / total_words

        # Check for over-repetition of few words
        top_word_count = max(word_counts.values()) if word_counts else 0
        repetition_ratio = top_word_count / total_words

        # Very low uniqueness or high repetition indicates gibberish
        gibberish_score = 0.0
        if uniqueness_ratio < 0.3:  # Less than 30% unique words
            gibberish_score += 0.5
        if repetition_ratio > 0.4:  # One word used more than 40% of the time
            gibberish_score += 0.5

        return {
            "uniqueness_ratio": uniqueness_ratio,
            "repetition_ratio": repetition_ratio,
            "gibberish_score": gibberish_score,
        }

    def _analyze_word_lengths(self, text: str) -> dict:
        """Analyze word length distribution"""
        words = re.findall(r"\b\w+\b", text)

        if not words:
            return {"avg_length": 0, "coherence_score": 0.0}

        lengths = [len(word) for word in words]
        avg_length = sum(lengths) / len(lengths)

        # Very short average word length indicates gibberish
        coherence_score = min(1.0, avg_length / 5.0)  # Normalize to 5 chars

        return {"avg_length": avg_length, "coherence_score": coherence_score}

    def _analyze_stop_word_ratio(self, text: str) -> float:
        """Analyze ratio of common stop words"""
        # Dynamic stop word detection based on frequency
        words = re.findall(r"\b\w+\b", text.lower())
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Most frequent words are likely stop words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Top 10 most frequent words
        top_words = [word for word, count in sorted_words[:10]]

        # Calculate how evenly distributed the content is
        # High concentration on few words = likely gibberish
        if len(sorted_words) < 3:
            return 1.0  # Too few words to analyze

        # Gini coefficient for word distribution
        total_count = sum(count for word, count in sorted_words)
        cumulative = 0
        gini = 0.0

        for i, (word, count) in enumerate(sorted_words):
            cumulative += count
            gini += (i + 1) * count

        gini = 1 - (2 * gini) / (len(sorted_words) * total_count)

        return gini  # High gini = uneven distribution = gibberish

    def _detect_template_patterns(self, text: str) -> float:
        """Detect template/placeholder patterns dynamically"""
        lines = text.split("\n")

        # Look for similar line structures
        line_patterns = []
        for line in lines:
            # Extract pattern: word types, punctuation, brackets
            pattern = re.sub(r"\w+", "W", line)  # Replace words with W
            pattern = re.sub(r"\d+", "N", pattern)  # Replace numbers with N
            line_patterns.append(pattern.strip())

        # Count pattern frequencies
        pattern_counts = {}
        for pattern in line_patterns:
            if pattern:  # Skip empty patterns
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Calculate template score
        total_lines = len([p for p in line_patterns if p])
        if total_lines == 0:
            return 0.0

        # High similarity in line structures indicates template content
        max_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
        template_score = max_pattern_count / total_lines

        return template_score

    def _calculate_concept_score(
        self, term: str, full_text: str, context: str
    ) -> float:
        """
        Calculate a relevance score for a concept based on various factors
        """
        score = 0.6  # Base score

        term_lower = term.lower()

        # Boost score for capitalized words (likely proper nouns or important terms)
        if term[0].isupper():
            score += 0.2

        # Boost score for longer words (likely more specific terms)
        if len(term) > 6:
            score += 0.1

        # Boost score for words that appear multiple times in the text
        word_count = full_text.lower().count(term_lower)
        if word_count > 1:
            score += min(0.2, word_count * 0.05)

        # Boost score for technical terms (contains numbers, underscores, etc.)
        if re.search(r"[0-9_]", term):
            score += 0.1

        # Boost score for terms that appear in headings or important positions
        lines = full_text.split("\n")
        for i, line in enumerate(lines):
            if term_lower in line.lower():
                # Boost for lines that look like headings
                if line.strip().startswith(
                    ("#", "##", "###", "-", "*", "1.", "2.", "3.")
                ):
                    score += 0.15
                    break
                # Boost for early lines (likely more important)
                if i < 10:
                    score += 0.05
                    break

        # Cap the score at 1.0
        return min(1.0, score)
