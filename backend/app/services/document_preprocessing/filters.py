import logging
from typing import List, Dict, Tuple

try:
    import langdetect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, language filtering disabled")

from .config import PreprocessingConfig


class QualityFilter:
    """Filters documents based on quality criteria"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def is_quality_document(self, text: str) -> Tuple[bool, str]:
        """
        Check if document meets quality criteria
        Returns: (is_quality, reason)
        """
        words = text.split()

        # Check minimum word count
        if len(words) < self.config.min_words_per_doc:
            return False, f"too_short_{len(words)}_words"

        # Check maximum word count
        if len(words) > self.config.max_words_per_doc:
            return False, f"too_long_{len(words)}_words"

        # Check for excessively long words (likely gibberish)
        if any(len(word) > self.config.max_word_length for word in words):
            return False, "long_words_found"

        # Check numeric token ratio
        if self.config.max_number_ratio < 1.0:
            numeric_ratio = len([w for w in words if w.isdigit()]) / len(words)
            if numeric_ratio > self.config.max_number_ratio:
                return False, f"too_many_numbers_{numeric_ratio:.2f}"

        return True, "quality"

    def filter_documents(
        self, documents: List[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter list of documents and return statistics"""
        filtered_docs = []
        filter_stats = {"total": len(documents), "kept": 0, "filtered": {}}

        for doc in documents:
            is_quality, reason = self.is_quality_document(doc)
            if is_quality:
                filtered_docs.append(doc)
                filter_stats["kept"] += 1
            else:
                filter_stats["filtered"][reason] = (
                    filter_stats["filtered"].get(reason, 0) + 1
                )

        return filtered_docs, filter_stats


class LanguageFilter:
    """Handles language detection and filtering"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.lang_detector_available = LANGDETECT_AVAILABLE

    def detect_language(self, text: str) -> str:
        """Detect the language of a text document"""
        if not self.lang_detector_available:
            return "unknown"

        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def is_allowed_language(self, text: str) -> Tuple[bool, str]:
        """Check if document is in an allowed language"""
        if not self.config.language_detection:
            return True, "detection_disabled"

        if not self.config.allowed_languages:
            return True, "no_language_filter"

        detected_lang = self.detect_language(text)
        is_allowed = detected_lang in self.config.allowed_languages

        return is_allowed, detected_lang

    def filter_by_language(
        self, documents: List[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter documents by language and return statistics"""
        filtered_docs = []
        lang_stats = {}

        for doc in documents:
            is_allowed, detected_lang = self.is_allowed_language(doc)
            if is_allowed:
                filtered_docs.append(doc)
            lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1

        return filtered_docs, lang_stats
