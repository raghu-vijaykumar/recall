import logging
from typing import List, Dict, Any, Tuple

from .config import PreprocessingConfig
from .cleaners import TextCleaner
from .filters import QualityFilter, LanguageFilter
from .deduplication import DuplicateDetector


class DocumentPreprocessor:
    """
    Main document preprocessing pipeline
    Combines all preprocessing steps with proper error handling
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.language_filter = LanguageFilter(self.config)
        self.duplicate_detector = DuplicateDetector(self.config)

    def preprocess_single_document(self, text: str) -> str:
        """Preprocess a single document through the full pipeline"""
        try:
            # Step 1: Basic text cleaning
            cleaned_text = self.cleaner.clean_text(text)

            # Step 2: Quality filtering
            is_quality, _ = self.quality_filter.is_quality_document(cleaned_text)
            if not is_quality:
                return None

            # Step 3: Language filtering
            is_allowed, _ = self.language_filter.is_allowed_language(cleaned_text)
            if not is_allowed:
                return None

            return cleaned_text

        except Exception as e:
            logging.warning(f"Error preprocessing document: {e}")
            return None

    def preprocess_documents(
        self, documents: List[str], verbose: bool = True
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Preprocess multiple documents through the full pipeline
        Returns: (processed_docs, statistics_dict)
        """
        if verbose:
            logging.info(f"Starting preprocessing of {len(documents)} documents")

        stats = {"input_count": len(documents), "stages": {}}

        current_docs = documents

        # Step 1: Basic text cleaning
        if verbose:
            logging.info("Cleaning text (URLs, emails, HTML, whitespace)...")
        cleaned_docs = [self.cleaner.clean_text(doc) for doc in current_docs]
        stats["stages"]["cleaning"] = {"output_count": len(cleaned_docs)}

        # Step 2: Quality filtering
        if verbose:
            logging.info("Quality filtering (length, numeric ratio, etc.)...")
        current_docs, quality_stats = self.quality_filter.filter_documents(cleaned_docs)
        stats["stages"]["quality_filter"] = quality_stats

        # Step 3: Language filtering
        if verbose:
            logging.info("Language filtering...")
        current_docs, lang_stats = self.language_filter.filter_by_language(current_docs)
        stats["stages"]["language_filter"] = lang_stats

        # Step 4: Deduplication
        if verbose:
            logging.info("Removing duplicates...")
        current_docs, duplicates_removed = self.duplicate_detector.remove_duplicates(
            current_docs
        )
        stats["stages"]["deduplication"] = {"duplicates_removed": duplicates_removed}

        # Final statistics
        stats["final_count"] = len(current_docs)
        stats["total_removed"] = stats["input_count"] - stats["final_count"]

        if verbose:
            logging.info(
                f"Preprocessing complete: {stats['final_count']} documents kept"
            )

        return current_docs, stats
