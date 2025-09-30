import logging
from typing import List, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, deduplication disabled")

from .config import PreprocessingConfig


class DuplicateDetector:
    """Detects and removes near-duplicate documents"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.sklearn_available = SKLEARN_AVAILABLE

    def find_duplicates(self, documents: List[str]) -> List[bool]:
        """
        Find near-duplicate documents
        Returns boolean mask: True = keep, False = duplicate
        """
        if not self.config.remove_duplicates or not self.sklearn_available:
            return [True] * len(documents)

        if len(documents) <= 1:
            return [True] * len(documents)

        try:
            # Create TF-IDF vectors for similarity comparison
            max_features = min(1000, len(documents) // 2)
            vectorizer = TfidfVectorizer(
                max_features=max_features, stop_words="english", ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Keep track of documents to keep
            keep_mask = [True] * len(documents)

            # For each document, check if it's too similar to any previous kept document
            for i in range(len(documents)):
                for j in range(i):
                    if (
                        keep_mask[j]
                        and similarity_matrix[i, j] > self.config.duplicate_threshold
                    ):
                        keep_mask[i] = (
                            False  # This is a duplicate of an earlier document
                        )
                        break

            return keep_mask

        except Exception as e:
            logging.warning(f"Duplicate detection failed: {e}")
            return [True] * len(documents)

    def remove_duplicates(self, documents: List[str]) -> Tuple[List[str], int]:
        """Remove duplicates and return count of removed documents"""
        if len(documents) <= 1:
            return documents, 0

        keep_mask = self.find_duplicates(documents)
        filtered_docs = [doc for doc, keep in zip(documents, keep_mask) if keep]
        duplicates_removed = len(documents) - len(filtered_docs)

        return filtered_docs, duplicates_removed
