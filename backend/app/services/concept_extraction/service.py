"""
Main service class for concept extraction with pluggable components.
"""

from typing import List, Dict, Any

from .base import ConceptExtractor, RankingAlgorithm


class ConceptExtractionService:
    """Main service for concept extraction with pluggable extractors and rankers"""

    def __init__(self, extractor: ConceptExtractor, ranker: RankingAlgorithm):
        self.extractor = extractor
        self.ranker = ranker

    def extract_and_rank_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts and rank them"""
        concepts = self.extractor.extract_concepts(content)
        ranked_concepts = self.ranker.rank_concepts(concepts, content)
        return ranked_concepts
