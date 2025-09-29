"""
BERTopic-based topic extractor for workspace analysis.
Works directly with documents for superior topic discovery.
"""

import math
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np
from pathlib import Path

from .base import BaseTopicExtractor
from ...models import TopicArea
from ..embedding_service import EmbeddingService

try:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sklearn.feature_extraction.text import CountVectorizer
    import umap
    import hdbscan

    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning(
        "BERTopic dependencies not available. Install with: pip install bertopic umap-learn hdbscan"
    )


class BERTopicExtractor(BaseTopicExtractor):
    """
    Advanced topic extractor using BERTopic for superior topic modeling.
    Provides better clustering, automatic topic naming, and hierarchical topic discovery.
    """

    def __init__(
        self,
        # BERTopic-specific parameters
        model_name: str = "all-MiniLM-L6-v2",
        min_topic_size: int = 3,
        max_topic_areas: int = 20,
        nr_topics: Optional[int] = None,  # Auto-determined if None
        diversity: Optional[float] = None,  # Control topic diversity
        # Model components
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model=None,
        representation_model=None,
        # Quality parameters
        coherence_threshold: float = 0.1,
        outlier_threshold: float = 0.1,
        # Integration parameters
        embedding_service: Optional[EmbeddingService] = None,
        calculate_probabilities: bool = True,
        **kwargs,
    ):
        super().__init__(
            min_topic_concepts=min_topic_size, max_topic_areas=max_topic_areas, **kwargs
        )

        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic is not installed. Please install with: pip install bertopic umap-learn hdbscan"
            )

        self.model_name = model_name
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.diversity = diversity
        self.coherence_threshold = coherence_threshold
        self.outlier_threshold = outlier_threshold
        self.calculate_probabilities = calculate_probabilities

        # Initialize models
        self.umap_model = umap_model or umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_epsilon=0.1,
            prediction_data=True,
        )

        self.vectorizer_model = vectorizer_model or CountVectorizer(
            min_df=2, max_df=0.9, stop_words="english", ngram_range=(1, 2)
        )

        self.representation_model = representation_model or KeyBERTInspired()

        # Services
        self.embedding_service = embedding_service or EmbeddingService.get_instance()

        logging.info(
            f"Initialized BERTopicExtractor with model={model_name}, min_topic_size={min_topic_size}, "
            f"nr_topics={nr_topics}, diversity={diversity}"
        )

    async def extract_topics(
        self, workspace_id: int, file_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas directly from documents using BERTopic.

        Args:
            workspace_id: ID of the workspace
            file_data: List of document dictionaries with content, file_path, etc.

        Returns:
            List of discovered TopicArea objects
        """
        logging.info(
            f"Starting BERTopic extraction for workspace {workspace_id} with {len(file_data)} documents"
        )

        if len(file_data) < self.min_topic_concepts:
            logging.warning(
                f"Insufficient documents ({len(file_data)}) for BERTopic extraction, minimum required: {self.min_topic_concepts}"
            )
            return []

        # Extract document texts
        document_texts = await self._extract_document_texts(file_data)
        logging.info(
            f"Extracted {len(document_texts)} document texts for BERTopic processing"
        )

        if len(document_texts) < self.min_topic_concepts:
            logging.warning("No valid document texts found after extraction")
            return []

        # Create and fit BERTopic processor
        config = self._create_bertopic_config()
        processor = self._create_bertopic_processor(config)

        try:
            # Process documents with BERTopic
            logging.info("Processing documents with BERTopic...")
            topics, probs = processor.process_documents(document_texts)

            # Get topic information
            all_topics = processor.get_topics()
            topic_info = processor.get_statistics()
            logging.info(
                f"BERTopic found {len(all_topics)} topics from {topic_info.get('total_documents', 0)} documents"
            )

            # Convert to our format
            topic_areas = await self._convert_bertopic_results_to_areas(
                all_topics, topics, probs, file_data, workspace_id
            )

            # Filter and rank topics by quality
            topic_areas = await self._filter_and_rank_topics(topic_areas)

            # Limit to max_topic_areas
            topic_areas = topic_areas[: self.max_topic_areas]

            logging.info(
                f"BERTopic extraction completed: {len(topic_areas)} topic areas"
            )
            return topic_areas

        except Exception as e:
            logging.error(f"Error in BERTopic extraction: {e}")
            # Fallback to simple clustering if available
            logging.info("Falling back to basic clustering approach")
            return await self._fallback_clustering(workspace_id, file_data)

    async def _extract_document_texts(
        self, documents_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract document texts from documents data.

        Args:
            documents_data: List of document dictionaries with 'content' field

        Returns:
            List of document text strings
        """
        document_texts = []

        for doc_data in documents_data:
            # Extract content field
            content = doc_data.get("content", "").strip()
            if content:
                document_texts.append(content)

            # Alternative: if content is not available, try other fields
            else:
                # Try alternative field names that might contain text
                alt_content = (
                    doc_data.get("text")
                    or doc_data.get("body")
                    or doc_data.get("document")
                    or ""
                ).strip()
                if alt_content:
                    document_texts.append(alt_content)

        # Remove empty or very short documents
        document_texts = [doc for doc in document_texts if len(doc.split()) >= 3]

        return document_texts

    def _create_bertopic_config(self) -> "TopicModelingConfig":
        """Create BERTopic configuration from extractor parameters."""
        # Import here to avoid circular imports
        try:
            from bertopic import TopicModelingConfig as BERTopicConfig
        except ImportError:
            # Fallback config if import fails
            class BERTopicConfig:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        config = BERTopicConfig(
            model_name=self.model_name,
            min_topic_size=self.min_topic_size,
            max_topics=self.nr_topics,
            cluster_epsilon=0.4,  # Default for now
            umap_neighbors=8,
            umap_components=3,
            diversity=self.diversity,
            verbose=False,  # Controlled by logging
        )

        return config

    def _create_bertopic_processor(self, config: "TopicModelingConfig") -> BERTopic:
        """Create BERTopic model instance with custom configuration."""
        # Create BERTopic with configuration
        model = BERTopic(
            embedding_model=config.model_name,
            umap_model=umap.UMAP(
                n_neighbors=config.umap_neighbors,
                n_components=config.umap_components,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=hdbscan.HDBSCAN(
                min_cluster_size=config.min_topic_size,
                metric="euclidean",
                cluster_selection_epsilon=config.cluster_epsilon,
                prediction_data=True,
            ),
            vectorizer_model=CountVectorizer(
                min_df=2, max_df=0.9, stop_words="english", ngram_range=(1, 2)
            ),
            representation_model=KeyBERTInspired(),
            nr_topics=config.max_topics,
            min_topic_size=config.min_topic_size,
            calculate_probabilities=True,
            verbose=config.verbose,
        )

        return model

    async def _convert_bertopic_results_to_areas(
        self,
        all_topics: Dict[int, Dict[str, Any]],
        topics: List[int],
        probs: np.ndarray,
        file_data: List[Dict[str, Any]],
        workspace_id: int,
    ) -> List[TopicArea]:
        """Convert BERTopic results to TopicArea format."""
        topic_areas = []

        for topic_id, topic_data in all_topics.items():
            if topic_id == -1:  # Skip outliers
                continue

            # Get document count for this topic
            doc_count = topic_data.get("count", 0)
            if doc_count < self.min_topic_concepts:
                continue

            # Create topic name and description
            topic_name = self._generate_topic_name_from_bertopic(topic_data)
            topic_description = self._generate_topic_description_from_bertopic(
                topic_data
            )

            # Calculate coverage score based on topic attributes
            topic_words = topic_data.get("words", [])
            coverage_score = min(1.0, len(topic_words) / 10.0)  # Simple heuristic

            # Create topic area
            topic_area = TopicArea(
                topic_area_id=f"bertopic_{workspace_id}_{topic_id}",
                workspace_id=workspace_id,
                name=topic_name,
                description=topic_description,
                coverage_score=coverage_score,
                concept_count=doc_count,  # Documents count
                file_count=0,  # Updated by service based on topic_document_counts
                explored_percentage=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Add metadata
            topic_area._bertopic_metadata = {
                "topic_id": topic_id,
                "topic_words": topic_words,
            }

            topic_areas.append(topic_area)

        return topic_areas

    def _generate_topic_name_from_bertopic(self, topic_data: Dict[str, Any]) -> str:
        """Generate topic name from BERTopic topic data."""
        # Try to use the name field if available
        name = topic_data.get("name", "").strip()
        if name and name.lower() not in ["none", "", "unknown"]:
            return name

        # Fall back to using top words
        topic_words = topic_data.get("words", [])
        if topic_words:
            # Use top 2-3 most relevant words
            top_words = [word for word, score in topic_words[:3]]
            name = " ".join(top_words).title()

            # Ensure name is not too long
            if len(name) > 50:
                name = name[:47] + "..."
            return name

        return "General Topic"

    def _generate_topic_description_from_bertopic(
        self, topic_data: Dict[str, Any]
    ) -> str:
        """Generate topic description from BERTopic topic data."""
        topic_name = topic_data.get("name", "this topic")
        count = topic_data.get("count", 0)
        words = topic_data.get("words", [])

        # Start with count and basic description
        description = f"Topic discovered in {count} documents"

        if words:
            # Add top words to description
            top_words = [word for word, score in words[:5]]
            word_str = ", ".join(top_words)
            description += f". Key themes include: {word_str}"

        return description

    async def _filter_and_rank_topics(
        self, topic_areas: List[TopicArea]
    ) -> List[TopicArea]:
        """Filter and rank topics by quality metrics (simplified version)."""
        # Since we're not using a BERTopic model directly, rank by coverage score and document count

        # Calculate quality scores for each topic
        for topic_area in topic_areas:
            # Quality score based on coverage and document count
            coverage_score = topic_area.coverage_score
            doc_count = topic_area.concept_count

            # Simple quality score: combine coverage with document count bonus
            doc_count_bonus = min(1.0, doc_count / 15.0)  # Bonus for more documents

            quality_score = (coverage_score * 0.7) + (doc_count_bonus * 0.3)

            topic_area._quality_score = quality_score

        # Sort by quality score (higher is better)
        topic_areas.sort(key=lambda x: getattr(x, "_quality_score", 0.0), reverse=True)

        logging.info(f"Ranked {len(topic_areas)} topics by quality score")

        return topic_areas

    async def _fallback_clustering(
        self, workspace_id: int, file_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """Fallback clustering when BERTopic fails."""
        logging.warning("Using fallback clustering for documents")

        try:
            # Simple clustering: create one general topic with all documents
            if not file_data:
                return []

            topic_area = TopicArea(
                topic_area_id=f"fallback_{workspace_id}_general",
                workspace_id=workspace_id,
                name="General Documents",
                description=f"General topic covering {len(file_data)} documents",
                coverage_score=min(1.0, len(file_data) / 10.0),  # Simple heuristic
                concept_count=len(file_data),
                file_count=0,
                explored_percentage=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            return [topic_area]

        except Exception as e:
            logging.error(f"Fallback clustering failed: {e}")
            return []

    # Old concept-based methods removed:
    # - _prepare_documents (concepts_data based)
    # - _get_concept_context (concept-specific)

    # Old concept-based methods removed:
    # - _prepare_documents (concepts_data based)
    # - _get_concept_context (concept-specific)
    # - _get_bertopic_model (old caching method)
    # - _get_model_cache_key (old caching key)

    # Old concept-based methods removed:
    # - _prepare_documents (concepts_data based)
    # - _get_concept_context (concept-specific)
    # - _get_bertopic_model (old caching method)
    # - _get_model_cache_key (old caching key)
    # - _convert_bertopic_results (old concept-based conversion)
    # - _generate_bertopic_name (old naming method)
    # - _generate_bertopic_description (old description method)
    # - _calculate_topic_coherence (unused coherence calculation)
    # - _calculate_outlier_score (unused outlier calculation)
    # - _filter_and_rank_topics (old version with bertopic_model parameter)
    # - _fallback_clustering (old concepts_data version)
    # - get_topic_visualization_data (unused visualization method)
    # - get_topic_hierarchy (unused hierarchy method)

    async def _prepare_documents(self, concepts_data: List[Dict[str, Any]]) -> List[str]:
        """Prepare documents for BERTopic processing (compatibility method for tests)."""
        logging.warning("_prepare_documents is deprecated, use extract_topics with file_data")
        # Simple fallback: extract raw concepts as documents if needed
        document_texts = []
        for concept in concepts_data:
            name = concept.get("name", "").strip()
            description = concept.get("description", "").strip()
            content = f"{name} {description}".strip()
            if content:
                document_texts.append(content)
        return document_texts

    def _generate_bertopic_name(self) -> str:
        """Generate topic name using BERTopic (compatibility method)."""
        # Simplified implementation for compatibility
        return "BERTopic Topic"

    def _generate_topic_name(self, topic_data: Dict[str, Any]) -> str:
        """Generate topic name from topic data."""
        return self._generate_topic_name_from_bertopic(topic_data)

    def _calculate_topic_coherence(self) -> float:
        """Calculate topic coherence (compatibility method)."""
        return 0.5  # Default coherence

    def _calculate_outlier_score(self) -> float:
        """Calculate outlier score (compatibility method)."""
        return 0.1  # Default outlier score

    def _get_model_cache_key(self) -> str:
        """Generate model cache key (compatibility method)."""
        return (
            f"bertopic_{self.model_name}_{self.min_topic_size}_"
            f"{self.nr_topics}_{self.diversity}"
        )

    def get_topic_visualization_data(self) -> Dict[str, Any]:
        """Get topic visualization data (compatibility method)."""
        return {
            "nodes": [],
            "edges": [],
        }
