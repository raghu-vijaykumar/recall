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
        min_topic_size: int = 5,
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

        # Documents are assumed to be preprocessed by preprocessing service
        # Extract text content from preprocessed data
        document_texts, valid_metadata = (
            await self._extract_text_from_preprocessed_documents(file_data)
        )

        if not document_texts or len(document_texts) < self.min_topic_concepts:
            logging.warning("No valid preprocessed documents found after extraction")
            return []

        # Create and fit BERTopic processor
        logging.info("[DEBUG] Creating BERTopic configuration...")
        config = self._create_bertopic_config()
        logging.info("[DEBUG] Creating BERTopic processor...")
        processor = self._create_bertopic_processor(config)

        try:
            # Add timeout and progress tracking
            logging.info(
                f"[DEBUG] Starting BERTopic fit_transform on {len(document_texts)} documents..."
            )
            import time

            start_time = time.time()

            # Process documents with BERTopic - use fit_transform instead of process_documents
            logging.info("Processing documents with BERTopic...")

            # Add progress callback if possible
            topics, probs = processor.fit_transform(document_texts)

            processing_time = time.time() - start_time
            logging.info(
                f"[DEBUG] BERTopic processing completed in {processing_time:.2f} seconds"
            )
            logging.info(
                f"[DEBUG] Topics shape: {topics.shape if hasattr(topics, 'shape') else len(topics) if topics is not None else 'None'}"
            )
            logging.info(
                f"[DEBUG] Probabilities shape: {probs.shape if hasattr(probs, 'shape') else len(probs) if probs is not None else 'None'}"
            )

            # Get topic information
            all_topics = processor.get_topics()  # Dict: {topic_id: [(word, prob), ...]}
            topic_info_df = (
                processor.get_topic_info()
            )  # DataFrame with Topic, Count, Name, etc.
            logging.info(
                f"BERTopic found {len(all_topics)} topics from {len(document_texts)} documents"
            )

            # Convert to our format
            topic_areas = await self._convert_bertopic_results_to_areas(
                all_topics, topic_info_df, topics, probs, file_data, workspace_id
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

    async def _extract_text_from_preprocessed_documents(
        self, documents_data: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract text content from preprocessed documents for BERTopic processing.

        Args:
            documents_data: List of preprocessed document dictionaries

        Returns:
            Tuple of (document_texts, valid_metadata)
        """
        document_texts = []
        valid_metadata = []

        for doc_data in documents_data:
            # Extract preprocessed text content (check multiple fields for compatibility)
            content_fields = ["content", "text", "processed_text"]
            text_content = ""
            for field in content_fields:
                candidate = doc_data.get(field, "").strip()
                if candidate:
                    text_content = candidate
                    break

            if text_content:
                # Ensure reasonable length (BERTopic performs better with reasonable document sizes)
                max_chars = 50000  # Limit to prevent memory issues in BERTopic
                if len(text_content) > max_chars:
                    text_content = text_content[:max_chars]
                    logging.debug(
                        f"Truncated document {doc_data.get('id', 'unknown')} to {max_chars} characters for BERTopic"
                    )

                document_texts.append(text_content)
                valid_metadata.append(doc_data)

        logging.info(
            f"Extracted {len(document_texts)} valid document texts from {len(documents_data)} preprocessed documents for BERTopic"
        )
        return document_texts, valid_metadata

    async def _extract_document_texts(
        self, documents_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract document texts from documents data with preprocessing.

        Args:
            documents_data: List of document dictionaries with 'content' field

        Returns:
            List of document text strings
        """
        document_texts = []

        for doc_data in documents_data:
            # Extract content field
            content = doc_data.get("content", "").strip()
            if not content:
                # Alternative: if content is not available, try other fields
                alt_content = (
                    doc_data.get("text")
                    or doc_data.get("body")
                    or doc_data.get("document")
                    or ""
                ).strip()
                if alt_content:
                    content = alt_content

            if content:
                # Apply preprocessing similar to prototype
                content = self._preprocess_document_content(content, doc_data)
                if content:  # Only add if preprocessing didn't filter it out
                    document_texts.append(content)

        logging.info(
            f"[DEBUG] After preprocessing: {len(document_texts)} documents from {len(documents_data)} files"
        )

        return document_texts

    def _preprocess_document_content(
        self, content: str, doc_data: Dict[str, Any]
    ) -> str:
        """
        Preprocess document content similar to prototype app.

        Args:
            content: Raw document content
            doc_data: Document metadata

        Returns:
            Preprocessed content or empty string if filtered out
        """
        # 1. Basic length filtering
        if len(content.split()) < 3:
            return ""

        # 2. Size limits (avoid extremely large documents that cause memory issues)
        max_chars = 50000  # Similar to workspace_analysis_service.py
        if len(content) > max_chars:
            logging.warning(
                f"[DEBUG] Truncating document {doc_data.get('name', 'unknown')} from {len(content)} to {max_chars} chars"
            )
            content = content[:max_chars]

        # 3. Filter by content type - avoid binary-looking content
        # Check if content has too many non-text characters
        text_chars = sum(
            1 for c in content if c.isalnum() or c.isspace() or c in ".,!?;:"
        )
        text_ratio = text_chars / len(content) if content else 0

        if text_ratio < 0.7:  # Less than 70% text characters, likely binary
            logging.debug(
                f"[DEBUG] Filtering out likely binary content for {doc_data.get('name', 'unknown')} (text ratio: {text_ratio:.2f})"
            )
            return ""

        # 4. Remove excessive whitespace
        import re

        content = re.sub(r"\s+", " ", content.strip())

        # 5. Limit to reasonable document count for BERTopic (prevents memory issues)
        # This will be handled at a higher level, but individual doc limit is good too

        return content

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
        all_topics: Dict[int, List[Tuple[str, float]]],
        topic_info_df: Any,  # pandas DataFrame
        topics: List[int],
        probs: np.ndarray,
        file_data: List[Dict[str, Any]],
        workspace_id: int,
    ) -> List[TopicArea]:
        """Convert BERTopic results to TopicArea format."""
        topic_areas = []

        # Process each row in the topic info DataFrame
        for _, row in topic_info_df.iterrows():
            topic_id = int(row["Topic"])

            if topic_id == -1:  # Skip outliers
                continue

            # Get document count for this topic
            doc_count = int(row["Count"])
            if doc_count < self.min_topic_concepts:
                continue

            # Get topic name (automatically generated by BERTopic)
            topic_name = str(row["Name"]).strip()

            # Get top words from all_topics dict
            topic_words = all_topics.get(topic_id, [])

            # Create topic description
            topic_description = self._generate_topic_description_from_bertopic(
                {"name": topic_name, "count": doc_count, "words": topic_words}
            )

            # Calculate coverage score based on document count and words
            coverage_score = min(1.0, (len(topic_words) + doc_count) / 20.0)

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
                "top_words": topic_words,
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

    async def _prepare_documents(
        self, concepts_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Prepare documents for BERTopic processing (compatibility method for tests)."""
        logging.warning(
            "_prepare_documents is deprecated, use extract_topics with file_data"
        )
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
