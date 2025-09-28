"""
BERTopic-based topic extractor for workspace analysis.
Uses transformer embeddings and advanced clustering for superior topic discovery.
"""

import math
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import uuid
import numpy as np
from collections import defaultdict

from .base import BaseTopicExtractor
from ...models import TopicArea, TopicConceptLink
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
        use_hierarchy: bool = True,
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
        self.use_hierarchy = use_hierarchy

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
        self._bertopic_model = None  # Lazy initialization
        self._model_cache = {}

        logging.info(
            f"Initialized BERTopicExtractor with model={model_name}, min_topic_size={min_topic_size}, "
            f"nr_topics={nr_topics}, diversity={diversity}"
        )

    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Extract topic areas using BERTopic.
        """
        logging.info(
            f"Starting BERTopic extraction for workspace {workspace_id} with {len(concepts_data)} concepts"
        )

        if len(concepts_data) < self.min_topic_concepts:
            logging.warning(
                f"Insufficient concepts ({len(concepts_data)}) for BERTopic extraction, minimum required: {self.min_topic_concepts}"
            )
            return [], []

        # Prepare documents for BERTopic
        documents = await self._prepare_documents(concepts_data)
        logging.info(f"Prepared {len(documents)} documents for BERTopic processing")

        # Get or create BERTopic model
        bertopic_model = await self._get_bertopic_model()

        try:
            # Fit BERTopic model
            logging.info("Fitting BERTopic model...")
            topics, probs = bertopic_model.fit_transform(documents)

            # Get topic information
            topic_info = bertopic_model.get_topic_info()
            logging.info(
                f"BERTopic found {len(topic_info)} topics (including outliers)"
            )

            # Convert to our format
            topic_areas, concept_links = await self._convert_bertopic_results(
                bertopic_model, topics, probs, concepts_data, workspace_id
            )

            # Filter and rank topics by quality
            topic_areas = await self._filter_and_rank_topics(
                topic_areas, bertopic_model
            )

            # Limit to max_topic_areas
            topic_areas = topic_areas[: self.max_topic_areas]

            logging.info(
                f"BERTopic extraction completed: {len(topic_areas)} topic areas, {len(concept_links)} concept links"
            )
            return topic_areas, concept_links

        except Exception as e:
            logging.error(f"Error in BERTopic extraction: {e}")
            # Fallback to simple clustering
            logging.info("Falling back to simple clustering approach")
            return await self._fallback_clustering(workspace_id, concepts_data)

    async def _prepare_documents(
        self, concepts_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Prepare concept data as documents for BERTopic"""
        documents = []

        for concept in concepts_data:
            # Create rich document representation
            doc_parts = []

            # Add concept name (most important)
            if concept.get("name"):
                doc_parts.append(concept["name"])

            # Add description if available
            if concept.get("description"):
                doc_parts.append(concept["description"])

            # Add context from related concepts if available
            # This could be enhanced with knowledge graph relationships
            context = self._get_concept_context(concept.get("id"))
            if context:
                doc_parts.extend(context)

            # Join parts into a coherent document
            document = " ".join(doc_parts).strip()
            if document:
                documents.append(document)
            else:
                # Fallback to just the name
                documents.append(concept.get("name", "unknown"))

        return documents

    def _get_concept_context(self, concept_id: str) -> List[str]:
        """Get additional context for a concept (placeholder for future enhancement)"""
        # This could be enhanced to query the knowledge graph for related concepts
        return []

    async def _get_bertopic_model(self) -> BERTopic:
        """Get or create BERTopic model with caching"""
        # Create cache key from configuration
        cache_key = self._get_model_cache_key()

        if cache_key in self._model_cache:
            logging.info(f"Using cached BERTopic model for key: {cache_key}")
            return self._model_cache[cache_key]

        # Create new model
        logging.info(f"Creating new BERTopic model with key: {cache_key}")

        # Initialize BERTopic with custom components
        bertopic_model = BERTopic(
            # Model configuration
            embedding_model=self.model_name,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            representation_model=self.representation_model,
            # Topic configuration
            nr_topics=self.nr_topics,
            min_topic_size=self.min_topic_size,
            # Behavior configuration
            calculate_probabilities=self.calculate_probabilities,
            verbose=True,
        )

        # Cache the model
        self._model_cache[cache_key] = bertopic_model

        return bertopic_model

    def _get_model_cache_key(self) -> str:
        """Generate cache key for model configuration"""
        return (
            f"{self.model_name}_{self.min_topic_size}_{self.nr_topics}_{self.diversity}"
        )

    async def _convert_bertopic_results(
        self,
        bertopic_model: BERTopic,
        topics: List[int],
        probs: np.ndarray,
        concepts_data: List[Dict[str, Any]],
        workspace_id: int,
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """Convert BERTopic results to our TopicArea and TopicConceptLink format"""

        # Get topic information
        topic_info = bertopic_model.get_topic_info()

        # Group concepts by topic
        topic_concepts = defaultdict(list)
        topic_probs = defaultdict(list)

        for i, (topic_id, concept) in enumerate(zip(topics, concepts_data)):
            if topic_id != -1:  # Skip outliers
                topic_concepts[topic_id].append(concept)
                if probs is not None and i < len(probs):
                    topic_probs[topic_id].append(probs[i])

        # Create topic areas
        topic_areas = []
        concept_links = []

        for topic_id in topic_concepts:
            if topic_id == -1:  # Skip outliers
                continue

            concepts = topic_concepts[topic_id]
            if len(concepts) < self.min_topic_concepts:
                continue

            # Get topic words and their scores
            topic_words = bertopic_model.get_topic(topic_id)

            # Create topic name from top words
            topic_name = self._generate_bertopic_name(topic_words)

            # Create topic description
            topic_description = self._generate_bertopic_description(
                topic_words, concepts
            )

            # Calculate topic metrics
            avg_relevance = sum(c.get("relevance_score", 0.5) for c in concepts) / len(
                concepts
            )

            # Get topic probabilities for this topic
            topic_prob_list = topic_probs.get(topic_id, [])
            avg_probability = np.mean(topic_prob_list) if topic_prob_list else 0.5

            # Create enhanced topic area
            topic_area = TopicArea(
                topic_area_id=f"bertopic_{workspace_id}_{topic_id}",
                workspace_id=workspace_id,
                name=topic_name,
                description=topic_description,
                coverage_score=min(1.0, avg_relevance * avg_probability),
                concept_count=len(concepts),
                file_count=0,  # Will be calculated by service
                explored_percentage=0.0,  # Will be calculated later
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Add BERTopic-specific metadata
            topic_area._bertopic_metadata = {
                "topic_id": topic_id,
                "topic_words": topic_words,
                "avg_probability": avg_probability,
                "coherence_score": self._calculate_topic_coherence(topic_words),
                "outlier_score": self._calculate_outlier_score(concepts),
            }

            topic_areas.append(topic_area)

            # Create concept links with BERTopic probabilities
            for concept in concepts:
                # Get probability for this concept in this topic
                concept_idx = concepts_data.index(concept)
                probability = 0.5  # Default

                if probs is not None and concept_idx < len(probs):
                    topic_probabilities = probs[concept_idx]
                    if topic_probabilities is not None and topic_id < len(
                        topic_probabilities
                    ):
                        probability = topic_probabilities[topic_id]

                link = TopicConceptLink(
                    topic_concept_link_id=str(uuid.uuid4()),
                    topic_area_id=topic_area.topic_area_id,
                    concept_id=concept["id"],
                    relevance_score=min(
                        1.0, concept.get("relevance_score", 0.5) * probability
                    ),
                    explored=False,  # Will be updated by service
                )

                # Add BERTopic-specific metadata to link
                link._bertopic_metadata = {
                    "topic_probability": probability,
                    "topic_id": topic_id,
                }

                concept_links.append(link)

        return topic_areas, concept_links

    def _generate_bertopic_name(self, topic_words: List[Tuple[str, float]]) -> str:
        """Generate topic name from BERTopic words"""
        if not topic_words:
            return "General Topic"

        # Use top 2-3 most relevant words
        top_words = [word for word, score in topic_words[:3]]
        name = " ".join(top_words).title()

        # Ensure name is not too long
        if len(name) > 50:
            name = name[:47] + "..."

        return name

    def _generate_bertopic_description(
        self, topic_words: List[Tuple[str, float]], concepts: List[Dict[str, Any]]
    ) -> str:
        """Generate topic description from BERTopic words and concepts"""
        # Start with concept names
        concept_names = [c.get("name", "") for c in concepts[:5]]
        concept_str = ", ".join([name for name in concept_names if name])

        if len(concepts) > 5:
            concept_str += f", and {len(concepts) - 5} more"

        # Add top words context
        if topic_words:
            word_str = ", ".join([word for word, score in topic_words[:5]])
            return f"Topic covering concepts like {concept_str}. Key themes: {word_str}"
        else:
            return f"Topic covering concepts like {concept_str}"

    def _calculate_topic_coherence(self, topic_words: List[Tuple[str, float]]) -> float:
        """Calculate coherence score for a topic based on word associations"""
        if len(topic_words) < 2:
            return 0.0

        # Simple coherence measure based on word scores
        scores = [score for word, score in topic_words[:5]]
        if not scores:
            return 0.0

        # Normalize scores and calculate variance (lower variance = higher coherence)
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Convert variance to coherence (inverse relationship)
        coherence = 1.0 / (1.0 + variance)

        return min(1.0, coherence)

    def _calculate_outlier_score(self, concepts: List[Dict[str, Any]]) -> float:
        """Calculate how likely this topic is to be an outlier"""
        if len(concepts) < 2:
            return 1.0

        # Simple heuristic: topics with very low average relevance are likely outliers
        avg_relevance = sum(c.get("relevance_score", 0.5) for c in concepts) / len(
            concepts
        )

        # Lower relevance = higher outlier score
        outlier_score = 1.0 - avg_relevance

        return min(1.0, outlier_score)

    async def _filter_and_rank_topics(
        self, topic_areas: List[TopicArea], bertopic_model: BERTopic
    ) -> List[TopicArea]:
        """Filter and rank topics by quality metrics"""

        # Calculate quality scores for each topic
        for topic_area in topic_areas:
            metadata = getattr(topic_area, "_bertopic_metadata", {})

            # Combine multiple quality metrics
            coherence = metadata.get("coherence_score", 0.0)
            outlier_score = metadata.get("outlier_score", 0.0)
            concept_count = topic_area.concept_count
            avg_probability = metadata.get("avg_probability", 0.5)

            # Quality score combines coherence, size, and probability
            # Penalize outliers and small topics
            size_bonus = min(1.0, concept_count / 10.0)  # Bonus for larger topics
            outlier_penalty = 1.0 - outlier_score

            quality_score = (
                coherence * 0.4
                + avg_probability * 0.3
                + size_bonus * 0.2
                + outlier_penalty * 0.1
            )

            topic_area._quality_score = quality_score

        # Filter by quality threshold
        filtered_topics = [
            ta
            for ta in topic_areas
            if getattr(ta, "_quality_score", 0.0) >= self.coherence_threshold
        ]

        # Sort by quality score
        filtered_topics.sort(
            key=lambda x: getattr(x, "_quality_score", 0.0), reverse=True
        )

        logging.info(
            f"Filtered {len(topic_areas)} topics to {len(filtered_topics)} high-quality topics "
            f"(threshold: {self.coherence_threshold})"
        )

        return filtered_topics

    async def _fallback_clustering(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """Fallback to simple clustering if BERTopic fails"""
        logging.warning("Using fallback clustering approach")

        # Simple approach: group by concept name similarity
        from .embedding_cluster_extractor import EmbeddingClusterExtractor

        fallback_extractor = EmbeddingClusterExtractor(
            embedding_service=self.embedding_service,
            similarity_threshold=0.6,
            min_topic_concepts=self.min_topic_concepts,
            max_topic_areas=self.max_topic_areas,
        )

        return await fallback_extractor.extract_topics(workspace_id, concepts_data)

    def get_topic_visualization_data(self, bertopic_model: BERTopic) -> Dict[str, Any]:
        """Get visualization data for BERTopic results"""
        try:
            # Get topic barchart data
            viz_data = {
                "topic_info": bertopic_model.get_topic_info().to_dict("records"),
                "topic_word_scores": {},
                "topic_sizes": bertopic_model.get_topic_info()["Count"].tolist(),
            }

            # Get word scores for each topic
            for topic_id in bertopic_model.get_topics():
                if topic_id != -1:  # Skip outliers
                    words = bertopic_model.get_topic(topic_id)
                    viz_data["topic_word_scores"][topic_id] = words

            return viz_data

        except Exception as e:
            logging.error(f"Error generating visualization data: {e}")
            return {}

    def get_topic_hierarchy(self, bertopic_model: BERTopic) -> Dict[int, List[int]]:
        """Extract hierarchical topic relationships"""
        try:
            # This is a simplified hierarchy extraction
            # In practice, you might use hierarchical BERTopic or custom clustering
            hierarchy = {}

            topic_info = bertopic_model.get_topic_info()
            topic_sizes = topic_info.set_index("Topic")["Count"].to_dict()

            # Group topics by size ranges (simple hierarchy)
            size_ranges = {"large": [], "medium": [], "small": []}

            for topic_id, size in topic_sizes.items():
                if topic_id == -1:
                    continue

                if size >= 20:
                    size_ranges["large"].append(topic_id)
                elif size >= 10:
                    size_ranges["medium"].append(topic_id)
                else:
                    size_ranges["small"].append(topic_id)

            # Create simple hierarchy (large topics contain medium, etc.)
            all_large = size_ranges["large"]
            all_medium = size_ranges["medium"]
            all_small = size_ranges["small"]

            for topic_id in all_medium:
                hierarchy[topic_id] = [
                    parent for parent in all_large if parent != topic_id
                ]

            for topic_id in all_small:
                hierarchy[topic_id] = [
                    parent for parent in all_medium + all_large if parent != topic_id
                ]

            return hierarchy

        except Exception as e:
            logging.error(f"Error extracting topic hierarchy: {e}")
            return {}
