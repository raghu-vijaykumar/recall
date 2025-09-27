"""
Embedding-based clustering topic extractor.
Uses embeddings and clustering algorithms to discover topic areas.
"""

import math
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base import BaseTopicExtractor
from ...models import TopicArea
from ..embedding_service import EmbeddingService


class EmbeddingClusterExtractor(BaseTopicExtractor):
    """
    Topic extractor that uses embeddings and clustering for topic discovery.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.7,
        coverage_weight: float = 0.6,
        explored_weight: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.similarity_threshold = similarity_threshold
        self.coverage_weight = coverage_weight
        self.explored_weight = explored_weight

    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Extract topic areas using embedding-based clustering.
        """
        if len(concepts_data) < self.min_topic_concepts:
            return []

        logging.info(f"Using embeddings for clustering {len(concepts_data)} concepts")

        # Use generalized clustering based on semantic similarity
        topic_areas = await self._cluster_generalized(workspace_id, concepts_data)

        # Limit to top topic areas
        topic_areas = sorted(
            topic_areas,
            key=lambda ta: ta.coverage_score * self.coverage_weight
            + ta.explored_percentage * self.explored_weight,
            reverse=True,
        )[: self.max_topic_areas]

        return topic_areas

    async def _cluster_generalized(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Generalized clustering using embeddings and simple centroid-based approach.
        """
        logging.info(
            f"Starting generalized clustering for {len(concepts_data)} concepts"
        )

        if not self.embedding_service:
            logging.warning(
                "No embedding service available, creating single general topic area"
            )
            # Fallback: create a single general topic area
            return await self._create_single_topic_area(workspace_id, concepts_data)

        # Ensure embedding service is initialized
        if not self.embedding_service.current_model:
            logging.info(
                "Embedding service not initialized, initializing with default model"
            )
            try:
                success = await self.embedding_service.initialize()
                if not success:
                    logging.error(
                        "Failed to initialize embedding service, falling back to single topic"
                    )
                    return await self._create_single_topic_area(
                        workspace_id, concepts_data
                    )
            except Exception as e:
                logging.error(
                    f"Error initializing embedding service: {e}, falling back to single topic"
                )
                return await self._create_single_topic_area(workspace_id, concepts_data)

        try:
            # Generate embeddings
            concept_names = [c["name"] for c in concepts_data]
            embeddings = await self.embedding_service.embed_texts(concept_names)
            logging.info(f"Generated embeddings for {len(embeddings)} concepts")

            # Simple clustering: group by similarity to centroids
            # Use a simple approach: find natural clusters by similarity
            clusters = self._simple_similarity_clustering(embeddings, concepts_data)

            # Convert clusters to topic areas
            topic_areas = []
            for cluster_id, cluster_concepts in clusters.items():
                if len(cluster_concepts) < self.min_topic_concepts:
                    continue

                topic_name = self._generate_topic_name(cluster_concepts)
                topic_description = self._generate_topic_description(cluster_concepts)

                avg_relevance = sum(
                    c["relevance_score"] for c in cluster_concepts
                ) / len(cluster_concepts)

                topic_area = TopicArea(
                    topic_area_id=f"topic_{workspace_id}_{cluster_id}",
                    workspace_id=workspace_id,
                    name=topic_name,
                    description=topic_description,
                    coverage_score=min(1.0, avg_relevance),
                    concept_count=len(cluster_concepts),
                    file_count=0,  # Will be calculated by service
                    explored_percentage=0.0,  # Will be calculated later
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                topic_areas.append(topic_area)

            logging.info(
                f"Created {len(topic_areas)} topic areas through generalized clustering"
            )
            return topic_areas

        except Exception as e:
            logging.error(
                f"Error in generalized clustering: {e}, falling back to single topic"
            )
            return await self._create_single_topic_area(workspace_id, concepts_data)

    def _simple_similarity_clustering(
        self, embeddings: List[List[float]], concepts_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simple clustering based on similarity thresholds.
        """
        clusters = {}
        cluster_id_counter = 0

        # Sort concepts by relevance score for processing order
        indexed_concepts = sorted(
            enumerate(concepts_data),
            key=lambda x: x[1]["relevance_score"],
            reverse=True,
        )

        for idx, concept in indexed_concepts:
            # Find the best cluster for this concept
            best_cluster = None
            best_similarity = -1

            for cluster_id, cluster_concepts in clusters.items():
                # Calculate average similarity to cluster members
                similarities = []
                for cluster_concept in cluster_concepts[
                    :5
                ]:  # Limit to avoid too many calculations
                    cluster_idx = concepts_data.index(cluster_concept)
                    sim = self._cosine_similarity(
                        embeddings[idx], embeddings[cluster_idx]
                    )
                    similarities.append(sim)

                avg_similarity = (
                    sum(similarities) / len(similarities) if similarities else 0
                )

                if (
                    avg_similarity > best_similarity and avg_similarity > 0.6
                ):  # Similarity threshold
                    best_similarity = avg_similarity
                    best_cluster = cluster_id

            if best_cluster:
                clusters[best_cluster].append(concept)
            else:
                # Create new cluster
                new_cluster_id = f"cluster_{cluster_id_counter}"
                clusters[new_cluster_id] = [concept]
                cluster_id_counter += 1

        # Filter clusters to ensure minimum size
        filtered_clusters = {
            cid: concepts
            for cid, concepts in clusters.items()
            if len(concepts) >= self.min_topic_concepts
        }

        # If we have too many clusters, merge smaller ones
        while len(filtered_clusters) > self.max_topic_areas:
            # Find smallest cluster
            smallest_cluster = min(filtered_clusters.items(), key=lambda x: len(x[1]))
            smallest_cid, smallest_concepts = smallest_cluster

            # Find best cluster to merge with
            best_merge_cid = None
            best_merge_sim = -1

            for cid, concepts in filtered_clusters.items():
                if cid == smallest_cid:
                    continue

                # Calculate similarity between cluster centroids
                sim = self._calculate_cluster_similarity(
                    embeddings, concepts_data, smallest_concepts, concepts
                )

                if sim > best_merge_sim:
                    best_merge_sim = sim
                    best_merge_cid = cid

            if best_merge_cid:
                filtered_clusters[best_merge_cid].extend(smallest_concepts)
                del filtered_clusters[smallest_cid]
            else:
                # If no good merge found, just remove the smallest
                del filtered_clusters[smallest_cid]

        return filtered_clusters

    def _calculate_cluster_similarity(
        self,
        embeddings: List[List[float]],
        concepts_data: List[Dict[str, Any]],
        cluster1: List[Dict[str, Any]],
        cluster2: List[Dict[str, Any]],
    ) -> float:
        """Calculate similarity between two clusters"""
        similarities = []
        for c1 in cluster1[:3]:  # Sample a few concepts
            for c2 in cluster2[:3]:
                idx1 = concepts_data.index(c1)
                idx2 = concepts_data.index(c2)
                sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0

    async def _create_single_topic_area(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """Create a single general topic area as fallback"""
        logging.info("Creating single general topic area")

        avg_relevance = sum(c["relevance_score"] for c in concepts_data) / len(
            concepts_data
        )

        topic_area = TopicArea(
            topic_area_id=f"topic_{workspace_id}_general",
            workspace_id=workspace_id,
            name="General Concepts",
            description=f"General topic area covering {len(concepts_data)} concepts",
            coverage_score=min(1.0, avg_relevance),
            concept_count=len(concepts_data),
            file_count=0,
            explored_percentage=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return [topic_area]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
