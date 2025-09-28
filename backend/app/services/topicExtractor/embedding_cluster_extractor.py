"""
Embedding-based clustering topic extractor.
Uses embeddings and clustering algorithms to discover topic areas.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import uuid

from .base import BaseTopicExtractor
from ...models import TopicArea, TopicConceptLink
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

        logging.info(
            f"Initialized EmbeddingClusterExtractor with similarity_threshold={similarity_threshold}, coverage_weight={coverage_weight}, explored_weight={explored_weight}"
        )

    async def extract_topics(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Extract topic areas using embedding-based clustering.
        Returns both topic areas and the concept links that connect them.
        """
        logging.info(
            f"Starting topic extraction for workspace {workspace_id} with {len(concepts_data)} concepts"
        )

        if len(concepts_data) < self.min_topic_concepts:
            logging.warning(
                f"Insufficient concepts ({len(concepts_data)}) for topic extraction, minimum required: {self.min_topic_concepts}"
            )
            return [], []

        logging.info(f"Using embeddings for clustering {len(concepts_data)} concepts")

        # Use generalized clustering based on semantic similarity
        topic_areas, concept_links = await self._cluster_generalized(
            workspace_id, concepts_data
        )

        logging.info(
            f"Clustering completed, found {len(topic_areas)} topic areas before filtering"
        )

        # Sort topic areas by score and limit to top areas
        topic_areas = sorted(
            topic_areas,
            key=lambda ta: ta.coverage_score * self.coverage_weight
            + ta.explored_percentage * self.explored_weight,
            reverse=True,
        )[: self.max_topic_areas]

        logging.info(
            f"After filtering, returning top {len(topic_areas)} topic areas (max allowed: {self.max_topic_areas})"
        )

        # Filter concept links to only include those for the selected topic areas
        selected_topic_ids = {ta.topic_area_id for ta in topic_areas}
        concept_links = [
            link for link in concept_links if link.topic_area_id in selected_topic_ids
        ]

        logging.info(
            f"Returning {len(topic_areas)} topic areas and {len(concept_links)} concept links"
        )
        return topic_areas, concept_links

    async def _cluster_generalized(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Generalized clustering using embeddings and simple centroid-based approach.
        Returns both topic areas and concept links.
        """
        logging.info(
            f"Starting generalized clustering for {len(concepts_data)} concepts"
        )

        if not self.embedding_service:
            logging.warning(
                "No embedding service available, creating single general topic area"
            )
            # Fallback: create a single general topic area
            topic_areas, concept_links = await self._create_single_topic_area(
                workspace_id, concepts_data
            )
            return topic_areas, concept_links

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
                    topic_areas, concept_links = await self._create_single_topic_area(
                        workspace_id, concepts_data
                    )
                    return topic_areas, concept_links
            except Exception as e:
                logging.error(
                    f"Error initializing embedding service: {e}, falling back to single topic"
                )
                topic_areas, concept_links = await self._create_single_topic_area(
                    workspace_id, concepts_data
                )
                return topic_areas, concept_links

        try:
            # Generate embeddings
            concept_names = [c["name"] for c in concepts_data]
            embeddings = await self.embedding_service.embed_texts(concept_names)
            logging.info(f"Generated embeddings for {len(embeddings)} concepts")

            # Simple clustering: group by similarity to centroids
            # Use a simple approach: find natural clusters by similarity
            clusters = self._simple_similarity_clustering(embeddings, concepts_data)

            # Convert clusters to topic areas and concept links
            topic_areas = []
            concept_links = []
            for cluster_id, cluster_concepts in clusters.items():
                if len(cluster_concepts) < self.min_topic_concepts:
                    continue

                topic_name = self._generate_topic_name(cluster_concepts)
                topic_description = self._generate_topic_description(cluster_concepts)

                avg_relevance = sum(
                    c["relevance_score"] for c in cluster_concepts
                ) / len(cluster_concepts)

                topic_area_id = f"topic_{workspace_id}_{cluster_id}"
                topic_area = TopicArea(
                    topic_area_id=topic_area_id,
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

                # Create concept links for this topic area
                for concept in cluster_concepts:
                    link = TopicConceptLink(
                        topic_concept_link_id=str(uuid.uuid4()),
                        topic_area_id=topic_area_id,
                        concept_id=concept["id"],
                        relevance_score=concept["relevance_score"],
                        explored=False,  # Will be updated by service
                    )
                    concept_links.append(link)

                logging.info(
                    f"Created topic area '{topic_name}' with {len(cluster_concepts)} concepts"
                )

            logging.info(
                f"Created {len(topic_areas)} topic areas and {len(concept_links)} concept links through generalized clustering"
            )
            return topic_areas, concept_links

        except Exception as e:
            logging.error(
                f"Error in generalized clustering: {e}, falling back to single topic"
            )
            topic_areas, concept_links = await self._create_single_topic_area(
                workspace_id, concepts_data
            )
            return topic_areas, concept_links

    def _simple_similarity_clustering(
        self, embeddings: List[List[float]], concepts_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Simple clustering based on similarity thresholds.
        """
        logging.info(
            f"Starting similarity clustering with {len(concepts_data)} concepts and {len(embeddings)} embeddings"
        )

        clusters = {}
        cluster_id_counter = 0

        # Sort concepts by relevance score for processing order
        indexed_concepts = sorted(
            enumerate(concepts_data),
            key=lambda x: x[1]["relevance_score"],
            reverse=True,
        )
        logging.info(f"Processing {len(indexed_concepts)} concepts in relevance order")

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
                logging.debug(
                    f"Added concept '{concept['name']}' to cluster {best_cluster} with similarity {best_similarity:.3f}"
                )
            else:
                # Create new cluster
                new_cluster_id = f"cluster_{cluster_id_counter}"
                clusters[new_cluster_id] = [concept]
                cluster_id_counter += 1
                logging.debug(
                    f"Created new cluster {new_cluster_id} for concept '{concept['name']}'"
                )

        logging.info(f"Initial clustering created {len(clusters)} clusters")

        # Filter clusters to ensure minimum size
        filtered_clusters = {
            cid: concepts
            for cid, concepts in clusters.items()
            if len(concepts) >= self.min_topic_concepts
        }
        logging.info(
            f"After filtering, {len(filtered_clusters)} clusters meet minimum size requirement ({self.min_topic_concepts})"
        )

        # If we have too many clusters, merge smaller ones
        initial_count = len(filtered_clusters)
        merge_count = 0
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
                merge_count += 1
                logging.debug(
                    f"Merged cluster {smallest_cid} ({len(smallest_concepts)} concepts) into {best_merge_cid} with similarity {best_merge_sim:.3f}"
                )
            else:
                # If no good merge found, just remove the smallest
                del filtered_clusters[smallest_cid]
                logging.debug(
                    f"Removed cluster {smallest_cid} ({len(smallest_concepts)} concepts) - no suitable merge found"
                )

        if merge_count > 0:
            logging.info(
                f"Merged {merge_count} clusters, final count: {len(filtered_clusters)}"
            )

        return filtered_clusters

    def _calculate_cluster_similarity(
        self,
        embeddings: List[List[float]],
        concepts_data: List[Dict[str, Any]],
        cluster1: List[Dict[str, Any]],
        cluster2: List[Dict[str, Any]],
    ) -> float:
        """Calculate similarity between two clusters"""
        logging.debug(
            f"Calculating similarity between clusters with {len(cluster1)} and {len(cluster2)} concepts"
        )

        similarities = []
        for c1 in cluster1[:3]:  # Sample a few concepts
            for c2 in cluster2[:3]:
                idx1 = concepts_data.index(c1)
                idx2 = concepts_data.index(c2)
                sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        logging.debug(
            f"Cluster similarity calculated: {avg_similarity:.3f} from {len(similarities)} concept pairs"
        )

        return avg_similarity

    async def _create_single_topic_area(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """Create a single general topic area as fallback"""
        logging.info(
            f"Creating single general topic area for workspace {workspace_id} with {len(concepts_data)} concepts"
        )

        avg_relevance = sum(c["relevance_score"] for c in concepts_data) / len(
            concepts_data
        )
        logging.info(
            f"Calculated average relevance: {avg_relevance:.3f} for {len(concepts_data)} concepts"
        )

        topic_area_id = f"topic_{workspace_id}_general"
        topic_area = TopicArea(
            topic_area_id=topic_area_id,
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

        # Create concept links for the general topic area
        concept_links = []
        for concept in concepts_data:
            link = TopicConceptLink(
                topic_concept_link_id=str(uuid.uuid4()),
                topic_area_id=topic_area_id,
                concept_id=concept["id"],
                relevance_score=concept["relevance_score"],
                explored=False,  # Will be updated by service
            )
            concept_links.append(link)

        logging.info(
            f"Created fallback topic area with {len(concept_links)} concept links"
        )
        return [topic_area], concept_links

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            logging.debug("Cosine similarity: zero vector detected, returning 0.0")
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        logging.debug(
            f"Cosine similarity calculated: {similarity:.3f} (dot_product={dot_product:.3f}, norm1={norm1:.3f}, norm2={norm2:.3f})"
        )
        return similarity
