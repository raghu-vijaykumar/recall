"""
Heuristic-based topic extractor for workspace analysis.
Uses embedding-based clustering to discover topic areas from documents.
Works directly with documents for heuristic topic discovery.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import uuid

from .base import BaseTopicExtractor
from ...models import TopicArea, TopicConceptLink
from ..embedding_service import EmbeddingService


class HeuristicExtractor(BaseTopicExtractor):
    """
    Heuristic topic extractor using embeddings and clustering for topic discovery.
    Works directly with documents using preprocessing and similarity clustering.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        similarity_threshold: float = 0.7,
        coverage_weight: float = 0.6,
        explored_weight: float = 0.4,
        # Document preprocessing parameters
        min_word_count: int = 5,
        max_cluster_ratio: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self.similarity_threshold = similarity_threshold
        self.coverage_weight = coverage_weight
        self.explored_weight = explored_weight
        self.min_word_count = min_word_count
        self.max_cluster_ratio = max_cluster_ratio

        logging.info(
            f"Initialized HeuristicExtractor with similarity_threshold={similarity_threshold}, coverage_weight={coverage_weight}, explored_weight={explored_weight}, min_word_count={min_word_count}, max_cluster_ratio={max_cluster_ratio}"
        )

    async def extract_topics(
        self, workspace_id: int, documents_data: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Extract topic areas directly from documents using heuristic embedding-based clustering.
        Returns both topic areas and document links connecting documents to topic areas.
        """
        logging.info(
            f"Starting heuristic extraction for workspace {workspace_id} with {len(documents_data)} documents"
        )

        if len(documents_data) < self.min_topic_concepts:
            logging.warning(
                f"Insufficient documents ({len(documents_data)}) for heuristic extraction, minimum required: {self.min_topic_concepts}"
            )
            return [], []

        logging.info(
            f"Preprocessing {len(documents_data)} documents for heuristic clustering"
        )

        # Preprocess documents for topic extraction
        processed_documents = await self._preprocess_documents(documents_data)

        if not processed_documents:
            logging.warning("No documents could be processed, creating fallback topic")
            return await self._create_fallback_topic_area(workspace_id, documents_data)

        # Use heuristic clustering based on semantic similarity
        topic_areas, document_links = await self._heuristic_clustering(
            workspace_id, processed_documents, documents_data
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

        # Filter document links to only include those for the selected topic areas
        selected_topic_ids = {ta.topic_area_id for ta in topic_areas}
        document_links = [
            link for link in document_links if link.topic_area_id in selected_topic_ids
        ]

        logging.info(
            f"Returning {len(topic_areas)} topic areas and {len(document_links)} document links"
        )
        return topic_areas, document_links

    async def _preprocess_documents(
        self, documents_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Preprocess documents for topic extraction by extracting text and preparing features.
        """
        processed_docs = []

        for idx, doc_data in enumerate(documents_data):
            # Extract text content
            text_content = await self._extract_document_text(doc_data)
            if not text_content:
                continue

            # Create enriched document representation
            processed_doc = {
                "id": doc_data.get("id") or f"doc_{idx}",
                "original_data": doc_data,
                "text": text_content,
                "word_count": len(text_content.split()),
                "file_path": doc_data.get("file_path", ""),
                "relevance_score": 1.0,  # Default relevance for documents
            }

            if processed_doc["word_count"] >= self.min_word_count:
                processed_docs.append(processed_doc)

        logging.info(
            f"Preprocessed {len(processed_docs)} documents with sufficient content (min {self.min_word_count} words)"
        )
        return processed_docs

    async def _extract_document_text(self, document_data: Dict[str, Any]) -> str:
        """Extract text content from document data."""
        # Primary content fields
        text_fields = ["content", "text", "body", "document"]

        for field in text_fields:
            content = document_data.get(field, "").strip()
            if content and len(content.split()) >= 3:  # At least 3 words
                return content

        # If no primary field has sufficient content, return empty
        return ""

    async def _heuristic_clustering(
        self,
        workspace_id: int,
        processed_documents: List[Dict[str, Any]],
        original_documents: List[Dict[str, Any]],
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """
        Heuristic clustering using embeddings and similarity-based approach.
        Returns both topic areas and document links.
        """
        logging.info(
            f"Starting heuristic clustering for {len(processed_documents)} processed documents"
        )

        if not self.embedding_service:
            logging.warning(
                "No embedding service available, creating general topic area"
            )
            return await self._create_single_topic_area(
                workspace_id, processed_documents, original_documents
            )

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
                        workspace_id, processed_documents, original_documents
                    )
            except Exception as e:
                logging.error(
                    f"Error initializing embedding service: {e}, falling back to single topic"
                )
                return await self._create_single_topic_area(
                    workspace_id, processed_documents, original_documents
                )

        try:
            # Generate embeddings for document texts
            document_texts = [doc["text"] for doc in processed_documents]
            embeddings = await self.embedding_service.embed_texts(document_texts)
            logging.info(f"Generated embeddings for {len(embeddings)} documents")

            # Perform similarity clustering
            clusters = self._similarity_clustering(embeddings, processed_documents)

            # Convert clusters to topic areas and document links
            topic_areas = []
            document_links = []
            for cluster_id, cluster_docs in clusters.items():
                if len(cluster_docs) < self.min_topic_concepts:
                    continue

                topic_name = self._generate_topic_name_from_documents(cluster_docs)
                topic_description = self._generate_topic_description_from_documents(
                    cluster_docs
                )

                avg_relevance = sum(
                    doc["relevance_score"] for doc in cluster_docs
                ) / len(cluster_docs)

                topic_area_id = f"heuristic_{workspace_id}_{cluster_id}"
                topic_area = TopicArea(
                    topic_area_id=topic_area_id,
                    workspace_id=workspace_id,
                    name=topic_name,
                    description=topic_description,
                    coverage_score=min(1.0, avg_relevance),
                    concept_count=len(cluster_docs),  # Documents treated as "concepts"
                    file_count=0,  # Will be set by service
                    explored_percentage=0.0,  # Will be calculated later
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                topic_areas.append(topic_area)

                # Create document links for this topic area
                for doc in cluster_docs:
                    original_doc_data = doc["original_data"]
                    concept_id = doc["id"]  # Use document ID

                    link = TopicConceptLink(
                        topic_concept_link_id=str(uuid.uuid4()),
                        topic_area_id=topic_area_id,
                        concept_id=concept_id,
                        relevance_score=doc["relevance_score"],
                        explored=False,  # Will be updated by service
                    )
                    document_links.append(link)

                logging.info(
                    f"Created topic area '{topic_name}' with {len(cluster_docs)} documents"
                )

            logging.info(
                f"Created {len(topic_areas)} topic areas and {len(document_links)} document links through heuristic clustering"
            )
            return topic_areas, document_links

        except Exception as e:
            logging.error(
                f"Error in heuristic clustering: {e}, falling back to single topic"
            )
            return await self._create_single_topic_area(
                workspace_id, processed_documents, original_documents
            )

    def _similarity_clustering(
        self, embeddings: List[List[float]], processed_documents: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Similarity-based document clustering.
        """
        logging.info(
            f"Starting similarity clustering with {len(processed_documents)} documents and {len(embeddings)} embeddings"
        )

        clusters = {}
        cluster_id_counter = 0

        # Sort documents by relevance score for processing order
        indexed_documents = sorted(
            enumerate(processed_documents),
            key=lambda x: x[1]["relevance_score"],
            reverse=True,
        )
        logging.info(
            f"Processing {len(indexed_documents)} documents in relevance order"
        )

        for idx, document in indexed_documents:
            # Find the best cluster for this document
            best_cluster = None
            best_similarity = -1

            for cluster_id, cluster_documents in clusters.items():
                # Calculate average similarity to cluster members
                similarities = []
                for cluster_document in cluster_documents[
                    :5  # Limit to avoid too many calculations
                ]:
                    cluster_idx = processed_documents.index(cluster_document)
                    sim = self._cosine_similarity(
                        embeddings[idx], embeddings[cluster_idx]
                    )
                    similarities.append(sim)

                avg_similarity = (
                    sum(similarities) / len(similarities) if similarities else 0
                )

                if (
                    avg_similarity > best_similarity
                    and avg_similarity > self.similarity_threshold
                ):
                    best_similarity = avg_similarity
                    best_cluster = cluster_id

            if best_cluster:
                clusters[best_cluster].append(document)
                logging.debug(
                    f"Added document '{document['id']}' to cluster {best_cluster} with similarity {best_similarity:.3f}"
                )
            else:
                # Create new cluster
                new_cluster_id = f"cluster_{cluster_id_counter}"
                clusters[new_cluster_id] = [document]
                cluster_id_counter += 1
                logging.debug(
                    f"Created new cluster {new_cluster_id} for document '{document['id']}'"
                )

        logging.info(f"Initial clustering created {len(clusters)} clusters")

        # Filter clusters to ensure minimum size
        filtered_clusters = {
            cid: documents
            for cid, documents in clusters.items()
            if len(documents) >= self.min_topic_concepts
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
            smallest_cid, smallest_docs = smallest_cluster

            # Find best cluster to merge with
            best_merge_cid = None
            best_merge_sim = -1

            for cid, documents in filtered_clusters.items():
                if cid == smallest_cid:
                    continue

                # Calculate similarity between cluster centroids
                sim = self._calculate_cluster_similarity(
                    embeddings, processed_documents, smallest_docs, documents
                )

                if sim > best_merge_sim:
                    best_merge_sim = sim
                    best_merge_cid = cid

            if best_merge_cid:
                filtered_clusters[best_merge_cid].extend(smallest_docs)
                del filtered_clusters[smallest_cid]
                merge_count += 1
                logging.debug(
                    f"Merged cluster {smallest_cid} ({len(smallest_docs)} docs) into {best_merge_cid} with similarity {best_merge_sim:.3f}"
                )
            else:
                # If no good merge found, just remove the smallest
                del filtered_clusters[smallest_cid]
                logging.debug(
                    f"Removed cluster {smallest_cid} ({len(smallest_docs)} docs) - no suitable merge found"
                )

        if merge_count > 0:
            logging.info(
                f"Merged {merge_count} clusters, final count: {len(filtered_clusters)}"
            )

        return filtered_clusters

    def _calculate_cluster_similarity(
        self,
        embeddings: List[List[float]],
        processed_documents: List[Dict[str, Any]],
        cluster1: List[Dict[str, Any]],
        cluster2: List[Dict[str, Any]],
    ) -> float:
        """Calculate similarity between two document clusters"""
        logging.debug(
            f"Calculating similarity between clusters with {len(cluster1)} and {len(cluster2)} documents"
        )

        similarities = []
        for d1 in cluster1[:3]:  # Sample a few documents
            for d2 in cluster2[:3]:
                idx1 = processed_documents.index(d1)
                idx2 = processed_documents.index(d2)
                sim = self._cosine_similarity(embeddings[idx1], embeddings[idx2])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        logging.debug(
            f"Cluster similarity calculated: {avg_similarity:.3f} from {len(similarities)} document pairs"
        )

        return avg_similarity

    async def _create_single_topic_area(
        self,
        workspace_id: int,
        processed_documents: List[Dict[str, Any]],
        original_documents: List[Dict[str, Any]],
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """Create a single general topic area as fallback"""
        logging.info(
            f"Creating single general topic area for workspace {workspace_id} with {len(processed_documents)} documents"
        )

        avg_relevance = sum(
            doc["relevance_score"] for doc in processed_documents
        ) / len(processed_documents)
        logging.info(
            f"Calculated average relevance: {avg_relevance:.3f} for {len(processed_documents)} documents"
        )

        topic_area_id = f"heuristic_{workspace_id}_general"
        topic_area = TopicArea(
            topic_area_id=topic_area_id,
            workspace_id=workspace_id,
            name="General Documents",
            description=f"General topic area covering {len(processed_documents)} documents using heuristic extraction",
            coverage_score=min(1.0, avg_relevance),
            concept_count=len(processed_documents),  # Documents as concepts
            file_count=0,
            explored_percentage=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Create document links for the general topic area
        document_links = []
        for doc in processed_documents:
            link = TopicConceptLink(
                topic_concept_link_id=str(uuid.uuid4()),
                topic_area_id=topic_area_id,
                concept_id=doc["id"],
                relevance_score=doc["relevance_score"],
                explored=False,  # Will be updated by service
            )
            document_links.append(link)

        logging.info(
            f"Created fallback topic area with {len(document_links)} document links"
        )
        return [topic_area], document_links

    async def _create_fallback_topic_area(
        self, workspace_id: int, original_documents: List[Dict[str, Any]]
    ) -> Tuple[List[TopicArea], List[TopicConceptLink]]:
        """Create a fallback topic when no documents can be processed"""
        logging.warning("Creating fallback topic area for unprocessed documents")

        topic_area_id = f"heuristic_{workspace_id}_fallback"
        topic_area = TopicArea(
            topic_area_id=topic_area_id,
            workspace_id=workspace_id,
            name="All Documents",
            description=f"Fallback topic area containing all {len(original_documents)} documents",
            coverage_score=0.5,  # Default score
            concept_count=len(original_documents),
            file_count=0,
            explored_percentage=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Create links for all documents
        document_links = []
        for i, doc_data in enumerate(original_documents):
            concept_id = doc_data.get("id") or f"doc_{i}"
            link = TopicConceptLink(
                topic_concept_link_id=str(uuid.uuid4()),
                topic_area_id=topic_area_id,
                concept_id=concept_id,
                relevance_score=0.5,  # Default relevance
                explored=False,
            )
            document_links.append(link)

        return [topic_area], document_links

    def _generate_topic_name_from_documents(
        self, cluster_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate topic name from clustered documents"""
        if not cluster_documents:
            return "General Topic"

        # Extract common words/themes from document texts
        # Simple approach: use first document's first few words + document count
        first_doc_text = cluster_documents[0]["text"]
        words = first_doc_text.split()[:4]  # First 4 words

        if len(cluster_documents) > 1:
            # Pluralize if multiple documents
            base_name = " ".join(words).title()
            topic_name = f"{base_name} and Related"
        else:
            topic_name = " ".join(words).title()

        # Ensure reasonable length
        if len(topic_name) > 50:
            topic_name = topic_name[:47] + "..."

        return topic_name

    def _generate_topic_description_from_documents(
        self, cluster_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate topic description from clustered documents"""
        if not cluster_documents:
            return "General topic description"

        doc_count = len(cluster_documents)
        word_counts = [doc["word_count"] for doc in cluster_documents]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        description = f"Heuristic topic containing {doc_count} document"
        if doc_count != 1:
            description += "s"

        description += f" with average {avg_words:.0f} words per document."

        # Add information about file paths if available
        file_paths = [
            doc.get("file_path", "")
            for doc in cluster_documents
            if doc.get("file_path")
        ]
        if file_paths and len(file_paths) <= 3:
            path_info = ", ".join([path.split("/")[-1] for path in file_paths[:3]])
            description += f" Files: {path_info}"

        return description

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
