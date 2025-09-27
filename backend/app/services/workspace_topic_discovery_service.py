"""
Workspace Topic Discovery Service for identifying major subject areas and learning paths
Analyzes entire workspaces to discover topics and recommend learning trajectories
"""

import json
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import (
    TopicArea,
    TopicAreaCreate,
    TopicConceptLink,
    TopicConceptLinkCreate,
    LearningPath,
    LearningPathCreate,
    LearningRecommendation,
    LearningRecommendationCreate,
    WorkspaceTopicAnalysis,
)
from . import KnowledgeGraphService, WorkspaceAnalysisService
from .embedding_service import EmbeddingService


class WorkspaceTopicDiscoveryService:
    """
    Service for discovering major topic areas in workspaces and generating learning recommendations.
    Uses clustering and analysis to identify subject areas and suggest learning paths.
    """

    def __init__(
        self,
        db: AsyncSession,
        kg_service: Optional[KnowledgeGraphService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        workspace_analysis_service: Optional["WorkspaceAnalysisService"] = None,
    ):
        self.db = db
        self.kg_service = kg_service or KnowledgeGraphService(db)
        self.embedding_service = embedding_service
        # Initialize embedding service if not provided
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService.get_instance()

        # Initialize workspace analysis service if not provided
        self.workspace_analysis_service = workspace_analysis_service

        # Analysis configuration
        self.min_topic_concepts = 3  # Minimum concepts per topic area
        self.max_topic_areas = 20  # Maximum topic areas to identify
        self.similarity_threshold = 0.7  # Threshold for concept clustering
        self.coverage_weight = 0.6  # Weight for coverage in scoring
        self.explored_weight = 0.4  # Weight for exploration in scoring

    async def analyze_workspace_topics(
        self, workspace_id: int, force_reanalysis: bool = False
    ) -> WorkspaceTopicAnalysis:
        """
        Analyze a workspace to discover major topic areas and generate learning insights.

        Args:
            workspace_id: ID of the workspace to analyze
            force_reanalysis: If True, re-analyze even if recent analysis exists

        Returns:
            Complete analysis results with topic areas, learning paths, and recommendations
        """
        logging.info(f"Starting workspace topic analysis for workspace {workspace_id}")

        # Check if recent analysis exists
        if not force_reanalysis:
            logging.info(f"Checking for recent analysis for workspace {workspace_id}")
            recent_analysis = await self._get_recent_analysis(workspace_id)
            if recent_analysis:
                logging.info(
                    f"Found recent analysis for workspace {workspace_id}, returning cached results"
                )
                return recent_analysis

        # Get all concepts and relationships for the workspace
        logging.info(f"Fetching concepts for workspace {workspace_id}")
        workspace_concepts = await self.kg_service.get_workspace_concepts(workspace_id)
        if not workspace_concepts:
            logging.info(
                f"No concepts found for workspace {workspace_id}, triggering workspace analysis to extract concepts"
            )

            # Get workspace path
            workspace_query = text(
                "SELECT folder_path FROM workspaces WHERE id = :workspace_id"
            )
            result = await self.db.execute(
                workspace_query, {"workspace_id": workspace_id}
            )
            workspace_record = result.fetchone()

            if not workspace_record:
                logging.error(f"Workspace {workspace_id} not found")
                return self._create_empty_analysis(workspace_id)

            workspace_path = workspace_record.folder_path
            logging.info(f"Workspace path: {workspace_path}")

            # Initialize workspace analysis service if not provided
            if self.workspace_analysis_service is None:
                self.workspace_analysis_service = WorkspaceAnalysisService(
                    self.db,
                    kg_service=self.kg_service,
                    embedding_service=self.embedding_service,
                )

            # Trigger workspace analysis to extract concepts
            try:
                logging.info(f"Starting workspace analysis for concept extraction")
                analysis_results = (
                    await self.workspace_analysis_service.analyze_workspace(
                        workspace_id=workspace_id,
                        workspace_path=workspace_path,
                        force_reanalysis=False,
                        file_paths=None,
                    )
                )
                logging.info(
                    f"Workspace analysis completed, extracted concepts: {analysis_results}"
                )

                # Re-fetch concepts after analysis
                workspace_concepts = await self.kg_service.get_workspace_concepts(
                    workspace_id
                )
                logging.info(
                    f"After analysis, found {len(workspace_concepts) if workspace_concepts else 0} concepts"
                )

            except Exception as e:
                logging.error(
                    f"Failed to analyze workspace for concept extraction: {e}"
                )
                return self._create_empty_analysis(workspace_id)

        # Extract concept data and limit to top concepts for performance
        concepts_data = []
        for item in workspace_concepts:
            concept = item["concept"]
            concepts_data.append(
                {
                    "id": concept.concept_id,
                    "name": concept.name,
                    "description": concept.description or "",
                    "relevance_score": item.get("relevance_score", 0.5),
                }
            )

        # Sort by relevance and limit for performance
        concepts_data.sort(key=lambda x: x["relevance_score"], reverse=True)
        max_concepts_for_clustering = 500  # Limit for performance
        if len(concepts_data) > max_concepts_for_clustering:
            concepts_data = concepts_data[:max_concepts_for_clustering]
            logging.info(
                f"Limited concepts to top {max_concepts_for_clustering} by relevance score for performance"
            )

        logging.info(
            f"Processing {len(concepts_data)} concepts for workspace {workspace_id}"
        )

        # Discover topic areas through clustering
        logging.info(f"Discovering topic areas for workspace {workspace_id}")
        topic_areas = await self._discover_topic_areas(workspace_id, concepts_data)
        logging.info(
            f"Discovered {len(topic_areas)} topic areas for workspace {workspace_id}"
        )

        # Calculate coverage and exploration metrics
        logging.info(f"Calculating topic metrics for workspace {workspace_id}")
        await self._calculate_topic_metrics(workspace_id, topic_areas)

        # Generate learning paths
        logging.info(f"Generating learning paths for workspace {workspace_id}")
        learning_paths = await self._generate_learning_paths(workspace_id, topic_areas)
        logging.info(
            f"Generated {len(learning_paths)} learning paths for workspace {workspace_id}"
        )

        # Generate learning recommendations
        logging.info(f"Generating recommendations for workspace {workspace_id}")
        recommendations = await self._generate_recommendations(
            workspace_id, topic_areas
        )
        logging.info(
            f"Generated {len(recommendations)} recommendations for workspace {workspace_id}"
        )

        # Create analysis result
        analysis = WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=topic_areas,
            total_concepts=len(concepts_data),
            total_files=await self._get_workspace_file_count(workspace_id),
            coverage_distribution={
                ta.topic_area_id: ta.coverage_score for ta in topic_areas
            },
            learning_paths=learning_paths,
            recommendations=recommendations,
            analysis_timestamp=datetime.utcnow(),
            next_analysis_suggested=datetime.utcnow() + timedelta(days=7),
        )

        # Store analysis results
        logging.info(f"Storing analysis results for workspace {workspace_id}")
        await self._store_analysis_results(analysis)

        # Commit the transaction to ensure data is persisted
        await self.db.commit()

        logging.info(f"Completed workspace topic analysis for workspace {workspace_id}")
        return analysis

    async def _discover_topic_areas(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Discover major topic areas by clustering concepts based on semantic similarity.
        """
        if len(concepts_data) < self.min_topic_concepts:
            return []

        topic_areas = []

        # Use generalized clustering based on semantic similarity
        logging.info(f"Using generalized clustering for {len(concepts_data)} concepts")
        topic_areas = await self._cluster_generalized(workspace_id, concepts_data)

        # Limit to top topic areas
        topic_areas = sorted(
            topic_areas,
            key=lambda ta: ta.coverage_score * self.coverage_weight
            + ta.explored_percentage * self.explored_weight,
            reverse=True,
        )[: self.max_topic_areas]

        return topic_areas

    async def _cluster_with_embeddings(
        self, workspace_id: int, concepts_data: List[Dict[str, Any]]
    ) -> List[TopicArea]:
        """
        Use embeddings to cluster concepts into topic areas.
        """
        logging.info(f"Using embeddings for clustering {len(concepts_data)} concepts")
        if not self.embedding_service:
            logging.info("No embedding service available, falling back to heuristics")
            return await self._cluster_with_heuristics(workspace_id, concepts_data)

        # Generate embeddings for concept names
        logging.info("Generating embeddings for concept names")
        concept_names = [c["name"] for c in concepts_data]
        embeddings = await self.embedding_service.embed_texts(concept_names)
        logging.info(f"Generated embeddings for {len(embeddings)} concepts")

        # Perform hierarchical clustering based on similarity
        logging.info("Starting hierarchical clustering")
        clusters = self._hierarchical_clustering(embeddings, concepts_data)
        logging.info(f"Created {len(clusters)} clusters")

        # Convert clusters to topic areas
        topic_areas = []
        for cluster_id, cluster_concepts in clusters.items():
            if len(cluster_concepts) < self.min_topic_concepts:
                continue

            # Generate topic name from cluster
            topic_name = self._generate_topic_name(cluster_concepts)
            topic_description = self._generate_topic_description(cluster_concepts)

            # Calculate initial metrics
            avg_relevance = sum(c["relevance_score"] for c in cluster_concepts) / len(
                cluster_concepts
            )

            topic_area = TopicArea(
                topic_area_id=f"topic_{workspace_id}_{cluster_id}",
                workspace_id=workspace_id,
                name=topic_name,
                description=topic_description,
                coverage_score=min(1.0, avg_relevance),
                concept_count=len(cluster_concepts),
                file_count=await self._get_topic_file_count(
                    workspace_id, cluster_concepts
                ),
                explored_percentage=0.0,  # Will be calculated later
            )
            topic_areas.append(topic_area)

        return topic_areas

    def _hierarchical_clustering(
        self, embeddings: List[List[float]], concepts_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform hierarchical clustering on embeddings.
        """
        logging.info(
            f"Starting hierarchical clustering with {len(concepts_data)} concepts"
        )

        # Simple agglomerative clustering implementation
        clusters = {
            f"cluster_{i}": [concept] for i, concept in enumerate(concepts_data)
        }
        logging.info(f"Initialized {len(clusters)} individual clusters")

        merge_count = 0
        # Continue merging until we have reasonable cluster sizes
        while len(clusters) > self.max_topic_areas:
            logging.info(
                f"Merge iteration {merge_count + 1}: {len(clusters)} clusters remaining"
            )

            # Find most similar clusters to merge
            best_similarity = -1
            best_pair = None

            cluster_items = list(clusters.items())
            for i, (cid1, concepts1) in enumerate(cluster_items):
                for j, (cid2, concepts2) in enumerate(cluster_items[i + 1 :], i + 1):
                    # Calculate average similarity between clusters
                    similarities = []
                    for c1 in concepts1:
                        for c2 in concepts2:
                            idx1 = concepts_data.index(c1)
                            idx2 = concepts_data.index(c2)
                            sim = self._cosine_similarity(
                                embeddings[idx1], embeddings[idx2]
                            )
                            similarities.append(sim)

                    avg_similarity = (
                        sum(similarities) / len(similarities) if similarities else 0
                    )

                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_pair = (cid1, cid2)

            if best_similarity < self.similarity_threshold or not best_pair:
                logging.info(
                    f"Stopping clustering: best similarity {best_similarity:.3f} < threshold {self.similarity_threshold} or no pairs found"
                )
                break

            # Merge the best pair
            cid1, cid2 = best_pair
            clusters[cid1].extend(clusters[cid2])
            del clusters[cid2]
            merge_count += 1

            # Log progress every 10 merges
            if merge_count % 10 == 0:
                logging.info(
                    f"Completed {merge_count} merges, {len(clusters)} clusters remaining"
                )

        logging.info(
            f"Clustering completed: {len(clusters)} final clusters from {len(concepts_data)} concepts"
        )
        return clusters

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
                    file_count=await self._get_topic_file_count(
                        workspace_id, cluster_concepts
                    ),
                    explored_percentage=0.0,
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
            file_count=await self._get_topic_file_count(workspace_id, concepts_data),
            explored_percentage=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return [topic_area]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _generate_topic_name(self, concepts: List[Dict[str, Any]]) -> str:
        """Generate a descriptive name for a topic cluster"""
        # Use the most common words across concept names
        all_words = []
        for concept in concepts:
            words = concept["name"].split()
            all_words.extend(words)

        word_counts = Counter(all_words)
        top_words = [word for word, count in word_counts.most_common(3)]

        # Create a title-case name
        if len(top_words) >= 2:
            return " ".join(top_words[:2]).title()
        elif top_words:
            return top_words[0].title()
        else:
            return "General Concepts"

    def _generate_topic_description(self, concepts: List[Dict[str, Any]]) -> str:
        """Generate a description for a topic cluster"""
        concept_names = [c["name"] for c in concepts[:5]]  # Use first 5 concepts
        names_str = ", ".join(concept_names)

        if len(concepts) > 5:
            names_str += f", and {len(concepts) - 5} more"

        return f"Topic area covering concepts like {names_str}"

    async def _calculate_topic_metrics(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> None:
        """
        Calculate coverage scores and exploration percentages for topic areas.
        """
        for topic_area in topic_areas:
            # Get concepts for this topic area
            topic_concepts = await self._get_topic_concepts(
                workspace_id, topic_area.topic_area_id
            )

            # Calculate exploration percentage based on quiz performance
            explored_count = 0
            for concept in topic_concepts:
                # Check if concept has been quizzed recently or has high performance
                if await self._is_concept_explored(workspace_id, concept["id"]):
                    explored_count += 1

            topic_area.explored_percentage = (
                explored_count / len(topic_concepts) if topic_concepts else 0.0
            )

            # Update coverage score based on multiple factors
            base_coverage = topic_area.coverage_score
            exploration_bonus = topic_area.explored_percentage * 0.2
            topic_area.coverage_score = min(1.0, base_coverage + exploration_bonus)

    async def _generate_learning_paths(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningPath]:
        """
        Generate recommended learning paths based on topic areas.
        """
        learning_paths = []

        # Sort topic areas by coverage and exploration
        sorted_topics = sorted(
            topic_areas,
            key=lambda ta: ta.coverage_score + ta.explored_percentage,
            reverse=True,
        )

        # Create comprehensive learning path
        if len(sorted_topics) >= 3:
            comprehensive_path = LearningPath(
                learning_path_id=f"path_{workspace_id}_comprehensive",
                workspace_id=workspace_id,
                name="Comprehensive Workspace Mastery",
                description="Complete learning path covering all major topics in your workspace",
                topic_areas=[ta.topic_area_id for ta in sorted_topics],
                estimated_hours=sum(
                    ta.concept_count * 2 for ta in sorted_topics
                ),  # Rough estimate
                difficulty_level=self._calculate_path_difficulty(sorted_topics),
                prerequisites=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            learning_paths.append(comprehensive_path)

        # Create focused learning paths for high-priority topics
        high_priority_topics = [ta for ta in sorted_topics if ta.coverage_score > 0.7][
            :5
        ]

        for i, topic in enumerate(high_priority_topics):
            focused_path = LearningPath(
                learning_path_id=f"path_{workspace_id}_focused_{i}",
                workspace_id=workspace_id,
                name=f"Deep Dive: {topic.name}",
                description=f"Focused learning path for mastering {topic.name}",
                topic_areas=[topic.topic_area_id],
                estimated_hours=topic.concept_count * 3,  # More intensive
                difficulty_level="intermediate",
                prerequisites=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            learning_paths.append(focused_path)

        return learning_paths

    async def _generate_recommendations(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> List[LearningRecommendation]:
        """
        Generate personalized learning recommendations.
        """
        recommendations = []

        for topic_area in topic_areas:
            # Recommendation based on low exploration
            if topic_area.explored_percentage < 0.3:
                recommendation = LearningRecommendation(
                    recommendation_id=f"rec_{workspace_id}_{topic_area.topic_area_id}_explore",
                    workspace_id=workspace_id,
                    recommendation_type="concept_gap",
                    topic_area_id=topic_area.topic_area_id,
                    priority_score=min(
                        1.0, (1.0 - topic_area.explored_percentage) * 0.8
                    ),
                    reason=f"You've only explored {topic_area.explored_percentage:.1%} of {topic_area.name}",
                    suggested_action=f"Take a quiz on {topic_area.name} to improve your understanding",
                    created_at=datetime.utcnow(),
                )
                recommendations.append(recommendation)

            # Recommendation based on high coverage but low exploration
            elif (
                topic_area.coverage_score > 0.8 and topic_area.explored_percentage < 0.5
            ):
                recommendation = LearningRecommendation(
                    recommendation_id=f"rec_{workspace_id}_{topic_area.topic_area_id}_practice",
                    workspace_id=workspace_id,
                    recommendation_type="quiz_performance",
                    topic_area_id=topic_area.topic_area_id,
                    priority_score=0.7,
                    reason=f"You have good coverage of {topic_area.name} but need more practice",
                    suggested_action=f"Practice quizzes on {topic_area.name} concepts",
                    created_at=datetime.utcnow(),
                )
                recommendations.append(recommendation)

        # Sort by priority
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def _calculate_path_difficulty(self, topic_areas: List[TopicArea]) -> str:
        """Calculate overall difficulty of a learning path"""
        avg_coverage = sum(ta.coverage_score for ta in topic_areas) / len(topic_areas)

        if avg_coverage < 0.4:
            return "beginner"
        elif avg_coverage < 0.7:
            return "intermediate"
        else:
            return "advanced"

    async def _get_topic_concepts(
        self, workspace_id: int, topic_area_id: str
    ) -> List[Dict[str, Any]]:
        """Get concepts associated with a topic area"""
        query = text(
            """
            SELECT c.*, tcl.relevance_score
            FROM concepts c
            JOIN topic_concept_links tcl ON c.concept_id = tcl.concept_id
            WHERE tcl.topic_area_id = :topic_area_id
            """
        )

        result = await self.db.execute(query, {"topic_area_id": topic_area_id})
        rows = result.fetchall()

        concepts = []
        for row in rows:
            concepts.append(
                {
                    "id": row.concept_id,
                    "name": row.name,
                    "description": row.description,
                    "relevance_score": row.relevance_score,
                }
            )

        return concepts

    async def _get_topic_file_count(
        self, workspace_id: int, concepts: List[Dict[str, Any]]
    ) -> int:
        """Get number of unique files covering the given concepts"""
        if not concepts:
            return 0

        concept_ids = [c["id"] for c in concepts]

        # Use proper parameter binding to avoid SQL injection and formatting issues
        placeholders = ",".join(f":concept_{i}" for i in range(len(concept_ids)))

        query_str = f"""
            SELECT COUNT(DISTINCT cf.file_id)
            FROM concept_files cf
            WHERE cf.concept_id IN ({placeholders})
            AND cf.workspace_id = :workspace_id
        """

        # Create parameter dictionary
        params = {
            f"concept_{i}": concept_id for i, concept_id in enumerate(concept_ids)
        }
        params["workspace_id"] = workspace_id

        query = text(query_str)
        result = await self.db.execute(query, params)
        return result.scalar() or 0

    async def _is_concept_explored(self, workspace_id: int, concept_id: str) -> bool:
        """Check if a concept has been sufficiently explored"""
        # Check quiz performance for this concept
        query = text(
            """
            SELECT COUNT(*) as quiz_count,
                   AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as avg_score
            FROM questions q
            JOIN answers a ON q.id = a.question_id
            WHERE q.kg_concept_ids LIKE :concept_pattern
            AND q.file_id IN (
                SELECT id FROM files WHERE workspace_id = :workspace_id
            )
            """
        )

        # SQLite doesn't support complex LIKE patterns easily, so we'll check differently
        concept_pattern = f"%{concept_id}%"

        result = await self.db.execute(
            query, {"concept_pattern": concept_pattern, "workspace_id": workspace_id}
        )

        row = result.fetchone()
        if row and row.quiz_count and row.quiz_count > 0:
            # Consider explored if quizzed at least 3 times with >70% accuracy
            return row.quiz_count >= 3 and (row.avg_score or 0) > 0.7

        return False

    async def _get_workspace_file_count(self, workspace_id: int) -> int:
        """Get total number of files in workspace"""
        query = text("SELECT COUNT(*) FROM files WHERE workspace_id = :workspace_id")
        result = await self.db.execute(query, {"workspace_id": workspace_id})
        return result.scalar() or 0

    async def _get_recent_analysis(
        self, workspace_id: int, max_age_hours: int = 24
    ) -> Optional[WorkspaceTopicAnalysis]:
        """Get recent analysis if it exists and is not too old"""
        # For now, we'll implement a simple check
        # In production, you'd store analysis metadata
        return None

    def _create_empty_analysis(self, workspace_id: int) -> WorkspaceTopicAnalysis:
        """Create empty analysis for workspaces with no concepts"""
        return WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=[],
            total_concepts=0,
            total_files=0,
            coverage_distribution={},
            learning_paths=[],
            recommendations=[],
            analysis_timestamp=datetime.utcnow(),
            next_analysis_suggested=datetime.utcnow() + timedelta(days=1),
        )

    async def _store_analysis_results(self, analysis: WorkspaceTopicAnalysis) -> None:
        """Store analysis results in database"""
        logging.info(f"Storing analysis results for workspace {analysis.workspace_id}")
        logging.info(f"Topic areas to store: {len(analysis.topic_areas)}")
        logging.info(f"Learning paths to store: {len(analysis.learning_paths)}")
        logging.info(f"Recommendations to store: {len(analysis.recommendations)}")

        try:
            # Store topic areas
            for i, topic_area in enumerate(analysis.topic_areas):
                logging.info(
                    f"Storing topic area {i+1}/{len(analysis.topic_areas)}: {topic_area.name}"
                )
                await self._store_topic_area(topic_area)

            # Store learning paths
            for i, learning_path in enumerate(analysis.learning_paths):
                logging.info(
                    f"Storing learning path {i+1}/{len(analysis.learning_paths)}: {learning_path.name}"
                )
                await self._store_learning_path(learning_path)

            # Store recommendations
            for i, recommendation in enumerate(analysis.recommendations):
                logging.info(
                    f"Storing recommendation {i+1}/{len(analysis.recommendations)}: {recommendation.reason[:50]}..."
                )
                await self._store_recommendation(recommendation)

            logging.info("Successfully stored all analysis results")
        except Exception as e:
            logging.error(f"Error storing analysis results: {e}")
            raise

    async def _store_topic_area(self, topic_area: TopicArea) -> None:
        """Store a topic area in the database"""
        query = text(
            """
            INSERT OR REPLACE INTO topic_areas
            (topic_area_id, workspace_id, name, description, coverage_score,
             concept_count, file_count, explored_percentage, created_at, updated_at)
            VALUES (:id, :workspace_id, :name, :description, :coverage_score,
                   :concept_count, :file_count, :explored_percentage, :created_at, :updated_at)
            """
        )

        await self.db.execute(
            query,
            {
                "id": topic_area.topic_area_id,
                "workspace_id": topic_area.workspace_id,
                "name": topic_area.name,
                "description": topic_area.description,
                "coverage_score": topic_area.coverage_score,
                "concept_count": topic_area.concept_count,
                "file_count": topic_area.file_count,
                "explored_percentage": topic_area.explored_percentage,
                "created_at": topic_area.created_at,
                "updated_at": topic_area.updated_at,
            },
        )

    async def _store_learning_path(self, learning_path: LearningPath) -> None:
        """Store a learning path in the database"""
        query = text(
            """
            INSERT OR REPLACE INTO learning_paths
            (learning_path_id, workspace_id, name, description, topic_areas,
             estimated_hours, difficulty_level, prerequisites, created_at, updated_at)
            VALUES (:id, :workspace_id, :name, :description, :topic_areas,
                   :estimated_hours, :difficulty_level, :prerequisites, :created_at, :updated_at)
            """
        )

        await self.db.execute(
            query,
            {
                "id": learning_path.learning_path_id,
                "workspace_id": learning_path.workspace_id,
                "name": learning_path.name,
                "description": learning_path.description,
                "topic_areas": json.dumps(learning_path.topic_areas),
                "estimated_hours": learning_path.estimated_hours,
                "difficulty_level": learning_path.difficulty_level,
                "prerequisites": (
                    json.dumps(learning_path.prerequisites)
                    if learning_path.prerequisites
                    else None
                ),
                "created_at": learning_path.created_at,
                "updated_at": learning_path.updated_at,
            },
        )

    async def _store_recommendation(
        self, recommendation: LearningRecommendation
    ) -> None:
        """Store a learning recommendation in the database"""
        query = text(
            """
            INSERT OR REPLACE INTO learning_recommendations
            (recommendation_id, workspace_id, user_id, recommendation_type,
             topic_area_id, concept_id, priority_score, reason, suggested_action, created_at)
            VALUES (:id, :workspace_id, :user_id, :recommendation_type,
                   :topic_area_id, :concept_id, :priority_score, :reason, :suggested_action, :created_at)
            """
        )

        await self.db.execute(
            query,
            {
                "id": recommendation.recommendation_id,
                "workspace_id": recommendation.workspace_id,
                "user_id": recommendation.user_id,
                "recommendation_type": recommendation.recommendation_type,
                "topic_area_id": recommendation.topic_area_id,
                "concept_id": recommendation.concept_id,
                "priority_score": recommendation.priority_score,
                "reason": recommendation.reason,
                "suggested_action": recommendation.suggested_action,
                "created_at": recommendation.created_at,
            },
        )

    async def get_workspace_topic_analysis(
        self, workspace_id: int
    ) -> Optional[WorkspaceTopicAnalysis]:
        """Retrieve stored topic analysis for a workspace"""
        # Get topic areas
        topic_areas_query = text(
            "SELECT * FROM topic_areas WHERE workspace_id = :workspace_id"
        )
        topic_areas_result = await self.db.execute(
            topic_areas_query, {"workspace_id": workspace_id}
        )
        topic_areas_rows = topic_areas_result.fetchall()

        topic_areas = []
        for row in topic_areas_rows:
            topic_areas.append(
                TopicArea(
                    topic_area_id=row.topic_area_id,
                    workspace_id=row.workspace_id,
                    name=row.name,
                    description=row.description,
                    coverage_score=row.coverage_score,
                    concept_count=row.concept_count,
                    file_count=row.file_count,
                    explored_percentage=row.explored_percentage,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

        if not topic_areas:
            return None

        # Get learning paths
        learning_paths_query = text(
            "SELECT * FROM learning_paths WHERE workspace_id = :workspace_id"
        )
        learning_paths_result = await self.db.execute(
            learning_paths_query, {"workspace_id": workspace_id}
        )
        learning_paths_rows = learning_paths_result.fetchall()

        learning_paths = []
        for row in learning_paths_rows:
            learning_paths.append(
                LearningPath(
                    learning_path_id=row.learning_path_id,
                    workspace_id=row.workspace_id,
                    name=row.name,
                    description=row.description,
                    topic_areas=json.loads(row.topic_areas) if row.topic_areas else [],
                    estimated_hours=row.estimated_hours,
                    difficulty_level=row.difficulty_level,
                    prerequisites=(
                        json.loads(row.prerequisites) if row.prerequisites else None
                    ),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

        # Get recommendations
        recommendations_query = text(
            "SELECT * FROM learning_recommendations WHERE workspace_id = :workspace_id ORDER BY priority_score DESC"
        )
        recommendations_result = await self.db.execute(
            recommendations_query, {"workspace_id": workspace_id}
        )
        recommendations_rows = recommendations_result.fetchall()

        recommendations = []
        for row in recommendations_rows:
            recommendations.append(
                LearningRecommendation(
                    recommendation_id=row.recommendation_id,
                    workspace_id=row.workspace_id,
                    user_id=row.user_id,
                    recommendation_type=row.recommendation_type,
                    topic_area_id=row.topic_area_id,
                    concept_id=row.concept_id,
                    priority_score=row.priority_score,
                    reason=row.reason,
                    suggested_action=row.suggested_action,
                    created_at=row.created_at,
                )
            )

        return WorkspaceTopicAnalysis(
            workspace_id=workspace_id,
            topic_areas=topic_areas,
            total_concepts=sum(ta.concept_count for ta in topic_areas),
            total_files=sum(ta.file_count for ta in topic_areas),
            coverage_distribution={
                ta.topic_area_id: ta.coverage_score for ta in topic_areas
            },
            learning_paths=learning_paths,
            recommendations=recommendations,
            analysis_timestamp=datetime.utcnow(),
            next_analysis_suggested=datetime.utcnow() + timedelta(days=7),
        )
