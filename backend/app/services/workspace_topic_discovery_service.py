"""
Workspace Topic Discovery Service for identifying major subject areas and learning paths
Analyzes entire workspaces to discover topics and recommend learning trajectories
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import (
    WorkspaceTopicAnalysis,
    TopicArea,
    LearningPath,
    LearningRecommendation,
)
from . import KnowledgeGraphService, WorkspaceAnalysisService
from .workspace_flattener import WorkspaceFlattener
from .embedding_service import EmbeddingService
from .topicExtractor import TopicExtractionService
from .recommendation import RecommendationService


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
        topic_extraction_service: Optional[TopicExtractionService] = None,
        recommendation_service: Optional[RecommendationService] = None,
    ):
        self.db = db
        self.kg_service = kg_service or KnowledgeGraphService(db)
        self.embedding_service = embedding_service
        # Initialize embedding service if not provided
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService.get_instance()

        # Initialize workspace analysis service if not provided
        self.workspace_analysis_service = workspace_analysis_service

        # Initialize topic extraction and recommendation services
        self.topic_extraction_service = (
            topic_extraction_service
            or TopicExtractionService(
                db=self.db, embedding_service=self.embedding_service
            )
        )
        self.recommendation_service = recommendation_service or RecommendationService()

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

            # Create flattened workspace file and analyze it
            try:
                logging.info(
                    f"Creating flattened workspace file for workspace {workspace_id}"
                )

                # Initialize workspace flattener
                flattener = WorkspaceFlattener(workspace_path)

                # Create flattened file
                flattened_file_path = flattener.flatten_workspace()
                logging.info(f"Created flattened file: {flattened_file_path}")

                # Initialize workspace analysis service if not provided
                if self.workspace_analysis_service is None:
                    self.workspace_analysis_service = WorkspaceAnalysisService(
                        self.db,
                        kg_service=self.kg_service,
                        embedding_service=self.embedding_service,
                    )

                # Analyze the flattened file
                logging.info(
                    f"Starting flattened file analysis for workspace {workspace_id}"
                )
                analysis_results = (
                    await self.workspace_analysis_service.analyze_flattened_file(
                        flattened_file_path, workspace_id
                    )
                )
                logging.info(f"Flattened file analysis completed: {analysis_results}")

                # Re-fetch concepts after analysis
                workspace_concepts = await self.kg_service.get_workspace_concepts(
                    workspace_id
                )
                logging.info(
                    f"After flattened analysis, found {len(workspace_concepts) if workspace_concepts else 0} concepts"
                )

            except Exception as e:
                logging.error(
                    f"Failed to create flattened file and analyze workspace: {e}"
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
        topic_areas = await self.topic_extraction_service.extract_topics(
            workspace_id, concepts_data
        )
        logging.info(
            f"Discovered {len(topic_areas)} topic areas for workspace {workspace_id}"
        )

        # Calculate coverage and exploration metrics
        logging.info(f"Calculating topic metrics for workspace {workspace_id}")
        await self._calculate_topic_metrics(workspace_id, topic_areas)

        # Generate learning paths
        logging.info(f"Generating learning paths for workspace {workspace_id}")
        learning_paths = self.recommendation_service.generate_learning_paths(
            workspace_id, topic_areas
        )
        logging.info(
            f"Generated {len(learning_paths)} learning paths for workspace {workspace_id}"
        )

        # Generate learning recommendations
        logging.info(f"Generating recommendations for workspace {workspace_id}")
        recommendations = self.recommendation_service.generate_recommendations(
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

    async def _calculate_topic_metrics(
        self, workspace_id: int, topic_areas: List[TopicArea]
    ) -> None:
        """
        Calculate coverage scores and exploration metrics for topic areas.
        Updates the topic areas in-place with calculated metrics.
        """
        logging.info(f"Calculating metrics for {len(topic_areas)} topic areas")

        for topic_area in topic_areas:
            # Get concepts for this topic area
            topic_concepts = await self._get_topic_concepts(
                workspace_id, topic_area.topic_area_id
            )
            topic_area.concept_count = len(topic_concepts)

            # Get file count for this topic area
            topic_area.file_count = await self._get_topic_file_count(
                workspace_id, topic_concepts
            )

            # Calculate coverage score (based on concept count and file coverage)
            # Normalize by total workspace concepts and files
            total_workspace_concepts = await self._get_workspace_concept_count(
                workspace_id
            )
            total_workspace_files = await self._get_workspace_file_count(workspace_id)

            concept_coverage = (
                topic_area.concept_count / total_workspace_concepts
                if total_workspace_concepts > 0
                else 0
            )
            file_coverage = (
                topic_area.file_count / total_workspace_files
                if total_workspace_files > 0
                else 0
            )

            # Weighted coverage score
            topic_area.coverage_score = (
                self.coverage_weight * concept_coverage
                + (1 - self.coverage_weight) * file_coverage
            )

            # Calculate exploration percentage
            explored_concepts = 0
            for concept in topic_concepts:
                if await self._is_concept_explored(workspace_id, concept["id"]):
                    explored_concepts += 1

            topic_area.explored_percentage = (
                explored_concepts / topic_area.concept_count
                if topic_area.concept_count > 0
                else 0
            )

            logging.info(
                f"Topic area '{topic_area.name}': {topic_area.concept_count} concepts, "
                f"{topic_area.file_count} files, coverage={topic_area.coverage_score:.3f}, "
                f"explored={topic_area.explored_percentage:.1%}"
            )

    async def _get_workspace_concept_count(self, workspace_id: int) -> int:
        """Get total number of concepts in workspace"""
        query = text(
            """
            SELECT COUNT(*) FROM concepts c
            JOIN concept_files cf ON c.concept_id = cf.concept_id
            WHERE cf.workspace_id = :workspace_id
            """
        )
        result = await self.db.execute(query, {"workspace_id": workspace_id})
        return result.scalar() or 0

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
