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
from ..database.workspace_topics import WorkspaceTopicsDatabase
from . import KnowledgeGraphService, WorkspaceAnalysisService
from .workspace_flattener import WorkspaceFlattener
from .embedding_service import EmbeddingService
from .topicExtractor import TopicExtractionService
from .recommendation import RecommendationService
from .concept_extraction.extractors import EntityRecognitionExtractor
from .concept_extraction.rankers import TFIDFRanking


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
                db=self.db,
                embedding_service=self.embedding_service,
                extractor_type="bertopic",  # Use BERTopic by default
                extractor_config={
                    "model_name": "all-MiniLM-L6-v2",
                    "min_topic_size": 20,
                    "diversity": 0.3,
                    "coherence_threshold": 0.1,
                },
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
                extractor = EntityRecognitionExtractor(use_spacy=True)
                ranker = TFIDFRanking()
                self.workspace_analysis_service = WorkspaceAnalysisService(
                    self.db,
                    extractor,
                    ranker,
                    kg_service=self.kg_service,
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
                    extractor = EntityRecognitionExtractor(use_spacy=True)
                    ranker = TFIDFRanking()
                    self.workspace_analysis_service = WorkspaceAnalysisService(
                        self.db,
                        extractor,
                        ranker,
                        kg_service=self.kg_service,
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
        max_concepts_for_clustering = 500000  # Limit for performance
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
        topic_areas, concept_links = await self.topic_extraction_service.extract_topics(
            workspace_id, concepts_data
        )
        logging.info(
            f"Discovered {len(topic_areas)} topic areas and {len(concept_links)} concept links for workspace {workspace_id}"
        )

        # Store concept links first so metrics calculation can use them
        logging.info(
            f"Storing {len(concept_links)} concept links for workspace {workspace_id}"
        )
        for concept_link in concept_links:
            await self._store_concept_link(concept_link)

        # Calculate coverage and exploration metrics using the consolidated method
        logging.info(f"Calculating topic metrics for workspace {workspace_id}")
        await self.topic_extraction_service.calculate_workspace_topic_metrics(
            workspace_id, topic_areas
        )

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
            total_files=await self.topic_extraction_service._get_workspace_file_count(
                workspace_id
            ),
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
            # Use database layer for storage
            db_workspace_topics = WorkspaceTopicsDatabase(self.db)

            # Store topic areas
            for i, topic_area in enumerate(analysis.topic_areas):
                logging.info(
                    f"Storing topic area {i+1}/{len(analysis.topic_areas)}: {topic_area.name}"
                )
                await db_workspace_topics.store_topic_area(topic_area)

            # Concept links are already stored before metrics calculation

            # Store learning paths
            for i, learning_path in enumerate(analysis.learning_paths):
                logging.info(
                    f"Storing learning path {i+1}/{len(analysis.learning_paths)}: {learning_path.name}"
                )
                await db_workspace_topics.store_learning_path(learning_path)

            # Store recommendations
            for i, recommendation in enumerate(analysis.recommendations):
                logging.info(
                    f"Storing recommendation {i+1}/{len(analysis.recommendations)}: {recommendation.reason[:50]}..."
                )
                await db_workspace_topics.store_recommendation(recommendation)

            logging.info("Successfully stored all analysis results")
        except Exception as e:
            logging.error(f"Error storing analysis results: {e}")
            raise

    async def _store_topic_area(self, topic_area: TopicArea) -> None:
        """Store a topic area in the database"""
        db_workspace_topics = WorkspaceTopicsDatabase(self.db)
        await db_workspace_topics.store_topic_area(topic_area)

    async def _store_concept_link(self, concept_link) -> None:
        """Store a concept link in the database"""
        db_workspace_topics = WorkspaceTopicsDatabase(self.db)
        await db_workspace_topics.store_concept_link(concept_link)

    async def _store_learning_path(self, learning_path: LearningPath) -> None:
        """Store a learning path in the database"""
        db_workspace_topics = WorkspaceTopicsDatabase(self.db)
        await db_workspace_topics.store_learning_path(learning_path)

    async def _store_recommendation(
        self, recommendation: LearningRecommendation
    ) -> None:
        """Store a learning recommendation in the database"""
        db_workspace_topics = WorkspaceTopicsDatabase(self.db)
        await db_workspace_topics.store_recommendation(recommendation)

    async def get_workspace_topic_analysis(
        self, workspace_id: int
    ) -> Optional[WorkspaceTopicAnalysis]:
        """Retrieve stored topic analysis for a workspace"""
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(self.db)
        return await db_workspace_topics.get_workspace_topic_analysis(workspace_id)
