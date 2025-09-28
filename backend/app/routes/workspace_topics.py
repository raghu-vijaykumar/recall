"""
Workspace Topics API routes for topic discovery and learning path recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from ..database import get_db
from ..database.workspace_topics import WorkspaceTopicsDatabase
from ..models import (
    TopicArea,
    LearningPath,
    LearningRecommendation,
    WorkspaceTopicAnalysis,
)
from ..services import (
    WorkspaceTopicDiscoveryService,
    KnowledgeGraphService,
    EmbeddingService,
)

router = APIRouter(prefix="/api/workspace-topics", tags=["workspace-topics"])


@router.post("/analyze/{workspace_id}")
async def analyze_workspace_topics(
    workspace_id: int,
    force_reanalysis: bool = False,
    db: AsyncSession = Depends(get_db),
) -> WorkspaceTopicAnalysis:
    """
    Analyze a workspace to discover major topic areas and generate learning insights.

    Args:
        workspace_id: ID of the workspace to analyze
        force_reanalysis: If True, re-analyze even if recent analysis exists

    Returns:
        Complete analysis results with topic areas, learning paths, and recommendations
    """
    logging.info(
        f"Received request to analyze workspace topics for workspace {workspace_id}"
    )
    try:
        # Initialize services
        logging.info("Initializing KnowledgeGraphService")
        kg_service = KnowledgeGraphService(db)

        logging.info("Initializing EmbeddingService")
        embedding_service = EmbeddingService()  # Initialize with default path
        await embedding_service.initialize()  # Initialize the embedding service

        logging.info("Initializing WorkspaceTopicDiscoveryService")
        from ..services.concept_extraction.extractors import EntityRecognitionExtractor
        from ..services.concept_extraction.rankers import TFIDFRanking

        extractor = EntityRecognitionExtractor(use_spacy=True)
        ranker = TFIDFRanking()
        from ..services import WorkspaceAnalysisService

        workspace_analysis_service = WorkspaceAnalysisService(
            db,
            extractor,
            ranker,
            kg_service=kg_service,
        )

        topic_service = WorkspaceTopicDiscoveryService(
            db, kg_service, embedding_service, workspace_analysis_service
        )

        # Perform analysis
        logging.info(f"Starting analysis for workspace {workspace_id}")
        analysis = await topic_service.analyze_workspace_topics(
            workspace_id, force_reanalysis
        )

        logging.info(f"Successfully completed analysis for workspace {workspace_id}")
        logging.info(
            f"Analysis results: {len(analysis.topic_areas)} topic areas, {len(analysis.learning_paths)} learning paths, {len(analysis.recommendations)} recommendations"
        )
        return analysis

    except Exception as e:
        logging.error(
            f"Error analyzing workspace topics for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze workspace topics: {str(e)}"
        )


@router.get("/{workspace_id}")
async def get_workspace_topic_analysis(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
) -> Optional[WorkspaceTopicAnalysis]:
    """
    Get stored topic analysis for a workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        Stored analysis results if available
    """
    try:
        # Initialize service
        kg_service = KnowledgeGraphService(db)
        embedding_service = EmbeddingService()  # Initialize with default path
        topic_service = WorkspaceTopicDiscoveryService(
            db, kg_service, embedding_service
        )

        # Get stored analysis
        analysis = await topic_service.get_workspace_topic_analysis(workspace_id)

        return analysis

    except Exception as e:
        logging.error(
            f"Error retrieving workspace topic analysis for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve workspace topic analysis: {str(e)}",
        )


@router.get("/{workspace_id}/topic-areas")
async def get_workspace_topic_areas(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[TopicArea]:
    """
    Get all topic areas for a workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        List of topic areas
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        return await db_workspace_topics.get_workspace_topic_areas(workspace_id)

    except Exception as e:
        logging.error(f"Error retrieving topic areas for workspace {workspace_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve topic areas: {str(e)}"
        )


@router.get("/{workspace_id}/learning-paths")
async def get_workspace_learning_paths(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[LearningPath]:
    """
    Get all learning paths for a workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        List of learning paths
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        return await db_workspace_topics.get_workspace_learning_paths(workspace_id)

    except Exception as e:
        logging.error(
            f"Error retrieving learning paths for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve learning paths: {str(e)}"
        )


@router.get("/{workspace_id}/recommendations")
async def get_workspace_recommendations(
    workspace_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> List[LearningRecommendation]:
    """
    Get learning recommendations for a workspace.

    Args:
        workspace_id: ID of the workspace
        limit: Maximum number of recommendations to return

    Returns:
        List of learning recommendations ordered by priority
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        return await db_workspace_topics.get_workspace_recommendations(
            workspace_id, limit
        )

    except Exception as e:
        logging.error(
            f"Error retrieving recommendations for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve recommendations: {str(e)}"
        )


@router.get("/topic-area/{topic_area_id}/concepts")
async def get_topic_area_concepts(
    topic_area_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[dict]:
    """
    Get all concepts associated with a specific topic area.

    Args:
        topic_area_id: ID of the topic area

    Returns:
        List of concepts with their relevance scores
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        return await db_workspace_topics.get_topic_area_concepts(topic_area_id)

    except Exception as e:
        logging.error(f"Error retrieving concepts for topic area {topic_area_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve topic area concepts: {str(e)}"
        )


@router.get("/topic-area/{topic_area_id}/files")
async def get_topic_area_files(
    topic_area_id: str,
    db: AsyncSession = Depends(get_db),
) -> List[dict]:
    """
    Get all files that contain concepts from a specific topic area.

    Args:
        topic_area_id: ID of the topic area

    Returns:
        List of files with concept coverage information
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        return await db_workspace_topics.get_topic_area_files(topic_area_id)

    except Exception as e:
        logging.error(f"Error retrieving files for topic area {topic_area_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve topic area files: {str(e)}"
        )
