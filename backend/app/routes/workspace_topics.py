"""
Workspace Topics API routes for topic discovery and learning path recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging
from sqlalchemy import text

from ..services.database import get_db
from ..database.workspace_topics import WorkspaceTopicsDatabase
from ..models import (
    TopicArea,
    LearningPath,
    LearningRecommendation,
    WorkspaceTopicAnalysis,
)
from ..services import WorkspaceAnalysisService

router = APIRouter()


@router.post("/analyze/{workspace_id}")
async def analyze_workspace_topics(
    workspace_id: int,
    extractor_type: str = "heuristic",  # Allow choosing extractor type
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Analyze a workspace to extract topic areas.

    Args:
        workspace_id: ID of the workspace to analyze
        extractor_type: "heuristic" or "bertopic" extractor to use

    Returns:
        Analysis results with topic statistics
    """
    logging.info(
        f"Received request to analyze workspace {workspace_id} using {extractor_type} extractor"
    )

    try:
        # Get workspace path from database
        workspace_query = await db.execute(
            text("SELECT folder_path FROM workspaces WHERE id = :workspace_id"),
            {"workspace_id": workspace_id},
        )
        workspace_record = workspace_query.fetchone()

        if not workspace_record:
            raise HTTPException(status_code=404, detail="Workspace not found")

        workspace_path = workspace_record.folder_path

        # Initialize streamlined workspace analysis service
        workspace_analysis_service = WorkspaceAnalysisService(
            db=db,
            extractor_type=extractor_type,
        )

        # Perform analysis
        logging.info(f"Starting topic analysis for workspace {workspace_id}")
        result = await workspace_analysis_service.analyze_workspace(
            workspace_id, workspace_path
        )

        logging.info(f"Successfully completed analysis for workspace {workspace_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(
            f"Error analyzing workspace topics for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze workspace topics: {str(e)}"
        )


@router.get("/{workspace_id}/summary")
async def get_workspace_topic_summary(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Get topic analysis summary for a workspace.

    Args:
        workspace_id: ID of the workspace

    Returns:
        Summary with topic area counts
    """
    try:
        # Use database layer
        db_workspace_topics = WorkspaceTopicsDatabase(db)
        topic_areas = await db_workspace_topics.get_workspace_topic_areas(workspace_id)

        return {
            "workspace_id": workspace_id,
            "total_topics": len(topic_areas),
            "topic_areas": [
                {
                    "id": ta.topic_area_id,
                    "name": ta.name,
                    "concept_count": ta.concept_count,
                    "coverage_score": ta.coverage_score,
                    "explored_percentage": ta.explored_percentage,
                }
                for ta in topic_areas
            ],
        }

    except Exception as e:
        logging.error(
            f"Error retrieving workspace topic summary for workspace {workspace_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve workspace topic summary: {str(e)}",
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
