"""
Workspace Topics API routes for topic discovery and learning path recommendations
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from ..database import get_db
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
        topic_service = WorkspaceTopicDiscoveryService(
            db, kg_service, embedding_service
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
        from sqlalchemy import text

        query = text(
            """
            SELECT * FROM topic_areas
            WHERE workspace_id = :workspace_id
            ORDER BY coverage_score DESC, explored_percentage DESC
            """
        )

        result = await db.execute(query, {"workspace_id": workspace_id})
        rows = result.fetchall()

        topic_areas = []
        for row in rows:
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

        return topic_areas

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
        import json
        from sqlalchemy import text

        query = text(
            """
            SELECT * FROM learning_paths
            WHERE workspace_id = :workspace_id
            ORDER BY estimated_hours ASC
            """
        )

        result = await db.execute(query, {"workspace_id": workspace_id})
        rows = result.fetchall()

        learning_paths = []
        for row in rows:
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

        return learning_paths

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
        from sqlalchemy import text

        query = text(
            """
            SELECT * FROM learning_recommendations
            WHERE workspace_id = :workspace_id
            ORDER BY priority_score DESC
            LIMIT :limit
            """
        )

        result = await db.execute(query, {"workspace_id": workspace_id, "limit": limit})
        rows = result.fetchall()

        recommendations = []
        for row in rows:
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

        return recommendations

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
        from sqlalchemy import text

        query = text(
            """
            SELECT c.*, tcl.relevance_score, tcl.explored
            FROM concepts c
            JOIN topic_concept_links tcl ON c.concept_id = tcl.concept_id
            WHERE tcl.topic_area_id = :topic_area_id
            ORDER BY tcl.relevance_score DESC
            """
        )

        result = await db.execute(query, {"topic_area_id": topic_area_id})
        rows = result.fetchall()

        concepts = []
        for row in rows:
            concepts.append(
                {
                    "concept_id": row.concept_id,
                    "name": row.name,
                    "description": row.description,
                    "relevance_score": row.relevance_score,
                    "explored": row.explored,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )

        return concepts

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
        from sqlalchemy import text

        query = text(
            """
            SELECT DISTINCT f.*, COUNT(cf.concept_id) as concept_count
            FROM files f
            JOIN concept_files cf ON f.id = cf.file_id
            JOIN topic_concept_links tcl ON cf.concept_id = tcl.concept_id
            WHERE tcl.topic_area_id = :topic_area_id
            GROUP BY f.id
            ORDER BY concept_count DESC
            """
        )

        result = await db.execute(query, {"topic_area_id": topic_area_id})
        rows = result.fetchall()

        files = []
        for row in rows:
            files.append(
                {
                    "id": row.id,
                    "workspace_id": row.workspace_id,
                    "name": row.name,
                    "path": row.path,
                    "file_type": row.file_type,
                    "size": row.size,
                    "concept_count": row.concept_count,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )

        return files

    except Exception as e:
        logging.error(f"Error retrieving files for topic area {topic_area_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve topic area files: {str(e)}"
        )
