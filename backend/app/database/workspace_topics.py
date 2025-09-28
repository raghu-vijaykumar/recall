"""
Database operations for workspace topics, topic areas, learning paths, and recommendations.
"""

import json
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import (
    TopicArea,
    LearningPath,
    LearningRecommendation,
    WorkspaceTopicAnalysis,
)


class WorkspaceTopicsDatabase:
    """Database operations for workspace topic-related data."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_workspace_topic_areas(self, workspace_id: int) -> List[TopicArea]:
        """
        Get all topic areas for a workspace.

        Args:
            workspace_id: ID of the workspace

        Returns:
            List of topic areas
        """
        query = text(
            """
            SELECT * FROM topic_areas
            WHERE workspace_id = :workspace_id
            ORDER BY coverage_score DESC, explored_percentage DESC
            """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
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

    async def get_workspace_learning_paths(
        self, workspace_id: int
    ) -> List[LearningPath]:
        """
        Get all learning paths for a workspace.

        Args:
            workspace_id: ID of the workspace

        Returns:
            List of learning paths
        """
        query = text(
            """
            SELECT * FROM learning_paths
            WHERE workspace_id = :workspace_id
            ORDER BY estimated_hours ASC
            """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
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

    async def get_workspace_recommendations(
        self, workspace_id: int, limit: int = 10
    ) -> List[LearningRecommendation]:
        """
        Get learning recommendations for a workspace.

        Args:
            workspace_id: ID of the workspace
            limit: Maximum number of recommendations to return

        Returns:
            List of learning recommendations ordered by priority
        """
        query = text(
            """
            SELECT * FROM learning_recommendations
            WHERE workspace_id = :workspace_id
            ORDER BY priority_score DESC
            LIMIT :limit
            """
        )

        result = await self.db.execute(
            query, {"workspace_id": workspace_id, "limit": limit}
        )
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

    async def get_topic_area_concepts(self, topic_area_id: str) -> List[dict]:
        """
        Get all concepts associated with a specific topic area.

        Args:
            topic_area_id: ID of the topic area

        Returns:
            List of concepts with their relevance scores
        """
        query = text(
            """
            SELECT c.*, tcl.relevance_score, tcl.explored
            FROM concepts c
            JOIN topic_concept_links tcl ON c.concept_id = tcl.concept_id
            WHERE tcl.topic_area_id = :topic_area_id
            ORDER BY tcl.relevance_score DESC
            """
        )

        result = await self.db.execute(query, {"topic_area_id": topic_area_id})
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

    async def get_topic_area_files(self, topic_area_id: str) -> List[dict]:
        """
        Get all files that contain concepts from a specific topic area.

        Args:
            topic_area_id: ID of the topic area

        Returns:
            List of files with concept coverage information
        """
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

        result = await self.db.execute(query, {"topic_area_id": topic_area_id})
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

    async def store_topic_area(self, topic_area: TopicArea) -> None:
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

    async def store_concept_link(self, concept_link) -> None:
        """Store a concept link in the database"""
        query = text(
            """
            INSERT OR REPLACE INTO topic_concept_links
            (topic_concept_link_id, topic_area_id, concept_id, relevance_score, explored)
            VALUES (:id, :topic_area_id, :concept_id, :relevance_score, :explored)
            """
        )

        await self.db.execute(
            query,
            {
                "id": concept_link.topic_concept_link_id,
                "topic_area_id": concept_link.topic_area_id,
                "concept_id": concept_link.concept_id,
                "relevance_score": concept_link.relevance_score,
                "explored": concept_link.explored,
            },
        )

    async def store_learning_path(self, learning_path: LearningPath) -> None:
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

    async def store_recommendation(
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
