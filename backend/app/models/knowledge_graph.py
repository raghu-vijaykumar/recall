"""
Knowledge Graph models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


# Simplified topic-only models for workspace topic discovery


class TopicAreaBase(BaseModel):
    """Major subject areas/topics identified in a workspace"""

    workspace_id: int = Field(..., description="Workspace ID")
    name: str = Field(..., description="Topic area name (e.g., 'Machine Learning')")
    description: str = Field(..., description="Topic area description")
    coverage_score: float = Field(
        ..., description="How well this area is covered (0-1)"
    )
    concept_count: int = Field(..., description="Number of concepts in this area")
    file_count: int = Field(..., description="Number of files covering this area")
    explored_percentage: float = Field(
        ..., description="Percentage of area explored (0-1)"
    )


class TopicAreaCreate(TopicAreaBase):
    topic_area_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique topic area ID"
    )


class TopicArea(TopicAreaBase):
    topic_area_id: str = Field(..., description="Unique topic area ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class LearningPathBase(BaseModel):
    """Recommended learning paths for users"""

    workspace_id: int = Field(..., description="Workspace ID")
    name: str = Field(..., description="Learning path name")
    description: str = Field(..., description="Learning path description")
    topic_areas: List[str] = Field(..., description="Topic area IDs in this path")
    estimated_hours: int = Field(..., description="Estimated hours to complete")
    difficulty_level: str = Field(
        ..., description="Overall difficulty: beginner, intermediate, advanced"
    )
    prerequisites: Optional[List[str]] = Field(
        None, description="Required prerequisite knowledge"
    )


class LearningPathCreate(LearningPathBase):
    learning_path_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique learning path ID"
    )


class LearningPath(LearningPathBase):
    learning_path_id: str = Field(..., description="Unique learning path ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class LearningRecommendationBase(BaseModel):
    """Specific recommendations for what to study next"""

    workspace_id: int = Field(..., description="Workspace ID")
    user_id: Optional[str] = Field(None, description="User ID (if multi-user)")
    recommendation_type: str = Field(
        ..., description="Type: quiz_performance, topic_gap, prerequisite, interest"
    )
    topic_area_id: Optional[str] = Field(None, description="Related topic area")
    priority_score: float = Field(..., description="Recommendation priority (0-1)")
    reason: str = Field(..., description="Why this is recommended")
    suggested_action: str = Field(..., description="What the user should do")


class LearningRecommendationCreate(LearningRecommendationBase):
    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation ID",
    )


class LearningRecommendation(LearningRecommendationBase):
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


# Topic Relationship models for the knowledge graph


class TopicRelationshipBase(BaseModel):
    """Base model for relationships between topics"""

    source_topic_id: str = Field(..., description="Source topic area ID")
    target_topic_id: str = Field(..., description="Target topic area ID")
    relationship_type: str = Field(
        ...,
        description="Type of relationship: relates_to, builds_on, contrasts_with, contains, precedes",
    )
    strength: float = Field(..., description="Relationship strength (0.0-1.0)")
    reasoning: Optional[str] = Field(
        None, description="LLM-generated reasoning for the relationship"
    )


class TopicRelationshipCreate(TopicRelationshipBase):
    relationship_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique relationship ID"
    )


class TopicRelationship(TopicRelationshipBase):
    relationship_id: str = Field(..., description="Unique relationship ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class WorkspaceTopicGraph(BaseModel):
    """Knowledge graph structure for a workspace"""

    workspace_id: int
    topics: List[TopicArea]
    relationships: List[TopicRelationship]


class WorkspaceTopicAnalysis(BaseModel):
    """Results of workspace-level topic analysis"""

    workspace_id: int
    topic_areas: List[TopicArea]
    total_files: int
    coverage_distribution: Dict[str, float]  # topic_area_id -> coverage_score
    learning_paths: List[LearningPath]
    recommendations: List[LearningRecommendation]
    analysis_timestamp: datetime
    next_analysis_suggested: Optional[datetime] = None
