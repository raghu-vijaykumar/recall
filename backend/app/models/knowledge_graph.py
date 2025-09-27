"""
Knowledge Graph models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid


class ConceptBase(BaseModel):
    name: str = Field(..., description="Concept name")
    description: Optional[str] = Field(None, description="Concept description")


class ConceptCreate(ConceptBase):
    concept_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique concept ID"
    )


class Concept(ConceptBase):
    concept_id: str = Field(..., description="Unique concept ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class RelationshipBase(BaseModel):
    source_concept_id: str = Field(..., description="Source concept ID")
    target_concept_id: str = Field(..., description="Target concept ID")
    type: str = Field(
        ..., description="Relationship type (relates_to, dives_deep_to, etc.)"
    )
    strength: Optional[float] = Field(
        None, description="Relationship strength/relevance score"
    )


class RelationshipCreate(RelationshipBase):
    relationship_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique relationship ID"
    )


class Relationship(RelationshipBase):
    relationship_id: str = Field(..., description="Unique relationship ID")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


class ConceptFileBase(BaseModel):
    concept_id: str = Field(..., description="Concept ID")
    file_id: int = Field(..., description="File ID")
    workspace_id: int = Field(..., description="Workspace ID")
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    relevance_score: Optional[float] = Field(None, description="Relevance score")
    last_accessed_at: Optional[datetime] = Field(
        None, description="Last accessed timestamp"
    )
    start_line: Optional[int] = Field(None, description="Starting line number in file")
    end_line: Optional[int] = Field(None, description="Ending line number in file")


class ConceptFileCreate(ConceptFileBase):
    concept_file_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique concept-file link ID",
    )


class ConceptFile(ConceptFileBase):
    concept_file_id: str = Field(..., description="Unique concept-file link ID")

    class Config:
        from_attributes = True


# New models for workspace-level topic discovery and learning paths


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


class TopicConceptLinkBase(BaseModel):
    """Links concepts to topic areas"""

    topic_area_id: str = Field(..., description="Topic area ID")
    concept_id: str = Field(..., description="Concept ID")
    relevance_score: float = Field(
        ..., description="How relevant this concept is to the topic"
    )
    explored: bool = Field(False, description="Whether user has explored this concept")


class TopicConceptLinkCreate(TopicConceptLinkBase):
    topic_concept_link_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique link ID"
    )


class TopicConceptLink(TopicConceptLinkBase):
    topic_concept_link_id: str = Field(..., description="Unique link ID")

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
        ..., description="Type: quiz_performance, concept_gap, prerequisite, interest"
    )
    topic_area_id: Optional[str] = Field(None, description="Related topic area")
    concept_id: Optional[str] = Field(None, description="Specific concept to study")
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


class WorkspaceTopicAnalysis(BaseModel):
    """Results of workspace-level topic analysis"""

    workspace_id: int
    topic_areas: List[TopicArea]
    total_concepts: int
    total_files: int
    coverage_distribution: Dict[str, float]  # topic_area_id -> coverage_score
    learning_paths: List[LearningPath]
    recommendations: List[LearningRecommendation]
    analysis_timestamp: datetime
    next_analysis_suggested: Optional[datetime] = None
