"""
Knowledge Graph models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List
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


class ConceptFileCreate(ConceptFileBase):
    concept_file_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique concept-file link ID",
    )


class ConceptFile(ConceptFileBase):
    concept_file_id: str = Field(..., description="Unique concept-file link ID")

    class Config:
        from_attributes = True
