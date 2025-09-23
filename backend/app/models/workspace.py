"""
Workspace models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class WorkspaceType(str, Enum):
    STUDY = "study"
    PROJECT = "project"
    NOTES = "notes"


class WorkspaceBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Workspace name")
    description: Optional[str] = Field(
        None, max_length=500, description="Workspace description"
    )
    type: WorkspaceType = Field(WorkspaceType.STUDY, description="Workspace type")
    color: Optional[str] = Field("#007bff", description="Workspace color theme")
    folder_path: Optional[str] = Field(
        None, description="Path to associated folder on file system"
    )


class WorkspaceCreate(WorkspaceBase):
    pass


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    type: Optional[WorkspaceType] = None
    color: Optional[str] = None
    folder_path: Optional[str] = Field(
        None, description="Path to associated folder on file system"
    )


class Workspace(WorkspaceBase):
    id: int = Field(..., description="Unique workspace ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    file_count: int = Field(default=0, description="Number of files in workspace")
    total_questions: int = Field(default=0, description="Total questions generated")
    completed_questions: int = Field(
        default=0, description="Questions answered correctly"
    )
    last_studied: Optional[datetime] = Field(
        default=None, description="Last study session timestamp"
    )

    class Config:
        from_attributes = True


class WorkspaceStats(BaseModel):
    workspace_id: int
    total_files: int
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    study_streak: int
    average_score: float
    last_study_date: Optional[datetime]
