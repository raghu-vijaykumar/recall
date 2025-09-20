"""
File models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
import os


class FileType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    PDF = "pdf"
    IMAGE = "image"
    OTHER = "other"


class FileBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="File name")
    path: str = Field(..., description="File path relative to workspace")
    file_type: FileType = Field(..., description="Type of file")
    size: int = Field(..., ge=0, description="File size in bytes")
    workspace_id: int = Field(..., description="Parent workspace ID")


class FileCreate(FileBase):
    content: Optional[str] = Field(None, description="File content for text files")


class FileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    path: Optional[str] = Field(None)
    content: Optional[str] = Field(None, description="Updated file content")


class FileItem(FileBase):
    id: int = Field(..., description="Unique file ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    content_hash: Optional[str] = Field(
        None, description="Content hash for change detection"
    )
    question_count: int = Field(
        0, description="Number of questions generated from this file"
    )
    last_processed: Optional[datetime] = Field(
        None, description="Last time file was processed for questions"
    )

    class Config:
        from_attributes = True


class FileStats(BaseModel):
    file_id: int
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    difficulty_score: float  # 0-1 scale
    last_studied: Optional[datetime]


class FileTreeNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    children: Optional[List["FileTreeNode"]] = None
    file_info: Optional[FileItem] = None


# Update forward reference
FileTreeNode.update_forward_refs()
