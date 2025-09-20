"""
Progress tracking models for the Recall application
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date


class ProgressBase(BaseModel):
    user_id: Optional[str] = Field(
        None, description="User identifier (for future multi-user support)"
    )
    workspace_id: int = Field(..., description="Workspace ID")
    file_id: Optional[int] = Field(
        None, description="File ID if tracking file-specific progress"
    )
    question_id: Optional[int] = Field(
        None, description="Question ID if tracking question-specific progress"
    )
    session_id: Optional[int] = Field(None, description="Quiz session ID")


class ProgressCreate(ProgressBase):
    action_type: str = Field(..., description="Type of progress action")
    metadata: Optional[Dict] = Field(None, description="Additional progress data")


class Progress(ProgressBase):
    id: int = Field(..., description="Unique progress ID")
    timestamp: datetime = Field(..., description="When the progress was recorded")
    action_type: str = Field(..., description="Type of action (study, quiz, etc.)")
    value: Optional[float] = Field(
        None, description="Numeric value (score, time, etc.)"
    )
    metadata: Optional[Dict] = Field(None, description="Additional progress data")

    class Config:
        from_attributes = True


class StudySession(BaseModel):
    date: date
    duration_minutes: int
    questions_answered: int
    correct_answers: int
    workspaces_studied: List[int]
    average_difficulty: str


class UserStats(BaseModel):
    total_study_time: int  # in minutes
    total_questions_answered: int
    total_correct_answers: int
    average_score: float
    current_streak: int
    longest_streak: int
    favorite_workspace: Optional[int]
    study_sessions_this_week: int
    study_sessions_this_month: int
    improvement_rate: float  # percentage improvement over time
    weak_topics: List[str]
    strong_topics: List[str]


class ProgressReport(BaseModel):
    period: str  # "week", "month", "all_time"
    start_date: date
    end_date: date
    total_sessions: int
    total_study_time: int
    total_questions: int
    average_score: float
    daily_progress: List[Dict[str, Any]]  # date -> stats
    workspace_breakdown: Dict[int, Dict[str, Any]]  # workspace_id -> stats
    improvement_trend: str


class Achievement(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    unlocked_at: Optional[datetime]
    progress: float  # 0-1 completion percentage
    target_value: int
    current_value: int


class GamificationStats(BaseModel):
    level: int
    experience_points: int
    points_to_next_level: int
    achievements_unlocked: List[Achievement]
    current_streak: int
    longest_streak: int
    total_study_time: int
    questions_mastered: int
    workspaces_completed: int
