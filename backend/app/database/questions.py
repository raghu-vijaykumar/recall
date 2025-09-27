"""
Database operations for questions table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class QuestionDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_question(self, data: Dict[str, Any]) -> int:
        """Create a new question"""
        return self.db.insert("questions", data)

    def get_question(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Get question by ID"""
        return self.db.get_by_id("questions", question_id)

    def get_questions_by_file(self, file_id: int) -> List[Dict[str, Any]]:
        """Get all questions for a file"""
        return self.db.execute_query(
            "SELECT * FROM questions WHERE file_id = ? ORDER BY created_at",
            (file_id,),
        )

    def get_questions_by_workspace(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get all questions for a workspace"""
        return self.db.execute_query(
            """
            SELECT q.* FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = ?
            ORDER BY q.created_at
        """,
            (workspace_id,),
        )

    def update_question(self, question_id: int, data: Dict[str, Any]) -> int:
        """Update question information"""
        return self.db.update("questions", question_id, data)

    def delete_question(self, question_id: int) -> int:
        """Delete a question"""
        return self.db.delete("questions", question_id)

    def update_question_stats(self, question_id: int, is_correct: bool) -> int:
        """Update question statistics after answering"""
        current = self.get_question(question_id)
        if not current:
            return 0

        times_asked = current["times_asked"] + 1
        times_correct = current["times_correct"] + (1 if is_correct else 0)

        return self.update_question(
            question_id,
            {
                "times_asked": times_asked,
                "times_correct": times_correct,
                "last_asked": datetime.now(),
            },
        )

    def get_questions_due_for_review(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get questions due for spaced repetition review"""
        return self.db.execute_query(
            """
            SELECT q.*, srd.next_review
            FROM questions q
            JOIN spaced_repetition_data srd ON q.id = srd.question_id
            WHERE srd.next_review <= ?
            ORDER BY srd.next_review ASC
            LIMIT ?
        """,
            (datetime.now(), limit),
        )

    def get_question_performance_stats(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get performance statistics for questions in workspace"""
        return self.db.execute_query(
            """
            SELECT
                q.id,
                q.difficulty,
                q.kg_concept_ids,
                q.times_asked,
                q.times_correct,
                AVG(a.time_taken) as avg_time
            FROM questions q
            JOIN files f ON q.file_id = f.id
            LEFT JOIN answers a ON q.id = a.question_id
            WHERE f.workspace_id = ?
            GROUP BY q.id, q.difficulty, q.kg_concept_ids
        """,
            (workspace_id,),
        )

    def count_questions_by_workspace(self, workspace_id: int) -> int:
        """Count questions in workspace"""
        result = self.db.execute_query(
            """
            SELECT COUNT(*) as count FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = ?
        """,
            (workspace_id,),
        )
        return result[0]["count"] if result else 0
