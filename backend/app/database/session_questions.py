"""
Database operations for session_questions table
"""

from typing import List, Dict, Any, Optional
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class SessionQuestionDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def add_question_to_session(self, data: Dict[str, Any]) -> int:
        """Add a question to a quiz session"""
        return self.db.insert("session_questions", data)

    def get_session_questions(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all questions for a session"""
        return self.db.execute_query(
            "SELECT * FROM session_questions WHERE session_id = ? ORDER BY question_order",
            (session_id,),
        )

    def update_answer(
        self, session_id: int, question_id: int, answer_data: Dict[str, Any]
    ) -> int:
        """Update answer for a session question"""
        return self.db.update(
            "session_questions",
            None,  # No primary key update
            {**answer_data},
            where_clause="session_id = ? AND question_id = ?",
            where_params=(session_id, question_id),
        )

    def get_session_results(self, session_id: int) -> List[Dict[str, Any]]:
        """Get results for all questions in a session"""
        return self.db.execute_query(
            """
            SELECT sq.*, q.question_text, q.correct_answer, q.options, q.explanation
            FROM session_questions sq
            JOIN questions q ON sq.question_id = q.id
            WHERE sq.session_id = ?
            ORDER BY sq.question_order
        """,
            (session_id,),
        )

    def count_correct_answers(self, session_id: int) -> int:
        """Count correct answers in a session"""
        result = self.db.execute_query(
            "SELECT COUNT(*) as correct FROM session_questions WHERE session_id = ? AND is_correct = 1",
            (session_id,),
        )
        return result[0]["correct"] if result else 0
