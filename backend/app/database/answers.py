"""
Database operations for answers table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class AnswerDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_answer(self, data: Dict[str, Any]) -> int:
        """Create a new answer"""
        return self.db.insert("answers", data)

    def get_answer(self, answer_id: int) -> Optional[Dict[str, Any]]:
        """Get answer by ID"""
        return self.db.get_by_id("answers", answer_id)

    def get_answers_by_question(self, question_id: int) -> List[Dict[str, Any]]:
        """Get all answers for a question"""
        return self.db.execute_query(
            "SELECT * FROM answers WHERE question_id = ? ORDER BY id DESC",
            (question_id,),
        )

    def get_answers_by_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all answers for a quiz session"""
        return self.db.execute_query(
            """
            SELECT a.* FROM answers a
            JOIN session_questions sq ON a.question_id = sq.question_id
            WHERE sq.session_id = ?
            ORDER BY a.created_at
        """,
            (session_id,),
        )

    def update_answer(self, answer_id: int, data: Dict[str, Any]) -> int:
        """Update answer information"""
        return self.db.update("answers", answer_id, data)

    def delete_answer(self, answer_id: int) -> int:
        """Delete an answer"""
        return self.db.delete("answers", answer_id)

    def get_answer_statistics(self, question_id: int) -> Dict[str, Any]:
        """Get statistics for answers to a question"""
        result = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total_answers,
                AVG(time_taken) as avg_time,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers
            FROM answers
            WHERE question_id = ?
        """,
            (question_id,),
        )
        if result:
            stats = result[0]
            stats["avg_time"] = stats["avg_time"] or 0
            stats["correct_answers"] = stats["correct_answers"] or 0
            return stats
        else:
            return {"total_answers": 0, "avg_time": 0, "correct_answers": 0}

    def get_recent_answers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent answers"""
        return self.db.execute_query(
            "SELECT * FROM answers ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
