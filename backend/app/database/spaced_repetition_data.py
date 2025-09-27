"""
Database operations for spaced_repetition_data table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class SpacedRepetitionDataDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def create_spaced_repetition_data(self, data: Dict[str, Any]) -> int:
        """Create spaced repetition data for a question"""
        return self.db.insert("spaced_repetition_data", data)

    def get_spaced_repetition_data(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Get spaced repetition data for a question"""
        result = self.db.execute_query(
            "SELECT * FROM spaced_repetition_data WHERE question_id = ?",
            (question_id,),
        )
        return result[0] if result else None

    def update_spaced_repetition_data(
        self, question_id: int, data: Dict[str, Any]
    ) -> int:
        """Update spaced repetition data"""
        return self.db.execute_update(
            "UPDATE spaced_repetition_data SET ease_factor = ?, interval_days = ?, review_count = ?, next_review = ?, kg_concept_id = ? WHERE question_id = ?",
            (
                data["ease_factor"],
                data["interval_days"],
                data["review_count"],
                data["next_review"],
                data.get("kg_concept_id"),
                question_id,
            ),
        )

    def get_questions_due_for_review(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get questions due for spaced repetition review"""
        return self.db.execute_query(
            "SELECT * FROM spaced_repetition_data WHERE next_review <= ? ORDER BY next_review ASC LIMIT ?",
            (datetime.now(), limit),
        )

    def get_overdue_questions(self) -> List[Dict[str, Any]]:
        """Get questions that are overdue for review"""
        return self.db.execute_query(
            "SELECT * FROM spaced_repetition_data WHERE next_review < ? ORDER BY next_review ASC",
            (datetime.now(),),
        )

    def delete_spaced_repetition_data(self, question_id: int) -> int:
        """Delete spaced repetition data for a question"""
        return self.db.execute_update(
            "DELETE FROM spaced_repetition_data WHERE question_id = ?",
            (question_id,),
        )
