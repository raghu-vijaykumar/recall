"""
Database operations for achievements table
"""

from typing import List, Dict, Any, Optional
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class AchievementDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def get_all_achievements(self) -> List[Dict[str, Any]]:
        """Get all achievements"""
        return self.db.get_all("achievements")

    def get_achievement(self, achievement_id: str) -> Optional[Dict[str, Any]]:
        """Get achievement by ID"""
        result = self.db.execute_query(
            "SELECT * FROM achievements WHERE id = ?",
            (achievement_id,),
        )
        return result[0] if result else None

    def create_achievement(self, data: Dict[str, Any]) -> int:
        """Create a new achievement"""
        return self.db.insert("achievements", data)

    def update_achievement(self, achievement_id: str, data: Dict[str, Any]) -> int:
        """Update achievement"""
        return self.db.execute_update(
            "UPDATE achievements SET name = ?, description = ?, icon = ?, target_value = ?, category = ? WHERE id = ?",
            (
                data["name"],
                data["description"],
                data["icon"],
                data["target_value"],
                data["category"],
                achievement_id,
            ),
        )
