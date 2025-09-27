"""
Database operations for user_achievements table
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.services.database import DatabaseService

logger = logging.getLogger(__name__)


class UserAchievementDatabase:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    def unlock_achievement(
        self, achievement_id: str, user_id: Optional[str] = None
    ) -> int:
        """Unlock an achievement for a user"""
        return self.db.insert(
            "user_achievements",
            {
                "achievement_id": achievement_id,
                "user_id": user_id,
                "unlocked_at": datetime.now(),
                "progress": 1.0,
                "current_value": 1,
            },
        )

    def get_user_achievements(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get achievements unlocked by user"""
        if user_id:
            return self.db.execute_query(
                "SELECT * FROM user_achievements WHERE user_id = ? ORDER BY unlocked_at DESC",
                (user_id,),
            )
        return self.db.get_all("user_achievements")

    def update_progress(
        self,
        achievement_id: str,
        user_id: Optional[str],
        progress: float,
        current_value: int,
    ) -> int:
        """Update progress on an achievement"""
        return self.db.execute_update(
            "UPDATE user_achievements SET progress = ?, current_value = ? WHERE achievement_id = ? AND user_id = ?",
            (progress, current_value, achievement_id, user_id),
        )

    def is_achievement_unlocked(
        self, achievement_id: str, user_id: Optional[str] = None
    ) -> bool:
        """Check if achievement is unlocked"""
        result = self.db.execute_query(
            "SELECT id FROM user_achievements WHERE achievement_id = ? AND user_id = ?",
            (achievement_id, user_id),
        )
        return len(result) > 0
