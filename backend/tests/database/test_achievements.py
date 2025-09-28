import pytest
from backend.app.database.achievements import AchievementDatabase
from backend.app.services.database import DatabaseService


class TestAchievementDatabase:
    """Test cases for AchievementDatabase class."""

    @pytest.fixture
    def achievement_db(self):
        """Create AchievementDatabase instance."""
        db_service = DatabaseService()
        return AchievementDatabase(db_service)

    def test_get_all_achievements_empty(self, achievement_db):
        """Test getting all achievements when none exist."""
        achievements = achievement_db.get_all_achievements()
        assert isinstance(achievements, list)
        assert len(achievements) == 0

    def test_create_and_get_achievement(self, achievement_db):
        """Test creating and retrieving an achievement."""
        # Create test achievement
        achievement_data = {
            "id": "test_achievement_1",
            "name": "Test Achievement",
            "description": "A test achievement",
            "icon": "trophy",
            "target_value": 10,
            "category": "study",
        }

        # Create achievement
        result = achievement_db.create_achievement(achievement_data)
        assert result > 0  # Should return row ID

        # Get the achievement back
        achievement = achievement_db.get_achievement("test_achievement_1")
        assert achievement is not None
        assert achievement["id"] == "test_achievement_1"
        assert achievement["name"] == "Test Achievement"
        assert achievement["description"] == "A test achievement"
        assert achievement["icon"] == "trophy"
        assert achievement["target_value"] == 10
        assert achievement["category"] == "study"

    def test_get_all_achievements_with_data(self, achievement_db):
        """Test getting all achievements when data exists."""
        # Create multiple achievements
        achievements_data = [
            {
                "id": "achievement_1",
                "name": "First Achievement",
                "description": "First test achievement",
                "icon": "star",
                "target_value": 5,
                "category": "study",
            },
            {
                "id": "achievement_2",
                "name": "Second Achievement",
                "description": "Second test achievement",
                "icon": "medal",
                "target_value": 15,
                "category": "quiz",
            },
        ]

        for data in achievements_data:
            achievement_db.create_achievement(data)

        # Get all achievements
        achievements = achievement_db.get_all_achievements()
        assert len(achievements) == 2

        # Check that both achievements are present
        ids = [a["id"] for a in achievements]
        assert "achievement_1" in ids
        assert "achievement_2" in ids

    def test_get_achievement_not_found(self, achievement_db):
        """Test getting a non-existent achievement."""
        achievement = achievement_db.get_achievement("non_existent_id")
        assert achievement is None

    def test_update_achievement(self, achievement_db):
        """Test updating an achievement."""
        # Create initial achievement
        initial_data = {
            "id": "update_test",
            "name": "Initial Name",
            "description": "Initial description",
            "icon": "initial_icon",
            "target_value": 5,
            "category": "initial_category",
        }
        achievement_db.create_achievement(initial_data)

        # Update achievement
        update_data = {
            "name": "Updated Name",
            "description": "Updated description",
            "icon": "updated_icon",
            "target_value": 10,
            "category": "updated_category",
        }
        result = achievement_db.update_achievement("update_test", update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        achievement = achievement_db.get_achievement("update_test")
        assert achievement["name"] == "Updated Name"
        assert achievement["description"] == "Updated description"
        assert achievement["icon"] == "updated_icon"
        assert achievement["target_value"] == 10
        assert achievement["category"] == "updated_category"

    def test_update_achievement_not_found(self, achievement_db):
        """Test updating a non-existent achievement."""
        update_data = {
            "name": "New Name",
            "description": "New description",
            "icon": "new_icon",
            "target_value": 10,
            "category": "new_category",
        }
        result = achievement_db.update_achievement("non_existent", update_data)
        assert result == 0  # Should affect 0 rows
