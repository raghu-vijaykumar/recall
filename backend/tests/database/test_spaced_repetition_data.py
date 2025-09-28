import pytest
from datetime import datetime, timedelta
from backend.app.database.spaced_repetition_data import SpacedRepetitionDataDatabase
from backend.app.services.database import DatabaseService


class TestSpacedRepetitionDataDatabase:
    """Test cases for SpacedRepetitionDataDatabase class."""

    @pytest.fixture
    def spaced_repetition_db(self):
        """Create SpacedRepetitionDataDatabase instance."""
        db_service = DatabaseService()
        return SpacedRepetitionDataDatabase(db_service)

    def test_create_and_get_spaced_repetition_data(self, spaced_repetition_db):
        """Test creating and retrieving spaced repetition data."""
        # Create workspace, file, and question first
        db_service = spaced_repetition_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Spaced Repetition Test Workspace", "type": "study"}
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "test.txt",
                "path": "/path/test.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create question
        question_id = db_service.insert(
            "questions",
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Test question?",
                "correct_answer": "A",
            },
        )

        # Create spaced repetition data
        next_review = datetime.now() + timedelta(days=1)
        data = {
            "question_id": question_id,
            "ease_factor": 2.5,
            "interval_days": 1,
            "review_count": 0,
            "next_review": next_review.isoformat(),
            "kg_concept_id": "concept_123",
        }

        result = spaced_repetition_db.create_spaced_repetition_data(data)
        assert result > 0

        # Get the data back
        retrieved_data = spaced_repetition_db.get_spaced_repetition_data(question_id)
        assert retrieved_data is not None
        assert retrieved_data["question_id"] == question_id
        assert retrieved_data["ease_factor"] == 2.5
        assert retrieved_data["interval_days"] == 1
        assert retrieved_data["review_count"] == 0
        assert retrieved_data["kg_concept_id"] == "concept_123"

    def test_get_spaced_repetition_data_not_found(self, spaced_repetition_db):
        """Test getting spaced repetition data for non-existent question."""
        data = spaced_repetition_db.get_spaced_repetition_data(99999)
        assert data is None

    def test_update_spaced_repetition_data(self, spaced_repetition_db):
        """Test updating spaced repetition data."""
        # Create workspace, file, question, and initial data
        db_service = spaced_repetition_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces",
            {"name": "Update Spaced Repetition Workspace", "type": "study"},
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "update.txt",
                "path": "/path/update.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create question
        question_id = db_service.insert(
            "questions",
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Update test question?",
                "correct_answer": "A",
            },
        )

        # Create initial spaced repetition data
        initial_next_review = datetime.now() + timedelta(days=1)
        initial_data = {
            "question_id": question_id,
            "ease_factor": 2.5,
            "interval_days": 1,
            "review_count": 0,
            "next_review": initial_next_review.isoformat(),
            "kg_concept_id": "concept_123",
        }
        spaced_repetition_db.create_spaced_repetition_data(initial_data)

        # Update the data
        new_next_review = datetime.now() + timedelta(days=3)
        update_data = {
            "ease_factor": 2.7,
            "interval_days": 3,
            "review_count": 1,
            "next_review": new_next_review.isoformat(),
            "kg_concept_id": "concept_456",
        }
        result = spaced_repetition_db.update_spaced_repetition_data(
            question_id, update_data
        )
        assert result == 1  # Should affect 1 row

        # Verify update
        data = spaced_repetition_db.get_spaced_repetition_data(question_id)
        assert data["ease_factor"] == 2.7
        assert data["interval_days"] == 3
        assert data["review_count"] == 1
        assert data["kg_concept_id"] == "concept_456"

    def test_get_questions_due_for_review_empty(self, spaced_repetition_db):
        """Test getting questions due for review when none exist."""
        questions = spaced_repetition_db.get_questions_due_for_review()
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_questions_due_for_review_with_data(self, spaced_repetition_db):
        """Test getting questions due for spaced repetition review."""
        # Create workspace, file, and questions
        db_service = spaced_repetition_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Due Review Workspace", "type": "study"}
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "review.txt",
                "path": "/path/review.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create questions
        question_ids = []
        for i in range(3):
            q_id = db_service.insert(
                "questions",
                {
                    "file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_text": f"Review question {i}?",
                    "correct_answer": "A",
                },
            )
            question_ids.append(q_id)

        # Create spaced repetition data with different due dates
        now = datetime.now()
        due_dates = [
            now - timedelta(hours=1),  # Due now
            now + timedelta(hours=1),  # Due later
            now - timedelta(hours=2),  # Due now
        ]

        for i, (q_id, due_date) in enumerate(zip(question_ids, due_dates)):
            data = {
                "question_id": q_id,
                "ease_factor": 2.5,
                "interval_days": 1,
                "review_count": i,
                "next_review": due_date.isoformat(),
                "kg_concept_id": f"concept_{i}",
            }
            spaced_repetition_db.create_spaced_repetition_data(data)

        # Get questions due for review
        due_questions = spaced_repetition_db.get_questions_due_for_review(limit=10)
        assert len(due_questions) == 2  # Two questions are due

        # Should be ordered by next_review ASC
        assert due_questions[0]["question_id"] == question_ids[2]  # Due 2 hours ago
        assert due_questions[1]["question_id"] == question_ids[0]  # Due 1 hour ago

    def test_get_overdue_questions_empty(self, spaced_repetition_db):
        """Test getting overdue questions when none exist."""
        questions = spaced_repetition_db.get_overdue_questions()
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_overdue_questions_with_data(self, spaced_repetition_db):
        """Test getting questions that are overdue for review."""
        # Create workspace, file, and questions
        db_service = spaced_repetition_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Overdue Workspace", "type": "study"}
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "overdue.txt",
                "path": "/path/overdue.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create questions
        question_ids = []
        for i in range(4):
            q_id = db_service.insert(
                "questions",
                {
                    "file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_text": f"Overdue question {i}?",
                    "correct_answer": "A",
                },
            )
            question_ids.append(q_id)

        # Create spaced repetition data with different due dates
        now = datetime.now()
        due_dates = [
            now - timedelta(hours=1),  # Overdue
            now + timedelta(hours=1),  # Not overdue
            now - timedelta(hours=2),  # Overdue
            now - timedelta(minutes=30),  # Overdue
        ]

        for i, (q_id, due_date) in enumerate(zip(question_ids, due_dates)):
            data = {
                "question_id": q_id,
                "ease_factor": 2.5,
                "interval_days": 1,
                "review_count": i,
                "next_review": due_date.isoformat(),
                "kg_concept_id": f"concept_{i}",
            }
            spaced_repetition_db.create_spaced_repetition_data(data)

        # Get overdue questions
        overdue_questions = spaced_repetition_db.get_overdue_questions()
        assert len(overdue_questions) == 3  # Three questions are overdue

        # Should be ordered by next_review ASC (most overdue first)
        assert overdue_questions[0]["question_id"] == question_ids[2]  # Due 2 hours ago
        assert overdue_questions[1]["question_id"] == question_ids[0]  # Due 1 hour ago
        assert (
            overdue_questions[2]["question_id"] == question_ids[3]
        )  # Due 30 minutes ago

    def test_delete_spaced_repetition_data(self, spaced_repetition_db):
        """Test deleting spaced repetition data."""
        # Create workspace, file, question, and data
        db_service = spaced_repetition_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces",
            {"name": "Delete Spaced Repetition Workspace", "type": "study"},
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "delete.txt",
                "path": "/path/delete.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create question
        question_id = db_service.insert(
            "questions",
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Delete test question?",
                "correct_answer": "A",
            },
        )

        # Create spaced repetition data
        next_review = datetime.now() + timedelta(days=1)
        data = {
            "question_id": question_id,
            "ease_factor": 2.5,
            "interval_days": 1,
            "review_count": 0,
            "next_review": next_review.isoformat(),
            "kg_concept_id": "concept_123",
        }
        spaced_repetition_db.create_spaced_repetition_data(data)

        # Verify it exists
        retrieved_data = spaced_repetition_db.get_spaced_repetition_data(question_id)
        assert retrieved_data is not None

        # Delete the data
        result = spaced_repetition_db.delete_spaced_repetition_data(question_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        retrieved_data = spaced_repetition_db.get_spaced_repetition_data(question_id)
        assert retrieved_data is None
