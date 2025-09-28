import pytest
from datetime import datetime, timedelta
from backend.app.database.questions import QuestionDatabase
from backend.app.services.database import DatabaseService


class TestQuestionDatabase:
    """Test cases for QuestionDatabase class."""

    @pytest.fixture
    def question_db(self):
        """Create QuestionDatabase instance."""
        db_service = DatabaseService()
        return QuestionDatabase(db_service)

    def test_create_and_get_question(self, question_db):
        """Test creating and retrieving a question."""
        # Create workspace and file first
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Question Test Workspace", "type": "study"}
        )
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

        # Create test question
        question_data = {
            "file_id": file_id,
            "question_type": "multiple_choice",
            "question_text": "What is 2+2?",
            "correct_answer": "4",
            "options": '["3", "4", "5", "6"]',
            "difficulty": "easy",
            "tags": '["math", "basic"]',
        }

        # Create question
        question_id = question_db.create_question(question_data)
        assert question_id > 0

        # Get the question back
        question = question_db.get_question(question_id)
        assert question is not None
        assert question["id"] == question_id
        assert question["file_id"] == file_id
        assert question["question_type"] == "multiple_choice"
        assert question["question_text"] == "What is 2+2?"
        assert question["correct_answer"] == "4"
        assert question["difficulty"] == "easy"

    def test_get_question_not_found(self, question_db):
        """Test getting a non-existent question."""
        question = question_db.get_question(99999)
        assert question is None

    def test_get_questions_by_file_empty(self, question_db):
        """Test getting questions for file with no questions."""
        # Create workspace and file
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty File Workspace", "type": "study"}
        )
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "empty.txt",
                "path": "/path/empty.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Get questions
        questions = question_db.get_questions_by_file(file_id)
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_questions_by_file_with_data(self, question_db):
        """Test getting questions for file with questions."""
        # Create workspace and file
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "File Questions Workspace", "type": "study"}
        )
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "questions.txt",
                "path": "/path/questions.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create multiple questions
        question_data = [
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Question 1?",
                "correct_answer": "A",
            },
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Question 2?",
                "correct_answer": "B",
            },
        ]

        for data in question_data:
            question_db.create_question(data)

        # Get questions
        questions = question_db.get_questions_by_file(file_id)
        assert len(questions) == 2

        # Should be ordered by created_at
        assert questions[0]["question_text"] == "Question 1?"
        assert questions[1]["question_text"] == "Question 2?"

    def test_get_questions_by_workspace_empty(self, question_db):
        """Test getting questions for workspace with no questions."""
        # Create workspace
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Workspace Questions", "type": "study"}
        )

        # Get questions
        questions = question_db.get_questions_by_workspace(workspace_id)
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_questions_by_workspace_with_data(self, question_db):
        """Test getting questions for workspace with questions."""
        # Create workspace
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Workspace Questions", "type": "study"}
        )

        # Create two files
        file_ids = []
        for i in range(2):
            file_id = db_service.insert(
                "files",
                {
                    "workspace_id": workspace_id,
                    "name": f"file{i}.txt",
                    "path": f"/path/file{i}.txt",
                    "file_type": "text",
                    "size": 100,
                },
            )
            file_ids.append(file_id)

        # Create questions for each file
        for i, file_id in enumerate(file_ids):
            question_db.create_question(
                {
                    "file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_text": f"Question for file {i}?",
                    "correct_answer": "A",
                }
            )

        # Get questions by workspace
        questions = question_db.get_questions_by_workspace(workspace_id)
        assert len(questions) == 2

    def test_update_question(self, question_db):
        """Test updating a question."""
        # Create workspace, file, and question
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Update Question Workspace", "type": "study"}
        )
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

        initial_data = {
            "file_id": file_id,
            "question_type": "multiple_choice",
            "question_text": "Old question?",
            "correct_answer": "A",
            "difficulty": "easy",
        }
        question_id = question_db.create_question(initial_data)

        # Update question
        update_data = {
            "question_text": "Updated question?",
            "correct_answer": "B",
            "difficulty": "medium",
        }
        result = question_db.update_question(question_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        question = question_db.get_question(question_id)
        assert question["question_text"] == "Updated question?"
        assert question["correct_answer"] == "B"
        assert question["difficulty"] == "medium"

    def test_delete_question(self, question_db):
        """Test deleting a question."""
        # Create workspace, file, and question
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Delete Question Workspace", "type": "study"}
        )
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

        question_data = {
            "file_id": file_id,
            "question_type": "multiple_choice",
            "question_text": "To delete?",
            "correct_answer": "A",
        }
        question_id = question_db.create_question(question_data)

        # Verify it exists
        question = question_db.get_question(question_id)
        assert question is not None

        # Delete question
        result = question_db.delete_question(question_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        question = question_db.get_question(question_id)
        assert question is None

    def test_update_question_stats(self, question_db):
        """Test updating question statistics."""
        # Create workspace, file, and question
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Stats Question Workspace", "type": "study"}
        )
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "stats.txt",
                "path": "/path/stats.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        question_data = {
            "file_id": file_id,
            "question_type": "multiple_choice",
            "question_text": "Stats question?",
            "correct_answer": "A",
            "times_asked": 5,
            "times_correct": 3,
        }
        question_id = question_db.create_question(question_data)

        # Update stats with correct answer
        result = question_db.update_question_stats(question_id, True)
        assert result == 1

        # Verify stats updated
        question = question_db.get_question(question_id)
        assert question["times_asked"] == 6
        assert question["times_correct"] == 4

        # Update stats with incorrect answer
        result = question_db.update_question_stats(question_id, False)
        assert result == 1

        # Verify stats updated again
        question = question_db.get_question(question_id)
        assert question["times_asked"] == 7
        assert question["times_correct"] == 4

    def test_update_question_stats_not_found(self, question_db):
        """Test updating stats for non-existent question."""
        result = question_db.update_question_stats(99999, True)
        assert result == 0

    def test_get_questions_due_for_review_empty(self, question_db):
        """Test getting questions due for review when none exist."""
        questions = question_db.get_questions_due_for_review()
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_questions_due_for_review_with_data(self, question_db):
        """Test getting questions due for spaced repetition review."""
        # Create workspace, file, and question
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Review Questions Workspace", "type": "study"}
        )
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

        # Create question
        question_id = question_db.create_question(
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Review question?",
                "correct_answer": "A",
            }
        )

        # Create spaced repetition data with due date in the past
        past_date = (datetime.now() - timedelta(days=1)).isoformat()
        db_service.insert(
            "spaced_repetition_data",
            {"question_id": question_id, "next_review": past_date},
        )

        # Get questions due for review
        due_questions = question_db.get_questions_due_for_review()
        assert len(due_questions) >= 1  # At least our question should be due

        # Check that our question is in the results
        question_ids = [q["id"] for q in due_questions]
        assert question_id in question_ids

    def test_get_question_performance_stats_empty(self, question_db):
        """Test getting performance stats for workspace with no questions."""
        # Create workspace
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Performance Workspace", "type": "study"}
        )

        # Get performance stats
        stats = question_db.get_question_performance_stats(workspace_id)
        assert isinstance(stats, list)
        assert len(stats) == 0

    def test_get_question_performance_stats_with_data(self, question_db):
        """Test getting performance stats for workspace with questions."""
        # Create workspace and file
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Performance Stats Workspace", "type": "study"}
        )
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "performance.txt",
                "path": "/path/performance.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create question
        question_id = question_db.create_question(
            {
                "file_id": file_id,
                "question_type": "multiple_choice",
                "question_text": "Performance question?",
                "correct_answer": "A",
                "difficulty": "medium",
                "times_asked": 5,
                "times_correct": 3,
            }
        )

        # Create some answers
        db_service.insert(
            "answers",
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 20,
            },
        )
        db_service.insert(
            "answers",
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 30,
            },
        )

        # Get performance stats
        stats = question_db.get_question_performance_stats(workspace_id)
        assert len(stats) == 1

        stat = stats[0]
        assert stat["id"] == question_id
        assert stat["difficulty"] == "medium"
        assert stat["times_asked"] == 5
        assert stat["times_correct"] == 3
        assert abs(stat["avg_time"] - 25) < 0.01  # (20 + 30) / 2

    def test_count_questions_by_workspace_empty(self, question_db):
        """Test counting questions in workspace with no questions."""
        # Create workspace
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Count Workspace", "type": "study"}
        )

        count = question_db.count_questions_by_workspace(workspace_id)
        assert count == 0

    def test_count_questions_by_workspace_with_data(self, question_db):
        """Test counting questions in workspace with questions."""
        # Create workspace
        db_service = question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Count Questions Workspace", "type": "study"}
        )

        # Create two files
        file_ids = []
        for i in range(2):
            file_id = db_service.insert(
                "files",
                {
                    "workspace_id": workspace_id,
                    "name": f"count{i}.txt",
                    "path": f"/path/count{i}.txt",
                    "file_type": "text",
                    "size": 100,
                },
            )
            file_ids.append(file_id)

        # Create questions for each file
        for file_id in file_ids:
            for j in range(2):  # 2 questions per file
                question_db.create_question(
                    {
                        "file_id": file_id,
                        "question_type": "multiple_choice",
                        "question_text": f"Question {j} for file {file_id}?",
                        "correct_answer": "A",
                    }
                )

        # Count questions
        count = question_db.count_questions_by_workspace(workspace_id)
        assert count == 4  # 2 files * 2 questions each
