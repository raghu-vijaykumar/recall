import pytest
from backend.app.database.answers import AnswerDatabase
from backend.app.services.database import DatabaseService


class TestAnswerDatabase:
    """Test cases for AnswerDatabase class."""

    @pytest.fixture
    def answer_db(self):
        """Create AnswerDatabase instance."""
        db_service = DatabaseService()
        return AnswerDatabase(db_service)

    def test_create_and_get_answer(self, answer_db):
        """Test creating and retrieving an answer."""
        # First create a question (required for foreign key)
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,  # Dummy file_id
                "question_type": "multiple_choice",
                "question_text": "Test question?",
                "correct_answer": "A",
            },
        )

        # Create test answer
        answer_data = {
            "question_id": question_id,
            "answer_text": "A",
            "is_correct": True,
            "time_taken": 30,
            "confidence_level": 4,
        }

        # Create answer
        answer_id = answer_db.create_answer(answer_data)
        assert answer_id > 0

        # Get the answer back
        answer = answer_db.get_answer(answer_id)
        assert answer is not None
        assert answer["id"] == answer_id
        assert answer["question_id"] == question_id
        assert answer["answer_text"] == "A"
        assert answer["is_correct"] == 1  # SQLite stores booleans as integers
        assert answer["time_taken"] == 30
        assert answer["confidence_level"] == 4

    def test_get_answer_not_found(self, answer_db):
        """Test getting a non-existent answer."""
        answer = answer_db.get_answer(99999)
        assert answer is None

    def test_get_answers_by_question_empty(self, answer_db):
        """Test getting answers for a question with no answers."""
        # Create a question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Empty question?",
                "correct_answer": "A",
            },
        )

        # Get answers
        answers = answer_db.get_answers_by_question(question_id)
        assert isinstance(answers, list)
        assert len(answers) == 0

    def test_get_answers_by_question_with_data(self, answer_db):
        """Test getting answers for a question with answers."""
        # Create a question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Question with answers?",
                "correct_answer": "A",
            },
        )

        # Create multiple answers
        answer_data = [
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 20,
            },
            {
                "question_id": question_id,
                "answer_text": "B",
                "is_correct": False,
                "time_taken": 35,
            },
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 15,
            },
        ]

        for data in answer_data:
            answer_db.create_answer(data)

        # Get answers
        answers = answer_db.get_answers_by_question(question_id)
        assert len(answers) == 3

        # Should be ordered by created_at DESC
        assert answers[0]["time_taken"] == 15  # Most recent first
        assert answers[1]["time_taken"] == 35
        assert answers[2]["time_taken"] == 20

    def test_get_answers_by_session_empty(self, answer_db):
        """Test getting answers for a session with no answers."""
        # Create a session
        db_service = answer_db.db
        session_id = db_service.insert(
            "quiz_sessions", {"workspace_id": 1, "status": "in_progress"}
        )

        # Get answers
        answers = answer_db.get_answers_by_session(session_id)
        assert isinstance(answers, list)
        assert len(answers) == 0

    def test_get_answers_by_session_with_data(self, answer_db):
        """Test getting answers for a session with answers."""
        db_service = answer_db.db

        # Create session
        session_id = db_service.insert(
            "quiz_sessions", {"workspace_id": 1, "status": "completed"}
        )

        # Create questions
        question_ids = []
        for i in range(2):
            q_id = db_service.insert(
                "questions",
                {
                    "file_id": 1,
                    "question_type": "multiple_choice",
                    "question_text": f"Question {i}?",
                    "correct_answer": "A",
                },
            )
            question_ids.append(q_id)

        # Link questions to session
        for i, q_id in enumerate(question_ids):
            db_service.insert(
                "session_questions",
                {"session_id": session_id, "question_id": q_id, "question_order": i},
            )

        # Create answers for the questions
        for q_id in question_ids:
            answer_db.create_answer(
                {
                    "question_id": q_id,
                    "answer_text": "A",
                    "is_correct": True,
                    "time_taken": 25,
                }
            )

        # Get answers by session
        answers = answer_db.get_answers_by_session(session_id)
        assert len(answers) == 2

    def test_update_answer(self, answer_db):
        """Test updating an answer."""
        # Create question and answer
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Update test?",
                "correct_answer": "A",
            },
        )

        initial_data = {
            "question_id": question_id,
            "answer_text": "B",
            "is_correct": False,
            "time_taken": 60,
            "confidence_level": 2,
        }
        answer_id = answer_db.create_answer(initial_data)

        # Update answer
        update_data = {
            "answer_text": "A",
            "is_correct": True,
            "time_taken": 45,
            "confidence_level": 5,
        }
        result = answer_db.update_answer(answer_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        answer = answer_db.get_answer(answer_id)
        assert answer["answer_text"] == "A"
        assert answer["is_correct"] == 1  # SQLite stores booleans as integers
        assert answer["time_taken"] == 45
        assert answer["confidence_level"] == 5

    def test_delete_answer(self, answer_db):
        """Test deleting an answer."""
        # Create question and answer
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Delete test?",
                "correct_answer": "A",
            },
        )

        answer_data = {
            "question_id": question_id,
            "answer_text": "A",
            "is_correct": True,
            "time_taken": 30,
        }
        answer_id = answer_db.create_answer(answer_data)

        # Verify it exists
        answer = answer_db.get_answer(answer_id)
        assert answer is not None

        # Delete answer
        result = answer_db.delete_answer(answer_id)
        assert result == 1  # Should affect 1 row

        # Verify it's gone
        answer = answer_db.get_answer(answer_id)
        assert answer is None

    def test_get_answer_statistics_empty(self, answer_db):
        """Test getting statistics for question with no answers."""
        # Create question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Stats test?",
                "correct_answer": "A",
            },
        )

        # Get statistics
        stats = answer_db.get_answer_statistics(question_id)
        assert stats["total_answers"] == 0
        assert stats["avg_time"] == 0
        assert stats["correct_answers"] == 0

    def test_get_answer_statistics_with_data(self, answer_db):
        """Test getting statistics for question with answers."""
        # Create question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Stats test?",
                "correct_answer": "A",
            },
        )

        # Create answers with different times and correctness
        answer_data = [
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 20,
            },
            {
                "question_id": question_id,
                "answer_text": "B",
                "is_correct": False,
                "time_taken": 40,
            },
            {
                "question_id": question_id,
                "answer_text": "A",
                "is_correct": True,
                "time_taken": 30,
            },
            {
                "question_id": question_id,
                "answer_text": "C",
                "is_correct": False,
                "time_taken": 50,
            },
        ]

        for data in answer_data:
            answer_db.create_answer(data)

        # Get statistics
        stats = answer_db.get_answer_statistics(question_id)
        assert stats["total_answers"] == 4
        assert stats["correct_answers"] == 2
        # Average time: (20 + 40 + 30 + 50) / 4 = 35
        assert abs(stats["avg_time"] - 35) < 0.01

    def test_get_recent_answers_empty(self, answer_db):
        """Test getting recent answers when none exist."""
        answers = answer_db.get_recent_answers()
        assert isinstance(answers, list)
        assert len(answers) == 0

    def test_get_recent_answers_with_data(self, answer_db):
        """Test getting recent answers with data."""
        # Create question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Recent test?",
                "correct_answer": "A",
            },
        )

        # Create multiple answers
        for i in range(5):
            answer_db.create_answer(
                {
                    "question_id": question_id,
                    "answer_text": f"Answer {i}",
                    "is_correct": i % 2 == 0,
                    "time_taken": 20 + i * 5,
                }
            )

        # Get recent answers
        answers = answer_db.get_recent_answers(limit=3)
        assert len(answers) == 3

        # Should be ordered by created_at DESC (most recent first)
        assert answers[0]["answer_text"] == "Answer 4"
        assert answers[1]["answer_text"] == "Answer 3"
        assert answers[2]["answer_text"] == "Answer 2"

    def test_get_recent_answers_limit(self, answer_db):
        """Test getting recent answers with limit."""
        # Create question
        db_service = answer_db.db
        question_id = db_service.insert(
            "questions",
            {
                "file_id": 1,
                "question_type": "multiple_choice",
                "question_text": "Limit test?",
                "correct_answer": "A",
            },
        )

        # Create multiple answers
        for i in range(5):
            answer_db.create_answer(
                {
                    "question_id": question_id,
                    "answer_text": f"Answer {i}",
                    "is_correct": True,
                    "time_taken": 20 + i * 5,
                }
            )

        # Get recent answers with limit
        answers = answer_db.get_recent_answers(limit=3)
        assert len(answers) == 3
