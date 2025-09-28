import pytest
from backend.app.database.session_questions import SessionQuestionDatabase
from backend.app.services.database import DatabaseService


class TestSessionQuestionDatabase:
    """Test cases for SessionQuestionDatabase class."""

    @pytest.fixture
    def session_question_db(self):
        """Create SessionQuestionDatabase instance."""
        db_service = DatabaseService()
        return SessionQuestionDatabase(db_service)

    def test_add_question_to_session(self, session_question_db):
        """Test adding a question to a quiz session."""
        # Create workspace, file, question, and session first
        db_service = session_question_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Session Question Test Workspace", "type": "study"}
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

        # Create session
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 1, "status": "created"},
        )

        # Add question to session
        session_question_data = {
            "session_id": session_id,
            "question_id": question_id,
            "question_order": 1,
            "user_answer": "A",
            "is_correct": True,
            "time_taken": 30,
            "confidence_level": 4,
        }

        result = session_question_db.add_question_to_session(session_question_data)
        assert result > 0

    def test_get_session_questions_empty(self, session_question_db):
        """Test getting questions for session with no questions."""
        # Create workspace and session
        db_service = session_question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Session Questions Workspace", "type": "study"}
        )
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 0, "status": "created"},
        )

        # Get session questions
        questions = session_question_db.get_session_questions(session_id)
        assert isinstance(questions, list)
        assert len(questions) == 0

    def test_get_session_questions_with_data(self, session_question_db):
        """Test getting questions for session with questions."""
        # Create workspace, file, questions, and session
        db_service = session_question_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Session Questions Workspace", "type": "study"}
        )

        # Create file
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

        # Create questions
        question_ids = []
        for i in range(3):
            q_id = db_service.insert(
                "questions",
                {
                    "file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_text": f"Question {i}?",
                    "correct_answer": "A",
                },
            )
            question_ids.append(q_id)

        # Create session
        session_id = db_service.insert(
            "quiz_sessions",
            {
                "workspace_id": workspace_id,
                "question_count": 3,
                "status": "in_progress",
            },
        )

        # Add questions to session
        for i, q_id in enumerate(question_ids):
            session_question_db.add_question_to_session(
                {
                    "session_id": session_id,
                    "question_id": q_id,
                    "question_order": i + 1,
                    "user_answer": "A",
                    "is_correct": True,
                    "time_taken": 20 + i * 10,
                }
            )

        # Get session questions
        questions = session_question_db.get_session_questions(session_id)
        assert len(questions) == 3

        # Should be ordered by question_order
        assert questions[0]["question_order"] == 1
        assert questions[1]["question_order"] == 2
        assert questions[2]["question_order"] == 3

    def test_update_answer(self, session_question_db):
        """Test updating answer for a session question."""
        # Create workspace, file, question, and session
        db_service = session_question_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Update Answer Workspace", "type": "study"}
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
                "question_text": "Update answer question?",
                "correct_answer": "A",
            },
        )

        # Create session
        session_id = db_service.insert(
            "quiz_sessions",
            {
                "workspace_id": workspace_id,
                "question_count": 1,
                "status": "in_progress",
            },
        )

        # Add question to session
        session_question_db.add_question_to_session(
            {
                "session_id": session_id,
                "question_id": question_id,
                "question_order": 1,
                "user_answer": "B",
                "is_correct": False,
                "time_taken": 60,
            }
        )

        # Update answer
        update_data = {
            "user_answer": "A",
            "is_correct": True,
            "time_taken": 45,
            "confidence_level": 5,
        }
        result = session_question_db.update_answer(session_id, question_id, update_data)
        assert result == 1  # Should affect 1 row

        # Verify update
        questions = session_question_db.get_session_questions(session_id)
        assert len(questions) == 1
        question = questions[0]
        assert question["user_answer"] == "A"
        assert question["is_correct"] == 1  # SQLite stores booleans as integers
        assert question["time_taken"] == 45
        assert question["confidence_level"] == 5

    def test_get_session_results_empty(self, session_question_db):
        """Test getting results for session with no questions."""
        # Create workspace and session
        db_service = session_question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Results Workspace", "type": "study"}
        )
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 0, "status": "created"},
        )

        # Get session results
        results = session_question_db.get_session_results(session_id)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_get_session_results_with_data(self, session_question_db):
        """Test getting results for session with questions."""
        # Create workspace, file, questions, and session
        db_service = session_question_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Session Results Workspace", "type": "study"}
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "results.txt",
                "path": "/path/results.txt",
                "file_type": "text",
                "size": 100,
            },
        )

        # Create questions
        question_ids = []
        for i in range(2):
            q_id = db_service.insert(
                "questions",
                {
                    "file_id": file_id,
                    "question_type": "multiple_choice",
                    "question_text": f"What is {i+1}?",
                    "correct_answer": str(i + 1),
                    "options": f'["{i}", "{i+1}", "{i+2}"]',
                    "explanation": f"The answer is {i+1}",
                },
            )
            question_ids.append(q_id)

        # Create session
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 2, "status": "completed"},
        )

        # Add questions to session
        for i, q_id in enumerate(question_ids):
            session_question_db.add_question_to_session(
                {
                    "session_id": session_id,
                    "question_id": q_id,
                    "question_order": i + 1,
                    "user_answer": str(i + 1),
                    "is_correct": True,
                    "time_taken": 25,
                }
            )

        # Get session results
        results = session_question_db.get_session_results(session_id)
        assert len(results) == 2

        # Check that question details are included
        for result in results:
            assert "question_text" in result
            assert "correct_answer" in result
            assert "options" in result
            assert "explanation" in result
            assert "user_answer" in result
            assert "is_correct" in result

    def test_count_correct_answers_empty(self, session_question_db):
        """Test counting correct answers for session with no questions."""
        # Create workspace and session
        db_service = session_question_db.db
        workspace_id = db_service.insert(
            "workspaces", {"name": "Empty Correct Count Workspace", "type": "study"}
        )
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 0, "status": "created"},
        )

        # Count correct answers
        count = session_question_db.count_correct_answers(session_id)
        assert count == 0

    def test_count_correct_answers_with_data(self, session_question_db):
        """Test counting correct answers for session with questions."""
        # Create workspace, file, questions, and session
        db_service = session_question_db.db

        # Create workspace
        workspace_id = db_service.insert(
            "workspaces", {"name": "Correct Count Workspace", "type": "study"}
        )

        # Create file
        file_id = db_service.insert(
            "files",
            {
                "workspace_id": workspace_id,
                "name": "count.txt",
                "path": "/path/count.txt",
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
                    "question_text": f"Question {i}?",
                    "correct_answer": "A",
                },
            )
            question_ids.append(q_id)

        # Create session
        session_id = db_service.insert(
            "quiz_sessions",
            {"workspace_id": workspace_id, "question_count": 4, "status": "completed"},
        )

        # Add questions to session with mixed correctness
        correct_answers = [True, False, True, False]
        for i, (q_id, is_correct) in enumerate(zip(question_ids, correct_answers)):
            session_question_db.add_question_to_session(
                {
                    "session_id": session_id,
                    "question_id": q_id,
                    "question_order": i + 1,
                    "user_answer": "A" if is_correct else "B",
                    "is_correct": is_correct,
                    "time_taken": 20,
                }
            )

        # Count correct answers
        count = session_question_db.count_correct_answers(session_id)
        assert count == 2  # Two correct answers
