"""
Tests for Quiz Improvements functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Question, QuestionCreate, SpacedRepetitionData
from app.services import QuizService


class TestSpacedRepetitionEngine:
    """Test the SM-2 Spaced Repetition Algorithm"""

    def test_calculate_next_review_successful(self):
        """Test next review calculation for successful recall"""
        from app.services.quiz_service import SpacedRepetitionEngine

        engine = SpacedRepetitionEngine()

        # Test first successful review (quality = 5)
        ease_factor, interval, next_review = engine.calculate_next_review(
            ease_factor=2.5,
            interval_days=1,
            review_count=0,
            quality=5,
            last_review=datetime.utcnow(),
        )

        assert ease_factor == 2.6  # 2.5 + (0.1 - 0.0)
        assert interval == 1  # First review always 1 day

        # Test second successful review
        ease_factor, interval, next_review = engine.calculate_next_review(
            ease_factor=2.6,
            interval_days=1,
            review_count=1,
            quality=4,
            last_review=datetime.utcnow(),
        )

        assert ease_factor == 2.6  # 2.6 + (0.1 - 0.1) = 2.6 + 0.0 = 2.6
        assert interval == 6  # Second review is 6 days

        # Test third successful review
        ease_factor, interval, next_review = engine.calculate_next_review(
            ease_factor=2.68,
            interval_days=6,
            review_count=2,
            quality=3,
            last_review=datetime.utcnow(),
        )

        assert ease_factor == 2.58  # 2.68 + (0.1 - 0.2) = 2.68 - 0.1 = 2.58
        assert interval == round(6 * 2.68)  # interval * ease_factor

    def test_calculate_next_review_failed(self):
        """Test next review calculation for failed recall"""
        from app.services.quiz_service import SpacedRepetitionEngine

        engine = SpacedRepetitionEngine()

        # Test failed review (quality < 3)
        ease_factor, interval, next_review = engine.calculate_next_review(
            ease_factor=2.5,
            interval_days=10,
            review_count=3,
            quality=2,
            last_review=datetime.utcnow(),
        )

        assert ease_factor == 2.3  # max(1.3, 2.5 - 0.2)
        assert interval == 1  # Failed reviews reset to 1 day

    def test_ease_factor_bounds(self):
        """Test ease factor stays within bounds"""
        from app.services.quiz_service import SpacedRepetitionEngine

        engine = SpacedRepetitionEngine()

        # Test minimum bound
        ease_factor, _, _ = engine.calculate_next_review(
            ease_factor=1.4,
            interval_days=1,
            review_count=0,
            quality=2,
            last_review=datetime.utcnow(),
        )

        assert ease_factor == 1.3  # Minimum bound

        # Test maximum bound (theoretically, though hard to reach)
        ease_factor, _, _ = engine.calculate_next_review(
            ease_factor=2.9,
            interval_days=1,
            review_count=0,
            quality=5,
            last_review=datetime.utcnow(),
        )

        assert ease_factor <= 3.0  # Maximum bound


class TestQuizService:
    """Test Quiz Service functionality"""

    @pytest.fixture
    def mock_llm_factory(self):
        """Mock LLM factory for testing"""
        mock_factory = MagicMock()
        mock_client = AsyncMock()
        mock_client.generate_text.return_value = """
        [
          {
            "question": "What is machine learning?",
            "type": "short_answer",
            "answer": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed",
            "explanation": "Machine learning uses algorithms to identify patterns in data",
            "confidence": 0.95
          }
        ]
        """
        mock_factory.get_client.return_value = mock_client
        return mock_factory

    @pytest.mark.asyncio
    async def test_generate_llm_questions(self, db_session, mock_llm_factory):
        """Test LLM question generation"""
        async with db_session() as session:
            quiz_service = QuizService(session, mock_llm_factory)

            questions = await quiz_service.generate_llm_questions(
                file_content="Machine learning is a method of data analysis that automates analytical model building.",
                file_id=1,
                count=1,
                question_types=["short_answer"],
                difficulty="medium",
            )

            assert len(questions) == 1
            question = questions[0]
            assert question.question_text == "What is machine learning?"
            assert question.question_type == "short_answer"
            assert question.generated_by_llm is True
            assert question.confidence_score == 0.95

    @pytest.mark.asyncio
    async def test_update_spaced_repetition_new(self, db_session):
        """Test updating spaced repetition for new question"""
        async with db_session() as session:
            quiz_service = QuizService(session)

            # Create a question first
            question_data = QuestionCreate(
                file_id=1,
                question_type="multiple_choice",
                question_text="What is Python?",
                correct_answer="A programming language",
                options=["A programming language", "A snake", "A database"],
            )

            # Save question to get ID
            saved_question = await quiz_service._save_question(question_data)

            # Update spaced repetition with good performance
            result = await quiz_service.update_spaced_repetition(
                question_id=saved_question.id, answer_quality=5
            )

            assert result.question_id == saved_question.id
            assert result.ease_factor == 2.6  # Initial 2.5 + 0.1
            assert result.interval_days == 1  # First review
            assert result.review_count == 1

    @pytest.mark.asyncio
    async def test_update_spaced_repetition_existing(self, db_session):
        """Test updating spaced repetition for existing data"""
        async with db_session() as session:
            quiz_service = QuizService(session)

            # Create a question
            question_data = QuestionCreate(
                file_id=1,
                question_type="true_false",
                question_text="Python is a programming language",
                correct_answer="True",
            )
            saved_question = await quiz_service._save_question(question_data)

            # First update
            await quiz_service.update_spaced_repetition(
                question_id=saved_question.id, answer_quality=5
            )

            # Second update
            result = await quiz_service.update_spaced_repetition(
                question_id=saved_question.id, answer_quality=4
            )

            assert result.question_id == saved_question.id
            assert result.review_count == 2
            assert result.interval_days == 6  # Second review interval

    @pytest.mark.asyncio
    async def test_get_questions_due_for_review(self, db_session):
        """Test getting questions due for review"""
        async with db_session() as session:
            quiz_service = QuizService(session)

            # Create questions and spaced repetition data
            question_data = QuestionCreate(
                file_id=1,
                question_type="multiple_choice",
                question_text="What is AI?",
                correct_answer="Artificial Intelligence",
            )
            saved_question = await quiz_service._save_question(question_data)

            # Create spaced repetition data that's due
            past_date = datetime.utcnow() - timedelta(days=1)
            await quiz_service.update_spaced_repetition(
                question_id=saved_question.id, answer_quality=5
            )

            # Manually update the next_review to be in the past
            from sqlalchemy import text

            await session.execute(
                text(
                    "UPDATE spaced_repetition_data SET next_review = :past_date WHERE question_id = :question_id"
                ),
                {"past_date": past_date, "question_id": saved_question.id},
            )
            await session.commit()

            # Get due questions
            due_questions = await quiz_service.get_questions_due_for_review(limit=10)

            assert len(due_questions) >= 1
            found_question = None
            for dq in due_questions:
                if dq["question"].id == saved_question.id:
                    found_question = dq
                    break

            assert found_question is not None
            assert found_question["question"].question_text == "What is AI?"

    @pytest.mark.asyncio
    async def test_get_adaptive_question_set(self, db_session):
        """Test getting adaptive question set"""
        async with db_session() as session:
            quiz_service = QuizService(session)

            # Create a file in workspace 1
            file_query = text(
                """
                INSERT INTO files (workspace_id, name, path, file_type, size, created_at, updated_at)
                VALUES (1, 'test.txt', 'test.txt', 'text', 100, :created_at, :updated_at)
            """
            )
            await session.execute(
                file_query,
                {"created_at": datetime.utcnow(), "updated_at": datetime.utcnow()},
            )
            await session.commit()

            # Get the file ID
            file_result = await session.execute(text("SELECT last_insert_rowid()"))
            file_id = file_result.scalar()

            # Create some questions for this file
            questions_data = [
                QuestionCreate(
                    file_id=file_id,
                    question_type="multiple_choice",
                    question_text=f"Question {i}",
                    correct_answer=f"Answer {i}",
                )
                for i in range(5)
            ]

            saved_questions = []
            for q_data in questions_data:
                saved_q = await quiz_service._save_question(q_data)
                saved_questions.append(saved_q)

            # Get adaptive set
            adaptive_questions = await quiz_service.get_adaptive_question_set(
                workspace_id=1, count=3
            )

            assert len(adaptive_questions) == 3
            # Should prioritize questions with fewer asks
            question_ids = {q.id for q in adaptive_questions}
            saved_ids = {q.id for q in saved_questions}
            assert question_ids.issubset(saved_ids)

    def test_build_generation_prompt(self):
        """Test LLM prompt building"""
        quiz_service = QuizService(None)

        prompt = quiz_service._build_generation_prompt(
            content="Python is a programming language.",
            count=2,
            question_types=["multiple_choice", "short_answer"],
            difficulty="easy",
        )

        assert "Python is a programming language." in prompt
        assert "2 diverse quiz questions" in prompt
        assert "multiple_choice, short_answer" in prompt
        assert "easy difficulty level" in prompt
        assert "JSON array" in prompt

    def test_parse_llm_response(self):
        """Test LLM response parsing"""
        quiz_service = QuizService(None)

        response = """
        [
          {
            "question": "What is Python?",
            "type": "short_answer",
            "answer": "A programming language",
            "explanation": "Python is a high-level programming language",
            "confidence": 0.9
          }
        ]
        """

        parsed = quiz_service._parse_llm_response(response)

        assert len(parsed) == 1
        assert parsed[0]["question"] == "What is Python?"
        assert parsed[0]["type"] == "short_answer"
        assert parsed[0]["confidence"] == 0.9

    def test_parse_llm_response_malformed(self):
        """Test parsing malformed LLM response"""
        quiz_service = QuizService(None)

        # Test with no JSON
        parsed = quiz_service._parse_llm_response("This is not JSON")
        assert parsed == []

        # Test with invalid JSON
        parsed = quiz_service._parse_llm_response('{"invalid": json}')
        assert parsed == []


class TestQuizAPI:
    """Test Quiz API endpoints"""

    @pytest.mark.asyncio
    async def test_generate_llm_questions_endpoint(self, client, db_session):
        """Test the LLM question generation endpoint"""
        # This would require setting up test client with proper fixtures
        # For now, just test that the endpoint exists and can be called
        pass

    @pytest.mark.asyncio
    async def test_spaced_repetition_update_endpoint(self, client, db_session):
        """Test spaced repetition update endpoint"""
        # Test with valid quality score
        pass

    @pytest.mark.asyncio
    async def test_spaced_repetition_update_invalid_quality(self, client):
        """Test spaced repetition with invalid quality score"""
        # Should return 400 for quality < 0 or > 5
        pass

    @pytest.mark.asyncio
    async def test_get_due_questions_endpoint(self, client, db_session):
        """Test getting questions due for review"""
        pass

    @pytest.mark.asyncio
    async def test_adaptive_quiz_endpoint(self, client, db_session):
        """Test adaptive quiz generation endpoint"""
        pass

    @pytest.mark.asyncio
    async def test_weak_areas_endpoint(self, client, db_session):
        """Test weak areas analysis endpoint"""
        pass
