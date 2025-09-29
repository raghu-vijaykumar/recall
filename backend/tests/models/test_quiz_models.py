import pytest
from datetime import datetime
from backend.app.models.quiz import (
    QuestionType,
    DifficultyLevel,
    QuestionBase,
    QuestionCreate,
    Question,
    Answer,
    SpacedRepetitionData,
    QuizSessionBase,
    QuizSessionCreate,
    QuizSession,
    QuizResult,
    QuizStats,
)


class TestQuestionType:
    """Test QuestionType enum."""

    def test_question_types(self):
        """Test all question type values."""
        assert QuestionType.MULTIPLE_CHOICE == "multiple_choice"
        assert QuestionType.TRUE_FALSE == "true_false"
        assert QuestionType.SHORT_ANSWER == "short_answer"
        assert QuestionType.FILL_BLANK == "fill_blank"


class TestDifficultyLevel:
    """Test DifficultyLevel enum."""

    def test_difficulty_levels(self):
        """Test all difficulty level values."""
        assert DifficultyLevel.EASY == "easy"
        assert DifficultyLevel.MEDIUM == "medium"
        assert DifficultyLevel.HARD == "hard"


class TestQuestionBase:
    """Test QuestionBase model."""

    def test_question_base_creation(self):
        """Test creating a QuestionBase instance."""
        question = QuestionBase(
            file_id=1,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question_text="What is 2+2?",
            correct_answer="4",
            options=["3", "4", "5"],
            explanation="Basic arithmetic",
            difficulty=DifficultyLevel.EASY,
            tags=["math", "basic"],
        )
        assert question.file_id == 1
        assert question.question_type == QuestionType.MULTIPLE_CHOICE
        assert question.question_text == "What is 2+2?"
        assert question.correct_answer == "4"
        assert question.options == ["3", "4", "5"]
        assert question.explanation == "Basic arithmetic"
        assert question.difficulty == DifficultyLevel.EASY
        assert question.tags == ["math", "basic"]


class TestQuestion:
    """Test Question model."""

    def test_question_creation(self):
        """Test creating a Question instance."""
        created_at = datetime.now()
        question = Question(
            id=1,
            file_id=1,
            question_type=QuestionType.MULTIPLE_CHOICE,
            question_text="What is 2+2?",
            correct_answer="4",
            created_at=created_at,
            times_asked=5,
            times_correct=4,
        )
        assert question.id == 1
        assert question.times_asked == 5
        assert question.times_correct == 4


class TestAnswer:
    """Test Answer model."""

    def test_answer_creation(self):
        """Test creating an Answer instance."""
        answer = Answer(
            question_id=1,
            answer_text="4",
            is_correct=True,
            time_taken=30,
            confidence_level=4,
        )
        assert answer.question_id == 1
        assert answer.is_correct is True
        assert answer.time_taken == 30
        assert answer.confidence_level == 4


class TestSpacedRepetitionData:
    """Test SpacedRepetitionData model."""

    def test_spaced_repetition_creation(self):
        """Test creating a SpacedRepetitionData instance."""
        next_review = datetime.now()
        data = SpacedRepetitionData(
            question_id=1,
            ease_factor=2.5,
            interval_days=1,
            review_count=3,
            next_review=next_review,
        )
        assert data.question_id == 1
        assert data.ease_factor == 2.5
        assert data.interval_days == 1
        assert data.review_count == 3
        assert data.next_review == next_review


class TestQuizSession:
    """Test QuizSession model."""

    def test_quiz_session_creation(self):
        """Test creating a QuizSession instance."""
        created_at = datetime.now()
        session = QuizSession(
            id=1,
            workspace_id=1,
            question_count=10,
            created_at=created_at,
            status="active",
            total_questions=10,
            correct_answers=7,
        )
        assert session.id == 1
        assert session.workspace_id == 1
        assert session.question_count == 10
        assert session.status == "active"
        assert session.correct_answers == 7


class TestQuizResult:
    """Test QuizResult model."""

    def test_quiz_result_creation(self):
        """Test creating a QuizResult instance."""
        result = QuizResult(
            session_id=1,
            total_questions=10,
            correct_answers=7,
            incorrect_answers=3,
            score_percentage=70.0,
            total_time=300,
            average_time_per_question=30.0,
            difficulty_breakdown={"easy": 5, "medium": 3, "hard": 2},
            question_results=[],
        )
        assert result.session_id == 1
        assert result.total_questions == 10
        assert result.correct_answers == 7
        assert result.score_percentage == 70.0


class TestQuizStats:
    """Test QuizStats model."""

    def test_quiz_stats_creation(self):
        """Test creating a QuizStats instance."""
        stats = QuizStats(
            total_sessions=5,
            total_questions_answered=50,
            total_correct=35,
            average_score=70.0,
            study_streak=3,
            favorite_difficulty=DifficultyLevel.MEDIUM,
            most_difficult_topic="Advanced Calculus",
            improvement_trend="improving",
        )
        assert stats.total_sessions == 5
        assert stats.average_score == 70.0
        assert stats.study_streak == 3
        assert stats.favorite_difficulty == DifficultyLevel.MEDIUM
