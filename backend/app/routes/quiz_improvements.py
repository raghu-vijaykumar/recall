"""
Quiz Improvement API routes for LLM generation and spaced repetition
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..database import get_db
from ..services import QuizService
from ..models import Question, Answer

router = APIRouter(prefix="/api/quiz", tags=["quiz-improvements"])


# Pydantic models for API requests/responses
class GenerateQuestionsRequest(BaseModel):
    file_content: str
    file_id: int
    count: int = 5
    question_types: Optional[List[str]] = None
    difficulty: Optional[str] = None
    concept_ids: Optional[List[str]] = None


class SpacedRepetitionUpdateRequest(BaseModel):
    question_id: int
    answer_quality: int  # 0-5 quality rating
    concept_id: Optional[str] = None


class AdaptiveQuizRequest(BaseModel):
    workspace_id: int
    count: int = 10
    user_performance: Optional[Dict[str, Any]] = None


class QuestionResponse(BaseModel):
    id: int
    file_id: int
    question_type: str
    question_text: str
    correct_answer: str
    options: Optional[List[str]]
    explanation: Optional[str]
    difficulty: str
    generated_by_llm: bool
    confidence_score: Optional[float]
    kg_concept_ids: Optional[List[str]]
    created_at: str
    times_asked: int
    times_correct: int
    last_asked: Optional[str]

    class Config:
        from_attributes = True


class SpacedRepetitionResponse(BaseModel):
    question_id: int
    ease_factor: float
    interval_days: int
    review_count: int
    next_review: str
    kg_concept_id: Optional[str]


class ReviewQuestionResponse(BaseModel):
    question: QuestionResponse
    spaced_repetition_data: SpacedRepetitionResponse


@router.post("/generate-llm", response_model=List[QuestionResponse])
async def generate_llm_questions(
    request: GenerateQuestionsRequest, db: AsyncSession = Depends(get_db)
):
    """
    Generate questions using LLM based on file content
    """
    quiz_service = QuizService(db)

    try:
        questions = await quiz_service.generate_llm_questions(
            file_content=request.file_content,
            file_id=request.file_id,
            count=request.count,
            question_types=request.question_types,
            difficulty=request.difficulty,
            concept_ids=request.concept_ids,
        )

        return [
            QuestionResponse(
                id=q.id,
                file_id=q.file_id,
                question_type=q.question_type,
                question_text=q.question_text,
                correct_answer=q.correct_answer,
                options=q.options,
                explanation=q.explanation,
                difficulty=q.difficulty,
                generated_by_llm=q.generated_by_llm,
                confidence_score=q.confidence_score,
                kg_concept_ids=q.kg_concept_ids,
                created_at=q.created_at.isoformat(),
                times_asked=q.times_asked,
                times_correct=q.times_correct,
                last_asked=q.last_asked.isoformat() if q.last_asked else None,
            )
            for q in questions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")


@router.post("/spaced-repetition/update", response_model=SpacedRepetitionResponse)
async def update_spaced_repetition(
    request: SpacedRepetitionUpdateRequest, db: AsyncSession = Depends(get_db)
):
    """
    Update spaced repetition data based on answer quality
    """
    if not (0 <= request.answer_quality <= 5):
        raise HTTPException(
            status_code=400, detail="Answer quality must be between 0 and 5"
        )

    quiz_service = QuizService(db)

    try:
        updated_data = await quiz_service.update_spaced_repetition(
            question_id=request.question_id,
            answer_quality=request.answer_quality,
            concept_id=request.concept_id,
        )

        return SpacedRepetitionResponse(
            question_id=updated_data.question_id,
            ease_factor=updated_data.ease_factor,
            interval_days=updated_data.interval_days,
            review_count=updated_data.review_count,
            next_review=updated_data.next_review.isoformat(),
            kg_concept_id=updated_data.kg_concept_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Spaced repetition update failed: {str(e)}"
        )


@router.get("/spaced-repetition/due", response_model=List[ReviewQuestionResponse])
async def get_questions_due_for_review(
    limit: int = 20,
    concept_ids: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Get questions due for spaced repetition review
    """
    quiz_service = QuizService(db)

    try:
        review_questions = await quiz_service.get_questions_due_for_review(
            limit=limit, concept_ids=concept_ids
        )

        return [
            ReviewQuestionResponse(
                question=QuestionResponse(
                    id=rq["question"].id,
                    file_id=rq["question"].file_id,
                    question_type=rq["question"].question_type,
                    question_text=rq["question"].question_text,
                    correct_answer=rq["question"].correct_answer,
                    options=rq["question"].options,
                    explanation=rq["question"].explanation,
                    difficulty=rq["question"].difficulty,
                    generated_by_llm=rq["question"].generated_by_llm,
                    confidence_score=rq["question"].confidence_score,
                    kg_concept_ids=rq["question"].kg_concept_ids,
                    created_at=rq["question"].created_at.isoformat(),
                    times_asked=rq["question"].times_asked,
                    times_correct=rq["question"].times_correct,
                    last_asked=(
                        rq["question"].last_asked.isoformat()
                        if rq["question"].last_asked
                        else None
                    ),
                ),
                spaced_repetition_data=SpacedRepetitionResponse(
                    question_id=rq["spaced_repetition_data"].question_id,
                    ease_factor=rq["spaced_repetition_data"].ease_factor,
                    interval_days=rq["spaced_repetition_data"].interval_days,
                    review_count=rq["spaced_repetition_data"].review_count,
                    next_review=rq["spaced_repetition_data"].next_review.isoformat(),
                    kg_concept_id=rq["spaced_repetition_data"].kg_concept_id,
                ),
            )
            for rq in review_questions
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get review questions: {str(e)}"
        )


@router.post("/generate-adaptive", response_model=List[QuestionResponse])
async def generate_adaptive_quiz(
    request: AdaptiveQuizRequest, db: AsyncSession = Depends(get_db)
):
    """
    Generate an adaptive question set based on user performance and knowledge graph
    """
    quiz_service = QuizService(db)

    try:
        questions = await quiz_service.get_adaptive_question_set(
            workspace_id=request.workspace_id,
            count=request.count,
            user_performance=request.user_performance,
        )

        return [
            QuestionResponse(
                id=q.id,
                file_id=q.file_id,
                question_type=q.question_type,
                question_text=q.question_text,
                correct_answer=q.correct_answer,
                options=q.options,
                explanation=q.explanation,
                difficulty=q.difficulty,
                generated_by_llm=q.generated_by_llm,
                confidence_score=q.confidence_score,
                kg_concept_ids=q.kg_concept_ids,
                created_at=q.created_at.isoformat(),
                times_asked=q.times_asked,
                times_correct=q.times_correct,
                last_asked=q.last_asked.isoformat() if q.last_asked else None,
            )
            for q in questions
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Adaptive quiz generation failed: {str(e)}"
        )


@router.get("/weak-areas")
async def get_weak_performance_areas(
    workspace_id: Optional[int] = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze performance patterns to identify weak areas
    """
    # This would implement weak area analysis
    # For now, return placeholder data

    # Simple implementation: get questions with low success rates
    query = """
        SELECT
            q.id,
            q.question_text,
            q.difficulty,
            q.kg_concept_ids,
            COUNT(a.id) as times_asked,
            SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as times_correct,
            (COUNT(a.id) - SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END)) * 1.0 / COUNT(a.id) as error_rate
        FROM questions q
        LEFT JOIN answers a ON q.id = a.question_id
        WHERE q.times_asked > 0
    """

    if workspace_id:
        query += " AND q.file_id IN (SELECT id FROM files WHERE workspace_id = :workspace_id)"

    query += """
        GROUP BY q.id, q.question_text, q.difficulty, q.kg_concept_ids
        HAVING COUNT(a.id) > 2  -- Only consider questions asked multiple times
        ORDER BY error_rate DESC
        LIMIT :limit
    """

    from sqlalchemy import text

    result = await db.execute(
        text(query), {"workspace_id": workspace_id, "limit": limit}
    )
    rows = result.fetchall()

    weak_areas = []
    for row in rows:
        import json

        kg_concept_ids = json.loads(row.kg_concept_ids) if row.kg_concept_ids else None

        weak_areas.append(
            {
                "question_id": row.id,
                "question_text": row.question_text,
                "difficulty": row.difficulty,
                "kg_concept_ids": kg_concept_ids,
                "times_asked": row.times_asked,
                "times_correct": row.times_correct,
                "error_rate": row.error_rate,
            }
        )

    return {"weak_areas": weak_areas}


@router.post("/voice-answer")
async def submit_voice_answer(
    question_id: int,
    audio_data: str,  # Base64 encoded audio
    db: AsyncSession = Depends(get_db),
):
    """
    Submit voice-based answer for processing
    """
    # This would integrate with speech-to-text service
    # For now, return placeholder response

    return {
        "question_id": question_id,
        "transcribed_answer": "Placeholder transcription",
        "confidence": 0.85,
        "is_correct": False,  # Would be determined by comparison
        "feedback": "Voice answer processing not yet implemented",
    }


@router.get("/stats/spaced-repetition")
async def get_spaced_repetition_stats(
    workspace_id: Optional[int] = None, db: AsyncSession = Depends(get_db)
):
    """
    Get spaced repetition statistics
    """
    # Get overall stats
    base_query = """
        SELECT
            COUNT(*) as total_questions,
            COUNT(CASE WHEN next_review <= datetime('now') THEN 1 END) as due_today,
            AVG(ease_factor) as avg_ease_factor,
            AVG(interval_days) as avg_interval,
            SUM(review_count) as total_reviews
        FROM spaced_repetition_data srd
        JOIN questions q ON srd.question_id = q.id
    """

    if workspace_id:
        base_query += " WHERE q.file_id IN (SELECT id FROM files WHERE workspace_id = :workspace_id)"

    from sqlalchemy import text

    result = await db.execute(text(base_query), {"workspace_id": workspace_id})
    stats = result.fetchone()

    return {
        "total_questions_tracked": stats.total_questions or 0,
        "questions_due_today": stats.due_today or 0,
        "average_ease_factor": stats.avg_ease_factor or 2.5,
        "average_interval_days": stats.avg_interval or 1,
        "total_reviews_completed": stats.total_reviews or 0,
    }
