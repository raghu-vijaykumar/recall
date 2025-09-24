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
    Analyze performance patterns to identify weak areas with advanced analytics
    """
    from sqlalchemy import text
    import json
    from datetime import datetime, timedelta

    # Get comprehensive performance data
    base_query = """
        SELECT
            q.id,
            q.question_text,
            q.question_type,
            q.difficulty,
            q.kg_concept_ids,
            q.created_at,
            COUNT(a.id) as times_asked,
            SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as times_correct,
            AVG(a.time_taken) as avg_time,
            MIN(a.created_at) as first_asked,
            MAX(a.created_at) as last_asked,
            SUM(CASE WHEN a.is_correct THEN 0 ELSE 1 END) as times_incorrect
        FROM questions q
        LEFT JOIN answers a ON q.id = a.question_id
        WHERE q.times_asked > 0
    """

    if workspace_id:
        base_query += " AND q.file_id IN (SELECT id FROM files WHERE workspace_id = :workspace_id)"

    base_query += """
        GROUP BY q.id, q.question_text, q.question_type, q.difficulty, q.kg_concept_ids, q.created_at
        HAVING COUNT(a.id) >= 3  -- Need sufficient data points
    """

    result = await db.execute(text(base_query), {"workspace_id": workspace_id})
    question_performance = result.fetchall()

    # Analyze patterns and calculate advanced metrics
    weak_areas = []
    now = datetime.utcnow()

    for row in question_performance:
        # Basic metrics
        accuracy = row.times_correct / row.times_asked if row.times_asked > 0 else 0
        error_rate = 1 - accuracy

        # Advanced metrics
        consistency_score = 1 - (
            row.times_incorrect / row.times_asked
        )  # Lower is more consistent failures

        # Recency-weighted performance (recent mistakes matter more)
        days_since_last_asked = (now - row.last_asked).days if row.last_asked else 30
        recency_weight = min(
            days_since_last_asked / 7, 2
        )  # Max 2x weight for very recent

        # Learning velocity (improvement over time)
        learning_velocity = 0
        if (
            row.first_asked
            and row.last_asked
            and (row.last_asked - row.first_asked).days > 7
        ):
            # Simplified: assume some learning if accuracy > 0.5 after multiple attempts
            learning_velocity = accuracy - 0.5 if accuracy > 0.5 else accuracy - 0.7

        # Difficulty-adjusted score (hard questions get more weight)
        difficulty_multiplier = {"easy": 0.8, "medium": 1.0, "hard": 1.3}.get(
            row.difficulty, 1.0
        )

        # Overall weakness score
        weakness_score = (
            (error_rate * 0.4)  # High error rate
            + ((1 - consistency_score) * 0.3)  # Inconsistent performance
            + (recency_weight * 0.2)  # Recent struggles
            + (difficulty_multiplier * 0.1)  # Difficulty consideration
        )

        # Parse concept IDs
        kg_concept_ids = json.loads(row.kg_concept_ids) if row.kg_concept_ids else []

        # Time-based analysis
        time_trend = "stable"
        if row.avg_time and row.times_asked > 5:
            # Analyze if user is taking longer (struggling) or shorter (mastering)
            recent_answers_query = text(
                """
                SELECT time_taken, is_correct, created_at
                FROM answers
                WHERE question_id = :question_id
                ORDER BY created_at DESC
                LIMIT 5
            """
            )

            recent_result = await db.execute(
                recent_answers_query, {"question_id": row.id}
            )
            recent_answers = recent_result.fetchall()

            if len(recent_answers) >= 3:
                recent_avg_time = sum(r.time_taken for r in recent_answers) / len(
                    recent_answers
                )
                if recent_avg_time > row.avg_time * 1.2:
                    time_trend = "struggling"  # Taking longer
                elif recent_avg_time < row.avg_time * 0.8:
                    time_trend = "improving"  # Taking less time

        weak_areas.append(
            {
                "question_id": row.id,
                "question_text": row.question_text,
                "question_type": row.question_type,
                "difficulty": row.difficulty,
                "kg_concept_ids": kg_concept_ids,
                "performance_metrics": {
                    "times_asked": row.times_asked,
                    "times_correct": row.times_correct,
                    "accuracy": accuracy,
                    "error_rate": error_rate,
                    "avg_time_seconds": row.avg_time,
                    "consistency_score": consistency_score,
                    "learning_velocity": learning_velocity,
                    "time_trend": time_trend,
                },
                "weakness_score": weakness_score,
                "recommendations": _generate_weakness_recommendations(
                    accuracy, consistency_score, time_trend, row.difficulty
                ),
                "last_asked_days": days_since_last_asked,
            }
        )

    # Sort by weakness score and return top issues
    weak_areas.sort(key=lambda x: x["weakness_score"], reverse=True)

    # Group by concepts for higher-level insights
    concept_weakness = {}
    for area in weak_areas[: limit * 2]:  # Get more for concept analysis
        for concept_id in area["kg_concept_ids"] or []:
            if concept_id not in concept_weakness:
                concept_weakness[concept_id] = {
                    "concept_id": concept_id,
                    "total_questions": 0,
                    "avg_weakness": 0,
                    "question_count": 0,
                }
            concept_weakness[concept_id]["total_questions"] += 1
            concept_weakness[concept_id]["avg_weakness"] += area["weakness_score"]
            concept_weakness[concept_id]["question_count"] += 1

    # Calculate concept-level insights
    concept_insights = []
    for concept_data in concept_weakness.values():
        if concept_data["question_count"] > 0:
            concept_data["avg_weakness"] /= concept_data["question_count"]
            concept_insights.append(concept_data)

    concept_insights.sort(key=lambda x: x["avg_weakness"], reverse=True)

    return {
        "weak_areas": weak_areas[:limit],
        "concept_insights": concept_insights[:5],  # Top 5 weak concepts
        "summary": {
            "total_questions_analyzed": len(question_performance),
            "weak_questions_found": len(
                [w for w in weak_areas if w["weakness_score"] > 0.5]
            ),
            "average_accuracy": (
                sum(w["performance_metrics"]["accuracy"] for w in weak_areas)
                / len(weak_areas)
                if weak_areas
                else 0
            ),
        },
    }


def _generate_weakness_recommendations(
    accuracy: float, consistency: float, time_trend: str, difficulty: str
) -> List[str]:
    """Generate personalized recommendations based on performance patterns"""
    recommendations = []

    if accuracy < 0.5:
        recommendations.append(
            "Focus on fundamental concepts - consider reviewing basic materials"
        )
    elif accuracy < 0.7:
        recommendations.append("Practice similar questions to build confidence")

    if consistency < 0.6:
        recommendations.append(
            "Work on consistency - review mistakes and understand why they occurred"
        )

    if time_trend == "struggling":
        recommendations.append("Take more time to understand concepts before answering")
    elif time_trend == "improving":
        recommendations.append("Great progress! Continue practicing at this pace")

    if difficulty == "hard" and accuracy < 0.6:
        recommendations.append(
            "Break down complex topics into smaller, manageable parts"
        )
    elif difficulty == "easy" and accuracy < 0.8:
        recommendations.append("Review basic concepts that may have been misunderstood")

    if not recommendations:
        recommendations.append("Continue practicing to maintain performance")

    return recommendations


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
