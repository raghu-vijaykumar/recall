"""
Quiz Service for LLM-powered question generation and spaced repetition
"""

import math
import os
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, UTC
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Question,
    QuestionCreate,
    SpacedRepetitionData,
    Answer,
    QuizSession,
)
from ..llm_clients import llm_client_factory as LLMFactory


class SpacedRepetitionEngine:
    """SM-2 Spaced Repetition Algorithm Implementation"""

    def __init__(self):
        self.default_ease_factor = 2.5
        self.min_ease_factor = 1.3
        self.max_ease_factor = 3.0

    def calculate_next_review(
        self,
        ease_factor: float,
        interval_days: int,
        review_count: int,
        quality: int,  # 0-5 quality rating
        last_review: Optional[datetime] = None,
    ) -> tuple[float, int, datetime]:
        """
        Calculate next review date using SM-2 algorithm

        Args:
            ease_factor: Current ease factor
            interval_days: Current interval in days
            review_count: Number of times reviewed
            quality: Quality of response (0-5)
            last_review: Last review date

        Returns:
            Tuple of (new_ease_factor, new_interval_days, next_review_date)
        """
        if quality < 3:
            # Failed response - reset to 1 day
            new_interval = 1
            new_ease_factor = max(self.min_ease_factor, ease_factor - 0.2)
            new_review_count = 0
        else:
            # Successful response
            if review_count == 0:
                new_interval = 1
            elif review_count == 1:
                new_interval = 6
            else:
                new_interval = round(interval_days * ease_factor)

            # Adjust ease factor based on quality
            quality_modifier = (
                5 - quality
            ) * 0.1  # 0.0 for quality=5, 0.2 for quality=3
            new_ease_factor = max(
                self.min_ease_factor,
                min(self.max_ease_factor, ease_factor + (0.1 - quality_modifier)),
            )
            new_review_count = review_count + 1

        # Calculate next review date
        base_date = last_review or datetime.now(UTC)
        next_review = base_date + timedelta(days=new_interval)

        return new_ease_factor, new_interval, next_review


class QuizService:
    def __init__(self, db: AsyncSession, llm_factory=None):
        self.db = db
        self.llm_factory = llm_factory or LLMFactory
        self.spaced_repetition = SpacedRepetitionEngine()

    async def generate_llm_questions(
        self,
        file_content: str,
        file_id: int,
        count: int = 5,
        question_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        concept_ids: Optional[List[str]] = None,
    ) -> List[Question]:
        """
        Generate questions using LLM based on file content

        Args:
            file_content: The text content of the file
            file_id: ID of the source file
            count: Number of questions to generate
            question_types: Types of questions to generate
            difficulty: Difficulty level
            concept_ids: Knowledge graph concept IDs to link

        Returns:
            List of generated Question objects
        """
        # Get LLM client
        llm_client = self.llm_factory.get_client()

        # Prepare prompt
        prompt = self._build_generation_prompt(
            file_content, count, question_types, difficulty
        )

        # Generate questions
        try:
            response = await llm_client.generate_text(prompt)
            questions_data = self._parse_llm_response(response)

            # Create Question objects
            questions = []
            for q_data in questions_data[:count]:
                question = QuestionCreate(
                    file_id=file_id,
                    question_type=q_data.get("type", "multiple_choice"),
                    question_text=q_data["question"],
                    correct_answer=q_data["answer"],
                    options=q_data.get("options"),
                    explanation=q_data.get("explanation"),
                    difficulty=difficulty or "medium",
                    generated_by_llm=True,
                    generation_prompt=prompt,
                    confidence_score=q_data.get("confidence", 0.8),
                    kg_concept_ids=concept_ids or [],
                )

                # Save to database
                saved_question = await self._save_question(question)
                questions.append(saved_question)

            return questions

        except Exception as e:
            # Fallback to basic question generation if LLM fails
            print(f"LLM generation failed: {e}")
            return []

    def _build_generation_prompt(
        self,
        content: str,
        count: int,
        question_types: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
    ) -> str:
        """Build the prompt for LLM question generation"""
        types_str = (
            ", ".join(question_types)
            if question_types
            else "multiple choice, true/false, short answer"
        )
        difficulty_str = f" at {difficulty} difficulty level" if difficulty else ""

        prompt = f"""
Generate {count} diverse quiz questions from the following content{difficulty_str}.
Focus on key concepts, facts, and relationships in the text.

Question types to include: {types_str}

Requirements:
- Mix different question types appropriately
- Ensure questions test understanding, not just memorization
- Provide clear, concise correct answers
- Include brief explanations where helpful
- For multiple choice questions, provide 4 options with one clearly correct
- Rate your confidence in each question (0.0-1.0)

Content:
{content[:4000]}  # Truncate if too long

Output format: JSON array of objects with this structure:
[
  {{
    "question": "Question text here",
    "type": "multiple_choice|true_false|short_answer|fill_blank",
    "answer": "Correct answer",
    "options": ["A", "B", "C", "D"] // only for multiple_choice
    "explanation": "Brief explanation",
    "confidence": 0.85
  }}
]
"""
        return prompt

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured question data"""
        import json

        try:
            # Try to extract JSON from response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try parsing the whole response
                return json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            print(f"Failed to parse LLM response: {response}")
            return []

    async def _save_question(self, question_data: QuestionCreate) -> Question:
        """Save a question to the database"""
        # Insert question
        query = text(
            """
            INSERT INTO questions (
                file_id, question_type, question_text, correct_answer,
                options, explanation, difficulty, tags,
                generated_by_llm, generation_prompt, confidence_score, kg_concept_ids,
                created_at, times_asked, times_correct
            )
            VALUES (
                :file_id, :question_type, :question_text, :correct_answer,
                :options, :explanation, :difficulty, :tags,
                :generated_by_llm, :generation_prompt, :confidence_score, :kg_concept_ids,
                :created_at, 0, 0
            )
        """
        )

        # Convert options list to JSON string if needed
        options_json = None
        if question_data.options:
            import json

            options_json = json.dumps(question_data.options)

        # Convert concept_ids list to JSON string
        kg_concept_ids_json = None
        if question_data.kg_concept_ids:
            import json

            kg_concept_ids_json = json.dumps(question_data.kg_concept_ids)

        result = await self.db.execute(
            query,
            {
                "file_id": question_data.file_id,
                "question_type": question_data.question_type,
                "question_text": question_data.question_text,
                "correct_answer": question_data.correct_answer,
                "options": options_json,
                "explanation": question_data.explanation,
                "difficulty": question_data.difficulty,
                "tags": None,  # Could be extended
                "generated_by_llm": question_data.generated_by_llm,
                "generation_prompt": question_data.generation_prompt,
                "confidence_score": question_data.confidence_score,
                "kg_concept_ids": kg_concept_ids_json,
                "created_at": datetime.now(UTC),
            },
        )

        await self.db.commit()

        # Get the inserted question ID
        question_id = result.lastrowid

        # Return Question object
        return Question(
            id=question_id,
            file_id=question_data.file_id,
            question_type=question_data.question_type,
            question_text=question_data.question_text,
            correct_answer=question_data.correct_answer,
            options=question_data.options,
            explanation=question_data.explanation,
            difficulty=question_data.difficulty,
            tags=question_data.tags,
            generated_by_llm=question_data.generated_by_llm,
            generation_prompt=question_data.generation_prompt,
            confidence_score=question_data.confidence_score,
            kg_concept_ids=question_data.kg_concept_ids,
            created_at=datetime.now(UTC),
            times_asked=0,
            times_correct=0,
            last_asked=None,
        )

    async def update_spaced_repetition(
        self, question_id: int, answer_quality: int, concept_id: Optional[str] = None
    ) -> SpacedRepetitionData:
        """
        Update spaced repetition data based on answer quality

        Args:
            question_id: ID of the question answered
            answer_quality: Quality rating (0-5)
            concept_id: Optional knowledge graph concept ID

        Returns:
            Updated SpacedRepetitionData object
        """
        # Get existing spaced repetition data
        query = text(
            """
            SELECT * FROM spaced_repetition_data
            WHERE question_id = :question_id
        """
        )
        result = await self.db.execute(query, {"question_id": question_id})
        row = result.fetchone()

        if row:
            # Update existing record
            ease_factor = row.ease_factor
            interval_days = row.interval_days
            review_count = row.review_count
            last_review = row.next_review or datetime.now(UTC)
            # Ensure last_review is a datetime object
            if isinstance(last_review, str):
                last_review = datetime.fromisoformat(last_review.replace("Z", "+00:00"))
        else:
            # Create new record
            ease_factor = self.spaced_repetition.default_ease_factor
            interval_days = 1
            review_count = 0
            last_review = datetime.now(UTC)

        # Calculate new values
        new_ease_factor, new_interval, next_review = (
            self.spaced_repetition.calculate_next_review(
                ease_factor, interval_days, review_count, answer_quality, last_review
            )
        )

        # Update or insert record
        if row:
            update_query = text(
                """
                UPDATE spaced_repetition_data
                SET ease_factor = :ease_factor,
                    interval_days = :interval_days,
                    review_count = :review_count,
                    next_review = :next_review,
                    kg_concept_id = :kg_concept_id
                WHERE question_id = :question_id
            """
            )
            await self.db.execute(
                update_query,
                {
                    "ease_factor": new_ease_factor,
                    "interval_days": new_interval,
                    "review_count": review_count + (1 if answer_quality >= 3 else 0),
                    "next_review": next_review,
                    "kg_concept_id": concept_id,
                    "question_id": question_id,
                },
            )
        else:
            insert_query = text(
                """
                INSERT INTO spaced_repetition_data (
                    question_id, ease_factor, interval_days, review_count,
                    next_review, kg_concept_id
                ) VALUES (
                    :question_id, :ease_factor, :interval_days, :review_count,
                    :next_review, :kg_concept_id
                )
            """
            )
            await self.db.execute(
                insert_query,
                {
                    "question_id": question_id,
                    "ease_factor": new_ease_factor,
                    "interval_days": new_interval,
                    "review_count": 1 if answer_quality >= 3 else 0,
                    "next_review": next_review,
                    "kg_concept_id": concept_id,
                },
            )

        await self.db.commit()

        # Return updated data
        return SpacedRepetitionData(
            id=row.id if row else None,  # This would need to be fetched properly
            question_id=question_id,
            ease_factor=new_ease_factor,
            interval_days=new_interval,
            review_count=review_count + (1 if answer_quality >= 3 else 0),
            next_review=next_review,
            kg_concept_id=concept_id,
        )

    async def get_questions_due_for_review(
        self, limit: int = 20, concept_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get questions due for spaced repetition review

        Args:
            limit: Maximum number of questions to return
            concept_ids: Optional filter by knowledge graph concepts

        Returns:
            List of dicts with question and spaced repetition data
        """
        base_query = """
            SELECT q.*, srd.ease_factor, srd.interval_days, srd.review_count,
                   srd.next_review, srd.kg_concept_id
            FROM questions q
            JOIN spaced_repetition_data srd ON q.id = srd.question_id
            WHERE srd.next_review <= :now
        """

        params = {"now": datetime.now(UTC), "limit": limit}

        if concept_ids:
            # Filter by concepts if provided
            concept_placeholders = ", ".join(
                f":concept_{i}" for i in range(len(concept_ids))
            )
            base_query += f" AND srd.kg_concept_id IN ({concept_placeholders})"
            for i, concept_id in enumerate(concept_ids):
                params[f"concept_{i}"] = concept_id

        base_query += " ORDER BY srd.next_review ASC LIMIT :limit"

        query = text(base_query)
        result = await self.db.execute(query, params)
        rows = result.fetchall()

        questions = []
        for row in rows:
            # Parse options and concept_ids from JSON
            import json

            options = json.loads(row.options) if row.options else None
            kg_concept_ids = (
                json.loads(row.kg_concept_ids) if row.kg_concept_ids else None
            )

            question = Question(
                id=row.id,
                file_id=row.file_id,
                question_type=row.question_type,
                question_text=row.question_text,
                correct_answer=row.correct_answer,
                options=options,
                explanation=row.explanation,
                difficulty=row.difficulty,
                tags=None,  # Could be extended
                generated_by_llm=row.generated_by_llm,
                generation_prompt=row.generation_prompt,
                confidence_score=row.confidence_score,
                kg_concept_ids=kg_concept_ids,
                created_at=row.created_at,
                times_asked=row.times_asked,
                times_correct=row.times_correct,
                last_asked=row.last_asked,
            )

            questions.append(
                {
                    "question": question,
                    "spaced_repetition_data": SpacedRepetitionData(
                        id=None,  # Not fetched
                        question_id=row.id,
                        ease_factor=row.ease_factor,
                        interval_days=row.interval_days,
                        review_count=row.review_count,
                        next_review=row.next_review,
                        kg_concept_id=row.kg_concept_id,
                    ),
                }
            )

        return questions

    async def get_adaptive_question_set(
        self,
        workspace_id: int,
        count: int = 10,
        user_performance: Optional[Dict[str, Any]] = None,
    ) -> List[Question]:
        """
        Generate an adaptive question set based on user performance and knowledge graph

        Args:
            workspace_id: Workspace to generate questions for
            count: Number of questions to return
            user_performance: Optional performance data for adaptation

        Returns:
            List of questions optimized for learning
        """
        # Get existing questions from workspace
        existing_questions = await self._get_workspace_questions(workspace_id)

        if not existing_questions:
            # If no questions exist, generate new ones using content analysis
            return await self._generate_adaptive_from_content(workspace_id, count)

        # Analyze user performance to determine focus areas
        performance_analysis = await self._analyze_user_performance(
            workspace_id, user_performance
        )

        # Select questions based on performance gaps and diversity
        selected_questions = self._select_adaptive_questions(
            existing_questions, performance_analysis, count
        )

        # If we need more questions, generate them
        if len(selected_questions) < count:
            additional_needed = count - len(selected_questions)
            new_questions = await self._generate_adaptive_from_content(
                workspace_id, additional_needed, performance_analysis
            )
            selected_questions.extend(new_questions)

        return selected_questions[:count]

    async def _get_workspace_questions(self, workspace_id: int) -> List[Question]:
        """Get all questions from a workspace"""
        query = text(
            """
            SELECT q.* FROM questions q
            JOIN files f ON q.file_id = f.id
            WHERE f.workspace_id = :workspace_id
        """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
        rows = result.fetchall()

        questions = []
        for row in rows:
            import json

            options = json.loads(row.options) if row.options else None
            kg_concept_ids = (
                json.loads(row.kg_concept_ids) if row.kg_concept_ids else None
            )

            question = Question(
                id=row.id,
                file_id=row.file_id,
                question_type=row.question_type,
                question_text=row.question_text,
                correct_answer=row.correct_answer,
                options=options,
                explanation=row.explanation,
                difficulty=row.difficulty,
                tags=None,
                generated_by_llm=row.generated_by_llm,
                generation_prompt=row.generation_prompt,
                confidence_score=row.confidence_score,
                kg_concept_ids=kg_concept_ids,
                created_at=row.created_at,
                times_asked=row.times_asked,
                times_correct=row.times_correct,
                last_asked=row.last_asked,
            )
            questions.append(question)

        return questions

    async def _analyze_user_performance(
        self, workspace_id: int, user_performance: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze user performance patterns"""
        # Get performance data from database
        query = text(
            """
            SELECT
                q.id,
                q.difficulty,
                q.kg_concept_ids,
                COUNT(a.id) as times_asked,
                SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as times_correct,
                AVG(a.time_taken) as avg_time
            FROM questions q
            JOIN files f ON q.file_id = f.id
            LEFT JOIN answers a ON q.id = a.question_id
            WHERE f.workspace_id = :workspace_id
            GROUP BY q.id, q.difficulty, q.kg_concept_ids
        """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
        rows = result.fetchall()

        performance_data = {
            "difficulty_performance": {"easy": [], "medium": [], "hard": []},
            "concept_performance": {},
            "overall_accuracy": 0,
            "total_questions": len(rows),
        }

        total_correct = 0
        total_asked = 0

        for row in rows:
            if row.times_asked and row.times_asked > 0:
                accuracy = row.times_correct / row.times_asked
                total_correct += row.times_correct
                total_asked += row.times_asked

                # Track difficulty performance
                performance_data["difficulty_performance"][row.difficulty].append(
                    accuracy
                )

                # Track concept performance
                import json

                concept_ids = (
                    json.loads(row.kg_concept_ids) if row.kg_concept_ids else []
                )
                for concept_id in concept_ids:
                    if concept_id not in performance_data["concept_performance"]:
                        performance_data["concept_performance"][concept_id] = []
                    performance_data["concept_performance"][concept_id].append(accuracy)

        performance_data["overall_accuracy"] = (
            total_correct / total_asked if total_asked > 0 else 0
        )

        return performance_data

    def _select_adaptive_questions(
        self,
        questions: List[Question],
        performance_analysis: Dict[str, Any],
        count: int,
    ) -> List[Question]:
        """Select questions adaptively based on performance analysis"""
        if not questions:
            return []

        # Score questions based on learning needs
        scored_questions = []

        for question in questions:
            score = 0

            # Prioritize questions user struggles with (lower accuracy)
            if question.times_asked > 0:
                accuracy = question.times_correct / question.times_asked
                # Lower accuracy = higher priority (inverse relationship)
                score += (1 - accuracy) * 2
            else:
                # Never asked questions get medium priority
                score += 1

            # Factor in difficulty balance
            difficulty_scores = {
                "easy": 0.5,  # Less priority for easy questions
                "medium": 1.0,  # Normal priority
                "hard": 1.5,  # Higher priority for challenging questions
            }
            score *= difficulty_scores.get(question.difficulty, 1.0)

            # Recency factor - prefer questions not asked recently
            if question.last_asked:
                days_since_asked = (datetime.now(UTC) - question.last_asked).days
                recency_boost = min(
                    days_since_asked / 7, 2
                )  # Max 2x boost after a week
                score *= 1 + recency_boost
            else:
                score *= 1.5  # Boost for never asked

            scored_questions.append((question, score))

        # Sort by score (highest first) and return top questions
        scored_questions.sort(key=lambda x: x[1], reverse=True)
        return [q for q, s in scored_questions[:count]]

    async def _generate_adaptive_from_content(
        self,
        workspace_id: int,
        count: int,
        performance_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Question]:
        """Generate new questions from workspace content based on performance gaps"""
        # Get workspace files
        files_query = text(
            "SELECT id, path FROM files WHERE workspace_id = :workspace_id"
        )
        files_result = await self.db.execute(
            files_query, {"workspace_id": workspace_id}
        )
        files = files_result.fetchall()

        if not files:
            return []

        generated_questions = []

        # Analyze content from files to generate questions
        for file_row in files[:5]:  # Limit to first 5 files for performance
            try:
                # Read file content (simplified - would use proper file reading)
                file_path = file_row.path
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()[:2000]  # Limit content size

                    # Generate questions based on content analysis
                    questions = await self.generate_llm_questions(
                        file_content=content,
                        file_id=file_row.id,
                        count=min(
                            3, count - len(generated_questions)
                        ),  # Distribute across files
                        question_types=[
                            "multiple_choice",
                            "short_answer",
                            "true_false",
                        ],
                        difficulty=self._determine_optimal_difficulty(
                            performance_analysis
                        ),
                    )

                    generated_questions.extend(questions)

                    if len(generated_questions) >= count:
                        break

            except Exception as e:
                print(f"Error generating questions from file {file_row.path}: {e}")
                continue

        return generated_questions[:count]

    def _determine_optimal_difficulty(
        self, performance_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """Determine optimal difficulty based on performance analysis"""
        if not performance_analysis:
            return "medium"

        # Analyze difficulty performance
        difficulty_performance = performance_analysis.get("difficulty_performance", {})

        # Calculate average performance per difficulty
        difficulty_averages = {}
        for difficulty, accuracies in difficulty_performance.items():
            if accuracies:
                difficulty_averages[difficulty] = sum(accuracies) / len(accuracies)
            else:
                difficulty_averages[difficulty] = 0.5  # Default

        # Choose difficulty where user performs around 70-80% accuracy
        target_accuracy = 0.75

        best_difficulty = "medium"
        best_distance = abs(difficulty_averages.get("medium", 0.5) - target_accuracy)

        for difficulty, avg_accuracy in difficulty_averages.items():
            distance = abs(avg_accuracy - target_accuracy)
            if distance < best_distance:
                best_difficulty = difficulty
                best_distance = distance

        return best_difficulty
