# Quiz Generation Improvements Design Document

## Overview

This document outlines the design for improving the quiz generation system in the Recall application. The improvements focus on two main areas:

1. **Refining the algorithm** to produce more diverse and challenging quizzes
2. **Implementing LLM-powered features** for advanced quiz generation capabilities

## Current State Analysis

### Existing Components
- **Models**: Well-defined Pydantic models for Questions, QuizSessions, Answers, and statistics
- **Frontend**: Basic quiz component with question display and answer submission
- **Backend**: Placeholder routes for quiz functionality
- **LLM Infrastructure**: Factory pattern with Gemini and Ollama client support

### Current Limitations
- Quiz generation is not implemented (placeholder routes)
- No algorithm for question selection or diversity
- No adaptive difficulty or spaced repetition
- Limited question types and interaction modes

## 1. Algorithm Improvements for Diverse and Challenging Quizzes

### Core Principles
- **Diversity**: Ensure questions cover different topics, difficulty levels, and question types
- **Challenge**: Adapt to user performance and maintain engagement
- **Coverage**: Ensure comprehensive testing of workspace content
- **Efficiency**: Minimize redundant questions while maximizing learning value

### Question Selection Algorithm

#### Content Analysis Phase
```python
class ContentAnalyzer:
    def analyze_file_content(self, file_content: str) -> Dict[str, Any]:
        """
        Extract key concepts, topics, and difficulty indicators from file content
        Returns: {
            'topics': List[str],
            'complexity_score': float,
            'key_concepts': List[str],
            'question_candidates': List[Dict]
        }
        """
```

#### Question Pool Generation
```python
class QuestionPoolGenerator:
    def generate_pool(self, workspace_files: List[File]) -> List[Question]:
        """
        Generate diverse question pool from workspace content
        - Extract key concepts and relationships
        - Generate multiple question types per concept
        - Balance difficulty distribution
        - Ensure topic coverage
        """
```

#### Selection Criteria
- **Topic Distribution**: Ensure questions cover all major topics proportionally
- **Difficulty Balance**: Mix easy (30%), medium (50%), hard (20%) questions
- **Question Type Mix**: Multiple choice (40%), true/false (30%), short answer (20%), fill-in-blank (10%)
- **Freshness**: Prioritize recently modified content and avoid recently asked questions

### Adaptive Difficulty Algorithm

#### Performance Tracking
```python
class PerformanceTracker:
    def update_performance(self, question_id: int, correct: bool, time_taken: int):
        """
        Track question performance metrics:
        - Success rate
        - Average response time
        - Difficulty perception
        - Topic mastery level
        """
```

#### Difficulty Adjustment
```python
class AdaptiveDifficultyEngine:
    def adjust_difficulty(self, user_performance: Dict) -> DifficultyLevel:
        """
        Adjust question difficulty based on:
        - Recent performance (last 10 questions)
        - Topic mastery levels
        - Session goals (learning vs. testing)
        - User preferences
        """
```

### Diversity Metrics
- **Topic Entropy**: Measure distribution across different subjects
- **Question Type Variance**: Ensure mix of cognitive skills tested
- **Difficulty Spread**: Avoid clustering at single difficulty level
- **Temporal Distribution**: Balance questions from different time periods

## 2. LLM-Powered Advanced Features

### Dynamic Q&A Generation

#### LLM Integration Architecture
```python
class LLMQuestionGenerator:
    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def generate_questions(self, content: str, count: int = 5) -> List[Question]:
        """
        Use LLM to generate contextually relevant questions
        - Analyze content structure and key concepts
        - Generate varied question types
        - Ensure factual accuracy
        - Provide explanations
        """
```

#### Prompt Engineering
```
Generate {count} diverse quiz questions from the following content.
Requirements:
- Mix of question types: multiple choice, true/false, short answer
- Cover different aspects: facts, concepts, applications
- Include explanations for correct answers
- Vary difficulty levels appropriately

Content: {file_content}

Output format: JSON array of question objects
```

### Spaced Repetition System

#### Algorithm Implementation
```python
class SpacedRepetitionEngine:
    def __init__(self):
        self.algorithm = "sm2"  # SuperMemo 2 algorithm

    def calculate_next_review(self, question: Question, performance: Answer) -> datetime:
        """
        Calculate optimal review interval based on:
        - Current ease factor
        - Performance quality (0-5 scale)
        - Number of successful reviews
        - Time since last review
        """
```

#### Review Scheduling
- **Initial Review**: Immediate feedback
- **First Repetition**: 1 day later
- **Subsequent Reviews**: 3, 7, 14, 30 days (adjusting based on performance)
- **Failed Items**: Reset to 1-day interval

### Weak Area Analysis

#### Performance Analytics
```python
class WeakAreaAnalyzer:
    def identify_weak_areas(self, user_answers: List[Answer]) -> Dict[str, float]:
        """
        Analyze performance patterns to identify:
        - Topics with low success rates
        - Question types causing difficulty
        - Time-based performance trends
        - Concept dependencies
        """
```

#### Targeted Review Generation
```python
class TargetedReviewGenerator:
    def generate_review_session(self, weak_areas: Dict) -> QuizSession:
        """
        Create focused review sessions:
        - Prioritize weak topics (70% of questions)
        - Include related concepts for reinforcement
        - Mix in stronger areas for confidence building
        - Adaptive difficulty based on mastery levels
        """
```

### Voice and Flashcard Modes

#### Voice Mode Implementation
```python
class VoiceQuizEngine:
    def __init__(self, tts_engine):
        self.tts = tts_engine

    def speak_question(self, question: Question):
        """
        Convert question text to speech
        - Use natural voice synthesis
        - Support different speeds
        - Handle special characters/technical terms
        """

    def process_voice_answer(self, audio_input) -> str:
        """
        Convert speech to text for answer processing
        - Handle various accents and speech patterns
        - Provide confidence scores
        - Support retry mechanisms
        """
```

#### Flashcard Mode
```python
class FlashcardEngine:
    def create_flashcard_deck(self, questions: List[Question]) -> List[Flashcard]:
        """
        Transform questions into flashcard format:
        - Question on front, answer on back
        - Include hints and explanations
        - Group by topic and difficulty
        """

    def adaptive_sequence(self, deck: List[Flashcard], performance: Dict) -> List[Flashcard]:
        """
        Order flashcards for optimal learning:
        - Prioritize difficult items
        - Space repetitions appropriately
        - Balance topic coverage
        """
```

## Implementation Roadmap

### Phase 1: Core Algorithm Improvements (Leveraging Knowledge Graph)
1. Implement content analysis for question generation, integrating with Knowledge Graph for concept extraction and relationship inference.
2. Add question pool management with diversity metrics, guided by Knowledge Graph concepts and relationships.
3. Integrate adaptive difficulty engine, informed by Knowledge Graph concept mastery levels.
4. Update quiz routes with new algorithms.

### Phase 2: LLM Integration
1. Integrate LLM clients for dynamic question generation.
2. Implement prompt templates for different question types.
3. Add LLM-generated question validation.
4. Update models to support LLM-generated content.

### Phase 3: Advanced Features (Leveraging Knowledge Graph)
1. Implement spaced repetition algorithm, prioritizing questions based on Knowledge Graph concept review schedules.
2. Add weak area analysis and targeted reviews, directly linking to Knowledge Graph concepts.
3. Integrate voice synthesis and recognition.
4. Develop flashcard mode interface.

### Phase 4: Analytics and Optimization
1. Add comprehensive performance tracking, including concept-level mastery from Knowledge Graph.
2. Implement A/B testing for algorithm improvements.
3. Add user feedback mechanisms.
4. Optimize for performance and scalability.

## Integration with Knowledge Graph

The Knowledge Graph (KG) will serve as a foundational component for enhancing quiz generation, providing structured insights into user workspace content.

### Backend Integration Points
The Quiz Generation Backend will interact with the Knowledge Graph Service via its API endpoints:
*   `POST /api/workspaces/{workspace_id}/analyze`: To trigger content analysis and update the KG for a workspace, which in turn enriches the data available for quiz generation.
*   `GET /api/workspaces/{workspace_id}/knowledge-graph`: To retrieve the graph for understanding concept relationships, aiding in question diversity and difficulty assessment.
*   `GET /api/workspaces/{workspace_id}/suggested-topics`: To get a prioritized list of topics for quiz generation, especially for targeted reviews and ensuring comprehensive coverage.
*   `GET /api/concepts/{concept_id}/files`: To retrieve relevant file snippets for question generation based on specific concepts identified by the KG.

### Enhanced Models (Leveraging Knowledge Graph)
```python
class LLMQuestion(Question):
    generated_by_llm: bool = True
    generation_prompt: Optional[str]
    confidence_score: float
    # Link to Knowledge Graph concept(s)
    kg_concept_ids: List[str] = []

class SpacedRepetitionData(BaseModel):
    ease_factor: float = 2.5
    interval_days: int = 1
    review_count: int = 0
    next_review: datetime
    # Link to Knowledge Graph concept(s)
    kg_concept_id: Optional[str]
```

## API Design

### New Endpoints
```
POST /api/quiz/generate-adaptive
- Generate quiz with adaptive difficulty, potentially informed by Knowledge Graph concept mastery.

POST /api/quiz/generate-llm
- Generate questions using LLM, with input potentially guided by Knowledge Graph concepts.

GET /api/quiz/spaced-repetition
- Get questions due for review, prioritized by Spaced Repetition and Knowledge Graph concept mastery.

POST /api/quiz/voice-answer
- Submit voice-based answer.

GET /api/quiz/weak-areas
- Get analysis of weak performance areas, directly linked to Knowledge Graph concepts.
```

## Frontend Enhancements

### New Components
- **AdaptiveQuiz**: Handles difficulty adjustment, potentially visualizing progress against Knowledge Graph concepts.
- **VoiceQuiz**: Integrates speech synthesis/recognition.
- **FlashcardMode**: Card-based learning interface.
- **ProgressAnalytics**: Visualizes weak areas and progress, potentially highlighting struggling concepts within a Knowledge Graph visualization.

### UI Improvements
- Progress indicators for spaced repetition, potentially showing progress on specific Knowledge Graph concepts.
- Voice controls and feedback.
- Flashcard flip animations.
- Performance heatmaps for topic analysis, potentially mapping to Knowledge Graph concepts.

## Testing Strategy

### Unit Tests
- Algorithm correctness (diversity metrics, difficulty adjustment, KG integration).
- LLM integration and prompt effectiveness.
- Spaced repetition calculations.
- Knowledge Graph API interactions.

### Integration Tests
- End-to-end quiz generation workflows, including KG data flow.
- Voice processing accuracy.
- Performance tracking accuracy.
- KG-informed targeted review generation.

### User Testing
- A/B testing of algorithm improvements.
- Usability testing for new modes.
- Performance impact assessment.

## Metrics and Monitoring

### Key Metrics
- Question diversity scores (informed by KG concept coverage).
- User engagement (session completion rates).
- Learning effectiveness (performance improvement over time, concept mastery from KG).
- System performance (generation speed, accuracy).

### Monitoring
- LLM usage and costs.
- Question quality ratings.
- User satisfaction surveys.
- Error rates and failure modes.
- Knowledge Graph service health and data consistency.

## Conclusion

These improvements will transform the quiz system from a basic question generator into an intelligent, adaptive learning platform. The combination of algorithmic refinements, LLM-powered features, and deep integration with the Knowledge Graph will provide users with personalized, effective learning experiences that adapt to their needs and maximize knowledge retention by understanding their knowledge landscape.
