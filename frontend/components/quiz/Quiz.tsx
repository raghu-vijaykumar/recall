import React, { useState, useEffect } from 'react';
import { API_BASE } from '../../src/core/api';
import { Quiz as QuizType } from '../../src/core/types';

interface QuizProps {
  currentWorkspaceId: number | null;
}

const Quiz: React.FC<QuizProps> = ({ currentWorkspaceId }) => {
  const [quiz, setQuiz] = useState<QuizType | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showResult, setShowResult] = useState(false);

  const generateQuiz = async () => {
    if (!currentWorkspaceId) {
      alert('Please select a workspace first');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/workspaces/${currentWorkspaceId}/quiz/generate`, {
        method: 'POST'
      });

      if (response.ok) {
        const quizData = await response.json();
        setQuiz(quizData);
        setSelectedAnswer(null);
        setShowResult(false);
      } else {
        const error = await response.text();
        console.error('Failed to generate quiz:', error);
        alert(`Failed to generate quiz: ${error}`);
      }
    } catch (error) {
      console.error('Failed to generate quiz:', error);
      alert('Failed to generate quiz');
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async (answerIndex: number) => {
    if (!quiz) return;

    setSelectedAnswer(answerIndex);
    setShowResult(true);

    // TODO: Implement proper answer submission to backend
    // For now, just show feedback
    const isCorrect = answerIndex === quiz.correct_answer;
    alert(isCorrect ? 'Correct!' : 'Incorrect. Try again!');
  };

  const resetQuiz = () => {
    setQuiz(null);
    setSelectedAnswer(null);
    setShowResult(false);
  };

    return (
    <div id="quiz-tab" className="tab-content active">
      <div className="quiz-header">
        <h2>Quiz Time</h2>
        <button
          className="btn-primary"
          onClick={generateQuiz}
          disabled={loading || !currentWorkspaceId}
        >
          {loading ? 'Generating...' : 'Generate Quiz'}
        </button>
      </div>
      <div className="quiz-container">
        {quiz ? (
          <div className="quiz-question">
            <h3>{quiz.question}</h3>
            <div className="quiz-options">
              {quiz.options.map((option: string, index: number) => (
                <button
                  key={index}
                  className={`quiz-option ${selectedAnswer === index ? 'selected' : ''}`}
                  onClick={() => !showResult && submitAnswer(index)}
                  disabled={showResult}
                >
                  {option}
                </button>
              ))}
            </div>
            {showResult && (
              <div className="quiz-result">
                <p>
                  {selectedAnswer === quiz.correct_answer
                    ? '✅ Correct!'
                    : `❌ Incorrect. The correct answer is: ${quiz.options[quiz.correct_answer]}`
                  }
                </p>
                <button className="btn-secondary" onClick={resetQuiz}>
                  Try Another Quiz
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="quiz-placeholder">
            <h3>Ready to test your knowledge?</h3>
            <p>Generate a quiz based on your workspace files to reinforce your learning.</p>
            {!currentWorkspaceId && (
              <p className="warning">Please select a workspace first.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Quiz;
