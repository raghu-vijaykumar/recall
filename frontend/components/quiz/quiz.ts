import { API_BASE } from '../../core/api.js';
import { Quiz } from '../../core/types.js';

export class QuizComponent {
  private currentWorkspaceId: number | null = null;

  constructor() {
    this.initialize();
  }

  private initialize() {
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    const generateBtn = document.getElementById('generate-quiz-btn');
    if (generateBtn) {
      generateBtn.addEventListener('click', () => {
        if (this.currentWorkspaceId) {
          this.generateQuiz(this.currentWorkspaceId);
        } else {
          alert('Please select a workspace first');
        }
      });
    }

    // Listen for workspace selection events
    window.addEventListener('workspace-selected', (e: any) => {
      this.setCurrentWorkspace(e.detail.workspaceId);
    });
  }

  setCurrentWorkspace(workspaceId: number) {
    this.currentWorkspaceId = workspaceId;
  }

  async generateQuiz(workspaceId: number) {
    try {
      const response = await fetch(`${API_BASE}/workspaces/${workspaceId}/quiz/generate`, {
        method: 'POST'
      });

      if (response.ok) {
        const quiz = await response.json();
        this.renderQuiz(quiz);
      } else {
        const error = await response.text();
        console.error('Failed to generate quiz:', error);
        alert(`Failed to generate quiz: ${error}`);
      }
    } catch (error) {
      console.error('Failed to generate quiz:', error);
      alert('Failed to generate quiz');
    }
  }

  private renderQuiz(quiz: Quiz) {
    const container = document.getElementById('quiz-container');
    if (!container) return;

    container.innerHTML = `
      <div class="quiz-question">
        <h3>${quiz.question}</h3>
        <div class="quiz-options">
          ${quiz.options.map((option: string, index: number) =>
            `<button class="quiz-option" onclick="window.quizComponent.submitAnswer(${index})">${option}</button>`
          ).join('')}
        </div>
      </div>
    `;
  }

  submitAnswer(answerIndex: number) {
    // TODO: Implement answer submission and feedback
    console.log('Answer submitted:', answerIndex);
    alert(`You selected answer ${answerIndex + 1}. Answer submission functionality will be implemented soon!`);
  }

  // Menu event handlers
  showQuizTab() {
    // Emit event to show quiz tab
    window.dispatchEvent(new CustomEvent('show-tab', { detail: { tab: 'quiz' } }));
  }

  getCurrentWorkspaceId(): number | null {
    return this.currentWorkspaceId;
  }
}

// Global instance for onclick handlers
declare global {
  interface Window {
    quizComponent: QuizComponent;
  }
}
