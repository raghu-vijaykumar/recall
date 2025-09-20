import { API_BASE } from '../../core/api.js';
import { WorkspaceStats } from '../../core/types.js';

export class ProgressComponent {
  private currentWorkspaceId: number | null = null;

  constructor() {
    this.initialize();
  }

  private initialize() {
    // Listen for workspace selection events
    window.addEventListener('workspace-selected', (e: any) => {
      this.setCurrentWorkspace(e.detail.workspaceId);
    });
  }

  setCurrentWorkspace(workspaceId: number) {
    this.currentWorkspaceId = workspaceId;
    this.loadProgress();
  }

  async loadProgress() {
    if (!this.currentWorkspaceId) return;

    try {
      const response = await fetch(`${API_BASE}/workspaces/${this.currentWorkspaceId}/stats`);
      if (response.ok) {
        const stats = await response.json();
        this.renderProgress(stats);
      } else {
        console.error('Failed to load progress:', response.status);
        this.renderEmptyProgress();
      }
    } catch (error) {
      console.error('Failed to load progress:', error);
      this.renderEmptyProgress();
    }
  }

  private renderProgress(stats: WorkspaceStats) {
    const container = document.getElementById('progress-container');
    if (!container) return;

    container.innerHTML = `
      <div class="progress-stats">
        <div class="stat-card">
          <h4>Total Files</h4>
          <p>${stats.total_files || 0}</p>
        </div>
        <div class="stat-card">
          <h4>Quizzes Taken</h4>
          <p>${stats.quizzes_taken || 0}</p>
        </div>
        <div class="stat-card">
          <h4>Average Score</h4>
          <p>${stats.average_score || 0}%</p>
        </div>
        <div class="stat-card">
          <h4>Total Quizzes Available</h4>
          <p>${stats.total_quizzes || 0}</p>
        </div>
      </div>

      <div class="progress-chart">
        <h4>Learning Progress</h4>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${this.calculateProgressPercentage(stats)}%"></div>
        </div>
        <p class="progress-text">${this.calculateProgressPercentage(stats).toFixed(1)}% Complete</p>
      </div>

      <div class="recent-activity">
        <h4>Recent Activity</h4>
        <div class="activity-list">
          <div class="activity-item">
            <span class="activity-icon">📝</span>
            <span class="activity-text">Files created: ${stats.total_files || 0}</span>
          </div>
          <div class="activity-item">
            <span class="activity-icon">🧠</span>
            <span class="activity-text">Quizzes completed: ${stats.quizzes_taken || 0}</span>
          </div>
          <div class="activity-item">
            <span class="activity-icon">📊</span>
            <span class="activity-text">Average performance: ${stats.average_score || 0}%</span>
          </div>
        </div>
      </div>
    `;
  }

  private renderEmptyProgress() {
    const container = document.getElementById('progress-container');
    if (!container) return;

    container.innerHTML = `
      <div class="empty-progress">
        <h4>No Progress Data Available</h4>
        <p>Select a workspace and start creating files or taking quizzes to see your progress here.</p>
        <div class="progress-placeholder">
          <div class="stat-card">
            <h4>Total Files</h4>
            <p>0</p>
          </div>
          <div class="stat-card">
            <h4>Quizzes Taken</h4>
            <p>0</p>
          </div>
          <div class="stat-card">
            <h4>Average Score</h4>
            <p>0%</p>
          </div>
        </div>
      </div>
    `;
  }

  private calculateProgressPercentage(stats: WorkspaceStats): number {
    const totalActivities = (stats.total_files || 0) + (stats.quizzes_taken || 0);
    const maxActivities = Math.max((stats.total_quizzes || 0) * 2, 10); // Assume 2 files per quiz topic, minimum 10
    return Math.min((totalActivities / maxActivities) * 100, 100);
  }

  // Menu event handlers
  showProgressTab() {
    // Emit event to show progress tab
    window.dispatchEvent(new CustomEvent('show-tab', { detail: { tab: 'progress' } }));
  }

  refreshProgress() {
    this.loadProgress();
  }

  getCurrentWorkspaceId(): number | null {
    return this.currentWorkspaceId;
  }
}

// Global instance for onclick handlers
declare global {
  interface Window {
    progressComponent: ProgressComponent;
  }
}
