import React, { useState, useEffect } from 'react';
import { API_BASE } from '../../core/api.js';
import { WorkspaceStats } from '../../core/types.js';

interface ProgressProps {
  currentWorkspaceId: number | null;
}

const Progress: React.FC<ProgressProps> = ({ currentWorkspaceId }) => {
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (currentWorkspaceId) {
      loadProgress();
    } else {
      setStats(null);
    }
  }, [currentWorkspaceId]);

  const loadProgress = async () => {
    if (!currentWorkspaceId) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/workspaces/${currentWorkspaceId}/stats`);
      if (response.ok) {
        const stats = await response.json();
        setStats(stats);
      } else {
        console.error('Failed to load progress:', response.status);
        setStats(null);
      }
    } catch (error) {
      console.error('Failed to load progress:', error);
      setStats(null);
    } finally {
      setLoading(false);
    }
  };

  const calculateProgressPercentage = (stats: WorkspaceStats): number => {
    const totalActivities = (stats.total_files || 0) + (stats.quizzes_taken || 0);
    const maxActivities = Math.max((stats.total_quizzes || 0) * 2, 10);
    return Math.min((totalActivities / maxActivities) * 100, 100);
  };

  if (loading) {
    return (
      <div id="progress-tab" className="tab-content">
        <div className="progress-header">
          <h2>Your Progress</h2>
        </div>
        <div className="progress-container">
          <div className="loading">Loading progress...</div>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div id="progress-tab" className="tab-content">
        <div className="progress-header">
          <h2>Your Progress</h2>
        </div>
        <div className="progress-container">
          <div className="empty-progress">
            <h4>No Progress Data Available</h4>
            <p>Select a workspace and start creating files or taking quizzes to see your progress here.</p>
            <div className="progress-placeholder">
              <div className="stat-card">
                <h4>Total Files</h4>
                <p>0</p>
              </div>
              <div className="stat-card">
                <h4>Quizzes Taken</h4>
                <p>0</p>
              </div>
              <div className="stat-card">
                <h4>Average Score</h4>
                <p>0%</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const progressPercentage = calculateProgressPercentage(stats);

  return (
    <div id="progress-tab" className="tab-content">
      <div className="progress-header">
        <h2>Your Progress</h2>
      </div>
      <div className="progress-container">
        <div className="progress-stats">
          <div className="stat-card">
            <h4>Total Files</h4>
            <p>{stats.total_files || 0}</p>
          </div>
          <div className="stat-card">
            <h4>Quizzes Taken</h4>
            <p>{stats.quizzes_taken || 0}</p>
          </div>
          <div className="stat-card">
            <h4>Average Score</h4>
            <p>{stats.average_score || 0}%</p>
          </div>
          <div className="stat-card">
            <h4>Total Quizzes Available</h4>
            <p>{stats.total_quizzes || 0}</p>
          </div>
        </div>

        <div className="progress-chart">
          <h4>Learning Progress</h4>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progressPercentage}%` }}></div>
          </div>
          <p className="progress-text">{progressPercentage.toFixed(1)}% Complete</p>
        </div>

        <div className="recent-activity">
          <h4>Recent Activity</h4>
          <div className="activity-list">
            <div className="activity-item">
              <span className="activity-icon">📝</span>
              <span className="activity-text">Files created: {stats.total_files || 0}</span>
            </div>
            <div className="activity-item">
              <span className="activity-icon">🧠</span>
              <span className="activity-text">Quizzes completed: {stats.quizzes_taken || 0}</span>
            </div>
            <div className="activity-item">
              <span className="activity-icon">📊</span>
              <span className="activity-text">Average performance: {stats.average_score || 0}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Progress;
