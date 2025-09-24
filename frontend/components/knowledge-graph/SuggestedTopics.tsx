import React, { useEffect, useState } from 'react';
import { ApiService } from '../../src/core/api';

interface SuggestedTopic {
  concept_id: string;
  name: string;
  relevance_score: number;
  description?: string;
  score_breakdown?: {
    frequency: number;
    recency: number;
    semantic: number;
  };
}

interface SuggestedTopicsProps {
  workspaceId: number;
  onTopicClick?: (topic: SuggestedTopic) => void;
  limit?: number;
}

export const SuggestedTopics: React.FC<SuggestedTopicsProps> = ({
  workspaceId,
  onTopicClick,
  limit = 10,
}) => {
  const [topics, setTopics] = useState<SuggestedTopic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSuggestedTopics();
  }, [workspaceId, limit]);

  const loadSuggestedTopics = async () => {
    try {
      setLoading(true);
      const response = await ApiService.get(
        `/knowledge-graph/workspaces/${workspaceId}/suggested-topics?limit=${limit}`
      );
      setTopics(response.topics || []);
      setError(null);
    } catch (err) {
      setError('Failed to load suggested topics');
      console.error('Error loading topics:', err);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number): string => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getScoreBarWidth = (score: number): string => {
    return `${Math.max(10, score * 100)}%`;
  };

  if (loading) {
    return (
      <div className="suggested-topics p-4">
        <h3 className="text-lg font-semibold mb-4">Suggested Topics</h3>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="suggested-topics p-4">
        <h3 className="text-lg font-semibold mb-4">Suggested Topics</h3>
        <div className="text-red-500 text-sm">
          {error}
          <button
            onClick={loadSuggestedTopics}
            className="ml-2 text-blue-500 hover:text-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="suggested-topics p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Suggested Topics</h3>
        <button
          onClick={loadSuggestedTopics}
          className="text-sm text-gray-500 hover:text-gray-700"
          title="Refresh topics"
        >
          ↻
        </button>
      </div>

      {topics.length === 0 ? (
        <div className="text-gray-500 text-sm">
          No topics available. Try analyzing your workspace first.
        </div>
      ) : (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {topics.map((topic, index) => (
            <div
              key={topic.concept_id}
              className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer transition-colors"
              onClick={() => onTopicClick?.(topic)}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-gray-900">
                      {topic.name}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${getScoreColor(topic.relevance_score)}`}>
                      {(topic.relevance_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  {topic.description && (
                    <p className="text-xs text-gray-600 line-clamp-2">
                      {topic.description}
                    </p>
                  )}
                </div>
                <span className="text-xs text-gray-400 ml-2">
                  #{index + 1}
                </span>
              </div>

              {/* Score breakdown visualization */}
              {topic.score_breakdown && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Relevance Breakdown</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Freq</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-blue-500 h-1.5 rounded-full"
                          style={{ width: getScoreBarWidth(topic.score_breakdown.frequency) }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 w-8">
                        {(topic.score_breakdown.frequency * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Rec</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-green-500 h-1.5 rounded-full"
                          style={{ width: getScoreBarWidth(topic.score_breakdown.recency) }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 w-8">
                        {(topic.score_breakdown.recency * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500 w-12">Sem</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-purple-500 h-1.5 rounded-full"
                          style={{ width: getScoreBarWidth(topic.score_breakdown.semantic) }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 w-8">
                        {(topic.score_breakdown.semantic * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Overall score bar */}
              <div className="mt-2">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Overall Relevance</span>
                  <span>{(topic.relevance_score * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: getScoreBarWidth(topic.relevance_score) }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {topics.length > 0 && (
        <div className="mt-4 text-xs text-gray-500 border-t pt-2">
          <p>
            <strong>Scoring:</strong> Topics are ranked by frequency (how often mentioned),
            recency (recent activity), and semantic relevance (concept relationships).
          </p>
        </div>
      )}
    </div>
  );
};
