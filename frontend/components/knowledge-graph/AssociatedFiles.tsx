import React, { useEffect, useState } from 'react';
import { ApiService } from '../../src/core/api';

interface AssociatedFile {
  concept_file_id: string;
  file_id: number;
  workspace_id: number;
  snippet?: string;
  relevance_score?: number;
  last_accessed_at?: string;
}

interface AssociatedFilesProps {
  conceptId: string;
  onFileClick?: (fileId: number) => void;
}

export const AssociatedFiles: React.FC<AssociatedFilesProps> = ({
  conceptId,
  onFileClick,
}) => {
  const [files, setFiles] = useState<AssociatedFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (conceptId) {
      loadAssociatedFiles();
    }
  }, [conceptId]);

  const loadAssociatedFiles = async () => {
    try {
      setLoading(true);
      const response = await ApiService.get(`/knowledge-graph/concepts/${conceptId}/files`);
      setFiles(response.files || []);
      setError(null);
    } catch (err) {
      setError('Failed to load associated files');
      console.error('Error loading files:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRelevanceColor = (score?: number): string => {
    if (!score) return 'text-gray-400';
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const formatLastAccessed = (dateString?: string): string => {
    if (!dateString) return 'Never';

    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffTime = Math.abs(now.getTime() - date.getTime());
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

      if (diffDays === 1) return 'Today';
      if (diffDays === 2) return 'Yesterday';
      if (diffDays <= 7) return `${diffDays - 1} days ago`;
      if (diffDays <= 30) return `${Math.ceil(diffDays / 7)} weeks ago`;
      return `${Math.ceil(diffDays / 30)} months ago`;
    } catch {
      return 'Unknown';
    }
  };

  const highlightSnippet = (snippet: string, conceptId: string): React.ReactNode => {
    // Simple highlighting - in a real app, you'd get the concept name
    // For now, just return the snippet as-is
    return (
      <span className="text-sm text-gray-700 leading-relaxed">
        {snippet.length > 150 ? `${snippet.substring(0, 150)}...` : snippet}
      </span>
    );
  };

  if (!conceptId) {
    return (
      <div className="associated-files p-4">
        <h3 className="text-lg font-semibold mb-4">Associated Files</h3>
        <div className="text-gray-500 text-sm">
          Select a concept to view associated files.
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="associated-files p-4">
        <h3 className="text-lg font-semibold mb-4">Associated Files</h3>
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-full mb-1"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="associated-files p-4">
        <h3 className="text-lg font-semibold mb-4">Associated Files</h3>
        <div className="text-red-500 text-sm">
          {error}
          <button
            onClick={loadAssociatedFiles}
            className="ml-2 text-blue-500 hover:text-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="associated-files p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Associated Files</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {files.length} file{files.length !== 1 ? 's' : ''}
          </span>
          <button
            onClick={loadAssociatedFiles}
            className="text-sm text-gray-500 hover:text-gray-700"
            title="Refresh files"
          >
            ↻
          </button>
        </div>
      </div>

      {files.length === 0 ? (
        <div className="text-gray-500 text-sm">
          No files associated with this concept.
        </div>
      ) : (
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {files
            .sort((a, b) => (b.relevance_score || 0) - (a.relevance_score || 0))
            .map((file, index) => (
              <div
                key={file.concept_file_id}
                className="border rounded-lg p-3 hover:bg-gray-50 cursor-pointer transition-colors"
                onClick={() => onFileClick?.(file.file_id)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-gray-900">
                        File #{file.file_id}
                      </span>
                      {file.relevance_score && (
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${getRelevanceColor(file.relevance_score)} bg-opacity-20`}>
                          {(file.relevance_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 mb-2">
                      Last accessed: {formatLastAccessed(file.last_accessed_at)}
                    </div>
                  </div>
                  <span className="text-xs text-gray-400">
                    #{index + 1}
                  </span>
                </div>

                {file.snippet && (
                  <div className="bg-gray-50 rounded p-2 border-l-2 border-blue-200">
                    <div className="text-xs text-gray-500 mb-1 font-medium">
                      Relevant snippet:
                    </div>
                    {highlightSnippet(file.snippet, conceptId)}
                  </div>
                )}

                {/* Relevance score visualization */}
                {file.relevance_score && (
                  <div className="mt-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Relevance</span>
                      <span>{(file.relevance_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div
                        className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${file.relevance_score * 100}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {files.length > 0 && (
        <div className="mt-4 text-xs text-gray-500 border-t pt-2">
          <p>
            <strong>Relevance:</strong> Higher scores indicate stronger connections
            between the concept and file content.
          </p>
        </div>
      )}
    </div>
  );
};
