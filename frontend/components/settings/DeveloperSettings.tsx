import React, { useState, useEffect } from 'react';
import { ApiService } from '../../src/core/api';

interface EmbeddingModel {
  name: string;
  dimensions: number;
  size_mb: number;
  description: string;
  recommended: boolean;
}

interface EmbeddingStats {
  model?: string;
  total_concepts?: number;
  dimensions?: number;
  error?: string;
}

const DeveloperSettings: React.FC = () => {
  const [availableModels, setAvailableModels] = useState<Record<string, EmbeddingModel>>({});
  const [currentModel, setCurrentModel] = useState<EmbeddingModel | null>(null);
  const [stats, setStats] = useState<EmbeddingStats>({});
  const [loading, setLoading] = useState(false);
  const [switching, setSwitching] = useState(false);
  const [message, setMessage] = useState<string>('');

  useEffect(() => {
    loadEmbeddingInfo();
  }, []);

  const loadEmbeddingInfo = async () => {
    try {
      const response = await ApiService.get('/api/knowledge-graph/embeddings/models');
      setAvailableModels(response.available_models || {});
      setCurrentModel(response.current_model || null);

      // Load stats if model is active
      if (response.current_model) {
        const statsResponse = await ApiService.get('/api/knowledge-graph/embeddings/stats');
        setStats(statsResponse);
      }
    } catch (error) {
      console.error('Failed to load embedding info:', error);
      setMessage('Failed to load embedding information');
    }
  };

  const initializeModel = async (modelName: string) => {
    setLoading(true);
    setMessage('');

    try {
      const response = await ApiService.post('/api/knowledge-graph/embeddings/initialize', {
        model_name: modelName
      });

      setMessage(response.message || 'Model initialized successfully');
      await loadEmbeddingInfo();
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Failed to initialize model');
    } finally {
      setLoading(false);
    }
  };

  const switchModel = async (newModelName: string, reembedAll: boolean = false) => {
    setSwitching(true);
    setMessage('');

    try {
      const response = await ApiService.post('/api/knowledge-graph/embeddings/switch-model', {
        new_model_name: newModelName,
        reembed_all: reembedAll
      });

      setMessage(response.message || `Successfully switched to ${newModelName}`);
      await loadEmbeddingInfo();
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Failed to switch model');
    } finally {
      setSwitching(false);
    }
  };

  const testEmbeddingSearch = async () => {
    const query = prompt('Enter a search query:');
    if (!query) return;

    try {
      const response = await ApiService.post('/api/knowledge-graph/embeddings/search', {
        query,
        limit: 5
      });

      console.log('Search results:', response.results);
      alert(`Found ${response.results.length} similar concepts. Check console for details.`);
    } catch (error: any) {
      alert('Search failed: ' + (error.response?.data?.detail || 'Unknown error'));
    }
  };

  return (
    <div className="developer-settings">
      <h2>Developer Settings</h2>
      <p className="text-muted">Configure advanced features like embedding models for workspace analysis.</p>

      <div className="settings-section">
        <h3>Embedding Models</h3>
        <p>Configure the AI model used for semantic analysis of your workspace files.</p>

        {message && (
          <div className={`alert ${message.includes('success') ? 'alert-success' : 'alert-error'}`}>
            {message}
          </div>
        )}

        {/* Current Model Status */}
        <div className="current-model-section">
          <h4>Current Model</h4>
          {currentModel ? (
            <div className="model-info">
              <div className="model-details">
                <strong>{currentModel.name}</strong>
                <p>{currentModel.description}</p>
                <small>
                  Dimensions: {currentModel.dimensions} |
                  Size: {currentModel.size_mb}MB |
                  Concepts: {stats.total_concepts || 0}
                </small>
              </div>
              <button
                className="btn btn-secondary"
                onClick={testEmbeddingSearch}
                disabled={!currentModel}
              >
                Test Search
              </button>
            </div>
          ) : (
            <p className="text-muted">No embedding model initialized</p>
          )}
        </div>

        {/* Available Models */}
        <div className="available-models-section">
          <h4>Available Models</h4>
          <div className="models-grid">
            {Object.entries(availableModels).map(([name, model]) => (
              <div key={name} className={`model-card ${model.recommended ? 'recommended' : ''}`}>
                <div className="model-header">
                  <h5>{name}</h5>
                  {model.recommended && <span className="badge badge-primary">Recommended</span>}
                </div>
                <p>{model.description}</p>
                <div className="model-specs">
                  <small>
                    {model.dimensions} dimensions • {model.size_mb}MB
                  </small>
                </div>
                <div className="model-actions">
                  {currentModel?.name === name ? (
                    <span className="text-success">Active</span>
                  ) : (
                    <>
                      <button
                        className="btn btn-sm btn-outline"
                        onClick={() => initializeModel(name)}
                        disabled={loading}
                      >
                        {loading ? 'Initializing...' : 'Initialize'}
                      </button>
                      {currentModel && (
                        <button
                          className="btn btn-sm btn-primary"
                          onClick={() => switchModel(name, false)}
                          disabled={switching}
                        >
                          {switching ? 'Switching...' : 'Switch'}
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Advanced Options */}
        {currentModel && (
          <div className="advanced-options">
            <h4>Advanced Options</h4>
            <div className="option-group">
              <button
                className="btn btn-warning"
                onClick={() => switchModel(currentModel.name, true)}
                disabled={switching}
              >
                {switching ? 'Re-embedding...' : 'Re-embed All Content'}
              </button>
              <small className="text-muted">
                Re-analyze all workspace files with the current model. This may take time.
              </small>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .developer-settings {
          padding: 20px;
          max-width: 800px;
        }

        .settings-section {
          margin-bottom: 30px;
          padding: 20px;
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          background: white;
        }

        .current-model-section, .available-models-section {
          margin-bottom: 20px;
        }

        .model-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .model-details small {
          color: #6c757d;
        }

        .models-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .model-card {
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          padding: 15px;
          background: white;
        }

        .model-card.recommended {
          border-color: #007bff;
          background: #f8f9ff;
        }

        .model-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .model-specs {
          margin: 10px 0;
        }

        .model-actions {
          display: flex;
          gap: 8px;
          align-items: center;
          margin-top: 10px;
        }

        .advanced-options {
          margin-top: 20px;
          padding-top: 20px;
          border-top: 1px solid #e1e5e9;
        }

        .option-group {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .alert {
          padding: 10px 15px;
          border-radius: 4px;
          margin-bottom: 15px;
        }

        .alert-success {
          background: #d4edda;
          color: #155724;
          border: 1px solid #c3e6cb;
        }

        .alert-error {
          background: #f8d7da;
          color: #721c24;
          border: 1px solid #f5c6cb;
        }

        .badge {
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 0.75em;
          font-weight: bold;
        }

        .badge-primary {
          background: #007bff;
          color: white;
        }

        .text-success {
          color: #28a745;
          font-weight: bold;
        }

        .text-muted {
          color: #6c757d;
        }

        .btn {
          padding: 8px 16px;
          border: 1px solid #007bff;
          border-radius: 4px;
          background: #007bff;
          color: white;
          cursor: pointer;
          text-decoration: none;
          display: inline-block;
        }

        .btn:hover {
          background: #0056b3;
        }

        .btn:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .btn-secondary {
          background: #6c757d;
          border-color: #6c757d;
        }

        .btn-secondary:hover {
          background: #545b62;
        }

        .btn-outline {
          background: transparent;
          color: #007bff;
        }

        .btn-outline:hover {
          background: #007bff;
          color: white;
        }

        .btn-warning {
          background: #ffc107;
          border-color: #ffc107;
          color: #212529;
        }

        .btn-warning:hover {
          background: #e0a800;
        }

        .btn-sm {
          padding: 4px 8px;
          font-size: 0.875em;
        }
      `}</style>
    </div>
  );
};

export default DeveloperSettings;
