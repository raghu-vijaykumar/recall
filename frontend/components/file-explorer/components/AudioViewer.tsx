import React, { useState, useEffect } from 'react';
import { File } from '../../../src/core/types';

interface AudioViewerProps {
  activeFile: File;
}

const AudioViewer: React.FC<AudioViewerProps> = ({ activeFile }) => {
  const [streamingUrl, setStreamingUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const setupStreaming = async () => {
      setLoading(true);
      setError(null);

      try {
        // Get the file server port
        const port = await (window as any).electronAPI.getFileServerPort();

        if (!port) {
          setError('File streaming server not available');
          setLoading(false);
          return;
        }

        // Create streaming URL - need to get relative path from workspace
        // For now, we'll use the full path and let the server handle it
        // In a real implementation, you'd want to compute the relative path from workspace root
        const relativePath = activeFile.path.replace(/\\/g, '/'); // Normalize path separators
        const url = `http://127.0.0.1:${port}/${encodeURIComponent(relativePath)}`;
        setStreamingUrl(url);
        setLoading(false);
      } catch (err: any) {
        console.error('Failed to setup audio streaming:', err);
        setError('Failed to setup audio streaming');
        setLoading(false);
      }
    };

    setupStreaming();
  }, [activeFile.path]);

  if (loading) {
    return (
      <div className="audio-viewer">
        <div className="audio-container">
          <div className="loading">Loading audio...</div>
        </div>
        <div className="file-info">
          <p><strong>File:</strong> {activeFile.name}</p>
          <p><strong>Path:</strong> {activeFile.path}</p>
          <p><strong>Size:</strong> {activeFile.size} bytes</p>
        </div>
      </div>
    );
  }

  if (error || !streamingUrl) {
    return (
      <div className="audio-viewer">
        <div className="audio-container">
          <div className="error-content">
            <h3>Failed to Load Audio</h3>
            <p>{error || 'Could not display the audio file.'}</p>
            <p>File: {activeFile.name}</p>
            <p>Path: {activeFile.path}</p>
          </div>
        </div>
        <div className="file-info">
          <p><strong>File:</strong> {activeFile.name}</p>
          <p><strong>Path:</strong> {activeFile.path}</p>
          <p><strong>Size:</strong> {activeFile.size} bytes</p>
        </div>
      </div>
    );
  }

  return (
    <div className="audio-viewer">
      <div className="audio-container">
        <audio
          src={streamingUrl}
          className="viewer-audio"
          controls
          onError={(e) => {
            console.error('Failed to load audio:', activeFile.path);
            setError('Failed to play audio');
          }}
        >
          Your browser does not support the audio element.
        </audio>
      </div>
      <div className="file-info">
        <p><strong>File:</strong> {activeFile.name}</p>
        <p><strong>Path:</strong> {activeFile.path}</p>
        <p><strong>Size:</strong> {activeFile.size} bytes</p>
      </div>
    </div>
  );
};

export default AudioViewer;
