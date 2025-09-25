import React, { useState, useEffect } from 'react';
import { File } from '../../../src/core/types';

interface PdfViewerProps {
  activeFile: File;
}

const PdfViewer: React.FC<PdfViewerProps> = ({ activeFile }) => {
  const [pdfData, setPdfData] = useState<{ base64: string; mimeType: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    (window as any).electronAPI.readFileBase64(activeFile.path)
      .then((data: { base64: string; mimeType: string }) => {
        setPdfData(data);
        setLoading(false);
      })
      .catch((err: any) => {
        console.error('Failed to load PDF:', err);
        setError('Failed to load PDF');
        setLoading(false);
      });
  }, [activeFile.path]);

  if (loading) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-container">
          <div className="loading">Loading PDF...</div>
        </div>
        <div className="file-info">
          <p><strong>File:</strong> {activeFile.name}</p>
          <p><strong>Path:</strong> {activeFile.path}</p>
          <p><strong>Size:</strong> {activeFile.size} bytes</p>
        </div>
      </div>
    );
  }

  if (error || !pdfData) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-container">
          <div className="error-content">
            <h3>Failed to Load PDF</h3>
            <p>{error || 'Could not display the PDF file.'}</p>
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
    <div className="pdf-viewer">
      <div className="pdf-container">
        <iframe
          src={`data:${pdfData.mimeType};base64,${pdfData.base64}`}
          className="viewer-pdf"
          title={activeFile.name}
        />
      </div>
      <div className="file-info">
        <p><strong>File:</strong> {activeFile.name}</p>
        <p><strong>Path:</strong> {activeFile.path}</p>
        <p><strong>Size:</strong> {activeFile.size} bytes</p>
      </div>
    </div>
  );
};

export default PdfViewer;
