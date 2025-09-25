import React, { useState, useEffect } from 'react';
import { File } from '../../../src/core/types';

interface ImageViewerProps {
  activeFile: File;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ activeFile }) => {
  const [imageData, setImageData] = useState<{ base64: string; mimeType: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imageRef = React.useRef<HTMLImageElement>(null);

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  };

  const handleFitToWindow = () => {
    if (imageSize.width && imageSize.height) {
      const container = document.querySelector('.image-container') as HTMLElement;
      if (container) {
        const containerRect = container.getBoundingClientRect();
        const scaleX = containerRect.width / imageSize.width;
        const scaleY = containerRect.height / imageSize.height;
        const fitScale = Math.min(scaleX, scaleY, 1); // Don't scale up smaller images
        setZoom(fitScale);
      }
    }
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    // Auto-fit on first load
    setTimeout(handleFitToWindow, 100);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (e.deltaY < 0) {
      handleZoomIn();
    } else {
      handleZoomOut();
    }
  };

  useEffect(() => {
    setLoading(true);
    setError(null);
    setZoom(1); // Reset zoom when loading new image
    (window as any).electronAPI.readFileBase64(activeFile.path)
      .then((data: { base64: string; mimeType: string }) => {
        setImageData(data);
        setLoading(false);
      })
      .catch((err: any) => {
        console.error('Failed to load image:', err);
        setError('Failed to load image');
        setLoading(false);
      });
  }, [activeFile.path]);

  if (loading) {
    return (
      <div className="image-viewer">
        <div className="image-container">
          <div className="loading">Loading image...</div>
        </div>
        <div className="file-info">
          <p><strong>File:</strong> {activeFile.name}</p>
          <p><strong>Path:</strong> {activeFile.path}</p>
          <p><strong>Size:</strong> {activeFile.size} bytes</p>
        </div>
      </div>
    );
  }

  if (error || !imageData) {
    return (
      <div className="image-viewer">
        <div className="image-container">
          <div className="error-content">
            <h3>Failed to Load Image</h3>
            <p>{error || 'Could not display the image file.'}</p>
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
    <div className="image-viewer">
      <div className="image-controls">
        <button onClick={handleZoomOut} title="Zoom Out">🔍-</button>
        <button onClick={handleFitToWindow} title="Fit to Window">📐</button>
        <button onClick={handleZoomIn} title="Zoom In">🔍+</button>
        <span className="zoom-level">{Math.round(zoom * 100)}%</span>
      </div>
      <div
        className="image-container scrollable"
        onWheel={handleWheel}
      >
        <img
          ref={imageRef}
          src={`data:${imageData.mimeType};base64,${imageData.base64}`}
          alt={activeFile.name}
          className="viewer-image"
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: 'top left',
          }}
          onLoad={handleImageLoad}
        />
      </div>
      <div className="file-info">
        <p><strong>File:</strong> {activeFile.name}</p>
        <p><strong>Path:</strong> {activeFile.path}</p>
        <p><strong>Size:</strong> {activeFile.size} bytes</p>
        <p><strong>Dimensions:</strong> {imageSize.width} × {imageSize.height}</p>
      </div>
    </div>
  );
};

export default ImageViewer;
