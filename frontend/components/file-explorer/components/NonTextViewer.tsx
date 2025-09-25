import React from 'react';
import { File } from '../../../src/core/types';
import ImageViewer from './ImageViewer';
import PdfViewer from './PdfViewer';
import VideoViewer from './VideoViewer';
import AudioViewer from './AudioViewer';

interface NonTextViewerProps {
  activeFile: File;
}

const NonTextViewer: React.FC<NonTextViewerProps> = ({ activeFile }) => {
  const isImage = /\.(jpg|jpeg|png|gif|bmp|webp|svg)$/i.test(activeFile.name);
  const isPdf = /\.pdf$/i.test(activeFile.name);
  const isVideo = /\.(mp4|avi|mov|mkv|webm|flv|wmv|mpg|mpeg|3gp|m4v)$/i.test(activeFile.name);
  const isAudio = /\.(mp3|wav|ogg|flac|aac|m4a|wma|aiff|au)$/i.test(activeFile.name);

  if (isImage) {
    return <ImageViewer activeFile={activeFile} />;
  }

  if (isPdf) {
    return <PdfViewer activeFile={activeFile} />;
  }

  if (isVideo) {
    return <VideoViewer activeFile={activeFile} />;
  }

  if (isAudio) {
    return <AudioViewer activeFile={activeFile} />;
  }

  return (
    <div className="unsupported-file-viewer">
      <div className="unsupported-content">
        <h3>Unsupported File Type</h3>
        <p>This file type is not supported for viewing.</p>
        <p>File: {activeFile.name}</p>
        <p>Type: {activeFile.file_type}</p>
      </div>
    </div>
  );
};

export default NonTextViewer;
