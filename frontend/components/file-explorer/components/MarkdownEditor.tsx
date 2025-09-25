import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Tab, MonacoEditor } from '../../../src/core/types';
import RichMarkdownRenderer from '../../../src/shared/RichMarkdownRenderer';

interface MarkdownEditorProps {
  activeFile: Tab | null;
  monacoEditorRef: React.MutableRefObject<MonacoEditor | null>;
}

const MarkdownEditor: React.FC<MarkdownEditorProps> = ({ activeFile, monacoEditorRef }) => {
  const [markdownContent, setMarkdownContent] = useState('');
  const [splitRatio, setSplitRatio] = useState(50); // Percentage for left panel
  const [isDragging, setIsDragging] = useState(false);
  const [isPreviewOnly, setIsPreviewOnly] = useState(false);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const localMonacoRef = useRef<MonacoEditor | null>(null);

  // Initialize Monaco editor for this markdown component
  useEffect(() => {
    const initializeLocalMonaco = () => {
      const checkMonaco = () => {
        if (typeof (window as any).require !== 'undefined') {
          (window as any).require.config({
            paths: {
              vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs'
            }
          });

          (window as any).require(['vs/editor/editor.main'], () => {
            if (editorContainerRef.current && !localMonacoRef.current) {
              localMonacoRef.current = (window as any).monaco.editor.create(editorContainerRef.current, {
                value: activeFile?.file?.content || '',
                language: 'markdown',
                theme: 'vs-dark',
                automaticLayout: true,
                fontSize: 14,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                wordWrap: 'on'
              });

              // Sync content changes to preview
              const editor = (localMonacoRef.current as any);
              if (editor && editor.onDidChangeModelContent) {
                editor.onDidChangeModelContent(() => {
                  if (localMonacoRef.current) {
                    const content = localMonacoRef.current.getValue();
                    setMarkdownContent(content);
                  }
                });
              }
            }
          });
        } else {
          setTimeout(checkMonaco, 100);
        }
      };

      checkMonaco();
    };

    initializeLocalMonaco();

    return () => {
      if (localMonacoRef.current) {
        (localMonacoRef.current as any).dispose();
        localMonacoRef.current = null;
      }
    };
  }, []);

  // Update content when active file changes
  useEffect(() => {
    if (activeFile?.file?.content) {
      setMarkdownContent(activeFile.file.content);
      if (localMonacoRef.current) {
        localMonacoRef.current.setValue(activeFile.file.content);
      }
    } else {
      setMarkdownContent('');
      if (localMonacoRef.current) {
        localMonacoRef.current.setValue('');
      }
    }
  }, [activeFile]);

  // Handle mouse drag for splitter
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    e.preventDefault();
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !splitContainerRef.current) return;

    const container = splitContainerRef.current;
    const rect = container.getBoundingClientRect();
    const newRatio = ((e.clientX - rect.left) / rect.width) * 100;

    // Constrain ratio between 20% and 80%
    const constrainedRatio = Math.max(20, Math.min(80, newRatio));
    setSplitRatio(constrainedRatio);
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Add global mouse event listeners
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  if (!activeFile) {
    return (
      <div className="markdown-editor-empty">
        <div className="welcome-content">
          <h2>Select a Markdown File</h2>
          <p>Open a .md or .markdown file to start editing with rich preview</p>
        </div>
      </div>
    );
  }

  return (
    <div className="markdown-editor">
      {/* Editor Toolbar */}
      <div className="markdown-toolbar">
        <div className="toolbar-left">
          <span className="file-info">
            {activeFile.name}
          </span>
        </div>
        <div className="toolbar-right">
          {!isPreviewOnly && (
            <span className="split-info">
              {Math.round(splitRatio)}% • {Math.round(100 - splitRatio)}%
            </span>
          )}
          <button
            className={`mode-toggle ${isPreviewOnly ? 'active' : ''}`}
            onClick={() => setIsPreviewOnly(!isPreviewOnly)}
            title={isPreviewOnly ? "Switch to Edit Mode" : "Switch to Preview Only"}
          >
            {isPreviewOnly ? '✏️ Edit' : '👁️ Preview'}
          </button>
        </div>
      </div>

      {/* Content Area */}
      {isPreviewOnly ? (
        <div className="markdown-preview-only">
          <div className="markdown-preview">
            <RichMarkdownRenderer
              content={markdownContent}
              className="markdown-preview-content"
            />
          </div>
        </div>
      ) : (
        <div
          ref={splitContainerRef}
          className="markdown-split-container"
          style={{ cursor: isDragging ? 'col-resize' : 'default' }}
        >
          {/* Left Panel - Monaco Editor */}
          <div
            className="markdown-editor-panel"
            style={{ width: `${splitRatio}%` }}
          >
            <div
              ref={editorContainerRef}
              className="monaco-editor"
              style={{ width: '100%', height: '100%' }}
            />
          </div>

          {/* Splitter */}
          <div
            className="markdown-splitter"
            onMouseDown={handleMouseDown}
          >
            <div className="splitter-handle"></div>
          </div>

          {/* Right Panel - Preview */}
          <div
            className="markdown-preview-panel"
            style={{ width: `${100 - splitRatio}%` }}
          >
            <div className="markdown-preview">
              <RichMarkdownRenderer
                content={markdownContent}
                className="markdown-preview-content"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MarkdownEditor;
