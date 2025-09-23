import React from 'react';
import { Tab, MonacoEditor } from '../../../src/core/types';
import { getFileIcon } from '../../../src/shared/utils';

interface EditorTabsProps {
  openFiles: Tab[];
  activeFileId: number | null;
  setActiveFileId: (id: number | null) => void;
  closeTab: (fileId: number) => void;
  monacoEditorRef: React.MutableRefObject<MonacoEditor | null>;
}

const EditorTabs: React.FC<EditorTabsProps> = ({
  openFiles,
  activeFileId,
  setActiveFileId,
  closeTab,
  monacoEditorRef,
}) => {
  return (
    <div className="tab-bar">
      {openFiles.map(tab => (
        <div
          key={tab.id}
          className={`tab ${activeFileId === tab.id ? 'active' : ''}`}
          onClick={() => {
            setActiveFileId(tab.id);
            if (monacoEditorRef.current) {
              monacoEditorRef.current.setValue(tab.file.content || '');
            }
          }}
        >
          <span className="tab-icon">{getFileIcon(tab.name)}</span>
          <span className="tab-name">{tab.name}</span>
          <button
            className="tab-close"
            onClick={(e) => {
              e.stopPropagation();
              closeTab(tab.id);
            }}
            title="Close (Ctrl+W)"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
};

export default EditorTabs;
