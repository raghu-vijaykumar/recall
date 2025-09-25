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
  // Group files by name to detect duplicates
  const nameCounts = openFiles.reduce((acc, tab) => {
    acc[tab.name] = (acc[tab.name] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const getDisplayName = (tab: Tab) => {
    if (nameCounts[tab.name] > 1) {
      // For duplicate names, show parent directory
      const pathParts = tab.file.path.split('/');
      const parentDir = pathParts.length > 1 ? pathParts[pathParts.length - 2] : '';
      return `${tab.name} (${parentDir})`;
    }
    return tab.name;
  };

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
          <span className="tab-name" title={tab.file.path}>{getDisplayName(tab)}</span>
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
