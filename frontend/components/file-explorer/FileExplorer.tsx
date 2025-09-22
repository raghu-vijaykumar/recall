import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../../core/api.js';
import { File, Tab, MonacoEditor } from '../../core/types.js';
import { getFileIcon, getFileTypeFromName, getDefaultContentForFile, showModal, hideModal } from '../../shared/utils.js';

interface FileExplorerProps {
  currentWorkspaceId: number | null;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ currentWorkspaceId }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [openFiles, setOpenFiles] = useState<Tab[]>([]);
  const [activeFileId, setActiveFileId] = useState<number | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [fileName, setFileName] = useState('');
  const monacoEditorRef = useRef<MonacoEditor | null>(null);

  useEffect(() => {
    if (currentWorkspaceId) {
      loadFiles();
    }
  }, [currentWorkspaceId]);

  useEffect(() => {
    initializeMonacoEditor();
  }, []);

  const loadFiles = async () => {
    if (!currentWorkspaceId) return;

    try {
      const response = await fetch(`${API_BASE}/files/workspace/${currentWorkspaceId}`);
      const files = await response.json();
      setFiles(files);
    } catch (error) {
      console.error('Failed to load files:', error);
    }
  };

  const loadFileContent = async (fileId: number) => {
    try {
      const response = await fetch(`${API_BASE}/files/${fileId}/content`);
      const data = await response.json();

      if (monacoEditorRef.current) {
        monacoEditorRef.current.setValue(data.content || '');
      }
    } catch (error) {
      console.error('Failed to load file content:', error);
    }
  };

  const handleCreateFile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fileName.trim() || !currentWorkspaceId) return;

    const fileData = {
      name: fileName,
      path: fileName,
      file_type: getFileTypeFromName(fileName),
      size: 0,
      workspace_id: currentWorkspaceId,
      content: getDefaultContentForFile(fileName)
    };

    try {
      const response = await fetch(`${API_BASE}/files/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fileData)
      });

      if (response.ok) {
        setShowCreateModal(false);
        setFileName('');
        await loadFiles();
      } else {
        const error = await response.text();
        alert(`Failed to create file: ${error}`);
      }
    } catch (error) {
      console.error('Failed to create file:', error);
      alert('Failed to create file');
    }
  };

  const openFileInTab = async (file: File) => {
    const existingTab = openFiles.find(f => f.id === file.id);
    if (existingTab) {
      setActiveFileId(file.id);
      await loadFileContent(file.id);
      return;
    }

    const tab: Tab = {
      id: file.id,
      name: file.name,
      file: file,
      isActive: false
    };
    setOpenFiles(prev => [...prev, tab]);
    setActiveFileId(file.id);
    await loadFileContent(file.id);
  };

  const closeTab = (fileId: number) => {
    setOpenFiles(prev => prev.filter(f => f.id !== fileId));
    if (activeFileId === fileId) {
      const remainingTabs = openFiles.filter(f => f.id !== fileId);
      if (remainingTabs.length > 0) {
        setActiveFileId(remainingTabs[remainingTabs.length - 1].id);
      } else {
        setActiveFileId(null);
      }
    }
  };

  const saveCurrentFile = async () => {
    if (!currentWorkspaceId || !monacoEditorRef.current) return;

    const content = monacoEditorRef.current.getValue();
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `file_${timestamp}.txt`;

    const fileData = {
      name: filename,
      path: filename,
      file_type: getFileTypeFromName(filename),
      size: content.length,
      workspace_id: currentWorkspaceId,
      content: content
    };

    try {
      const response = await fetch(`${API_BASE}/files/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fileData)
      });

      if (response.ok) {
        alert(`File saved as: ${filename}`);
        await loadFiles();
      } else {
        const error = await response.text();
        alert(`Failed to save file: ${error}`);
      }
    } catch (error) {
      console.error('Failed to save file:', error);
      alert('Failed to save file');
    }
  };

  const scanCurrentWorkspaceFolder = async () => {
    if (!currentWorkspaceId) {
      alert('Please select a workspace first');
      return;
    }

    try {
      const workspaceResponse = await fetch(`${API_BASE}/workspaces/${currentWorkspaceId}`);
      if (workspaceResponse.ok) {
        const workspace = await workspaceResponse.json();

        if (!workspace.folder_path) {
          alert('This workspace is not linked to a folder. To scan files, first create a workspace from a folder using File → Open Folder.');
          return;
        }
      } else {
        alert('Could not verify workspace information. Please try again.');
        return;
      }
    } catch (error) {
      console.error('Failed to check workspace:', error);
      alert('Failed to check workspace information');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/files/scan-folder/${currentWorkspaceId}`, {
        method: 'POST'
      });

      if (response.ok) {
        const scanResult = await response.json();
        alert(`Folder scan completed!\nFound and added ${scanResult.files_created} new files.`);
        await loadFiles();
      } else {
        const error = await response.text();
        alert(`Failed to scan folder: ${error}`);
      }
    } catch (error) {
      console.error('Failed to scan folder:', error);
      alert('Failed to scan folder');
    }
  };

  const initializeMonacoEditor = () => {
    const checkMonaco = () => {
      if (typeof (window as any).require !== 'undefined') {
        (window as any).require.config({
          paths: {
            vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs'
          }
        });

        (window as any).require(['vs/editor/editor.main'], () => {
          const container = document.getElementById('monaco-editor');
          if (container) {
            monacoEditorRef.current = (window as any).monaco.editor.create(container, {
              value: '// Welcome to Recall\n// Start editing your files here',
              language: 'plaintext',
              theme: 'vs-dark',
              automaticLayout: true,
              fontSize: 14,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              wordWrap: 'on'
            });
            console.log('Monaco Editor initialized successfully');
          }
        });
      } else {
        setTimeout(checkMonaco, 100);
      }
    };

    checkMonaco();
  };

  return (
    <div id="files-tab" className="tab-content">
      <div className="files-container">
        <div className="sidebar">
          <div className="sidebar-header">
            <h3>EXPLORER</h3>
            <div className="sidebar-actions">
              <button
                className="sidebar-btn"
                title="Scan folder for new files"
                onClick={scanCurrentWorkspaceFolder}
              >
                🔄
              </button>
              <button
                className="sidebar-btn"
                title="New File"
                onClick={() => setShowCreateModal(true)}
              >
                ➕
              </button>
            </div>
          </div>
          <div className="file-tree">
            {files.map(file => (
              <div
                key={file.id}
                className={`file-item ${activeFileId === file.id ? 'active' : ''}`}
                onClick={() => openFileInTab(file)}
              >
                <span className="file-icon">{getFileIcon(file.name)}</span>
                <span className="file-name">{file.name}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="main-editor">
          <div className="tab-bar">
            {openFiles.map(tab => (
              <div
                key={tab.id}
                className={`tab ${activeFileId === tab.id ? 'active' : ''}`}
                onClick={() => setActiveFileId(tab.id)}
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

          <div className="editor-container">
            <div id="monaco-editor" className="monaco-editor"></div>
            {openFiles.length === 0 && (
              <div className="welcome-screen">
                <div className="welcome-content">
                  <h2>Welcome to Recall</h2>
                  <p>Select a file from the explorer to start editing</p>
                  <div className="welcome-actions">
                    <button
                      className="btn-primary"
                      onClick={() => setShowCreateModal(true)}
                    >
                      Create New File
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {showCreateModal && (
        <div className="modal">
          <div className="modal-content">
            <h3>Create New File</h3>
            <form onSubmit={handleCreateFile}>
              <input
                type="text"
                placeholder="filename.ext (e.g., script.py, notes.md)"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                required
              />
              <div className="modal-actions">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setShowCreateModal(false);
                    setFileName('');
                  }}
                >
                  Cancel
                </button>
                <button type="submit" className="btn-primary">Create</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileExplorer;
