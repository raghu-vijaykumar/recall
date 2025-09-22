import React, { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../../core/api.js';
import { File, Tab, MonacoEditor, FolderTreeNode, Workspace } from '../../core/types.js';
import { getFileIcon, getFileTypeFromName, getDefaultContentForFile, showModal, hideModal } from '../../shared/utils.js';

interface FileExplorerProps {
  currentWorkspaceId: number | null;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ currentWorkspaceId }) => {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [folderTree, setFolderTree] = useState<FolderTreeNode[]>([]);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [openFiles, setOpenFiles] = useState<Tab[]>([]);
  const [activeFileId, setActiveFileId] = useState<number | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [fileName, setFileName] = useState('');
  const monacoEditorRef = useRef<MonacoEditor | null>(null);

  useEffect(() => {
    if (currentWorkspaceId) {
      loadWorkspace();
    }
  }, [currentWorkspaceId]);

  useEffect(() => {
    if (workspace?.folder_path) {
      loadFolderTree();
    }
  }, [workspace]);

  useEffect(() => {
    initializeMonacoEditor();
  }, []);

  const loadWorkspace = async () => {
    if (!currentWorkspaceId) return;

    console.log('Loading workspace:', currentWorkspaceId);
    try {
      const response = await fetch(`${API_BASE}/workspaces/${currentWorkspaceId}`);
      const workspaceData = await response.json();
      console.log('Workspace data:', workspaceData);
      setWorkspace(workspaceData);
    } catch (error) {
      console.error('Failed to load workspace:', error);
    }
  };

  const loadFolderTree = async () => {
    if (!workspace?.folder_path) {
      console.log('No folder_path in workspace:', workspace);
      return;
    }

    console.log('Loading folder tree for:', workspace.folder_path);
    try {
      const tree = await (window as any).electronAPI.getFolderTree(workspace.folder_path);
      console.log('Folder tree loaded:', tree);
      setFolderTree(tree);
    } catch (error) {
      console.error('Failed to load folder tree:', error);
    }
  };

  const toggleDirectory = (path: string) => {
    setExpandedDirs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const openFile = async (node: FolderTreeNode) => {
    if (!workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${node.path}`;

    try {
      const content = await (window as any).electronAPI.readFileContent(fullPath);

      // Create a temporary file object for the tab
      const tempFile: File = {
        id: Date.now(), // Temporary ID
        name: node.name,
        path: node.path,
        file_type: getFileTypeFromName(node.name),
        size: content.length,
        workspace_id: currentWorkspaceId!,
        content: content
      };

      const existingTab = openFiles.find(f => f.file.path === node.path);
      if (existingTab) {
        setActiveFileId(existingTab.id);
        if (monacoEditorRef.current) {
          monacoEditorRef.current.setValue(content);
        }
        return;
      }

      const tab: Tab = {
        id: tempFile.id,
        name: tempFile.name,
        file: tempFile,
        isActive: false
      };

      setOpenFiles(prev => [...prev, tab]);
      setActiveFileId(tempFile.id);

      if (monacoEditorRef.current) {
        monacoEditorRef.current.setValue(content);
      }
    } catch (error) {
      console.error('Failed to open file:', error);
      alert('Failed to open file');
    }
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
        // Refresh folder tree after creating file
        if (workspace?.folder_path) {
          loadFolderTree();
        }
      } else {
        const error = await response.text();
        alert(`Failed to create file: ${error}`);
      }
    } catch (error) {
      console.error('Failed to create file:', error);
      alert('Failed to create file');
    }
  };

  const scanCurrentWorkspaceFolder = async (showAlerts: boolean = true) => {
    if (!currentWorkspaceId) {
      if (showAlerts) alert('Please select a workspace first');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/files/scan-folder/${currentWorkspaceId}`, {
        method: 'POST'
      });

      if (response.ok) {
        const scanResult = await response.json();
        if (showAlerts) alert(`Folder scan completed!\nFound and added ${scanResult.files_created} new files.`);
        // Refresh folder tree after scanning
        if (workspace?.folder_path) {
          loadFolderTree();
        }
      } else {
        const error = await response.text();
        if (showAlerts) alert(`Failed to scan folder: ${error}`);
      }
    } catch (error) {
      console.error('Failed to scan folder:', error);
      if (showAlerts) alert('Failed to scan folder');
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

  const renderTreeNode = (node: FolderTreeNode, level: number = 0): React.ReactNode => {
    const isExpanded = expandedDirs.has(node.path);
    const isActive = openFiles.some(tab => tab.file.path === node.path);

    return (
      <div key={node.path}>
        <div
          className={`tree-item ${isActive ? 'active' : ''}`}
          style={{ paddingLeft: `${level * 16 + 8}px` }}
          onClick={() => node.type === 'directory' ? toggleDirectory(node.path) : openFile(node)}
        >
          {node.type === 'directory' ? (
            <span className="tree-icon">
              {isExpanded ? '📂' : '📁'}
            </span>
          ) : (
            <span className="tree-icon">{getFileIcon(node.name)}</span>
          )}
          <span className="tree-name">{node.name}</span>
        </div>
        {node.type === 'directory' && isExpanded && node.children && (
          <div>
            {node.children.map(child => renderTreeNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div id="files-tab" className="tab-content active">
      <div className="files-container">
        <div className="sidebar">
          <div className="sidebar-header">
            <h3>EXPLORER</h3>
            <div className="sidebar-actions">
              <button
                className="sidebar-btn"
                title="Refresh folder tree"
                onClick={() => loadFolderTree()}
              >
                🔄
              </button>
              <button
                className="sidebar-btn"
                title="Scan folder for new files"
                onClick={() => scanCurrentWorkspaceFolder(true)}
              >
                📂
              </button>
            </div>
          </div>
          <div className="file-tree">
            {workspace?.folder_path ? (
              folderTree.length > 0 ? (
                folderTree.map(node => renderTreeNode(node))
              ) : (
                <div className="empty-tree">
                  <p>No files found in workspace folder</p>
                  <button
                    className="btn-secondary"
                    onClick={() => loadFolderTree()}
                  >
                    Refresh
                  </button>
                </div>
              )
            ) : (
              <div className="empty-tree">
                <p>No folder linked to this workspace</p>
                <p>Use "File → Open Folder" to link a folder</p>
              </div>
            )}
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
