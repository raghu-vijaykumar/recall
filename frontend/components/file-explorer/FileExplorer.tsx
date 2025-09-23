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
  const [recentlyOpened, setRecentlyOpened] = useState<Tab[]>([]);
  const [activeFileId, setActiveFileId] = useState<number | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [fileName, setFileName] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [activeView, setActiveView] = useState<'explorer' | 'search'>('explorer');
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

  useEffect(() => {
    try {
      const saved = localStorage.getItem('recentlyOpened');
      if (saved) {
        try {
          setRecentlyOpened(JSON.parse(saved));
        } catch (error) {
          console.error('Failed to parse recentlyOpened from localStorage:', error);
        }
      }
    } catch (error) {
      console.error('Failed to access localStorage for recentlyOpened:', error);
    }

    try {
      const savedOpen = localStorage.getItem('openFiles');
      if (savedOpen) {
        try {
          const parsed = JSON.parse(savedOpen);
          setOpenFiles(parsed);
          if (parsed.length > 0) {
            setActiveFileId(parsed[parsed.length - 1].id);
          }
        } catch (error) {
          console.error('Failed to parse openFiles from localStorage:', error);
        }
      }
    } catch (error) {
      console.error('Failed to access localStorage for openFiles:', error);
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('openFiles', JSON.stringify(openFiles));
    } catch (error) {
      console.error('Failed to save openFiles to localStorage:', error);
    }
  }, [openFiles]);

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
    const remainingTabs = openFiles.filter(f => f.id !== fileId);
    setOpenFiles(remainingTabs);
    if (activeFileId === fileId) {
      if (remainingTabs.length > 0) {
        const newActiveId = remainingTabs[remainingTabs.length - 1].id;
        setActiveFileId(newActiveId);
        if (monacoEditorRef.current) {
          const newActiveTab = remainingTabs.find(t => t.id === newActiveId);
          if (newActiveTab) {
            monacoEditorRef.current.setValue(newActiveTab.file.content || '');
          }
        }
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



  const filterTreeBySearch = (nodes: FolderTreeNode[], query: string): FolderTreeNode[] => {
    if (!query.trim()) return nodes;

    const lowerQuery = query.toLowerCase();

    return nodes.reduce((filtered: FolderTreeNode[], node) => {
      const matchesName = node.name.toLowerCase().includes(lowerQuery);

      if (node.type === 'directory' && node.children) {
        const filteredChildren = filterTreeBySearch(node.children, query);
        if (matchesName || filteredChildren.length > 0) {
          filtered.push({
            ...node,
            children: filteredChildren
          });
        }
      } else if (matchesName) {
        filtered.push(node);
      }

      return filtered;
    }, []);
  };

  const performContentSearch = async () => {
    if (!searchQuery.trim() || !workspace?.folder_path) return;

    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE}/search/content`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace_id: currentWorkspaceId,
          query: searchQuery,
          folder_path: workspace.folder_path
        })
      });

      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
      } else {
        console.error('Search failed:', response.statusText);
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const renderSearchResults = () => {
    // Group results by directory path
    const groupedResults = searchResults.reduce((groups: any, result: any) => {
      const pathParts = result.path.split('/');
      const dirPath = pathParts.slice(0, -1).join('/') || '/';

      if (!groups[dirPath]) {
        groups[dirPath] = [];
      }
      groups[dirPath].push(result);
      return groups;
    }, {});

    return Object.entries(groupedResults).map(([dirPath, files]: [string, any]) => (
      <div key={dirPath} className="search-group">
        <div className="search-group-header">
          <span className="search-group-path">{dirPath === '/' ? 'Root' : dirPath}</span>
        </div>
        <div className="search-group-files">
          {files.map((result: any) => (
            <div
              key={result.path}
              className="search-result-item"
              onClick={() => openSearchResult(result)}
            >
              <div className="search-result-header">
                <span className="search-result-icon">{getFileIcon(result.name)}</span>
                <span className="search-result-name">{result.name}</span>
              </div>
              {result.matches && result.matches.length > 0 && (
                <div className="search-result-matches">
                  {result.matches.slice(0, 3).map((match: any, index: number) => (
                    <div key={index} className="search-match">
                      <span className="search-match-line">Line {match.line}:</span>
                      <span className="search-match-text">{match.text}</span>
                    </div>
                  ))}
                  {result.matches.length > 3 && (
                    <div className="search-match-more">
                      ... and {result.matches.length - 3} more matches
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    ));
  };

  const openSearchResult = async (result: any) => {
    if (!workspace?.folder_path) return;

    const fullPath = `${workspace.folder_path}/${result.path}`;

    try {
      const content = await (window as any).electronAPI.readFileContent(fullPath);

      const tempFile: File = {
        id: Date.now(),
        name: result.name,
        path: result.path,
        file_type: getFileTypeFromName(result.name),
        size: content.length,
        workspace_id: currentWorkspaceId!,
        content: content
      };

      const existingTab = openFiles.find(f => f.file.path === result.path);
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
      console.error('Failed to open search result:', error);
      alert('Failed to open file');
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

            // Set content for active file if loaded from localStorage
            if (activeFileId) {
              const activeTab = openFiles.find(t => t.id === activeFileId);
              if (activeTab && monacoEditorRef.current) {
                monacoEditorRef.current.setValue(activeTab.file.content || '');
              }
            }
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
          <div className="icon-bar">
            <button
              className={`icon-btn ${activeView === 'explorer' ? 'active' : ''}`}
              onClick={() => setActiveView('explorer')}
              title="File Explorer"
            >
              📁
            </button>
            <button
              className={`icon-btn ${activeView === 'search' ? 'active' : ''}`}
              onClick={() => setActiveView('search')}
              title="Search Files"
            >
              🔍
            </button>
          </div>
          <div className="sidebar-header">
            <h3>{activeView === 'explorer' ? 'EXPLORER' : 'SEARCH'}</h3>
            <div className="sidebar-actions">
              {activeView === 'explorer' && (
                <button
                  className="sidebar-btn"
                  title="Refresh folder tree"
                  onClick={() => loadFolderTree()}
                >
                  🔄
                </button>
              )}
            </div>
          </div>
          <div className="search-container">
            <input
              type="text"
              placeholder="Search files..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            {searchQuery && (
              <button
                className="clear-search-btn"
                onClick={() => setSearchQuery('')}
                title="Clear search"
              >
                ×
              </button>
            )}
          </div>
          {activeView === 'explorer' ? (
            <div className="file-tree">
              {workspace?.folder_path ? (
                folderTree.length > 0 ? (
                  (() => {
                    const filteredTree = searchQuery ? filterTreeBySearch(folderTree, searchQuery) : folderTree;
                    return filteredTree.length > 0 ? (
                      filteredTree.map(node => renderTreeNode(node))
                    ) : (
                      <div className="empty-tree">
                        <p>No files match your search</p>
                        <button
                          className="btn-secondary"
                          onClick={() => setSearchQuery('')}
                        >
                          Clear search
                        </button>
                      </div>
                    );
                  })()
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
          ) : (
            <div className="search-results">
              <div className="search-input-container">
                <input
                  type="text"
                  placeholder="Search file contents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="search-input"
                  onKeyPress={(e) => e.key === 'Enter' && performContentSearch()}
                />
                <button
                  className="search-btn"
                  onClick={performContentSearch}
                  disabled={isSearching || !searchQuery.trim()}
                >
                  {isSearching ? '🔄' : '🔍'}
                </button>
                {searchQuery && (
                  <button
                    className="clear-search-btn"
                    onClick={() => {
                      setSearchQuery('');
                      setSearchResults([]);
                    }}
                    title="Clear search"
                  >
                    ×
                  </button>
                )}
              </div>
              <div className="search-results-list">
                {searchResults.length > 0 ? (
                  renderSearchResults()
                ) : searchQuery && !isSearching ? (
                  <div className="empty-search">
                    <p>No files found containing "{searchQuery}"</p>
                  </div>
                ) : isSearching ? (
                  <div className="searching">
                    <p>Searching...</p>
                  </div>
                ) : (
                  <div className="empty-search">
                    <p>Enter a search term to find files by content</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="main-editor">
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
