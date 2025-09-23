import { useState, useEffect, useRef } from 'react';
import { API_BASE } from '../../../src/core/api';
import { File, Tab, MonacoEditor, FolderTreeNode, Workspace } from '../../../src/core/types';

export const useFileExplorerState = () => {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [folderTree, setFolderTree] = useState<FolderTreeNode[]>([]);
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  const [openFiles, setOpenFiles] = useState<Tab[]>([]);
  const [recentlyOpened, setRecentlyOpened] = useState<Tab[]>([]);
  const [activeFileId, setActiveFileId] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [activeView, setActiveView] = useState<'explorer' | 'search'>('explorer');
  const monacoEditorRef = useRef<MonacoEditor | null>(null);

  // File operations state
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; item: FolderTreeNode | null } | null>(null);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showMoveModal, setShowMoveModal] = useState(false);
  const [showCopyModal, setShowCopyModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [selectedItem, setSelectedItem] = useState<FolderTreeNode | null>(null);
  const [newName, setNewName] = useState('');
  const [copiedItemPath, setCopiedItemPath] = useState<string | null>(null);
  const [draggedItem, setDraggedItem] = useState<FolderTreeNode | null>(null);
  const [createBasePath, setCreateBasePath] = useState<string>('');

  return {
    workspace,
    setWorkspace,
    folderTree,
    setFolderTree,
    expandedDirs,
    setExpandedDirs,
    openFiles,
    setOpenFiles,
    recentlyOpened,
    setRecentlyOpened,
    activeFileId,
    setActiveFileId,
    searchQuery,
    setSearchQuery,
    searchResults,
    setSearchResults,
    isSearching,
    setIsSearching,
    activeView,
    setActiveView,
    monacoEditorRef,
    contextMenu,
    setContextMenu,
    showRenameModal,
    setShowRenameModal,
    showMoveModal,
    setShowMoveModal,
    showCopyModal,
    setShowCopyModal,
    showDeleteConfirm,
    setShowDeleteConfirm,
    selectedItem,
    setSelectedItem,
    newName,
    setNewName,
    copiedItemPath,
    setCopiedItemPath,
    draggedItem,
    setDraggedItem,
    createBasePath,
    setCreateBasePath,
  };
};

export const useWorkspaceLoading = (currentWorkspaceId: number | null, setWorkspace: (workspace: Workspace | null) => void) => {
  useEffect(() => {
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

    if (currentWorkspaceId) {
      loadWorkspace();
    }
  }, [currentWorkspaceId, setWorkspace]);
};

export const useFolderTreeLoading = (workspace: Workspace | null, setFolderTree: (tree: FolderTreeNode[]) => void) => {
  useEffect(() => {
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

    if (workspace?.folder_path) {
      loadFolderTree();
    }
  }, [workspace, setFolderTree]);
};

export const useLocalStoragePersistence = (
  setRecentlyOpened: (tabs: Tab[]) => void,
  setOpenFiles: (tabs: Tab[]) => void,
  setActiveFileId: (id: number | null) => void,
  openFiles: Tab[]
) => {
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
  }, [setRecentlyOpened, setOpenFiles, setActiveFileId]);

  useEffect(() => {
    try {
      localStorage.setItem('openFiles', JSON.stringify(openFiles));
    } catch (error) {
      console.error('Failed to save openFiles to localStorage:', error);
    }
  }, [openFiles]);
};

export const useMonacoEditor = (
  monacoEditorRef: React.MutableRefObject<MonacoEditor | null>,
  activeFileId: number | null,
  openFiles: Tab[]
) => {
  useEffect(() => {
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

    initializeMonacoEditor();
  }, [monacoEditorRef, activeFileId, openFiles]);
};

export const useFileSystemWatcher = (workspace: Workspace | null, loadFolderTree: () => void) => {
  useEffect(() => {
    const handleFileSystemChange = (event: any, data: any) => {
      console.log('File system changed:', data);
      loadFolderTree();
    };

    (window as any).electronAPI.onFileSystemChange(handleFileSystemChange);

    return () => {
      (window as any).electronAPI.offFileSystemChange(handleFileSystemChange);
    };
  }, [workspace, loadFolderTree]);
};

export const useWorkspaceNotification = (workspace: Workspace | null) => {
  useEffect(() => {
    if (workspace?.folder_path) {
      (window as any).electronAPI.notifyWorkspaceChanged(workspace.folder_path);
    }
  }, [workspace]);
};
