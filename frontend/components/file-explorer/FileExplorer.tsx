import React, { useState } from 'react';
import {
  useFileExplorerState,
  useWorkspaceLoading,
  useFolderTreeLoading,
  useLocalStoragePersistence,
  useMonacoEditor,
  useFileSystemWatcher,
  useWorkspaceNotification,
} from './hooks/fileExplorerHooks.js';
import { useFileOperations } from './hooks/useFileOperations.js';
import { useContextMenu } from './hooks/useContextMenu.js';
import { useDragAndDrop } from './hooks/useDragAndDrop.js';
import { useFileManagement } from './hooks/useFileManagement.js';
import { filterTreeBySearch } from './utils/fileExplorerUtils.js';
import FileTree from './components/FileTree';
import FileSearch from './components/FileSearch';
import EditorTabs from './components/EditorTabs';
import FileOperationModals from './components/FileOperationModals';
import FileContextMenu from './components/FileContextMenu';
import Progress from '../progress/Progress';
import Quiz from '../quiz/Quiz';
import { KnowledgeGraph } from '../knowledge-graph/KnowledgeGraph';
import Chat from '../chat/Chat';
import { useTheme } from '../../src/core/ThemeContext';

interface FileExplorerProps {
  currentWorkspaceId: number | null;
}

const FileExplorer: React.FC<FileExplorerProps> = ({ currentWorkspaceId }) => {
  const { isDark } = useTheme();
  const {
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
  } = useFileExplorerState();


  const [showCreateModal, setShowCreateModal] = useState(false);
  const [fileName, setFileName] = useState('');

  // Custom hooks
  useWorkspaceLoading(currentWorkspaceId, setWorkspace);
  useFolderTreeLoading(workspace, setFolderTree);
  useLocalStoragePersistence(setRecentlyOpened, setOpenFiles, setActiveFileId, openFiles);
  useMonacoEditor(monacoEditorRef, activeFileId, openFiles);
  useFileSystemWatcher(workspace, () => loadFolderTree());
  useWorkspaceNotification(workspace);

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

  // Use hooks for file operations, context menu, drag and drop, and file management
  const { handleCreateFile, handleCreateDirectory, handleRename, handleMove, handleCopy, handleDelete } = useFileOperations(
    workspace,
    selectedItem,
    newName,
    fileName,
    createBasePath,
    setShowCreateModal,
    setShowRenameModal,
    setShowMoveModal,
    setShowCopyModal,
    setShowDeleteConfirm,
    setFileName,
    setNewName,
    setSelectedItem,
    setCreateBasePath,
    loadFolderTree
  );

  const { handleContextMenu, handleContextMenuAction } = useContextMenu(
    contextMenu,
    workspace,
    copiedItemPath,
    setContextMenu,
    setSelectedItem,
    setCreateBasePath,
    setShowCreateModal,
    setNewName,
    setShowRenameModal,
    setShowMoveModal,
    setCopiedItemPath,
    setShowDeleteConfirm,
    handleCreateDirectory,
    loadFolderTree
  );

  const { handleDragStart, handleDragOver, handleDrop } = useDragAndDrop(
    draggedItem,
    workspace,
    setDraggedItem,
    loadFolderTree
  );

  const { openFile, closeTab, openSearchResult } = useFileManagement(
    workspace,
    currentWorkspaceId,
    openFiles,
    activeFileId,
    monacoEditorRef,
    setOpenFiles,
    setActiveFileId
  );







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
              title="Search"
            >
              🔍
            </button>
            <button
              className={`icon-btn ${activeView === 'progress' ? 'active' : ''}`}
              onClick={() => setActiveView('progress')}
              title="Progress"
            >
              📊
            </button>
            <button
              className={`icon-btn ${activeView === 'knowledge-graph' ? 'active' : ''}`}
              onClick={() => setActiveView('knowledge-graph')}
              title="Knowledge Graph"
            >
              🧠
            </button>
            <button
              className={`icon-btn ${activeView === 'quiz' ? 'active' : ''}`}
              onClick={() => setActiveView('quiz')}
              title="Quiz"
            >
              ❓
            </button>
            <button
              className={`icon-btn ${activeView === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveView('chat')}
              title="AI Chat"
            >
              💬
            </button>
          </div>
          <div className="sidebar-header">
            <h3>
              {activeView === 'explorer' ? 'EXPLORER' :
               activeView === 'search' ? 'SEARCH' :
               activeView === 'progress' ? 'PROGRESS' :
               activeView === 'knowledge-graph' ? 'KNOWLEDGE GRAPH' :
               activeView === 'quiz' ? 'QUIZ' :
               activeView === 'chat' ? 'AI CHAT' : 'EXPLORER'}
            </h3>
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
          {activeView === 'explorer' && (
            <>
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
              <FileTree
                folderTree={folderTree}
                expandedDirs={expandedDirs}
                openFiles={openFiles}
                draggedItem={draggedItem}
                searchQuery={searchQuery}
                toggleDirectory={toggleDirectory}
                openFile={openFile}
                handleContextMenu={handleContextMenu}
                handleDragStart={handleDragStart}
                handleDragOver={handleDragOver}
                handleDrop={handleDrop}
                filterTreeBySearch={filterTreeBySearch}
              />
            </>
          )}
          {activeView === 'search' && (
            <>
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
              <FileSearch
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                searchResults={searchResults}
                setSearchResults={setSearchResults}
                isSearching={isSearching}
                setIsSearching={setIsSearching}
                currentWorkspaceId={currentWorkspaceId}
                workspace={workspace}
                openSearchResult={openSearchResult}
              />
            </>
          )}
          {activeView === 'progress' && (
            <div className="sidebar-content">
              <Progress currentWorkspaceId={currentWorkspaceId} />
            </div>
          )}
          {activeView === 'knowledge-graph' && (
            <div className="sidebar-content">
              {currentWorkspaceId ? (
                <KnowledgeGraph workspaceId={currentWorkspaceId} />
              ) : (
                <div className="placeholder-content">
                  <h4>Please select a workspace</h4>
                  <p>The Knowledge Graph requires a workspace to be selected.</p>
                </div>
              )}
            </div>
          )}
          {activeView === 'quiz' && (
            <div className="sidebar-content">
              <Quiz currentWorkspaceId={currentWorkspaceId} />
            </div>
          )}
          {activeView === 'chat' && (
            <div className="sidebar-content">
              <Chat currentWorkspaceId={currentWorkspaceId} />
            </div>
          )}
        </div>

        <div className="main-editor">
          <EditorTabs
            openFiles={openFiles}
            activeFileId={activeFileId}
            setActiveFileId={setActiveFileId}
            closeTab={closeTab}
            monacoEditorRef={monacoEditorRef}
          />

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

      <FileOperationModals
        showCreateModal={showCreateModal}
        showRenameModal={showRenameModal}
        showMoveModal={showMoveModal}
        showCopyModal={showCopyModal}
        showDeleteConfirm={showDeleteConfirm}
        fileName={fileName}
        newName={newName}
        selectedItem={selectedItem}
        createBasePath={createBasePath}
        setFileName={setFileName}
        setNewName={setNewName}
        setSelectedItem={setSelectedItem}
        setCreateBasePath={setCreateBasePath}
        setShowCreateModal={setShowCreateModal}
        setShowRenameModal={setShowRenameModal}
        setShowMoveModal={setShowMoveModal}
        setShowCopyModal={setShowCopyModal}
        setShowDeleteConfirm={setShowDeleteConfirm}
        handleCreateFile={handleCreateFile}
        handleRename={handleRename}
        handleMove={handleMove}
        handleCopy={handleCopy}
        handleDelete={handleDelete}
      />

      <FileContextMenu
        contextMenu={contextMenu}
        copiedItemPath={copiedItemPath}
        handleContextMenuAction={handleContextMenuAction}
        setContextMenu={setContextMenu}
      />
    </div>
  );
};

export default FileExplorer;
