/// <reference types="../../electron/preload" />
import React, { useState, useEffect } from 'react';
import Workspaces from '../components/workspaces/Workspaces';
// TODO: Import other components when converted
// import FileExplorer from './components/file-explorer/FileExplorer';
// import Quiz from './components/quiz/Quiz';
// import Progress from './components/progress/Progress';

const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState('workspaces');
  const [currentWorkspaceId, setCurrentWorkspaceId] = useState<number | null>(null);
  const [showCreateWorkspaceModal, setShowCreateWorkspaceModal] = useState(false);
  const [folderToCreateWorkspace, setFolderToCreateWorkspace] = useState<{ path: string, name: string } | null>(null);

  useEffect(() => {
    // Listen for tab switching events
    const handleShowTab = (e: any) => {
      setCurrentTab(e.detail.tab);
    };

    // Listen for workspace selection events
    const handleWorkspaceSelected = (e: any) => {
      setCurrentWorkspaceId(e.detail.workspaceId);
      setCurrentTab('files'); // Switch to files tab
    };

    // Listen for Electron menu events
    if (window.menuEvents) {
      window.menuEvents.on('menu-create-workspace', () => {
        setShowCreateWorkspaceModal(true);
      });
      window.menuEvents.on('menu-show-workspaces', () => setCurrentTab('workspaces'));
      window.menuEvents.on('menu-show-files', () => setCurrentTab('files'));
      window.menuEvents.on('menu-show-quiz', () => setCurrentTab('quiz'));
      window.menuEvents.on('menu-show-progress', () => setCurrentTab('progress'));
      window.menuEvents.on('menu-toggle-theme', () => {
        // This event is handled by the main process, but we might want to
        // update the UI if the theme changes. For now, just log.
        console.log('Theme toggle event received in renderer.');
      });
      window.menuEvents.on('folder-selected', (folderInfo: { path: string, name: string }) => {
        console.log('Folder selected:', folderInfo);
        setFolderToCreateWorkspace(folderInfo);
        setCurrentTab('workspaces'); // Switch to workspaces tab to show creation
      });
    }


    window.addEventListener('show-tab', handleShowTab);
    window.addEventListener('workspace-selected', handleWorkspaceSelected);

    return () => {
      window.removeEventListener('show-tab', handleShowTab);
      window.removeEventListener('workspace-selected', handleWorkspaceSelected);
      if (window.menuEvents) {
        window.menuEvents.off('menu-create-workspace', () => {
          setShowCreateWorkspaceModal(true);
        });
        window.menuEvents.off('menu-show-workspaces', () => setCurrentTab('workspaces'));
        window.menuEvents.off('menu-show-files', () => setCurrentTab('files'));
        window.menuEvents.off('menu-show-quiz', () => setCurrentTab('quiz'));
        window.menuEvents.off('menu-show-progress', () => setCurrentTab('progress'));
        window.menuEvents.off('menu-toggle-theme', () => {
          console.log('Theme toggle event received in renderer.');
        });
        window.menuEvents.off('folder-selected', (folderInfo: { path: string, name: string }) => {
          console.log('Folder selected:', folderInfo);
        });
      }
    };
  }, []);

  const renderTabContent = () => {
    switch (currentTab) {
      case 'workspaces':
        return <Workspaces
          showCreateModal={showCreateWorkspaceModal}
          setShowCreateModal={setShowCreateWorkspaceModal}
          folderToCreate={folderToCreateWorkspace}
          setFolderToCreate={setFolderToCreateWorkspace}
        />;
      case 'files':
        return <div>Files Tab - Coming Soon</div>; // TODO: FileExplorer component
      case 'quiz':
        return <div>Quiz Tab - Coming Soon</div>; // TODO: Quiz component
      case 'progress':
        return <div>Progress Tab - Coming Soon</div>; // TODO: Progress component
      default:
        return <Workspaces
          showCreateModal={showCreateWorkspaceModal}
          setShowCreateModal={setShowCreateWorkspaceModal}
          folderToCreate={folderToCreateWorkspace}
          setFolderToCreate={setFolderToCreateWorkspace}
        />;
    }
  };

  return (
    <div id="app">
      {/* Tab Navigation */}
      <nav className="tab-navigation">
        <button
          className={currentTab === 'workspaces' ? 'active' : ''}
          onClick={() => setCurrentTab('workspaces')}
        >
          Workspaces
        </button>
        <button
          className={currentTab === 'files' ? 'active' : ''}
          onClick={() => setCurrentTab('files')}
          disabled={!currentWorkspaceId}
        >
          Files
        </button>
        <button
          className={currentTab === 'quiz' ? 'active' : ''}
          onClick={() => setCurrentTab('quiz')}
          disabled={!currentWorkspaceId}
        >
          Quiz
        </button>
        <button
          className={currentTab === 'progress' ? 'active' : ''}
          onClick={() => setCurrentTab('progress')}
          disabled={!currentWorkspaceId}
        >
          Progress
        </button>
      </nav>

      {/* Main Content */}
      <main id="main">
        {renderTabContent()}
      </main>

      {/* Modals container */}
      <div id="modals-container"></div>
    </div>
  );
};

export default App;
