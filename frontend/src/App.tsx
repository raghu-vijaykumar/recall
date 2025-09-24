/// <reference types="../../electron/preload" />
import React, { useState, useEffect } from 'react';
import Workspaces from '../components/workspaces/Workspaces';
// Import migrated components
import FileExplorer from '../components/file-explorer/FileExplorer';
import Quiz from '../components/quiz/Quiz';
import Progress from '../components/progress/Progress';
import { KnowledgeGraph } from '../components/knowledge-graph/KnowledgeGraph';
import { ThemeProvider, useTheme } from './core/ThemeContext';

const AppContent: React.FC = () => {
  const { toggleTheme } = useTheme();
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
      window.menuEvents.on('menu-show-knowledge-graph', () => setCurrentTab('knowledge-graph'));
      window.menuEvents.on('menu-show-quiz', () => setCurrentTab('quiz'));
      window.menuEvents.on('menu-show-progress', () => setCurrentTab('progress'));
      window.menuEvents.on('menu-toggle-theme', () => {
        // Toggle theme using our centralized theme system
        toggleTheme();
      });
      window.menuEvents.on('folder-selected', (folderInfo: { path: string, name: string }) => {
        console.log('Folder selected:', folderInfo);
        setFolderToCreateWorkspace(folderInfo);
        setCurrentTab('workspaces'); // Switch to workspaces tab to show creation
        setShowCreateWorkspaceModal(true); // Automatically show create modal
        console.log('Modal should now be visible');
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
        window.menuEvents.off('menu-show-knowledge-graph', () => setCurrentTab('knowledge-graph'));
        window.menuEvents.off('menu-show-quiz', () => setCurrentTab('quiz'));
        window.menuEvents.off('menu-show-progress', () => setCurrentTab('progress'));
        window.menuEvents.off('menu-toggle-theme', () => {
          toggleTheme();
        });
        window.menuEvents.off('folder-selected', (folderInfo: { path: string, name: string }) => {
          console.log('Folder selected:', folderInfo);
        });
      }
    };
  }, [toggleTheme]);

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
        return <FileExplorer currentWorkspaceId={currentWorkspaceId} />;
      case 'knowledge-graph':
        return currentWorkspaceId ? <KnowledgeGraph workspaceId={currentWorkspaceId} /> : <div>Please select a workspace first</div>;
      case 'quiz':
        return <Quiz currentWorkspaceId={currentWorkspaceId} />;
      case 'progress':
        return <Progress currentWorkspaceId={currentWorkspaceId} />;
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
      {/* Main Content */}
      <main id="main">
        {renderTabContent()}
      </main>

      {/* Modals container */}
      <div id="modals-container"></div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
};

export default App;
