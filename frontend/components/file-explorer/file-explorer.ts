import { API_BASE } from '../../core/api.js';
import { File, Tab, MonacoEditor } from '../../core/types.js';
import { getFileIcon, getFileTypeFromName, getDefaultContentForFile, showModal, hideModal } from '../../shared/utils.js';

export class FileExplorerComponent {
  private currentWorkspaceId: number | null = null;
  private openFiles: Tab[] = [];
  private activeFileId: number | null = null;
  private monacoEditor: MonacoEditor | null = null;

  constructor() {
    this.initialize();
  }

  private initialize() {
    this.setupEventHandlers();
    this.initializeMonacoEditor();
  }

  private setupEventHandlers() {
    const uploadBtn = document.getElementById('upload-file-btn');
    const scanBtn = document.getElementById('scan-folder-btn');
    const fileModal = document.getElementById('create-file-modal');
    const fileForm = document.getElementById('create-file-form');
    const cancelFileBtn = document.getElementById('cancel-file-create');
    const createFirstFileBtn = document.getElementById('create-first-file-btn');

    if (uploadBtn) {
      uploadBtn.addEventListener('click', () => {
        this.showCreateFileModal();
      });
    }

    if (scanBtn) {
      scanBtn.addEventListener('click', async () => {
        await this.scanCurrentWorkspaceFolder();
      });
    }

    if (cancelFileBtn) {
      cancelFileBtn.addEventListener('click', () => {
        hideModal('create-file-modal');
        if (fileForm) (fileForm as HTMLFormElement).reset();
      });
    }

    if (fileForm) {
      fileForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await this.handleCreateFile();
      });
    }

    if (createFirstFileBtn) {
      createFirstFileBtn.addEventListener('click', () => {
        this.showCreateFileModal();
      });
    }

    // Enhanced keyboard shortcuts for tab management
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 's':
            e.preventDefault();
            this.saveCurrentFile();
            break;
          case 'n':
            e.preventDefault();
            this.showCreateFileModal();
            break;
          case 'w':
            e.preventDefault();
            if (this.activeFileId !== null) {
              this.closeTab(this.activeFileId);
            }
            break;
          case 'Tab':
            e.preventDefault();
            if (e.shiftKey) {
              this.switchToPreviousTab();
            } else {
              this.switchToNextTab();
            }
            break;
          case '1':
          case '2':
          case '3':
          case '4':
          case '5':
          case '6':
          case '7':
          case '8':
          case '9':
            e.preventDefault();
            const tabIndex = parseInt(e.key) - 1;
            if (tabIndex < this.openFiles.length) {
              this.switchToTab(this.openFiles[tabIndex].id);
            }
            break;
        }
      }
    });

    // Listen for workspace selection events
    window.addEventListener('workspace-selected', (e: any) => {
      this.setCurrentWorkspace(e.detail.workspaceId);
    });
  }

  setCurrentWorkspace(workspaceId: number) {
    this.currentWorkspaceId = workspaceId;
    this.loadFiles();
  }

  async loadFiles() {
    if (!this.currentWorkspaceId) return [];

    try {
      const response = await fetch(`${API_BASE}/files/workspace/${this.currentWorkspaceId}`);
      const files = await response.json();
      this.renderFileTree(files);
      return files;
    } catch (error) {
      console.error('Failed to load files:', error);
      return [];
    }
  }

  private renderFileTree(files: File[]) {
    const tree = document.getElementById('file-tree');
    if (!tree) return;

    tree.innerHTML = '';

    files.forEach(file => {
      const item = document.createElement('div');
      item.className = 'file-item';
      if (file.id === this.activeFileId) {
        item.classList.add('active');
      }

      const icon = getFileIcon(file.name);
      item.innerHTML = `
        <span class="file-icon">${icon}</span>
        <span class="file-name">${file.name}</span>
      `;
      item.addEventListener('click', () => this.openFileInTab(file));
      tree.appendChild(item);
    });
  }

  async loadFileContent(fileId: number) {
    try {
      const response = await fetch(`${API_BASE}/files/${fileId}/content`);
      const data = await response.json();

      if (this.monacoEditor) {
        this.monacoEditor.setValue(data.content || '');
      }
    } catch (error) {
      console.error('Failed to load file content:', error);
    }
  }

  private showCreateFileModal() {
    if (!this.currentWorkspaceId) {
      alert('Please select a workspace first');
      return;
    }

    showModal('create-file-modal');
    // Focus on the input field
    const input = document.getElementById('file-name') as HTMLInputElement;
    if (input) {
      setTimeout(() => input.focus(), 100);
    }
  }

  async handleCreateFile() {
    const input = document.getElementById('file-name') as HTMLInputElement;
    const filename = input?.value?.trim();

    if (!filename) {
      alert('Please enter a filename');
      return;
    }

    const fileData = {
      name: filename,
      path: filename,
      file_type: getFileTypeFromName(filename),
      size: 0,
      workspace_id: this.currentWorkspaceId,
      content: getDefaultContentForFile(filename)
    };

    try {
      const response = await fetch(`${API_BASE}/files/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fileData)
      });

      if (response.ok) {
        const newFile = await response.json();
        console.log('File created:', newFile);

        // Close modal and reset form
        hideModal('create-file-modal');
        const form = document.getElementById('create-file-form') as HTMLFormElement;
        if (form) form.reset();

        // Load the new file content in editor
        if (this.monacoEditor) {
          this.monacoEditor.setValue(fileData.content);
        }

        // Refresh file list
        await this.loadFiles();
      } else {
        const error = await response.text();
        alert(`Failed to create file: ${error}`);
      }
    } catch (error) {
      console.error('Failed to create file:', error);
      alert('Failed to create file');
    }
  }

  async saveCurrentFile() {
    if (!this.currentWorkspaceId) {
      alert('No workspace selected');
      return;
    }

    if (!this.monacoEditor) {
      alert('Editor not initialized');
      return;
    }

    const content = this.monacoEditor.getValue();

    // For now, use a simple filename based on timestamp
    // In a real implementation, you'd track the current file being edited
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `file_${timestamp}.txt`;

    const fileData = {
      name: filename,
      path: filename,
      file_type: getFileTypeFromName(filename),
      size: content.length,
      workspace_id: this.currentWorkspaceId,
      content: content
    };

    try {
      const response = await fetch(`${API_BASE}/files/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fileData)
      });

      if (response.ok) {
        console.log('File saved successfully');
        alert(`File saved as: ${filename}`);
        await this.loadFiles();
      } else {
        const error = await response.text();
        alert(`Failed to save file: ${error}`);
      }
    } catch (error) {
      console.error('Failed to save file:', error);
      alert('Failed to save file');
    }
  }

  // Monaco Editor Setup
  private initializeMonacoEditor() {
    // Wait for Monaco loader to be available (loaded from HTML)
    const checkMonaco = () => {
      // @ts-ignore
      if (typeof window.require !== 'undefined') {
        // @ts-ignore
        window.require.config({
          paths: {
            vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs'
          }
        });

        // @ts-ignore
        window.require(['vs/editor/editor.main'], () => {
          const container = document.getElementById('monaco-editor');
          if (container) {
            // @ts-ignore
            this.monacoEditor = window.monaco.editor.create(container, {
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
        // Retry after a short delay
        setTimeout(checkMonaco, 100);
      }
    };

    checkMonaco();
  }

  async openFileInTab(file: File) {
    // Check if file is already open
    const existingTab = this.openFiles.find(f => f.id === file.id);
    if (existingTab) {
      this.switchToTab(file.id);
      return;
    }

    // Add to open files
    const tab: Tab = {
      id: file.id,
      name: file.name,
      file: file,
      isActive: false
    };
    this.openFiles.push(tab);
    this.activeFileId = file.id;

    // Update UI
    this.renderTabs();
    this.renderFileTree(await this.loadFiles()); // Re-render file tree to update active state

    // Load file content
    await this.loadFileContent(file.id);

    // Hide welcome screen
    this.hideWelcomeScreen();
  }

  async switchToTab(fileId: number) {
    this.activeFileId = fileId;
    const file = this.openFiles.find(f => f.id === fileId);
    if (file) {
      await this.loadFileContent(file.id);
    }
    this.renderTabs();
    await this.loadFiles(); // Re-render file tree to update active state
  }

  async closeTab(fileId: number) {
    const index = this.openFiles.findIndex(f => f.id === fileId);
    if (index === -1) return;

    this.openFiles.splice(index, 1);

    // If closing active tab, switch to another tab or show welcome screen
    if (this.activeFileId === fileId) {
      if (this.openFiles.length > 0) {
        const newActiveIndex = Math.min(index, this.openFiles.length - 1);
        this.activeFileId = this.openFiles[newActiveIndex].id;
        if (this.activeFileId !== null) {
          await this.loadFileContent(this.activeFileId);
        }
      } else {
        this.activeFileId = null;
        this.showWelcomeScreen();
      }
    }

    this.renderTabs();
    await this.loadFiles(); // Re-render file tree to update active state
  }

  async switchToNextTab() {
    if (this.openFiles.length === 0) return;

    const currentIndex = this.openFiles.findIndex(f => f.id === this.activeFileId);
    const nextIndex = (currentIndex + 1) % this.openFiles.length;
    await this.switchToTab(this.openFiles[nextIndex].id);
  }

  async switchToPreviousTab() {
    if (this.openFiles.length === 0) return;

    const currentIndex = this.openFiles.findIndex(f => f.id === this.activeFileId);
    const prevIndex = currentIndex === 0 ? this.openFiles.length - 1 : currentIndex - 1;
    await this.switchToTab(this.openFiles[prevIndex].id);
  }

  private renderTabs() {
    const tabBar = document.getElementById('tab-bar');
    if (!tabBar) return;

    tabBar.innerHTML = '';

    this.openFiles.forEach((tab, index) => {
      const tabElement = document.createElement('div');
      tabElement.className = 'tab';
      if (tab.id === this.activeFileId) {
        tabElement.classList.add('active');
      }

      const icon = getFileIcon(tab.name);
      tabElement.innerHTML = `
        <span class="tab-icon">${icon}</span>
        <span class="tab-name">${tab.name}</span>
        <button class="tab-close" onclick="window.fileExplorerComponent.closeTab(${tab.id})" title="Close (Ctrl+W)">×</button>
      `;

      // Left click to switch tabs
      tabElement.addEventListener('click', (e) => {
        if (!(e.target as HTMLElement).classList.contains('tab-close')) {
          this.switchToTab(tab.id);
        }
      });

      // Right click for context menu
      tabElement.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        this.showTabContextMenu(e, tab.id, index);
      });

      // Middle click to close
      tabElement.addEventListener('auxclick', (e) => {
        if (e.button === 1) { // Middle mouse button
          e.preventDefault();
          this.closeTab(tab.id);
        }
      });

      tabBar.appendChild(tabElement);
    });
  }

  private showTabContextMenu(e: MouseEvent, fileId: number, index: number) {
    // Remove existing context menu
    const existingMenu = document.querySelector('.tab-context-menu');
    if (existingMenu) {
      existingMenu.remove();
    }

    const menu = document.createElement('div');
    menu.className = 'tab-context-menu';
    menu.style.position = 'fixed';
    menu.style.left = `${e.clientX}px`;
    menu.style.top = `${e.clientY}px`;
    menu.style.zIndex = '10000';

    menu.innerHTML = `
      <div class="context-menu-item" onclick="window.fileExplorerComponent.closeTab(${fileId})">Close</div>
      <div class="context-menu-item" onclick="window.fileExplorerComponent.closeOtherTabs(${fileId})">Close Others</div>
      <div class="context-menu-item" onclick="window.fileExplorerComponent.closeAllTabs()">Close All</div>
      <div class="context-menu-separator"></div>
      <div class="context-menu-item" onclick="window.fileExplorerComponent.switchToTab(${fileId})">Keep Open</div>
    `;

    document.body.appendChild(menu);

    // Close menu when clicking elsewhere
    const closeMenu = (e: Event) => {
      if (!menu.contains(e.target as Node)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };

    setTimeout(() => {
      document.addEventListener('click', closeMenu);
    }, 10);
  }

  closeOtherTabs(fileId: number) {
    const fileToKeep = this.openFiles.find(f => f.id === fileId);
    if (!fileToKeep) return;

    this.openFiles = [fileToKeep];
    this.activeFileId = fileId;
    this.renderTabs();
    this.loadFiles(); // Re-render file tree to update active state
  }

  closeAllTabs() {
    this.openFiles = [];
    this.activeFileId = null;
    this.renderTabs();
    this.showWelcomeScreen();
    this.loadFiles(); // Re-render file tree to clear active state
  }

  private showWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcome-screen');
    const monacoEditor = document.getElementById('monaco-editor');

    if (welcomeScreen) welcomeScreen.style.display = 'flex';
    if (monacoEditor) monacoEditor.style.display = 'none';
  }

  private hideWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcome-screen');
    const monacoEditor = document.getElementById('monaco-editor');

    if (welcomeScreen) welcomeScreen.style.display = 'none';
    if (monacoEditor) monacoEditor.style.display = 'block';
  }

  async scanCurrentWorkspaceFolder() {
    if (!this.currentWorkspaceId) {
      alert('Please select a workspace first');
      return;
    }

    // First check if the workspace has a folder path
    try {
      const workspaceResponse = await fetch(`${API_BASE}/workspaces/${this.currentWorkspaceId}`);
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
      const response = await fetch(`${API_BASE}/files/scan-folder/${this.currentWorkspaceId}`, {
        method: 'POST'
      });

      if (response.ok) {
        const scanResult = await response.json();
        console.log('Folder scan completed:', scanResult);
        alert(`Folder scan completed!\nFound and added ${scanResult.files_created} new files.`);

        // Refresh the file list
        await this.loadFiles();
      } else {
        const error = await response.text();
        console.error('Failed to scan folder:', error);
        alert(`Failed to scan folder: ${error}`);
      }
    } catch (error) {
      console.error('Failed to scan folder:', error);
      alert('Failed to scan folder');
    }
  }

  // Menu event handlers
  showFilesTab() {
    // Emit event to show files tab
    window.dispatchEvent(new CustomEvent('show-tab', { detail: { tab: 'files' } }));
  }

  showNewFileModal() {
    this.showCreateFileModal();
  }

  refreshFiles() {
    this.loadFiles();
  }

  getCurrentWorkspaceId(): number | null {
    return this.currentWorkspaceId;
  }

  getActiveFileId(): number | null {
    return this.activeFileId;
  }
}

// Global instance for onclick handlers
declare global {
  interface Window {
    fileExplorerComponent: FileExplorerComponent;
  }
}
