const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("versions", {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron,
  ping: () => ipcRenderer.invoke("ping"),
});

contextBridge.exposeInMainWorld("darkMode", {
  toggle: () => ipcRenderer.invoke("dark-mode:toggle"),
  system: () => ipcRenderer.invoke("dark-mode:system"),
  get: () => ipcRenderer.invoke("dark-mode:get"),
});

contextBridge.exposeInMainWorld("menuEvents", {
  on: (channel: string, callback: (...args: any[]) => void) => {
    const validChannels = [
      'menu-new-file',
      'menu-save-file',
      'menu-create-workspace',
      'menu-open-workspace',
      'menu-refresh-files',
      'menu-show-workspaces',
      'menu-show-files',
      'menu-show-knowledge-graph',
      'menu-show-quiz',
      'menu-show-progress',
      'menu-toggle-theme',
      'folder-selected'
    ];

    if (validChannels.includes(channel)) {
      ipcRenderer.on(channel, (_: any, ...args: any[]) => callback(...args));
    }
  },
  off: (channel: string, callback: (...args: any[]) => void) => {
    ipcRenderer.off(channel, callback);
  }
});

contextBridge.exposeInMainWorld("electronAPI", {
  readHtmlFile: (filePath: string) => ipcRenderer.invoke("read-html-file", filePath),
  getFolderTree: (folderPath: string) => ipcRenderer.invoke("get-folder-tree", folderPath),
  readFileContent: (filePath: string) => ipcRenderer.invoke("read-file-content", filePath),
  readFileBase64: (filePath: string) => ipcRenderer.invoke("read-file-base64", filePath),
  getFileStats: (filePath: string) => ipcRenderer.invoke("get-file-stats", filePath),
  getFileServerPort: () => ipcRenderer.invoke("get-file-server-port"),

  // File operations
  createFile: (basePath: string, name: string) => ipcRenderer.invoke("file-operations:create", { basePath, type: 'file', name }),
  createDirectory: (basePath: string, name: string) => ipcRenderer.invoke("file-operations:create", { basePath, type: 'directory', name }),
  renameFile: (oldPath: string, newName: string) => ipcRenderer.invoke("file-operations:rename", { oldPath, newName }),
  moveFile: (sourcePath: string, destinationPath: string) => ipcRenderer.invoke("file-operations:move", { sourcePath, destinationPath }),
  copyFile: (sourcePath: string, destinationPath: string) => ipcRenderer.invoke("file-operations:copy", { sourcePath, destinationPath }),
  deleteFile: (targetPath: string) => ipcRenderer.invoke("file-operations:delete", { targetPath }),

  // File system watching
  onFileSystemChange: (callback: (event: any, data: any) => void) => {
    ipcRenderer.on('file-system-changed', callback);
  },
  offFileSystemChange: (callback: (event: any, data: any) => void) => {
    ipcRenderer.off('file-system-changed', callback);
  },

  // Workspace management
  notifyWorkspaceChanged: (workspacePath: string) => ipcRenderer.send('workspace-changed', workspacePath),
});
