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
});
