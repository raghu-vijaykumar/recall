import { IpcRenderer } from 'electron';

declare global {
  interface Window {
    versions: {
      node: () => string;
      chrome: () => string;
      electron: () => string;
      ping: () => Promise<string>;
    };
    darkMode: {
      toggle: () => Promise<boolean>;
      system: () => Promise<void>;
      get: () => Promise<boolean>;
    };
    menuEvents: {
      on: (channel: string, callback: (...args: any[]) => void) => void;
      off: (channel: string, callback: (...args: any[]) => void) => void;
    };
    electronAPI: {
      readHtmlFile: (filePath: string) => Promise<string>;
      getFolderTree: (folderPath: string) => Promise<any[]>;
    readFileContent: (filePath: string) => Promise<string>;
    readFileBase64: (filePath: string) => Promise<{ base64: string; mimeType: string }>;

      // File operations
      createFile: (basePath: string, name: string) => Promise<{ success: boolean; path: string }>;
      createDirectory: (basePath: string, name: string) => Promise<{ success: boolean; path: string }>;
      renameFile: (oldPath: string, newName: string) => Promise<{ success: boolean; oldPath: string; newPath: string }>;
      moveFile: (sourcePath: string, destinationPath: string) => Promise<{ success: boolean; sourcePath: string; destinationPath: string }>;
      copyFile: (sourcePath: string, destinationPath: string) => Promise<{ success: boolean; sourcePath: string; destinationPath: string }>;
      deleteFile: (targetPath: string) => Promise<{ success: boolean; path: string }>;

      // File system watching
      onFileSystemChange: (callback: (event: any, data: any) => void) => void;
      offFileSystemChange: (callback: (event: any, data: any) => void) => void;

      // Workspace management
      notifyWorkspaceChanged: (workspacePath: string) => void;
    };
  }
}
