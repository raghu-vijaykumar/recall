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
    };
  }
}
