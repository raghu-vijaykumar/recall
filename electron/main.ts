import { app, BrowserWindow, ipcMain, nativeTheme, Menu, dialog, protocol } from "electron";
import path from "node:path";
import os from "os";
import { spawn } from "child_process";
import fs from "fs";
import fsp from "fs/promises"; // Import fs.promises for async file operations
import mime from "mime-types"; // Import mime-types for better MIME detection
import log from 'electron-log'; // Import electron-log

// Configure electron-log
log.transports.file.level = 'info';
log.transports.file.maxSize = 5 * 1024 * 1024; // 5MB
log.transports.file.format = '[{y}-{m}-{d} {h}:{i}:{s}.{ms}] [{level}] {text}';
log.transports.file.fileName = 'main.log';

// Set the log file path to a known location in user data
log.transports.file.resolvePath = () => path.join(app.getPath('userData'), 'logs', 'main.log');

// Handle uncaught exceptions and unhandled rejections to prevent Windows error dialogs
process.on('uncaughtException', (error) => {
  log.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  log.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Global variables for backend management
let backendProcess: any = null;
let backendReady = false;

// Backend management functions
function setupDatabase() {
  const dbDir = path.join(os.homedir(), '.recall');
  const dbPath = path.join(dbDir, 'recall.db');

  // Create database directory if it doesn't exist
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }

  // Database will be created on the fly by the backend if it doesn't exist
  return dbPath;
}

function startBackend(port: number): Promise<number> {
  return new Promise(async (resolve, reject) => {
    const dbPath = setupDatabase();
    const backendExecutablePath = path.join(__dirname, '..', 'backend', 'dist', 'recall-backend.exe');

    log.info(`Attempting to start backend server on port ${port}...`);
    log.info('Database path:', dbPath);
    log.info('Backend executable:', backendExecutablePath);

    const env = {
      ...process.env,
      DATABASE_PATH: dbPath,
      PORT: port.toString(), // Pass the port to the backend
    };

    backendProcess = spawn(backendExecutablePath, [], {
      cwd: path.join(__dirname, '..', 'backend'),
      env: env,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    const checkReady = (output: string) => {
      if (output.includes('Application startup complete') && !backendReady) {
        backendReady = true;
        log.info(`Backend server is ready on port ${port}!`);
        resolve(port);
      }
    };

    backendProcess.stdout.on('data', (data: Buffer) => {
      const output = data.toString();
      log.info('Backend stdout:', output);
      checkReady(output);
    });

    backendProcess.stderr.on('data', (data: Buffer) => {
      const output = data.toString();
      log.error('Backend stderr:', output);
      if (output.includes('error while attempting to bind on address')) {
        reject(new Error('Port in use'));
      }
      checkReady(output);
    });

    backendProcess.on('close', (code: number) => {
      log.info(`Backend process exited with code ${code}`);
      backendReady = false;
    });

    backendProcess.on('error', (error: Error) => {
      log.error('Failed to start backend:', error);
      reject(error);
    });

    setTimeout(() => {
      if (!backendReady) {
        log.error('Backend startup timeout');
        reject(new Error('Backend startup timeout'));
      }
    }, 30000);
  });
}

async function findAvailablePortAndStartBackend(startPort: number, maxRetries: number): Promise<number> {
  let currentPort = startPort;
  for (let i = 0; i < maxRetries; i++) {
    try {
      await startBackend(currentPort);
      return currentPort;
    } catch (error: any) {
      if (error.message === 'Port in use') {
        log.warn(`Port ${currentPort} is in use, trying next port...`);
        currentPort++;
      } else {
        throw error; // Re-throw other errors
      }
    }
  }
  throw new Error(`Failed to start backend after ${maxRetries} retries. All ports from ${startPort} to ${currentPort - 1} are in use.`);
}

function stopBackend(): Promise<void> {
  return new Promise((resolve) => {
    if (backendProcess) {
      log.info('Stopping backend server...');

      // On Windows, SIGTERM may not work reliably, so we use a more aggressive approach
      if (process.platform === 'win32') {
        // For Windows, try SIGKILL immediately since SIGTERM often doesn't work
        backendProcess.kill('SIGKILL');
        setTimeout(() => {
          resolve();
        }, 1000); // Give it 1 second to terminate
      } else {
        // For Unix-like systems, try SIGTERM first, then SIGKILL if needed
        backendProcess.kill('SIGTERM');

        const timeout = setTimeout(() => {
          if (backendProcess && !backendProcess.killed) {
            backendProcess.kill('SIGKILL');
          }
          resolve();
        }, 3000); // Reduced timeout since we have lifespan handler now

        backendProcess.on('close', () => {
          clearTimeout(timeout);
          resolve();
        });
      }
    } else {
      resolve();
    }
  });
}

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      // webSecurity: false, // Disable web security for development - Re-enable for production
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  if (app.isPackaged || !process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(`app://./index.html`);
  } else {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
    win.webContents.openDevTools();
  }

  // Create application menu
  createMenu(win);
};

const createMenu = (win: any) => {
  const template: any[] = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New File',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            win.webContents.send('menu-new-file');
          }
        },
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            win.webContents.send('menu-save-file');
          }
        },
        { type: 'separator' },
        {
          label: 'Workspaces',
          click: () => {
            win.webContents.send('menu-show-workspaces');
          }
        },
        {
          label: 'Files',
          click: () => {
            win.webContents.send('menu-show-files');
          }
        },
        {
          label: 'Quiz',
          click: () => {
            win.webContents.send('menu-show-quiz');
          }
        },
        {
          label: 'Progress',
          click: () => {
            win.webContents.send('menu-show-progress');
          }
        },
        { type: 'separator' },
        {
          label: 'Open Folder',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            openFolderDialog();
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Workspace',
      submenu: [
        {
          label: 'Create Workspace',
          click: () => {
            win.webContents.send('menu-create-workspace');
          }
        },
        {
          label: 'Open Workspace',
          click: () => {
            win.webContents.send('menu-open-workspace');
          }
        },
        { type: 'separator' },
        {
          label: 'Refresh Files',
          accelerator: 'F5',
          click: () => {
            win.webContents.send('menu-refresh-files');
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectall' }
      ]
    },
    {
      label: 'View',
      submenu: [
        {
          label: 'Workspaces',
          click: () => {
            win.webContents.send('menu-show-workspaces');
          }
        },
        {
          label: 'Files',
          click: () => {
            win.webContents.send('menu-show-files');
          }
        },
        {
          label: 'Quiz',
          click: () => {
            win.webContents.send('menu-show-quiz');
          }
        },
        {
          label: 'Progress',
          click: () => {
            win.webContents.send('menu-show-progress');
          }
        },
        { type: 'separator' },
        { role: 'reload' },
        { role: 'forcereload' },
        { role: 'toggledevtools' },
        { type: 'separator' },
        { role: 'resetzoom' },
        { role: 'zoomin' },
        { role: 'zoomout' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Settings',
      submenu: [
        {
          label: 'Toggle Theme',
          accelerator: 'CmdOrCtrl+T',
          click: () => {
            win.webContents.send('menu-toggle-theme');
          }
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About Recall',
          click: () => {
            dialog.showMessageBox(win, {
              type: 'info',
              title: 'About Recall',
              message: 'Recall - Study App',
              detail: 'A VSCode-like study application for managing workspaces and files.'
            });
          }
        }
      ]
    }
  ];

  // macOS specific menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services', submenu: [] },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideothers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });

    // Window menu for macOS
    template.splice(5, 0, {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

ipcMain.handle("dark-mode:toggle", () => {
  if (nativeTheme.shouldUseDarkColors) {
    nativeTheme.themeSource = "light";
  } else {
    nativeTheme.themeSource = "dark";
  }
  return nativeTheme.shouldUseDarkColors;
});

ipcMain.handle("dark-mode:system", () => {
  nativeTheme.themeSource = "system";
});

ipcMain.handle("dark-mode:get", () => {
  return nativeTheme.shouldUseDarkColors;
});

app.whenReady().then(async () => {
  // Register a custom protocol to serve frontend files with correct MIME types
  protocol.handle('app', async (request) => {
    let filePath = path.join(__dirname, '..', 'frontend', request.url.slice('app://./'.length));
    if (request.url === 'app://./index.html') {
      filePath = path.join(__dirname, '..', 'frontend', 'index.html');
    }

    try {
      const fileContent = await fsp.readFile(filePath);
      const mimeType = mime.lookup(filePath) || 'application/octet-stream';
      // Create a new ArrayBuffer from the Buffer to avoid SharedArrayBuffer issues
      const arrayBuffer = new Uint8Array(fileContent).buffer;
      return new Response(arrayBuffer, {
        headers: {
          'Content-Type': mimeType,
        },
      });
    } catch (error) {
      log.error(`Failed to load file: ${filePath}`, error);
      return new Response('File not found', { status: 404 });
    }
  });

  ipcMain.handle("ping", () => "pong");

ipcMain.handle("read-html-file", async (event, componentPath: string) => {
    try {
      // Resolve the path relative to the frontend dist directory
      const fullPath = path.join(__dirname, '..', 'dist', 'frontend', 'components', componentPath);
      const fileContent = await fsp.readFile(fullPath, 'utf8');
      return fileContent;
    } catch (error) {
      log.error(`Failed to read HTML file: ${componentPath}`, error);
      throw new Error(`Failed to read HTML file: ${componentPath}`);
    }
  });

ipcMain.handle("get-folder-tree", async (event, folderPath: string) => {
    try {
      const buildTree = async (dirPath: string, relativePath: string = ""): Promise<any> => {
        const items: any[] = [];

        try {
          const entries = await fsp.readdir(dirPath, { withFileTypes: true });

          for (const entry of entries) {
            // Skip hidden files/directories and common unwanted ones
            if (entry.name.startsWith('.') ||
                entry.name === 'node_modules' ||
                entry.name === '__pycache__' ||
                entry.name === '.git') {
              continue;
            }

            const fullPath = path.join(dirPath, entry.name);
            const itemRelativePath = relativePath ? `${relativePath}/${entry.name}` : entry.name;

            if (entry.isDirectory()) {
              const children = await buildTree(fullPath, itemRelativePath);
              items.push({
                name: entry.name,
                path: itemRelativePath,
                type: 'directory',
                children: children
              });
            } else {
              // Only include text-based files
              const ext = path.extname(entry.name).toLowerCase();
              const textExtensions = ['.txt', '.md', '.markdown', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yml', '.yaml'];
              if (textExtensions.includes(ext) || !ext) {
                items.push({
                  name: entry.name,
                  path: itemRelativePath,
                  type: 'file'
                });
              }
            }
          }
        } catch (error) {
          log.error(`Error reading directory ${dirPath}:`, error);
        }

        // Sort: directories first, then files, alphabetically
        return items.sort((a, b) => {
          if (a.type !== b.type) {
            return a.type === 'directory' ? -1 : 1;
          }
          return a.name.localeCompare(b.name);
        });
      };

      return await buildTree(folderPath);
    } catch (error) {
      log.error(`Failed to get folder tree for ${folderPath}:`, error);
      throw new Error(`Failed to get folder tree: ${error}`);
    }
  });

ipcMain.handle("read-file-content", async (event, filePath: string) => {
    try {
      const content = await fsp.readFile(filePath, 'utf-8');
      return content;
    } catch (error) {
      log.error(`Failed to read file ${filePath}:`, error);
      throw new Error(`Failed to read file: ${error}`);
    }
  });

  try {
    // Start the backend server
    log.info('Initializing Recall application...');
    const backendPort = await findAvailablePortAndStartBackend(8000, 3); // Try ports 8000, 8001, 8002

    // Create the main window after backend is ready
    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });

  } catch (error) {
    log.error('Failed to start backend:', error);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", async (event) => {
  event.preventDefault();
  await stopBackend();
  app.quit();
});

// Handle app quit on macOS
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Folder dialog and workspace creation
async function openFolderDialog() {
  const windows = BrowserWindow.getAllWindows();
  if (windows.length === 0) return;

  const win = windows[0];

  try {
    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory'],
      title: 'Select Folder to Open as Workspace'
    }) as any;

    if (!result.canceled && result.filePaths.length > 0) {
      const folderPath = result.filePaths[0];
      const folderName = path.basename(folderPath);

      // Send the folder path to the renderer process
      win.webContents.send('folder-selected', {
        path: folderPath,
        name: folderName
      });
    }
  } catch (err) {
    log.error('Error opening folder dialog:', err);
  }
}
