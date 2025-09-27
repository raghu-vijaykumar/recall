import { app, BrowserWindow, ipcMain, nativeTheme, Menu, dialog, protocol } from "electron";
import path from "node:path";
import os from "os";
import { spawn } from "child_process";
import fs from "fs";
import fsp from "fs/promises"; // Import fs.promises for async file operations
import mime from "mime-types"; // Import mime-types for better MIME detection
import log from 'electron-log'; // Import electron-log
import http from "http";

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

// Utility function to copy directories recursively
async function copyDirectory(source: string, destination: string): Promise<void> {
  await fsp.mkdir(destination, { recursive: true });
  const entries = await fsp.readdir(source, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(source, entry.name);
    const destPath = path.join(destination, entry.name);

    if (entry.isDirectory()) {
      await copyDirectory(srcPath, destPath);
    } else {
      await fsp.copyFile(srcPath, destPath);
    }
  }
}

// Global variables for backend management
let backendProcess: any = null;
let backendReady = false;

// Global variables for file streaming server
let fileServer: http.Server | null = null;
let fileServerPort: number = 0;

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

// File streaming server functions
function startFileServer(workspacePath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    // Find an available port starting from 3000
    let port = 3000;
    const maxPort = 3100;

    const tryPort = (currentPort: number) => {
      if (currentPort > maxPort) {
        reject(new Error('No available ports for file server'));
        return;
      }

      const server = http.createServer(async (req, res) => {
        if (!req.url) {
          res.writeHead(400);
          res.end('Bad Request');
          return;
        }

        try {
          // Decode URL and construct file path
          const decodedPath = decodeURIComponent(req.url);
          const filePath = path.join(workspacePath, decodedPath);

          // Security check: ensure the file is within the workspace
          const resolvedPath = path.resolve(filePath);
          const resolvedWorkspace = path.resolve(workspacePath);

          if (!resolvedPath.startsWith(resolvedWorkspace)) {
            res.writeHead(403);
            res.end('Forbidden');
            return;
          }

          // Check if file exists
          const stats = await fsp.stat(filePath);
          if (!stats.isFile()) {
            res.writeHead(404);
            res.end('Not Found');
            return;
          }

          // Set appropriate headers with custom MIME type handling
          let mimeType = mime.lookup(filePath) || 'application/octet-stream';

          // Handle MKV files specifically (mime-types library may not recognize them)
          if (filePath.toLowerCase().endsWith('.mkv')) {
            mimeType = 'video/x-matroska';
          }

          res.writeHead(200, {
            'Content-Type': mimeType,
            'Content-Length': stats.size,
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache'
          });

          // Stream the file
          const fileStream = fs.createReadStream(filePath);
          fileStream.pipe(res);

          fileStream.on('error', (error) => {
            log.error('File streaming error:', error);
            if (!res.headersSent) {
              res.writeHead(500);
              res.end('Internal Server Error');
            }
          });

        } catch (error) {
          log.error('File server error:', error);
          if (!res.headersSent) {
            res.writeHead(404);
            res.end('Not Found');
          }
        }
      });

      server.listen(currentPort, '127.0.0.1', () => {
        log.info(`File streaming server started on port ${currentPort}`);
        fileServer = server;
        fileServerPort = currentPort;
        resolve(currentPort);
      });

      server.on('error', (err: any) => {
        if (err.code === 'EADDRINUSE') {
          tryPort(currentPort + 1);
        } else {
          reject(err);
        }
      });
    };

    tryPort(port);
  });
}

function stopFileServer(): Promise<void> {
  return new Promise((resolve) => {
    if (fileServer) {
      log.info('Stopping file streaming server...');
      fileServer.close(() => {
        fileServer = null;
        fileServerPort = 0;
        resolve();
      });
    } else {
      resolve();
    }
  });
}

let currentWorkspacePath: string | null = null;
let fileWatcher: fs.FSWatcher | null = null;

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

  // Listen for workspace changes to set up file watching
  ipcMain.on('workspace-changed', (event, workspacePath: string) => {
    setupFileWatcher(workspacePath);
  });

  return win;
};

async function setupFileWatcher(workspacePath: string) {
  // Clean up existing watcher
  if (fileWatcher) {
    fileWatcher.close();
    fileWatcher = null;
  }

  // Stop existing file server
  await stopFileServer();

  currentWorkspacePath = workspacePath;

  try {
    // Start file streaming server for the workspace
    await startFileServer(workspacePath);

    fileWatcher = fs.watch(workspacePath, { recursive: true }, (eventType, filename) => {
      if (filename) {
        // Send file system change event to renderer
        const windows = BrowserWindow.getAllWindows();
        windows.forEach(win => {
          win.webContents.send('file-system-changed', {
            eventType,
            filename,
            fullPath: path.join(workspacePath, filename)
          });
        });
      }
    });

    log.info(`File watcher and streaming server set up for workspace: ${workspacePath}`);
  } catch (error) {
    log.error(`Failed to set up file watcher/server for ${workspacePath}:`, error);
  }
}

const createMenu = (win: any) => {
  const template: any[] = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            win.webContents.send('menu-save-file');
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
          label: 'Knowledge Graph',
          click: () => {
            win.webContents.send('menu-show-knowledge-graph');
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
          label: 'Developer Settings',
          accelerator: 'CmdOrCtrl+,',
          click: () => {
            win.webContents.send('menu-show-settings');
          }
        },
        { type: 'separator' },
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
              items.push({
                name: entry.name,
                path: itemRelativePath,
                type: 'file'
              });
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

ipcMain.handle("read-file-base64", async (event, filePath: string) => {
    try {
      const buffer = await fsp.readFile(filePath);
      const base64 = buffer.toString('base64');
      const mimeType = mime.lookup(filePath) || 'application/octet-stream';
      return { base64, mimeType };
    } catch (error) {
      log.error(`Failed to read file as base64 ${filePath}:`, error);
      throw new Error(`Failed to read file: ${error}`);
    }
  });

ipcMain.handle("get-file-stats", async (event, filePath: string) => {
    try {
      const stats = await fsp.stat(filePath);
      return {
        size: stats.size,
        mtime: stats.mtime,
        ctime: stats.ctime,
        isDirectory: stats.isDirectory(),
        isFile: stats.isFile()
      };
    } catch (error) {
      log.error(`Failed to get stats for file ${filePath}:`, error);
      throw new Error(`Failed to get file stats: ${error}`);
    }
  });

ipcMain.handle("get-file-server-port", () => {
    return fileServerPort;
  });

ipcMain.handle("file-operations:create", async (event, { basePath, type, name }: { basePath: string; type: 'file' | 'directory'; name: string }) => {
    try {
      const fullPath = path.join(basePath, name);

      if (type === 'file') {
        await fsp.writeFile(fullPath, '');
      } else {
        await fsp.mkdir(fullPath, { recursive: true });
      }

      log.info(`Created ${type}: ${fullPath}`);
      return { success: true, path: fullPath };
    } catch (error) {
      log.error(`Failed to create ${type} ${basePath}/${name}:`, error);
      throw new Error(`Failed to create ${type}: ${error}`);
    }
  });

ipcMain.handle("file-operations:rename", async (event, { oldPath, newName }: { oldPath: string; newName: string }) => {
    try {
      const newPath = path.join(path.dirname(oldPath), newName);
      await fsp.rename(oldPath, newPath);

      log.info(`Renamed ${oldPath} to ${newPath}`);
      return { success: true, oldPath, newPath };
    } catch (error) {
      log.error(`Failed to rename ${oldPath} to ${newName}:`, error);
      throw new Error(`Failed to rename: ${error}`);
    }
  });

ipcMain.handle("file-operations:move", async (event, { sourcePath, destinationPath }: { sourcePath: string; destinationPath: string }) => {
    try {
      const destPath = path.join(destinationPath, path.basename(sourcePath));
      await fsp.rename(sourcePath, destPath);

      log.info(`Moved ${sourcePath} to ${destPath}`);
      return { success: true, sourcePath, destinationPath: destPath };
    } catch (error) {
      log.error(`Failed to move ${sourcePath} to ${destinationPath}:`, error);
      throw new Error(`Failed to move: ${error}`);
    }
  });

ipcMain.handle("file-operations:copy", async (event, { sourcePath, destinationPath }: { sourcePath: string; destinationPath: string }) => {
    try {
      const destPath = path.join(destinationPath, path.basename(sourcePath));

      const stats = await fsp.stat(sourcePath);
      if (stats.isDirectory()) {
        await copyDirectory(sourcePath, destPath);
      } else {
        await fsp.copyFile(sourcePath, destPath);
      }

      log.info(`Copied ${sourcePath} to ${destPath}`);
      return { success: true, sourcePath, destinationPath: destPath };
    } catch (error) {
      log.error(`Failed to copy ${sourcePath} to ${destinationPath}:`, error);
      throw new Error(`Failed to copy: ${error}`);
    }
  });

ipcMain.handle("file-operations:delete", async (event, { targetPath }: { targetPath: string }) => {
    try {
      const stats = await fsp.stat(targetPath);
      if (stats.isDirectory()) {
        await fsp.rm(targetPath, { recursive: true, force: true });
      } else {
        await fsp.unlink(targetPath);
      }

      log.info(`Deleted ${targetPath}`);
      return { success: true, path: targetPath };
    } catch (error) {
      log.error(`Failed to delete ${targetPath}:`, error);
      throw new Error(`Failed to delete: ${error}`);
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
  //await stopBackend();
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
