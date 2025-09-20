import { app, BrowserWindow, ipcMain, nativeTheme, Menu, dialog } from "electron";
import path from "node:path";
import { spawn } from "child_process";
import fs from "fs";

// Handle uncaught exceptions and unhandled rejections to prevent Windows error dialogs
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Global variables for backend management
let backendProcess: any = null;
let backendReady = false;

// Backend management functions
function setupDatabase() {
  const userDataPath = app.getPath('userData');
  const dbDir = path.join(userDataPath, 'database');
  const dbPath = path.join(dbDir, 'recall.db');
  const schemaPath = path.join(dbDir, 'schema.sql');

  // Create database directory if it doesn't exist
  if (!fs.existsSync(dbDir)) {
    fs.mkdirSync(dbDir, { recursive: true });
  }

  // Copy database file if it doesn't exist
  const sourceDbPath = path.join(__dirname, '..', 'database', 'recall.db');
  if (fs.existsSync(sourceDbPath) && !fs.existsSync(dbPath)) {
    fs.copyFileSync(sourceDbPath, dbPath);
    console.log('Database copied to user data directory');
  }

  // Copy schema file
  const sourceSchemaPath = path.join(__dirname, '..', 'database', 'schema.sql');
  if (fs.existsSync(sourceSchemaPath)) {
    fs.copyFileSync(sourceSchemaPath, schemaPath);
  }

  return dbPath;
}

function startBackend() {
  return new Promise((resolve, reject) => {
    const dbPath = setupDatabase();
    const backendExecutablePath = path.join(__dirname, '..', 'backend', 'dist', 'recall-backend.exe');

    console.log('Starting backend server...');
    console.log('Database path:', dbPath);
    console.log('Backend executable:', backendExecutablePath);

    // Set environment variables for the backend
    const env = {
      ...process.env,
      DATABASE_PATH: dbPath,
    };

    // Start the PyInstaller-generated backend executable
    backendProcess = spawn(backendExecutablePath, [], {
      cwd: path.join(__dirname, '..', 'backend'),
      env: env,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    // Handle backend output
    const checkReady = (output: string) => {
      if (output.includes('Application startup complete') && !backendReady) {
        backendReady = true;
        console.log('Backend server is ready!');
        resolve(true);
      }
    };

    backendProcess.stdout.on('data', (data: Buffer) => {
      const output = data.toString();
      console.log('Backend stdout:', output);
      checkReady(output);
    });

    backendProcess.stderr.on('data', (data: Buffer) => {
      const output = data.toString();
      console.log('Backend stderr:', output);
      checkReady(output);
    });

    backendProcess.on('close', (code: number) => {
      console.log(`Backend process exited with code ${code}`);
      backendReady = false;
    });

    backendProcess.on('error', (error: Error) => {
      console.error('Failed to start backend:', error);
      reject(error);
    });

    // Timeout after 30 seconds
    setTimeout(() => {
      if (!backendReady) {
        console.error('Backend startup timeout');
        reject(new Error('Backend startup timeout'));
      }
    }, 30000);
  });
}

function stopBackend() {
  if (backendProcess) {
    console.log('Stopping backend server...');
    backendProcess.kill('SIGTERM');

    // Force kill after 5 seconds
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) {
        backendProcess.kill('SIGKILL');
      }
    }, 5000);
  }
}

const createWindow = () => {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      webSecurity: false, // Disable web security for development
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  if (app.isPackaged) {
    win.loadFile(path.join(__dirname, "..", "frontend", "index.html"));
  } else {
    win.loadURL("http://localhost:3000");
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
  ipcMain.handle("ping", () => "pong");

  try {
    // Start the backend server
    console.log('Initializing Recall application...');
    await startBackend();

    // Create the main window after backend is ready
    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });

  } catch (error) {
    console.error('Failed to start backend:', error);
    app.quit();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    stopBackend();
    app.quit();
  }
});

app.on("before-quit", () => {
  stopBackend();
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
    console.error('Error opening folder dialog:', err);
  }
}
