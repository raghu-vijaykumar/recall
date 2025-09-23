# Electron Design Document

## Overview
The Electron layer provides the desktop application framework for Recall, managing the application lifecycle, native OS integration, and communication between the frontend React app and the Python backend. It creates a seamless desktop experience while maintaining security through context isolation.

## Architecture

### Main Process (`main.ts`)
The main process handles application-level concerns and runs in a Node.js environment.

#### Key Responsibilities
- **Backend Management**: Spawns and manages the Python FastAPI backend process
- **Window Creation**: Creates and configures the main BrowserWindow
- **Menu System**: Application menu with keyboard shortcuts
- **IPC Handling**: Inter-process communication with renderer process
- **Protocol Registration**: Custom `app://` protocol for packaged app
- **Lifecycle Management**: Clean startup/shutdown handling

### Preload Script (`preload.ts`)
Provides a secure bridge between main and renderer processes.

#### Security Features
- **Context Isolation**: Renderer cannot directly access Node.js APIs
- **Type Safety**: TypeScript definitions for exposed APIs
- **Limited API Surface**: Only necessary functions exposed

#### Exposed APIs
- **Menu Events**: Application menu event handling
- **File Operations**: Secure file system access
- **Theme Management**: Dark/light mode toggling
- **IPC Communication**: Typed IPC channels

### Process Architecture

#### Backend Process Management
- **Automatic Startup**: Backend starts when Electron app launches
- **Port Management**: Finds available port (8000+) to avoid conflicts
- **Health Monitoring**: Monitors backend readiness
- **Clean Shutdown**: Ensures backend terminates properly on app exit

#### Window Management
- **Single Window**: Main application window (1000x700)
- **Web Preferences**: Security settings (context isolation, no node integration)
- **Development Mode**: DevTools and hot reload in development
- **Production Mode**: Optimized loading from custom protocol

### IPC Communication

#### Main → Renderer
- **Menu Events**: Navigation and action triggers
- **File Operations**: Folder selection results
- **Theme Changes**: Dark/light mode updates

#### Renderer → Main
- **File System Requests**: Folder tree, file content reading
- **Theme Toggle**: Dark mode preference changes
- **Ping/Pong**: Health check communication

### Menu System

#### Application Menu Structure
- **File Menu**: New file, save, workspaces, files, quiz, progress, exit
- **Workspace Menu**: Create workspace, open workspace, refresh files
- **Edit Menu**: Standard edit operations (undo, redo, cut, copy, paste)
- **View Menu**: Tab switching, reload, dev tools, zoom controls
- **Settings Menu**: Theme toggle
- **Help Menu**: About dialog

#### Keyboard Shortcuts
- **Ctrl+N**: New file
- **Ctrl+S**: Save file
- **Ctrl+O**: Open folder
- **F5**: Refresh files
- **Ctrl+T**: Toggle theme
- **Ctrl+Q**: Quit application

### File System Integration

#### Folder Operations
- **Native Dialog**: OS folder picker for workspace creation
- **Tree Building**: Recursive directory structure creation
- **File Filtering**: Text-based files only (.txt, .md, .py, .js, etc.)
- **Hidden File Exclusion**: Skips .git, node_modules, __pycache__, etc.

#### File Content Access
- **Async Reading**: Non-blocking file content retrieval
- **MIME Type Handling**: Proper content-type headers
- **Error Handling**: Graceful failure for inaccessible files

### Custom Protocol (`app://`)

#### Purpose
- **Packaged App Support**: Serve frontend files in production
- **MIME Type Correction**: Proper content-type for all assets
- **Security**: Isolated from file:// protocol vulnerabilities

#### Implementation
- **Protocol Handler**: Intercepts app:// requests
- **File Resolution**: Maps URLs to frontend dist directory
- **Buffer Handling**: Proper binary data handling for assets

### Theme Management

#### Dark/Light Mode
- **System Integration**: Respects OS theme preference
- **Manual Toggle**: User can override system setting
- **IPC Communication**: Theme changes communicated to renderer

### Logging and Error Handling

#### Electron-Log Integration
- **File Logging**: Logs written to `userData/logs/main.log`
- **Log Rotation**: 5MB max size, automatic rotation
- **Error Capture**: Uncaught exceptions and unhandled rejections
- **Process Monitoring**: Backend stdout/stderr logging

### Build and Packaging

#### Development
- **Hot Reload**: Vite dev server integration
- **DevTools**: Automatic DevTools opening in development
- **Source Maps**: Debug support

#### Production
- **Forge Configuration**: Electron Forge build setup
- **Backend Bundling**: PyInstaller executable inclusion
- **Asset Optimization**: Minified frontend bundle

### Security Considerations

#### Context Isolation
- **No Direct Node Access**: Renderer cannot require() Node modules
- **Preload Bridge**: All main process communication through preload
- **Input Validation**: All IPC messages validated

#### Process Separation
- **Main Process**: Privileged, handles system operations
- **Renderer Process**: Unprivileged, handles UI only
- **Backend Process**: Separate Python process for business logic

### Dependencies
- **Electron**: Desktop application framework
- **Electron Forge**: Build and packaging tool
- **Electron Log**: Logging framework
- **Mime-Types**: MIME type detection
- **TypeScript**: Type safety for main process

### Platform-Specific Features

#### Windows
- **Process Termination**: SIGKILL for reliable backend shutdown
- **Menu Adjustments**: Windows-specific menu structure

#### macOS
- **Dock Integration**: Application menu in dock
- **Window Management**: macOS-specific window controls

#### Linux
- **Generic Handling**: Cross-platform compatibility
- **Process Management**: Unix signal handling

### Performance Optimizations

#### Startup Time
- **Parallel Initialization**: Backend and window creation in parallel
- **Lazy Loading**: Components loaded on demand
- **Caching**: File system cache for folder trees

#### Memory Management
- **Process Cleanup**: Proper backend process termination
- **Event Listener Cleanup**: Prevents memory leaks
- **Resource Disposal**: File handles and buffers properly closed

### Error Recovery

#### Backend Failures
- **Automatic Restart**: Attempts to restart failed backend
- **Port Retry**: Finds alternative ports if initial port occupied
- **Graceful Degradation**: App continues with limited functionality

#### IPC Errors
- **Timeout Handling**: IPC calls with reasonable timeouts
- **Fallback Behavior**: Graceful handling of communication failures
- **User Notification**: Clear error messages for users
