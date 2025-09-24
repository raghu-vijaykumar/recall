# Directory Structure and Build Process Design Document

## Overview

This document provides a comprehensive overview of the Recall application's directory structure and build process. The application is a desktop study app built with a Python Flask backend, React/TypeScript frontend, and Electron wrapper.

## Monorepo Structure for Electron Applications

### Conceptual Organization

This project uses a monorepo structure that enables seamless integration between multiple technology stacks within a single Electron application. The key insight is organizing code by **responsibility and technology** rather than by deployment unit, allowing for:

- **Unified Version Control**: Single repository for all components
- **Atomic Changes**: Cross-component changes in single commits
- **Shared Tooling**: Common build scripts and configurations
- **Integrated Testing**: End-to-end tests across all layers
- **Simplified Dependencies**: Clear visibility of inter-component relationships

### Core Component Separation

```
recall/
├── backend/                 # Python API server (FastAPI)
├── frontend/                # React/TypeScript UI
├── electron/                # Desktop application wrapper
├── wiki/                    # Documentation and specifications
├── scripts/                 # Development and deployment utilities
└── [build configs]         # Unified build orchestration
```

#### Backend (`/backend`)
- **Technology**: Python FastAPI with SQLAlchemy
- **Responsibility**: Data persistence, business logic, API endpoints
- **Build Output**: Standalone executable (`recall-backend.exe`)
- **Runtime**: Separate process communicating via HTTP

#### Frontend (`/frontend`)
- **Technology**: React + TypeScript + Vite
- **Responsibility**: User interface and user experience
- **Build Output**: Static web assets (HTML, JS, CSS)
- **Runtime**: Served by Electron's built-in web server

#### Electron (`/electron`)
- **Technology**: Node.js + TypeScript
- **Responsibility**: Desktop integration, process management, security
- **Build Output**: Compiled JavaScript for main/preload processes
- **Runtime**: Native desktop application wrapper

#### Wiki (`/wiki`)
- **Technology**: Markdown + custom tooling
- **Responsibility**: Documentation, design specs, architectural decisions
- **Build Output**: Static documentation (optional integration)
- **Runtime**: External reference material

### Integration Through Build System

The monorepo's power lies in its **unified build orchestration** that treats these disparate technologies as a cohesive whole:

#### Single Command Build Process
```bash
make build  # Orchestrates all components
├── make build-frontend     # React → static assets
├── make build-electron     # TS → compiled JS
├── make build-backend-pyinstaller  # Python → executable
└── [integration]           # Combine into final package
```

#### Component Communication Architecture
```
┌─────────────────┐    HTTP    ┌─────────────────┐
│   Electron      │◄──────────►│     Backend     │
│   (Main Process)│            │  (Python exe)  │
└─────────────────┘            └─────────────────┘
         │                              │
         │ IPC                          │
         ▼                              │
┌─────────────────┐                     │
│   Frontend      │◄────────────────────┘
│   (React SPA)   │    HTTP API calls
└─────────────────┘
```

#### Runtime Process Management
1. **Electron launches** → Creates desktop window
2. **Backend starts** → Separate process with HTTP API
3. **Frontend loads** → Served via custom `app://` protocol
4. **IPC bridges** → Secure communication between processes

### Benefits for Electron App Development

#### Development Experience
- **Hot Reloading**: Frontend changes reflect immediately
- **Unified Debugging**: Single IDE workspace for all code
- **Shared Scripts**: Common development commands
- **Integrated Testing**: Cross-component test suites

#### Build Optimization
- **Parallel Building**: Components build simultaneously
- **Incremental Builds**: Only rebuild changed components
- **Artifact Reuse**: Cache intermediate build outputs
- **Unified Packaging**: Single distributable with all components

#### Future Extensibility
The structure naturally accommodates new components:

```
recall/
├── backend/                 # Existing Python API
├── frontend/                # Existing React UI
├── electron/                # Existing desktop wrapper
├── wiki/                    # Existing documentation
├── mobile/                  # Future: React Native app
├── cli/                     # Future: Command-line interface
├── shared/                  # Future: Cross-platform utilities
├── extensions/              # Future: Plugin system
└── [build configs]         # Unified orchestration
```

#### Technology Migration
- **Gradual Updates**: Migrate components independently
- **Compatibility Layers**: Maintain API contracts between components
- **Staged Rollouts**: Deploy component updates separately
- **Rollback Safety**: Version pinning and dependency management

### Build Integration Patterns

#### Configuration Inheritance
- **Shared Tooling**: Common linting, formatting, testing configs
- **Environment Variables**: Unified configuration across components
- **Dependency Management**: Centralized package management

#### Cross-Component Dependencies
- **API Contracts**: Defined interfaces between backend/frontend
- **Shared Types**: TypeScript definitions used across Electron/React
- **Build Dependencies**: Frontend build waits for backend API availability

#### Deployment Flexibility
- **Component Isolation**: Deploy backend/frontend independently
- **Version Pinning**: Lock compatible versions of components
- **Feature Flags**: Enable/disable features across the stack
- **Environment Parity**: Consistent configurations across dev/staging/prod

## Build Process

### Overview

The build process is orchestrated through a Makefile that coordinates building the backend, frontend, and Electron components. The application uses:

- **PyInstaller** for packaging the Python backend into a standalone executable
- **Vite** for building the React frontend
- **TypeScript** for compiling Electron scripts
- **Electron Forge** for creating distributable desktop packages

### Build Targets

#### Setup (`make setup`)
1. Installs Node.js dependencies via npm
2. Creates Python virtual environment
3. Installs Python dependencies from requirements.txt

#### Build (`make build`)
1. Runs tests first
2. Builds frontend with Vite
3. Compiles Electron TypeScript files
4. Packages backend with PyInstaller
5. Copies built assets to distribution directory

#### Development (`make dev`)
1. Kills any existing backend processes
2. Runs frontend development server
3. Starts Electron in development mode

#### Distribution (`make dist`)
1. Cleans previous builds
2. Runs full build process
3. Creates distributable packages using Electron Forge

#### Testing (`make test`)
1. Activates Python virtual environment
2. Runs pytest on backend tests

### Build Configuration Files

#### build.yaml
- Defines files to include in the packaged application
- Specifies Electron Forge build options
- Configures Windows-specific packaging (NSIS installer, portable version)
- Excludes development artifacts and unnecessary files

#### vite.config.ts
- Configures Vite for React frontend building
- Sets root directory to `frontend/`
- Outputs built files to `../dist/frontend`
- Configures development server on port 3000

#### tsconfig.electron.json
- Extends main TypeScript config
- Configures CommonJS module system for Electron
- Outputs compiled files to `dist/electron`
- Includes only Electron-related TypeScript files

#### Makefile
- Provides high-level build orchestration
- Handles cross-platform compatibility (Windows/macOS/Linux)
- Manages process cleanup and virtual environment activation
- Coordinates multi-stage build process

### Development Workflow

1. **Initial Setup**: Run `make setup` to install all dependencies
2. **Development**: Use `make dev` to start development servers
3. **Testing**: Run `make test` to execute backend tests
4. **Building**: Use `make build` for production builds
5. **Packaging**: Run `make dist` to create distributable packages
6. **Installation**: Use `make install` for full clean rebuild and install

### Key Build Dependencies

#### Frontend Build
- **Vite**: Fast build tool and development server
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **ESLint/Prettier**: Code quality tools

#### Backend Build
- **PyInstaller**: Python application packaging
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **pytest**: Testing framework

#### Desktop Packaging
- **Electron Forge**: Electron application packaging
- **Electron Builder**: Alternative packaging tool
- **NSIS**: Windows installer creation

### Build Output Structure

```
dist/
├── frontend/               # Built React application
│   ├── assets/
│   ├── index.html
│   └── (other static files)
├── electron/               # Compiled Electron scripts
│   ├── main.js
│   ├── preload.js
│   └── preload.d.ts
└── backend/
    └── recall-backend.exe  # Packaged Python executable
```

### Packaging and Distribution

The final distributable package includes:
- Electron executable
- Packaged Python backend
- Built React frontend
- Static assets
- Database migration scripts

Packages are created for Windows (NSIS installer and portable versions) and can be extended to other platforms.

### Continuous Integration

The build process is designed to be CI/CD friendly with:
- Automated testing before builds
- Clean build artifacts management
- Cross-platform compatibility
- Deterministic build outputs

This structure ensures reliable, reproducible builds across different development and deployment environments.

## Build and Packaging Internals

### Backend Packaging with PyInstaller

The Python backend is packaged into a standalone executable using PyInstaller with specific configuration:

#### PyInstaller Command Structure
```bash
python -m PyInstaller \
  --onefile \
  --name recall-backend \
  main.py \
  --hidden-import uvicorn \
  --collect-all uvicorn \
  --add-data "app/static;app/static" \
  --add-data "migrations;migrations"
```

#### Key PyInstaller Options Explained
- **`--onefile`**: Creates a single executable file instead of a directory bundle
- **`--hidden-import uvicorn`**: Explicitly includes uvicorn module (FastAPI's ASGI server)
- **`--collect-all uvicorn`**: Includes all uvicorn submodules and dependencies
- **`--add-data "app/static;app/static"`**: Includes static assets with proper path resolution
- **`--add-data "migrations;migrations"`**: Includes database migration scripts

#### Build Environment Setup
The backend is built in an isolated virtual environment:
1. Creates temporary virtual environment (`backend/.venv-build`)
2. Installs PyInstaller and all project dependencies
3. Runs PyInstaller with the specified configuration
4. Outputs `recall-backend.exe` to `backend/dist/`

### Frontend Build with Vite

The React/TypeScript frontend uses Vite for fast, optimized builds:

#### Vite Configuration Details
```typescript
export default defineConfig({
  plugins: [react()],
  root: 'frontend',
  build: {
    outDir: '../dist/frontend',
    emptyOutDir: true,
  },
  base: './',
  server: {
    port: 3000,
  },
})
```

#### Build Process Flow
1. **Entry Resolution**: Starts from `frontend/index.html`
2. **Dependency Analysis**: Analyzes imports and builds dependency graph
3. **Code Transformation**: Transpiles TypeScript to JavaScript, processes JSX
4. **Asset Processing**: Optimizes CSS, images, and other assets
5. **Bundle Generation**: Creates optimized chunks with code splitting
6. **Output Generation**: Produces `dist/frontend/` with all static assets

### Electron Compilation

Electron TypeScript files are compiled using the TypeScript compiler:

#### TypeScript Configuration for Electron
```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "module": "CommonJS",
    "outDir": "dist/electron",
    "rootDir": "electron",
    "noEmitOnError": false
  },
  "include": ["electron/**/*.ts", "electron/**/*.d.ts"],
  "exclude": ["node_modules", "dist", "frontend"]
}
```

#### Compilation Output
- **Input**: `electron/main.ts`, `electron/preload.ts`
- **Output**: `dist/electron/main.js`, `dist/electron/preload.js`
- **Module System**: CommonJS (required for Electron main process)

### Electron Forge Packaging

Electron Forge orchestrates the final packaging process:

#### Forge Configuration Structure
```javascript
module.exports = {
  build: {
    files: ["dist/electron/**/*", "dist/frontend/**/*"],
  },
  packagerConfig: {
    asar: false,  // Disable ASAR packaging for easier debugging
    ignore: [     // Files to exclude from package
      /\.ts$/,    // TypeScript source files
      /src$/,     // Source directories
      /tsconfig\.json$/,
      /\.git$/,
      /backend\//, // Backend source (only include executable)
    ],
    extraResource: ["backend/dist/recall-backend.exe"],
  },
  makers: [      // Package formats to create
    { name: "@electron-forge/maker-squirrel" }, // Windows NSIS installer
    { name: "@electron-forge/maker-zip" },     // Cross-platform ZIP
    { name: "@electron-forge/maker-deb" },     // Debian packages
    { name: "@electron-forge/maker-rpm" },     // RPM packages
  ],
  plugins: [     // Security and functionality plugins
    new FusesPlugin({ /* Electron fuses configuration */ })
  ]
};
```

#### Packaging Process Details

1. **File Collection**: Gathers all files specified in `build.files` and `packagerConfig.extraResource`
2. **ASAR Creation**: Optionally creates ASAR archive (disabled in this config for debugging)
3. **Platform-Specific Packaging**: Creates installers appropriate for each target platform
4. **Code Signing**: Applies security fuses to prevent code modification

#### Electron Fuses Security
```javascript
new FusesPlugin({
  version: FuseVersion.V1,
  [FuseV1Options.RunAsNode]: false,                    // Prevent Node.js execution
  [FuseV1Options.EnableCookieEncryption]: true,       // Encrypt cookies
  [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false, // Disable NODE_OPTIONS
  [FuseV1Options.EnableNodeCliInspectArguments]: false, // Disable dev tools
  [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true, // Validate ASAR integrity
  [FuseV1Options.OnlyLoadAppFromAsar]: true,          // Only load from ASAR
})
```

### Runtime Architecture

#### Backend Startup Process
1. **Executable Launch**: `recall-backend.exe` starts the PyInstaller bundle
2. **Environment Detection**: Checks for frozen environment (`sys.frozen`)
3. **Logging Setup**: Configures file and console logging
4. **Port Binding**: Attempts to bind to specified PORT environment variable
5. **Uvicorn Server**: Starts FastAPI application with uvicorn ASGI server

#### Electron Main Process Flow
1. **App Ready**: Waits for Electron app ready event
2. **Protocol Registration**: Registers custom `app://` protocol for serving frontend
3. **Backend Startup**: Launches backend executable and waits for ready signal
4. **Window Creation**: Creates BrowserWindow after backend confirmation
5. **IPC Setup**: Establishes communication channels between processes

#### Inter-Process Communication
- **Backend ↔ Frontend**: HTTP API calls through localhost
- **Main Process ↔ Renderer**: Electron IPC for file operations and system integration
- **File System Monitoring**: Real-time file watching with fs.watch
- **Process Management**: Backend lifecycle management with spawn/kill

### Development vs Production Differences

#### Development Mode
- **Hot Reloading**: Vite dev server with HMR
- **Source Maps**: Full debugging capabilities
- **Console Access**: DevTools always available
- **Loose Security**: Some Electron security features relaxed

#### Production Mode
- **Optimized Bundles**: Minified and tree-shaken code
- **Security Hardened**: All Electron fuses enabled
- **Self-Contained**: Includes all dependencies
- **Code Signed**: Ready for distribution

### Build Artifacts Cleanup

The Makefile includes comprehensive cleanup targets:
- **Frontend**: Removes `dist/frontend` directory
- **Backend**: Removes `backend/dist` and `backend/build` directories
- **Electron**: Removes `dist/electron` directory
- **Virtual Environments**: Removes temporary build environments

This ensures clean, reproducible builds across different environments and prevents stale artifacts from affecting builds.
