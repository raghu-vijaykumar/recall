# Frontend Design Document

## Overview
The Recall frontend is a React-based single-page application built with TypeScript that provides the user interface for the study application. It features a tabbed interface for managing workspaces, exploring files, taking quizzes, and tracking progress.

## Architecture

### Core Framework
- **React 18+**: Component-based UI framework
- **TypeScript**: Type-safe JavaScript for better development experience
- **Vite**: Fast build tool and development server

### Application Structure

#### Main Application (`src/App.tsx`)
- **State Management**: Local state for current tab, selected workspace, and modal visibility
- **Event Handling**: Custom events for tab switching and workspace selection
- **Menu Integration**: Electron menu event listeners for navigation
- **Tab Rendering**: Conditional rendering based on current tab state

#### Component Organization
```
frontend/
├── src/
│   ├── App.tsx                 # Main application component
│   └── index.tsx              # Application entry point
├── components/
│   ├── workspaces/            # Workspace management
│   ├── file-explorer/         # File browsing interface
│   ├── quiz/                  # Quiz taking interface
│   └── progress/              # Progress tracking
├── core/
│   ├── api.ts                 # API client functions
│   └── types.ts               # TypeScript type definitions
└── shared/
    ├── utils.ts               # Shared utility functions
    └── html-loader.ts         # HTML component loading
```

### Key Components

#### 1. Workspaces Component
- **Purpose**: Manage study workspaces (folders containing study materials)
- **Features**:
  - List existing workspaces
  - Create new workspaces from folders
  - Workspace statistics and metadata
  - Modal dialogs for creation/editing

#### 2. File Explorer Component
- **Purpose**: Browse and view files within selected workspace
- **Features**:
  - Tree view of folder structure
  - File content preview
  - File type filtering (text-based files)
  - Search within workspace files

#### 3. Quiz Component
- **Purpose**: Generate and take quizzes based on workspace content
- **Features**:
  - Quiz generation from LLM
  - Question types (multiple choice, etc.)
  - Answer submission and scoring
  - Session tracking

#### 4. Progress Component
- **Purpose**: Track study progress and achievements
- **Features**:
  - Study session history
  - Quiz performance metrics
  - Achievement system
  - Progress visualization

### State Management
- **Local State**: React useState hooks for component-level state
- **Event System**: Custom DOM events for cross-component communication
- **No Global State**: Simple architecture without Redux/Zustand

### API Integration (`core/api.ts`)
- **REST Client**: Functions for backend API calls
- **Error Handling**: Consistent error handling for API responses
- **Type Safety**: TypeScript interfaces for API responses

### Type Definitions (`core/types.ts`)
- **API Types**: Mirrors backend Pydantic models
- **Component Props**: Interface definitions for component properties
- **Event Types**: Custom event type definitions

### Electron Integration
- **Preload Script**: Type-safe IPC communication
- **Menu Events**: Integration with application menu
- **File System Access**: IPC handlers for file operations
- **Custom Protocol**: `app://` protocol for packaged app

### Styling
- **CSS Modules**: Scoped styling with `style.css`
- **Theme Support**: Dark/light mode toggle
- **Responsive Design**: Adapts to window resizing

### Build and Development
- **Vite Configuration**: Custom build setup in `vite.config.ts`
- **TypeScript Config**: Separate configs for Electron and web
- **Development Server**: Hot reload during development
- **Production Build**: Optimized bundle for Electron packaging

### Key Features

#### Tab-Based Navigation
- **Workspaces Tab**: Workspace management and creation
- **Files Tab**: File browsing and content viewing
- **Quiz Tab**: Interactive quiz taking
- **Progress Tab**: Study analytics and achievements

#### Workspace Integration
- **Folder Selection**: Native OS folder picker via Electron
- **Auto-Switching**: Automatic tab switching after workspace selection
- **Context Awareness**: Components adapt based on selected workspace

#### File Operations
- **Tree View**: Hierarchical file browser
- **Content Loading**: Async file content retrieval
- **Type Filtering**: Only text-based files displayed

#### Quiz System
- **LLM Generation**: AI-powered question creation
- **Session Management**: Quiz attempt tracking
- **Scoring**: Automatic answer evaluation

#### Progress Tracking
- **Session History**: Study session logging
- **Statistics**: Performance metrics and trends
- **Gamification**: Achievement system

### Dependencies
- **React/React-DOM**: UI framework
- **TypeScript**: Type checking
- **Vite**: Build tool
- **Electron Types**: Type definitions for Electron APIs

### Development Workflow
- **Component Development**: Modular, reusable components
- **Type Safety**: Full TypeScript coverage
- **Hot Reload**: Fast development iteration
- **Electron Integration**: Seamless desktop app experience
