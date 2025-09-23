# Backend Design Document

## Overview
The Recall backend is a FastAPI-based REST API that serves as the core data and business logic layer for the Recall study application. It manages workspaces, files, quiz generation, progress tracking, and search functionality.

## Architecture

### Core Components

#### 1. FastAPI Application (`app.py`)
- **Purpose**: Main application entry point with CORS configuration and router inclusion
- **Key Features**:
  - Lifespan management for clean startup/shutdown
  - CORS middleware for Electron app integration
  - Static file serving for file assets
  - Router mounting for different API endpoints

#### 2. Data Models (`models/`)
- **Workspace Models**: Define workspace types (folder-based, custom), creation/update operations, and statistics
- **File Models**: Handle file metadata, types, tree structures, and statistics
- **Quiz Models**: Define question types, difficulty levels, quiz sessions, and results
- **Progress Models**: Track study sessions, achievements, and gamification stats

All models use Pydantic for validation and serialization.

#### 3. API Routes (`routes/`)
- **Workspaces Router**: CRUD operations for workspace management
- **Files Router**: File system operations and metadata retrieval
- **Quiz Router**: Quiz generation and session management
- **Progress Router**: Progress tracking and statistics
- **Search Router**: Full-text search across workspace content

#### 4. Services Layer (`services/`)
- **DatabaseService**: SQLite operations with connection management and utility methods
- **MigrationService**: Database schema versioning and migration handling
- **WorkspaceService**: Business logic for workspace operations
- **FileService**: File system interactions and metadata processing

#### 5. LLM Integration (`llm_clients/`)
- **Factory Pattern**: Singleton factory for LLM client management
- **Supported Providers**: Gemini and Ollama
- **Features**:
  - Rate limiting
  - Retry mechanisms
  - Configurable API keys and settings
- **Usage**: Quiz generation and content analysis

### Database Design

#### SQLite Database
- **Location**: `~/.recall/recall.db`
- **Migration System**: Version-controlled schema changes
- **Key Tables**:
  - `workspaces`: Workspace metadata and configuration
  - `files`: File metadata and relationships
  - `quiz_sessions`: Quiz attempt tracking
  - `progress`: Study progress and achievements
  - `search_index`: Full-text search data

#### Migration System
- **Version Control**: Sequential migration files in `migrations/`
- **Automatic Application**: Migrations applied on startup
- **Rollback Support**: Ability to revert schema changes

### API Design

#### RESTful Endpoints
- **Base URL**: `/api/`
- **Response Format**: JSON with consistent error handling
- **Authentication**: None (local application)

#### Key Endpoints
- `GET/POST/PUT/DELETE /api/workspaces` - Workspace management
- `GET/POST /api/files` - File operations
- `POST /api/quiz/generate` - Quiz creation
- `GET/POST /api/progress` - Progress tracking
- `GET /api/search` - Content search

### Error Handling
- **HTTP Status Codes**: Standard REST status codes
- **Error Responses**: Consistent JSON error format
- **Logging**: Comprehensive logging with electron-log integration

### Dependencies
- **FastAPI**: Web framework
- **SQLite3**: Database
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server
- **LLM Libraries**: Google AI SDK, Ollama client

## Deployment
- **Packaging**: PyInstaller for executable distribution
- **Process Management**: Electron main process handles backend lifecycle
- **Port Management**: Automatic port selection (8000+) to avoid conflicts
