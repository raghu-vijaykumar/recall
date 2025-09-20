-- Migration: V001__initial_schema
-- Description: Initial database schema for Recall Study App
-- Created: 2025-09-20T09:26:00

-- Workspaces table
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    type TEXT NOT NULL DEFAULT 'study',
    color TEXT DEFAULT '#007bff',
    folder_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Files table
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT,
    question_count INTEGER DEFAULT 0,
    last_processed DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    UNIQUE(workspace_id, path)
);

-- Questions table
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    question_text TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    options TEXT, -- JSON array for multiple choice
    explanation TEXT,
    difficulty TEXT DEFAULT 'medium',
    tags TEXT, -- JSON array
    times_asked INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    last_asked DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

-- Quiz sessions table
CREATE TABLE IF NOT EXISTS quiz_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    file_ids TEXT, -- JSON array of file IDs
    question_count INTEGER DEFAULT 10,
    difficulty_filter TEXT,
    question_types TEXT, -- JSON array
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    current_question_index INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    correct_answers INTEGER DEFAULT 0,
    total_time INTEGER DEFAULT 0,
    status TEXT DEFAULT 'created',
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Session questions junction table
CREATE TABLE IF NOT EXISTS session_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    question_order INTEGER NOT NULL,
    user_answer TEXT,
    is_correct BOOLEAN,
    time_taken INTEGER,
    confidence_level INTEGER,
    answered_at DATETIME,
    FOREIGN KEY (session_id) REFERENCES quiz_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
    UNIQUE(session_id, question_id)
);

-- Progress tracking table
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    workspace_id INTEGER NOT NULL,
    file_id INTEGER,
    question_id INTEGER,
    session_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    action_type TEXT NOT NULL,
    value REAL,
    metadata TEXT, -- JSON
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES quiz_sessions(id) ON DELETE CASCADE
);

-- Achievements table
CREATE TABLE IF NOT EXISTS achievements (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    icon TEXT NOT NULL,
    target_value INTEGER NOT NULL,
    category TEXT NOT NULL
);

-- User achievements junction table
CREATE TABLE IF NOT EXISTS user_achievements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    achievement_id TEXT NOT NULL,
    unlocked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    progress REAL DEFAULT 0,
    current_value INTEGER DEFAULT 0,
    FOREIGN KEY (achievement_id) REFERENCES achievements(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_files_workspace ON files(workspace_id);
CREATE INDEX IF NOT EXISTS idx_questions_file ON questions(file_id);
CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON quiz_sessions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_progress_workspace ON progress(workspace_id);
CREATE INDEX IF NOT EXISTS idx_progress_timestamp ON progress(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_questions_session ON session_questions(session_id);

-- Triggers to update updated_at timestamps
CREATE TRIGGER IF NOT EXISTS update_workspace_timestamp
    AFTER UPDATE ON workspaces
    FOR EACH ROW
    BEGIN
        UPDATE workspaces SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER IF NOT EXISTS update_file_timestamp
    AFTER UPDATE ON files
    FOR EACH ROW
    BEGIN
        UPDATE files SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
