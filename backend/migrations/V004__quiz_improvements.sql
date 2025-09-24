-- Migration: V004__quiz_improvements
-- Description: Add LLM and spaced repetition support to quiz system, plus Knowledge Graph tables
-- Created: 2025-09-24T14:31:00

-- Knowledge Graph Tables

-- Concepts table: represents individual topics or ideas extracted from workspace
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY, -- UUID
    name TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Relationships table: represents connections between concepts
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY, -- UUID
    source_concept_id TEXT NOT NULL,
    target_concept_id TEXT NOT NULL,
    type TEXT NOT NULL, -- 'relates_to', 'dives_deep_to', etc.
    strength REAL, -- optional strength/relevance score
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    FOREIGN KEY (target_concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
);

-- Concept-files junction table: links concepts to files where they appear
CREATE TABLE IF NOT EXISTS concept_files (
    concept_file_id TEXT PRIMARY KEY, -- UUID
    concept_id TEXT NOT NULL,
    file_id INTEGER NOT NULL, -- references existing files table
    workspace_id INTEGER NOT NULL, -- references existing workspaces table
    snippet TEXT, -- relevant text snippet
    relevance_score REAL, -- relevance score
    last_accessed_at DATETIME,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Quiz Improvements Tables

-- Add columns to questions table for LLM-generated questions
ALTER TABLE questions ADD COLUMN generated_by_llm BOOLEAN DEFAULT FALSE;
ALTER TABLE questions ADD COLUMN generation_prompt TEXT;
ALTER TABLE questions ADD COLUMN confidence_score REAL;
ALTER TABLE questions ADD COLUMN kg_concept_ids TEXT; -- JSON array of concept IDs

-- Create answers table for tracking individual question performance
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    answer_text TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    time_taken INTEGER NOT NULL, -- Time in seconds
    confidence_level INTEGER, -- User's confidence rating (1-5)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

-- Create spaced_repetition_data table for spaced repetition algorithm
CREATE TABLE IF NOT EXISTS spaced_repetition_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    ease_factor REAL DEFAULT 2.5,
    interval_days INTEGER DEFAULT 1,
    review_count INTEGER DEFAULT 0,
    next_review DATETIME,
    kg_concept_id TEXT, -- Optional link to knowledge graph concept
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_files_concept ON concept_files(concept_id);
CREATE INDEX IF NOT EXISTS idx_concept_files_file ON concept_files(file_id);
CREATE INDEX IF NOT EXISTS idx_concept_files_workspace ON concept_files(workspace_id);
CREATE INDEX IF NOT EXISTS idx_answers_question ON answers(question_id);
CREATE INDEX IF NOT EXISTS idx_answers_created_at ON answers(created_at);
CREATE INDEX IF NOT EXISTS idx_spaced_repetition_question ON spaced_repetition_data(question_id);
CREATE INDEX IF NOT EXISTS idx_spaced_repetition_next_review ON spaced_repetition_data(next_review);
