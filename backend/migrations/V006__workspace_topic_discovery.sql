-- Migration: V006__workspace_topic_discovery
-- Description: Add tables for workspace-level topic discovery and learning paths
-- Created: 2025-09-27

-- Topic areas table: Major subject areas identified in workspaces
CREATE TABLE topic_areas (
    topic_area_id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    coverage_score REAL NOT NULL CHECK (coverage_score >= 0.0 AND coverage_score <= 1.0),
    concept_count INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    explored_percentage REAL NOT NULL DEFAULT 0.0 CHECK (explored_percentage >= 0.0 AND explored_percentage <= 1.0),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Topic-concept links table: Links concepts to topic areas
CREATE TABLE topic_concept_links (
    topic_concept_link_id TEXT PRIMARY KEY,
    topic_area_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    relevance_score REAL NOT NULL CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    explored BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (topic_area_id) REFERENCES topic_areas(topic_area_id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    UNIQUE(topic_area_id, concept_id)
);

-- Learning paths table: Recommended learning paths for users
CREATE TABLE learning_paths (
    learning_path_id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    topic_areas TEXT NOT NULL, -- JSON array of topic_area_ids
    estimated_hours INTEGER NOT NULL,
    difficulty_level TEXT NOT NULL CHECK (difficulty_level IN ('beginner', 'intermediate', 'advanced')),
    prerequisites TEXT, -- JSON array of prerequisite knowledge
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- Learning recommendations table: Specific recommendations for what to study next
CREATE TABLE learning_recommendations (
    recommendation_id TEXT PRIMARY KEY,
    workspace_id INTEGER NOT NULL,
    user_id TEXT, -- Optional user ID for multi-user support
    recommendation_type TEXT NOT NULL CHECK (recommendation_type IN ('quiz_performance', 'concept_gap', 'prerequisite', 'interest')),
    topic_area_id TEXT,
    concept_id TEXT,
    priority_score REAL NOT NULL CHECK (priority_score >= 0.0 AND priority_score <= 1.0),
    reason TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (topic_area_id) REFERENCES topic_areas(topic_area_id) ON DELETE SET NULL,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX idx_topic_areas_workspace ON topic_areas(workspace_id);
CREATE INDEX idx_topic_areas_coverage ON topic_areas(coverage_score);
CREATE INDEX idx_topic_concept_links_topic ON topic_concept_links(topic_area_id);
CREATE INDEX idx_topic_concept_links_concept ON topic_concept_links(concept_id);
CREATE INDEX idx_topic_concept_links_explored ON topic_concept_links(explored);
CREATE INDEX idx_learning_paths_workspace ON learning_paths(workspace_id);
CREATE INDEX idx_learning_recommendations_workspace ON learning_recommendations(workspace_id);
CREATE INDEX idx_learning_recommendations_type ON learning_recommendations(recommendation_type);
CREATE INDEX idx_learning_recommendations_priority ON learning_recommendations(priority_score DESC);
