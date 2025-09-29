-- Migration: V008__topics_only_architecture
-- Description: Implement simplified topics-only architecture by removing concepts and adding topic relationships
-- Created: 2025-09-29

-- Remove concept-related tables from the schema
DROP TABLE IF EXISTS topic_concept_links;
DROP TABLE IF EXISTS concept_files;
DROP TABLE IF EXISTS relationships;
DROP TABLE IF EXISTS concepts;

-- Update workspace_topics table to be purely topic-focused (removing concept references)
-- Note: We'll keep topic_areas but modify to be truly topic-only

-- Create new knowledge graph tables for topic-to-topic relationships
CREATE TABLE topic_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_topic_id TEXT NOT NULL,
    target_topic_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('relates_to', 'builds_on', 'contrasts_with', 'contains', 'precedes')),
    strength REAL NOT NULL DEFAULT 0.5 CHECK (strength >= 0.0 AND strength <= 1.0),
    reasoning TEXT, -- LLM-generated reasoning for the relationship
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_topic_id) REFERENCES topic_areas(topic_area_id) ON DELETE CASCADE,
    FOREIGN KEY (target_topic_id) REFERENCES topic_areas(topic_area_id) ON DELETE CASCADE,
    UNIQUE(source_topic_id, target_topic_id, relationship_type)
);

-- Add file change detection fields to workspace_files (re-purposed for topic discovery)
-- The existing content_hash column will be used for change detection
-- Add columns for topic discovery change tracking
ALTER TABLE files ADD COLUMN topic_discovery_processed BOOLEAN DEFAULT FALSE;
ALTER TABLE files ADD COLUMN topic_discovery_processed_at DATETIME;

-- Rename topic_areas to make it clear these are the core topics
-- (keeping the same structure but ensuring no concept dependencies)

-- Add indexes for topic relationships
CREATE INDEX idx_topic_relationships_source ON topic_relationships(source_topic_id);
CREATE INDEX idx_topic_relationships_target ON topic_relationships(target_topic_id);
CREATE INDEX idx_topic_relationships_type ON topic_relationships(relationship_type);
CREATE INDEX idx_topic_relationships_strength ON topic_relationships(strength DESC);

-- Add indexes for file change detection
CREATE INDEX idx_files_topic_processed ON files(topic_discovery_processed, topic_discovery_processed_at);

-- Update learning_paths table to reference topic_areas (already does)
-- learning_paths.topic_areas column already stores topic_area_ids as JSON array

-- Update learning_recommendations to reference topic_areas (already does)
-- learning_recommendations.topic_area_id already references topic_areas.topic_area_id
