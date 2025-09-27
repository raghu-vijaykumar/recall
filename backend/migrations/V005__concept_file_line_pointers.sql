-- Migration: V005__concept_file_line_pointers
-- Description: Add line number pointers to concept_files table for precise navigation
-- Created: 2025-09-26

-- Add line number columns to concept_files table
ALTER TABLE concept_files ADD COLUMN start_line INTEGER;
ALTER TABLE concept_files ADD COLUMN end_line INTEGER;

-- Create index for line-based queries (optional, for performance)
CREATE INDEX IF NOT EXISTS idx_concept_files_lines ON concept_files(start_line, end_line);
