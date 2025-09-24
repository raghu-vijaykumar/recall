"""
Workspace Analysis Service for extracting concepts and relationships from user files
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import ConceptCreate, RelationshipCreate, ConceptFileCreate
from . import KnowledgeGraphService


class WorkspaceAnalysisService:
    """
    Service for analyzing workspace files to extract concepts and build knowledge graphs
    """

    def __init__(
        self, db: AsyncSession, kg_service: Optional[KnowledgeGraphService] = None
    ):
        self.db = db
        self.kg_service = kg_service or KnowledgeGraphService(db)

        # File types to analyze
        self.supported_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
        }

        # Common stop words for preprocessing
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }

    async def analyze_workspace(
        self,
        workspace_id: int,
        workspace_path: str,
        force_reanalysis: bool = False,
        file_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a workspace to extract concepts and relationships

        Args:
            workspace_id: ID of the workspace
            workspace_path: Path to the workspace directory
            force_reanalysis: If True, re-analyze all files
            file_paths: Specific files to analyze (for incremental updates)

        Returns:
            Analysis results with statistics
        """
        start_time = datetime.utcnow()

        # Get files to analyze
        if file_paths:
            files_to_analyze = [Path(workspace_path) / fp for fp in file_paths]
        else:
            files_to_analyze = self._discover_files(workspace_path)

        analyzed_files = 0
        concepts_created = 0
        relationships_created = 0
        errors = []

        for file_path in files_to_analyze:
            try:
                if not file_path.exists() or not file_path.is_file():
                    continue

                # Check if file needs analysis
                if not force_reanalysis and await self._file_already_analyzed(
                    file_path, workspace_id
                ):
                    continue

                # Analyze file content
                file_concepts, file_relationships = await self._analyze_file(
                    file_path, workspace_id
                )

                concepts_created += len(file_concepts)
                relationships_created += len(file_relationships)
                analyzed_files += 1

            except Exception as e:
                errors.append(f"Error analyzing {file_path}: {str(e)}")

        end_time = datetime.utcnow()

        return {
            "workspace_id": workspace_id,
            "files_analyzed": analyzed_files,
            "concepts_created": concepts_created,
            "relationships_created": relationships_created,
            "errors": errors,
            "duration_seconds": (end_time - start_time).total_seconds(),
        }

    def _discover_files(self, workspace_path: str) -> List[Path]:
        """Discover all analyzable files in the workspace"""
        workspace_dir = Path(workspace_path)
        if not workspace_dir.exists() or not workspace_dir.is_dir():
            return []

        files = []
        for ext in self.supported_extensions:
            files.extend(workspace_dir.rglob(f"*{ext}"))

        return files

    async def _file_already_analyzed(self, file_path: Path, workspace_id: int) -> bool:
        """Check if file has already been analyzed"""
        query = text(
            """
            SELECT COUNT(*) FROM concept_files cf
            JOIN files f ON cf.file_id = f.id
            WHERE f.path = :file_path AND cf.workspace_id = :workspace_id
            """
        )

        result = await self.db.execute(
            query, {"file_path": str(file_path), "workspace_id": workspace_id}
        )

        count = result.scalar()
        return count > 0

    async def _analyze_file(
        self, file_path: Path, workspace_id: int
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Analyze a single file to extract concepts and relationships

        Returns:
            Tuple of (concepts_data, relationships_data)
        """
        # Read and preprocess content
        content = self._read_file_content(file_path)
        if not content:
            return [], []

        cleaned_content = self._preprocess_text(content)

        # Extract concepts
        concepts = self._extract_concepts(cleaned_content)

        # Get or create file record
        file_id = await self._get_or_create_file_record(file_path, workspace_id)

        # Store concepts and relationships
        stored_concepts = []
        stored_relationships = []

        for concept_data in concepts:
            # Create concept
            concept = await self.kg_service.create_concept(
                ConceptCreate(
                    name=concept_data["name"],
                    description=concept_data.get("description", ""),
                )
            )
            stored_concepts.append(concept)

            # Create concept-file link
            await self.kg_service.create_concept_file_link(
                ConceptFileCreate(
                    concept_id=concept.concept_id,
                    file_id=file_id,
                    workspace_id=workspace_id,
                    snippet=concept_data.get("snippet", ""),
                    relevance_score=concept_data.get("score", 0.5),
                )
            )

        # Infer relationships between concepts
        relationships = self._infer_relationships(stored_concepts, cleaned_content)
        for rel_data in relationships:
            try:
                relationship = await self.kg_service.create_relationship(
                    RelationshipCreate(
                        source_concept_id=rel_data["source_id"],
                        target_concept_id=rel_data["target_id"],
                        type=rel_data["type"],
                        strength=rel_data.get("strength", 0.5),
                    )
                )
                stored_relationships.append(relationship)
            except Exception:
                # Skip duplicate relationships
                pass

        return stored_concepts, stored_relationships

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding handling"""
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Try with errors='ignore'
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        except Exception:
            return ""

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not text:
            return ""

        # Remove markdown formatting
        text = re.sub(r"#+\s*", "", text)  # Headers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Links

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove extra whitespace
        text = text.strip()

        return text

    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from text using simple heuristics
        In production, this would use spaCy, KeyBERT, or other NLP libraries
        """
        concepts = []

        if not text:
            return concepts

        # Simple noun phrase extraction (basic implementation)
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Extract potential concepts (capitalized words, technical terms)
            words = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", sentence)

            for word in words:
                word_lower = word.lower()
                if word_lower not in self.stop_words and len(word) > 3:
                    # Find context snippet
                    snippet_start = max(0, sentence.find(word) - 50)
                    snippet_end = min(
                        len(sentence), sentence.find(word) + len(word) + 50
                    )
                    snippet = sentence[snippet_start:snippet_end].strip()

                    concepts.append(
                        {
                            "name": word,
                            "description": f"Concept mentioned in context: {snippet[:100]}...",
                            "snippet": snippet,
                            "score": 0.7,  # Basic relevance score
                        }
                    )

        # Remove duplicates and limit
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept["name"].lower() not in seen:
                seen.add(concept["name"].lower())
                unique_concepts.append(concept)

        return unique_concepts[:50]  # Limit concepts per file

    def _infer_relationships(
        self, concepts: List[Any], text: str
    ) -> List[Dict[str, Any]]:
        """
        Infer relationships between concepts based on co-occurrence and context
        """
        relationships = []

        # Simple co-occurrence based relationships
        concept_names = [c.name.lower() for c in concepts]

        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue

                name1 = concept1.name.lower()
                name2 = concept2.name.lower()

                # Check if concepts appear in same sentences
                sentences = re.split(r"[.!?]+", text)
                co_occurrences = 0

                for sentence in sentences:
                    if name1 in sentence.lower() and name2 in sentence.lower():
                        co_occurrences += 1

                if co_occurrences > 0:
                    # Determine relationship type based on context
                    rel_type = "relates_to"

                    # Check for hierarchical indicators
                    if any(
                        word in text.lower()
                        for word in ["type of", "kind of", "subtype", "extends"]
                    ):
                        rel_type = "dives_deep_to"

                    relationships.append(
                        {
                            "source_id": concept1.concept_id,
                            "target_id": concept2.concept_id,
                            "type": rel_type,
                            "strength": min(
                                0.9, co_occurrences * 0.3
                            ),  # Scale strength
                        }
                    )

        return relationships

    async def _get_or_create_file_record(
        self, file_path: Path, workspace_id: int
    ) -> int:
        """Get or create file record in database"""
        # Check if file exists
        query = text(
            "SELECT id FROM files WHERE path = :path AND workspace_id = :workspace_id"
        )

        result = await self.db.execute(
            query, {"path": str(file_path), "workspace_id": workspace_id}
        )

        file_record = result.fetchone()

        if file_record:
            return file_record.id

        # Create new file record
        insert_query = text(
            """
            INSERT INTO files (workspace_id, name, path, file_type, size, created_at, updated_at)
            VALUES (:workspace_id, :name, :path, :file_type, :size, :created_at, :updated_at)
            """
        )

        file_size = file_path.stat().st_size if file_path.exists() else 0
        file_type = file_path.suffix.lstrip(".") if file_path.suffix else "unknown"

        await self.db.execute(
            insert_query,
            {
                "workspace_id": workspace_id,
                "name": file_path.name,
                "path": str(file_path),
                "file_type": file_type,
                "size": file_size,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )

        await self.db.commit()

        # Get the new file ID
        result = await self.db.execute(text("SELECT last_insert_rowid()"))
        return result.scalar()
