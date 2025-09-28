"""
Flattened File Workspace Analysis Service for extracting concepts and relationships from workspace files
Specialized for analyzing pre-flattened workspace content for improved performance
"""

import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import ConceptCreate, ConceptFileCreate
from . import KnowledgeGraphService
from .relationship_builders.base import RelationshipBuilder

# Concept extraction imports
from .concept_extraction.base import ConceptExtractor, RankingAlgorithm


class WorkspaceAnalysisService:
    """
    Specialized service for analyzing pre-flattened workspace files to extract concepts and build knowledge graphs.
    Optimized for analyzing workspace content that has been pre-processed into a single flattened file.
    Uses pluggable concept extractors and ranking algorithms for flexibility.
    """

    def __init__(
        self,
        db: AsyncSession,
        extractor: ConceptExtractor,
        ranker: RankingAlgorithm,
        kg_service: Optional[KnowledgeGraphService] = None,
        relationship_builder: Optional[RelationshipBuilder] = None,
    ):
        self.db = db
        self.extractor = extractor
        self.ranker = ranker
        self.kg_service = kg_service or KnowledgeGraphService(db)
        self.relationship_builder = relationship_builder

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

    async def analyze_workspace(
        self,
        workspace_id: int,
        workspace_path: str,
        use_flattened_file: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze a workspace using a pre-flattened file.

        Args:
            workspace_id: ID of the workspace to analyze
            workspace_path: Path to the workspace directory
            use_flattened_file: If True, force use of flattened file if available

        Returns:
            Analysis results with statistics
        """
        # Find the most recent flattened file
        flattened_file_path = self._find_recent_flattened_file(workspace_path)

        if not flattened_file_path:
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": ["No flattened file found in workspace"],
                "duration_seconds": 0.0,
                "message": "No flattened file found",
            }

        logging.info(
            f"[FLATTENED_ANALYSIS] Using flattened file for workspace {workspace_id}: {flattened_file_path}"
        )
        return await self.analyze_flattened_file(flattened_file_path, workspace_id)

    def _find_recent_flattened_file(self, workspace_path: str) -> Optional[str]:
        """Find the most recent flattened file in the workspace directory"""
        workspace_dir = Path(workspace_path)
        if not workspace_dir.exists() or not workspace_dir.is_dir():
            return None

        # Look for flattened files with the pattern flattened_workspace1_*.txt
        flattened_files = list(workspace_dir.glob("flattened_workspace1_*.txt"))

        if not flattened_files:
            return None

        # Return the most recently modified one
        most_recent = max(flattened_files, key=lambda f: f.stat().st_mtime)
        return str(most_recent)

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

        # Get the new file ID
        result = await self.db.execute(text("SELECT last_insert_rowid()"))
        return result.scalar()

    async def analyze_flattened_file(
        self, flattened_file_path: str, workspace_id: int
    ) -> Dict[str, Any]:
        """
        Analyze a flattened workspace file containing all workspace content.

        Args:
            flattened_file_path: Path to the flattened file
            workspace_id: ID of the workspace

        Returns:
            Analysis results with statistics
        """
        start_time = datetime.utcnow()
        logging.info(
            f"[FLATTENED_ANALYSIS] Starting flattened file analysis for workspace {workspace_id}"
        )
        logging.info(f"[FLATTENED_ANALYSIS] Flattened file path: {flattened_file_path}")

        flattened_path = Path(flattened_file_path)
        if not flattened_path.exists():
            logging.error(
                f"[FLATTENED_ANALYSIS] Flattened file does not exist: {flattened_file_path}"
            )
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": [f"Flattened file not found: {flattened_file_path}"],
                "duration_seconds": 0.0,
                "message": "Flattened file not found",
            }

        # Read the flattened file content
        content = self._read_file_content(flattened_path)
        if not content:
            logging.error(f"[FLATTENED_ANALYSIS] Flattened file is empty or unreadable")
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": ["Flattened file is empty or unreadable"],
                "duration_seconds": 0.0,
                "message": "Flattened file is empty",
            }

        content_length = len(content)
        logging.info(
            f"[FLATTENED_ANALYSIS] Read {content_length} characters from flattened file"
        )

        # Extract concepts using the configured extractor
        logging.info(f"[FLATTENED_ANALYSIS] Extracting concepts from flattened content")
        raw_concepts = self.extractor.extract_concepts(content)

        # Apply ranking using the configured ranker
        concepts_data = self.ranker.rank_concepts(raw_concepts, content)

        logging.info(
            f"[FLATTENED_ANALYSIS] Extracted and ranked {len(concepts_data)} concepts from flattened file"
        )

        if not concepts_data:
            logging.info(f"[FLATTENED_ANALYSIS] No concepts found in flattened file")
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 1,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": [],
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "message": "No concepts found in flattened file",
            }

        # Create or get file record for the flattened file
        logging.info(
            f"[FLATTENED_ANALYSIS] Creating/updating file record for flattened file"
        )
        file_id = await self._get_or_create_file_record(flattened_path, workspace_id)

        # Store concepts
        stored_concepts = []
        logging.info(f"[FLATTENED_ANALYSIS] Storing {len(concepts_data)} concepts")

        for i, concept_data in enumerate(concepts_data):
            if (i + 1) % 1000 == 0:  # Log progress every 10 concepts
                logging.info(
                    f"[FLATTENED_ANALYSIS] Processed {i + 1}/{len(concepts_data)} concepts"
                )

            try:
                # Create concept
                concept = await self.kg_service.create_concept(
                    ConceptCreate(
                        name=concept_data["name"],
                        description=concept_data.get("description", ""),
                    )
                )
                stored_concepts.append(concept)

                # Create concept-file link with line pointers (no snippet)
                await self.kg_service.create_concept_file_link(
                    ConceptFileCreate(
                        concept_id=concept.concept_id,
                        file_id=file_id,
                        workspace_id=workspace_id,
                        relevance_score=concept_data.get("score", 0.5),
                        start_line=concept_data.get("start_line"),
                        end_line=concept_data.get("end_line"),
                    )
                )
            except Exception as e:
                logging.warning(
                    f"[FLATTENED_ANALYSIS] Failed to store concept {concept_data['name']}: {e}"
                )
                continue

        logging.info(f"[FLATTENED_ANALYSIS] Stored {len(stored_concepts)} concepts")

        # Relationship inference removed
        stored_relationships = []

        # Build relationships using the configured relationship builder
        if self.relationship_builder and len(stored_concepts) > 1:
            logging.info(
                f"[FLATTENED_ANALYSIS] Building relationships using {self.relationship_builder.get_builder_type()} relationship builder"
            )
            try:
                relationship_results = (
                    await self.relationship_builder.build_relationships(workspace_id)
                )
                logging.info(
                    f"[FLATTENED_ANALYSIS] Relationship building completed: {relationship_results.get('relationships_created', 0)} relationships created"
                )
            except Exception as e:
                logging.error(f"[FLATTENED_ANALYSIS] Error building relationships: {e}")

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logging.info(
            f"[FLATTENED_ANALYSIS] Flattened file analysis completed in {duration:.2f} seconds"
        )

        return {
            "workspace_id": workspace_id,
            "files_analyzed": 1,
            "concepts_created": len(stored_concepts),
            "relationships_created": len(stored_relationships),
            "errors": [],
            "duration_seconds": duration,
        }
