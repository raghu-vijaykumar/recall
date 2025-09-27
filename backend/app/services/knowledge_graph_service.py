"""
Knowledge Graph Service for managing concepts, relationships, and concept-file links
"""

import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, UTC
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Concept,
    Relationship,
    ConceptFile,
    ConceptCreate,
    RelationshipCreate,
    ConceptFileCreate,
)


class KnowledgeGraphService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_concept(self, concept_data: ConceptCreate) -> Concept:
        """Create a new concept"""
        # Generate UUID if not provided
        if not hasattr(concept_data, "concept_id") or not concept_data.concept_id:
            concept_data.concept_id = str(uuid.uuid4())

        # Create concept record
        query = text(
            """
            INSERT INTO concepts (concept_id, name, description, created_at, updated_at)
            VALUES (:concept_id, :name, :description, :created_at, :updated_at)
        """
        )

        await self.db.execute(
            query,
            {
                "concept_id": concept_data.concept_id,
                "name": concept_data.name,
                "description": concept_data.description,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
            },
        )

        # Commit the concept creation
        await self.db.commit()

        # Return the created concept
        return Concept(
            concept_id=concept_data.concept_id,
            name=concept_data.name,
            description=concept_data.description,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID"""
        query = text("SELECT * FROM concepts WHERE concept_id = :concept_id")
        result = await self.db.execute(query, {"concept_id": concept_id})
        row = result.fetchone()

        if row:
            return Concept(
                concept_id=row.concept_id,
                name=row.name,
                description=row.description,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        return None

    async def update_concept(
        self,
        concept_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Concept]:
        """Update a concept"""
        updates = {}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description

        if not updates:
            return await self.get_concept(concept_id)

        updates["updated_at"] = datetime.now(UTC)

        # Build dynamic update query
        set_clause = ", ".join(f"{key} = :{key}" for key in updates.keys())
        query = text(f"UPDATE concepts SET {set_clause} WHERE concept_id = :concept_id")

        updates["concept_id"] = concept_id

        await self.db.execute(query, updates)
        await self.db.commit()

        return await self.get_concept(concept_id)

    async def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept (cascade will handle relationships and concept_files)"""
        query = text("DELETE FROM concepts WHERE concept_id = :concept_id")
        result = await self.db.execute(query, {"concept_id": concept_id})
        await self.db.commit()
        return result.rowcount > 0

    async def create_relationship(
        self, relationship_data: RelationshipCreate
    ) -> Relationship:
        """Create a relationship between concepts"""
        # Check if relationship already exists
        existing_query = text(
            """
            SELECT relationship_id FROM relationships
            WHERE (source_concept_id = :source AND target_concept_id = :target)
               OR (source_concept_id = :target AND target_concept_id = :source)
        """
        )

        result = await self.db.execute(
            existing_query,
            {
                "source": relationship_data.source_concept_id,
                "target": relationship_data.target_concept_id,
            },
        )

        existing = result.fetchone()
        if existing:
            # Return existing relationship instead of creating duplicate
            return await self._get_relationship_by_id(existing.relationship_id)

        # Generate UUID if not provided
        if (
            not hasattr(relationship_data, "relationship_id")
            or not relationship_data.relationship_id
        ):
            relationship_data.relationship_id = str(uuid.uuid4())

        query = text(
            """
            INSERT INTO relationships (relationship_id, source_concept_id, target_concept_id,
                                     type, strength, created_at)
            VALUES (:relationship_id, :source_concept_id, :target_concept_id,
                   :type, :strength, :created_at)
        """
        )

        await self.db.execute(
            query,
            {
                "relationship_id": relationship_data.relationship_id,
                "source_concept_id": relationship_data.source_concept_id,
                "target_concept_id": relationship_data.target_concept_id,
                "type": relationship_data.type,
                "strength": relationship_data.strength,
                "created_at": datetime.now(UTC),
            },
        )

        # Commit the relationship creation
        await self.db.commit()

        return Relationship(
            relationship_id=relationship_data.relationship_id,
            source_concept_id=relationship_data.source_concept_id,
            target_concept_id=relationship_data.target_concept_id,
            type=relationship_data.type,
            strength=relationship_data.strength,
            created_at=datetime.now(UTC),
        )

    async def _get_relationship_by_id(self, relationship_id: str) -> Relationship:
        """Get a relationship by ID"""
        query = text(
            "SELECT * FROM relationships WHERE relationship_id = :relationship_id"
        )
        result = await self.db.execute(query, {"relationship_id": relationship_id})
        row = result.fetchone()

        if row:
            return Relationship(
                relationship_id=row.relationship_id,
                source_concept_id=row.source_concept_id,
                target_concept_id=row.target_concept_id,
                type=row.type,
                strength=row.strength,
                created_at=row.created_at,
            )
        return None

    async def get_relationships_for_concept(
        self, concept_id: str
    ) -> List[Relationship]:
        """Get all relationships for a concept (as source or target)"""
        query = text(
            """
            SELECT * FROM relationships
            WHERE source_concept_id = :concept_id OR target_concept_id = :concept_id
        """
        )
        result = await self.db.execute(query, {"concept_id": concept_id})
        rows = result.fetchall()

        relationships = []
        for row in rows:
            relationships.append(
                Relationship(
                    relationship_id=row.relationship_id,
                    source_concept_id=row.source_concept_id,
                    target_concept_id=row.target_concept_id,
                    type=row.type,
                    strength=row.strength,
                    created_at=row.created_at,
                )
            )

        return relationships

    async def create_concept_file_link(
        self, link_data: ConceptFileCreate
    ) -> ConceptFile:
        """Create a link between a concept and a file"""
        # Generate UUID if not provided
        if not hasattr(link_data, "concept_file_id") or not link_data.concept_file_id:
            link_data.concept_file_id = str(uuid.uuid4())

        query = text(
            """
            INSERT INTO concept_files (concept_file_id, concept_id, file_id, workspace_id,
                                     snippet, relevance_score, last_accessed_at, start_line, end_line)
            VALUES (:concept_file_id, :concept_id, :file_id, :workspace_id,
                   :snippet, :relevance_score, :last_accessed_at, :start_line, :end_line)
        """
        )

        await self.db.execute(
            query,
            {
                "concept_file_id": link_data.concept_file_id,
                "concept_id": link_data.concept_id,
                "file_id": link_data.file_id,
                "workspace_id": link_data.workspace_id,
                "snippet": link_data.snippet,
                "relevance_score": link_data.relevance_score,
                "last_accessed_at": link_data.last_accessed_at,
                "start_line": link_data.start_line,
                "end_line": link_data.end_line,
            },
        )

        # Commit the concept-file link creation
        await self.db.commit()

        return ConceptFile(
            concept_file_id=link_data.concept_file_id,
            concept_id=link_data.concept_id,
            file_id=link_data.file_id,
            workspace_id=link_data.workspace_id,
            snippet=link_data.snippet,
            relevance_score=link_data.relevance_score,
            last_accessed_at=link_data.last_accessed_at,
            start_line=link_data.start_line,
            end_line=link_data.end_line,
        )

    async def get_files_for_concept(self, concept_id: str) -> List[ConceptFile]:
        """Get all files linked to a concept"""
        query = text("SELECT * FROM concept_files WHERE concept_id = :concept_id")
        result = await self.db.execute(query, {"concept_id": concept_id})
        rows = result.fetchall()

        files = []
        for row in rows:
            files.append(
                ConceptFile(
                    concept_file_id=row.concept_file_id,
                    concept_id=row.concept_id,
                    file_id=row.file_id,
                    workspace_id=row.workspace_id,
                    snippet=row.snippet,
                    relevance_score=row.relevance_score,
                    last_accessed_at=row.last_accessed_at,
                    start_line=row.start_line,
                    end_line=row.end_line,
                )
            )

        return files

    async def get_concepts_for_file(self, file_id: int) -> List[ConceptFile]:
        """Get all concepts linked to a file"""
        query = text("SELECT * FROM concept_files WHERE file_id = :file_id")
        result = await self.db.execute(query, {"file_id": file_id})
        rows = result.fetchall()

        concepts = []
        for row in rows:
            concepts.append(
                ConceptFile(
                    concept_file_id=row.concept_file_id,
                    concept_id=row.concept_id,
                    file_id=row.file_id,
                    workspace_id=row.workspace_id,
                    snippet=row.snippet,
                    relevance_score=row.relevance_score,
                    last_accessed_at=row.last_accessed_at,
                    start_line=row.start_line,
                    end_line=row.end_line,
                )
            )

        return concepts

    async def search_concepts(self, query_str: str, limit: int = 50) -> List[Concept]:
        """Search concepts by name or description"""
        search_query = text(
            """
            SELECT * FROM concepts
            WHERE name LIKE :query OR description LIKE :query
            ORDER BY name
            LIMIT :limit
        """
        )

        result = await self.db.execute(
            search_query, {"query": f"%{query_str}%", "limit": limit}
        )
        rows = result.fetchall()

        concepts = []
        for row in rows:
            concepts.append(
                Concept(
                    concept_id=row.concept_id,
                    name=row.name,
                    description=row.description,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

        return concepts

    async def get_workspace_concepts(self, workspace_id: int) -> List[Dict[str, Any]]:
        """Get all concepts linked to files in a workspace"""
        query = text(
            """
            SELECT DISTINCT c.*, cf.relevance_score, cf.last_accessed_at
            FROM concepts c
            JOIN concept_files cf ON c.concept_id = cf.concept_id
            WHERE cf.workspace_id = :workspace_id
            ORDER BY cf.relevance_score DESC, cf.last_accessed_at DESC
        """
        )

        # Retry logic for SQLite database locked errors
        import asyncio

        max_retries = 10
        for attempt in range(max_retries):
            try:
                result = await self.db.execute(query, {"workspace_id": workspace_id})
                rows = result.fetchall()
                break
            except Exception as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    # Longer exponential backoff for database locks
                    delay = 0.5 * (2**attempt)  # 0.5s, 1s, 2s, 4s, 8s...
                    print(
                        f"[KG] Database locked, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise e

        concepts = []
        for row in rows:
            concepts.append(
                {
                    "concept": Concept(
                        concept_id=row.concept_id,
                        name=row.name,
                        description=row.description,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    ),
                    "relevance_score": row.relevance_score,
                    "last_accessed_at": row.last_accessed_at,
                }
            )

        return concepts

    async def get_concept_graph(
        self, concept_id: str, depth: int = 2
    ) -> Dict[str, Any]:
        """Get a subgraph around a concept with relationships"""
        # This is a simplified implementation - in practice you'd want more sophisticated
        # graph traversal with proper depth limiting

        # Get the central concept
        central_concept = await self.get_concept(concept_id)
        if not central_concept:
            return {"nodes": [], "edges": []}

        nodes = [central_concept]
        edges = []

        # Get relationships
        relationships = await self.get_relationships_for_concept(concept_id)

        # Add related concepts to nodes
        concept_ids = {concept_id}
        for rel in relationships:
            edges.append(
                {
                    "id": rel.relationship_id,
                    "source": rel.source_concept_id,
                    "target": rel.target_concept_id,
                    "type": rel.type,
                    "strength": rel.strength,
                }
            )

            # Add target concept if not already included
            if rel.target_concept_id not in concept_ids:
                target_concept = await self.get_concept(rel.target_concept_id)
                if target_concept:
                    nodes.append(target_concept)
                    concept_ids.add(rel.target_concept_id)

            # Add source concept if not already included
            if rel.source_concept_id not in concept_ids:
                source_concept = await self.get_concept(rel.source_concept_id)
                if source_concept:
                    nodes.append(source_concept)
                    concept_ids.add(rel.source_concept_id)

        return {
            "nodes": [
                {"id": c.concept_id, "name": c.name, "description": c.description}
                for c in nodes
            ],
            "edges": edges,
        }
