"""
Embedding-based relationship builder for workspace analysis.
Builds relationships between concepts using semantic similarity from embeddings.
"""

import math
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base import RelationshipBuilder
from ...models import RelationshipCreate
from ..embedding_service import EmbeddingService


class EmbeddingRelationshipBuilder(RelationshipBuilder):
    """
    Relationship builder that uses embeddings to create semantic relationships
    between concepts based on their similarity in vector space.
    """

    def __init__(
        self,
        db: AsyncSession,
        kg_service: Any,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the embedding relationship builder.

        Args:
            db: Database session
            kg_service: Knowledge graph service instance
            embedding_service: Embedding service for generating embeddings
            similarity_threshold: Minimum similarity score for creating relationships
        """
        super().__init__(db, kg_service)
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold

    async def build_relationships(self, workspace_id: int) -> Dict[str, Any]:
        """
        Build embedding-based relationships between concepts in the workspace.

        Args:
            workspace_id: ID of the workspace to build relationships for

        Returns:
            Dictionary with relationship building results
        """
        if not self.embedding_service:
            logging.warning("[EMBEDDING_RELATIONSHIPS] No embedding service available")
            return {
                "relationships_created": 0,
                "errors": ["No embedding service available"],
            }

        logging.info(
            f"[EMBEDDING_RELATIONSHIPS] Building relationships for workspace {workspace_id}"
        )

        try:
            # Get all concepts for this workspace
            query = text(
                """
                SELECT DISTINCT c.concept_id, c.name
                FROM concepts c
                JOIN concept_files cf ON c.concept_id = cf.concept_id
                WHERE cf.workspace_id = :workspace_id
            """
            )

            result = await self.db.execute(query, {"workspace_id": workspace_id})
            concept_rows = result.fetchall()

            if len(concept_rows) < 2:
                logging.info(
                    f"[EMBEDDING_RELATIONSHIPS] Not enough concepts ({len(concept_rows)}) to build relationships"
                )
                return {
                    "relationships_created": 0,
                    "errors": [],
                }

            concepts = [
                {"id": row.concept_id, "name": row.name} for row in concept_rows
            ]
            logging.info(
                f"[EMBEDDING_RELATIONSHIPS] Processing {len(concepts)} concepts"
            )

            # Generate embeddings for all concept names
            concept_names = [c["name"] for c in concepts]
            embeddings = await self.embedding_service.embed_texts(concept_names)

            # Calculate similarity matrix and create relationships
            relationships_created = 0
            for i, concept1 in enumerate(concepts):
                for j, concept2 in enumerate(concepts):
                    if i >= j:
                        continue

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                    if similarity >= self.similarity_threshold:
                        try:
                            # Create relationship
                            await self.kg_service.create_relationship(
                                RelationshipCreate(
                                    source_concept_id=concept1["id"],
                                    target_concept_id=concept2["id"],
                                    type="semantically_related",
                                    strength=float(similarity),
                                )
                            )
                            relationships_created += 1
                        except Exception as e:
                            # Skip duplicate relationships
                            if "UNIQUE constraint failed" not in str(e):
                                logging.warning(
                                    f"[EMBEDDING_RELATIONSHIPS] Error creating relationship: {e}"
                                )

            logging.info(
                f"[EMBEDDING_RELATIONSHIPS] Created {relationships_created} relationships"
            )
            return {
                "relationships_created": relationships_created,
                "errors": [],
            }

        except Exception as e:
            error_msg = f"Error building embedding relationships: {str(e)}"
            logging.error(f"[EMBEDDING_RELATIONSHIPS] {error_msg}")
            return {
                "relationships_created": 0,
                "errors": [error_msg],
            }

    def get_builder_type(self) -> str:
        """Get the type of this relationship builder."""
        return "embedding"

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
