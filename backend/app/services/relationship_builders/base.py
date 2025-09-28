"""
Base relationship builder for workspace analysis.
Provides the interface for building relationships between concepts.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import RelationshipCreate


class RelationshipBuilder(ABC):
    """
    Abstract base class for relationship builders.
    Defines the interface for building relationships between concepts in a workspace.
    """

    def __init__(self, db: AsyncSession, kg_service: Any):
        """
        Initialize the relationship builder.

        Args:
            db: Database session
            kg_service: Knowledge graph service instance
        """
        self.db = db
        self.kg_service = kg_service

    @abstractmethod
    async def build_relationships(self, workspace_id: int) -> Dict[str, Any]:
        """
        Build relationships between concepts in the workspace.

        Args:
            workspace_id: ID of the workspace to build relationships for

        Returns:
            Dictionary with relationship building results, including:
            - relationships_created: Number of relationships created
            - errors: List of any errors encountered
        """
        pass

    @abstractmethod
    def get_builder_type(self) -> str:
        """
        Get the type/name of this relationship builder.

        Returns:
            String identifier for the builder type
        """
        pass
