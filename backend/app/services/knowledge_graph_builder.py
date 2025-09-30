"""
Knowledge Graph Builder for creating topic-to-topic relationships.
Provides base class architecture with LLM implementation for building topic relationships.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    TopicArea,
    TopicRelationship,
    TopicRelationshipCreate,
    WorkspaceTopicGraph,
)
from ..llm_clients import LLMClientFactory


class BaseKnowledgeGraphBuilder(ABC):
    """
    Abstract base class for building knowledge graphs of topic relationships.
    Defines the interface for discovering and creating relationships between topics.
    """

    def __init__(self, db: AsyncSession, llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge graph builder.

        Args:
            db: Database session for storing relationships
            llm_config: Optional configuration for LLM clients
        """
        self.db = db
        self.llm_config = llm_config or {}

    @abstractmethod
    async def build_topic_relationships(
        self, topic_areas: List[TopicArea], workspace_id: int
    ) -> List[TopicRelationship]:
        """
        Build relationships between topic areas.

        Args:
            topic_areas: List of topic areas to analyze for relationships
            workspace_id: ID of the workspace containing these topics

        Returns:
            List of discovered topic relationships
        """
        pass

    def _create_topic_relationship(
        self,
        source_topic: TopicArea,
        target_topic: TopicArea,
        relationship_type: str,
        strength: float,
        reasoning: str,
    ) -> TopicRelationshipCreate:
        """Create a topic relationship model."""
        return TopicRelationshipCreate(
            source_topic_id=source_topic.topic_area_id,
            target_topic_id=target_topic.topic_area_id,
            relationship_type=relationship_type,
            strength=strength,
            reasoning=reasoning,
        )

    async def _store_relationships(
        self, relationships: List[TopicRelationshipCreate]
    ) -> List[TopicRelationship]:
        """Store topic relationships in the database."""
        stored_relationships = []

        for relationship_data in relationships:
            # Check if relationship already exists
            existing_query = """
                SELECT relationship_id FROM topic_relationships
                WHERE (source_topic_id = ? AND target_topic_id = ?)
                   OR (source_topic_id = ? AND target_topic_id = ?)
            """

            result = await self.db.execute(
                existing_query,
                (
                    relationship_data.source_topic_id,
                    relationship_data.target_topic_id,
                    relationship_data.target_topic_id,
                    relationship_data.source_topic_id,
                ),
            )

            existing = result.fetchone()
            if existing:
                # Update existing relationship
                update_query = """
                    UPDATE topic_relationships
                    SET relationship_type = ?, strength = ?, reasoning = ?, updated_at = ?
                    WHERE relationship_id = ?
                """
                await self.db.execute(
                    update_query,
                    (
                        relationship_data.relationship_type,
                        relationship_data.strength,
                        relationship_data.reasoning,
                        datetime.now(UTC),
                        existing.relationship_id,
                    ),
                )
            else:
                # Create new relationship
                insert_query = """
                    INSERT INTO topic_relationships
                    (relationship_id, source_topic_id, target_topic_id, relationship_type, strength, reasoning, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """
                await self.db.execute(
                    insert_query,
                    (
                        relationship_data.relationship_id,
                        relationship_data.source_topic_id,
                        relationship_data.target_topic_id,
                        relationship_data.relationship_type,
                        relationship_data.strength,
                        relationship_data.reasoning,
                        datetime.now(UTC),
                        datetime.now(UTC),
                    ),
                )

            # Create TopicRelationship model
            stored_relationships.append(
                TopicRelationship(
                    relationship_id=relationship_data.relationship_id,
                    source_topic_id=relationship_data.source_topic_id,
                    target_topic_id=relationship_data.target_topic_id,
                    relationship_type=relationship_data.relationship_type,
                    strength=relationship_data.strength,
                    reasoning=relationship_data.reasoning,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )

        await self.db.commit()
        return stored_relationships


class LLMKnowledgeGraphBuilder(BaseKnowledgeGraphBuilder):
    """
    LLM-based implementation of knowledge graph builder.
    Uses language models to analyze topic descriptions and discover relationships.
    """

    RELATIONSHIP_TYPES = {
        "relates_to": "Topics are related or connected in some way",
        "builds_on": "One topic provides foundation for another",
        "contrasts_with": "Topics present opposing or contrasting ideas",
        "contains": "One topic encompasses or includes another as a subtopic",
        "precedes": "One topic should be learned before another",
    }

    def __init__(self, db: AsyncSession, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(db, llm_config)
        from ..llm_clients import llm_client_factory

        self.llm_client = llm_client_factory.get_client(
            self.llm_config.get("provider", "gemini")
        )

    async def build_topic_relationships(
        self, topic_areas: List[TopicArea], workspace_id: int
    ) -> List[TopicRelationship]:
        """
        Use LLM to analyze topic areas and discover relationships between them.

        Args:
            topic_areas: List of topic areas to analyze
            workspace_id: Workspace ID for context

        Returns:
            List of discovered topic relationships
        """
        if len(topic_areas) < 2:
            logging.info("Need at least 2 topics to build relationships")
            return []

        relationships = []

        # Analyze topics in pairs for relationships
        for i, source_topic in enumerate(topic_areas):
            for target_topic in topic_areas[i + 1 :]:
                relationship = await self._analyze_topic_pair(
                    source_topic, target_topic
                )
                if relationship:
                    relationships.append(relationship)

        # Store relationships in database
        stored_relationships = await self._store_relationships(relationships)

        logging.info(f"Built {len(stored_relationships)} topic relationships")
        return stored_relationships

    async def _analyze_topic_pair(
        self, topic1: TopicArea, topic2: TopicArea
    ) -> Optional[TopicRelationshipCreate]:
        """Analyze a pair of topics to determine their relationship."""

        prompt = f"""
        Analyze the relationship between these two topics and determine the most appropriate relationship type:

        Topic 1: "{topic1.name}"
        Description: "{topic1.description}"

        Topic 2: "{topic2.name}"
        Description: "{topic2.description}"

        Relationship types available:
        - relates_to: Topics are related or connected in some meaningful way
        - builds_on: One topic provides necessary foundation for understanding the other
        - contrasts_with: Topics present opposing ideas, methods, or approaches
        - contains: One topic encompasses the other as a subtopic or component
        - precedes: One topic should logically be learned or understood before the other

        If there's no meaningful relationship between these topics, respond with "no_relationship".

        Please respond with ONLY a JSON object in this exact format:
        {{
            "relationship_type": "relates_to|builds_on|contrasts_with|contains|precedes",
            "strength": 0.0-1.0,
            "reasoning": "brief explanation",
            "direction": "topic1_to_topic2|topic2_to_topic1|bidirectional"
        }}

        If no relationship, respond with: {{"relationship_type": "no_relationship"}}
        """

        try:
            response = await self.llm_client.generate(prompt)
            result = self._parse_llm_response(response)

            if result.get("relationship_type") == "no_relationship":
                return None

            # Determine source and target based on direction
            direction = result.get("direction", "bidirectional")
            if direction == "topic2_to_topic1":
                source_topic, target_topic = topic2, topic1
            else:
                source_topic, target_topic = topic1, topic2

            return self._create_topic_relationship(
                source_topic=source_topic,
                target_topic=target_topic,
                relationship_type=result["relationship_type"],
                strength=result.get("strength", 0.5),
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logging.error(f"Error analyzing topic pair: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response JSON."""
        import json

        try:
            # Extract JSON from response (LLMs might add extra text)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)

            return {"relationship_type": "no_relationship"}

        except json.JSONDecodeError:
            logging.warning(f"Failed to parse LLM response: {response}")
            return {"relationship_type": "no_relationship"}

    async def get_workspace_topic_graph(self, workspace_id: int) -> WorkspaceTopicGraph:
        """Get the complete topic graph for a workspace."""
        # Get all topic areas
        query = """
            SELECT * FROM topic_areas WHERE workspace_id = ?
            ORDER BY coverage_score DESC
        """
        result = await self.db.execute(query, (workspace_id,))
        topic_rows = result.fetchall()

        topics = []
        for row in topic_rows:
            topics.append(
                TopicArea(
                    topic_area_id=row.topic_area_id,
                    workspace_id=row.workspace_id,
                    name=row.name,
                    description=row.description,
                    coverage_score=row.coverage_score,
                    file_count=row.file_count,
                    explored_percentage=row.explored_percentage,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            )

        # Get all relationships for these topics
        if topics:
            topic_ids = [t.topic_area_id for t in topics]
            placeholders = ",".join("?" * len(topic_ids))
            relationship_query = f"""
                SELECT * FROM topic_relationships
                WHERE source_topic_id IN ({placeholders})
                   OR target_topic_id IN ({placeholders})
                ORDER BY strength DESC, created_at DESC
            """

            result = await self.db.execute(relationship_query, topic_ids + topic_ids)
            relationship_rows = result.fetchall()

            relationships = []
            for row in relationship_rows:
                relationships.append(
                    TopicRelationship(
                        relationship_id=row.relationship_id,
                        source_topic_id=row.source_topic_id,
                        target_topic_id=row.target_topic_id,
                        relationship_type=row.relationship_type,
                        strength=row.strength,
                        reasoning=row.reasoning,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                )
        else:
            relationships = []

        return WorkspaceTopicGraph(
            workspace_id=workspace_id, topics=topics, relationships=relationships
        )
