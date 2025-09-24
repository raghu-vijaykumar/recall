"""
Tests for Knowledge Graph functionality
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Concept,
    Relationship,
    ConceptFile,
    ConceptCreate,
    RelationshipCreate,
    ConceptFileCreate,
)
from app.services import KnowledgeGraphService


class TestKnowledgeGraphModels:
    """Test Knowledge Graph Pydantic models"""

    def test_concept_create(self):
        """Test ConceptCreate model"""
        concept_data = ConceptCreate(
            name="Machine Learning",
            description="A field of study that gives computers the ability to learn without being explicitly programmed",
        )
        assert concept_data.name == "Machine Learning"
        assert (
            concept_data.description
            == "A field of study that gives computers the ability to learn without being explicitly programmed"
        )
        assert concept_data.concept_id is not None
        assert isinstance(uuid.UUID(concept_data.concept_id), uuid.UUID)

    def test_relationship_create(self):
        """Test RelationshipCreate model"""
        relationship_data = RelationshipCreate(
            source_concept_id=str(uuid.uuid4()),
            target_concept_id=str(uuid.uuid4()),
            type="relates_to",
            strength=0.8,
        )
        assert relationship_data.type == "relates_to"
        assert relationship_data.strength == 0.8
        assert relationship_data.relationship_id is not None

    def test_concept_file_create(self):
        """Test ConceptFileCreate model"""
        concept_file_data = ConceptFileCreate(
            concept_id=str(uuid.uuid4()),
            file_id=1,
            workspace_id=1,
            snippet="Machine learning is a subset of AI",
            relevance_score=0.9,
        )
        assert concept_file_data.snippet == "Machine learning is a subset of AI"
        assert concept_file_data.relevance_score == 0.9
        assert concept_file_data.concept_file_id is not None


class TestKnowledgeGraphService:
    """Test Knowledge Graph Service"""

    @pytest.mark.asyncio
    async def test_create_concept(self, db_session):
        """Test creating a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

        concept_data = ConceptCreate(
            name="Neural Networks",
            description="A series of algorithms that mimic the operations of a human brain",
        )

        concept = await kg_service.create_concept(concept_data)

        assert concept.name == "Neural Networks"
        assert (
            concept.description
            == "A series of algorithms that mimic the operations of a human brain"
        )
        assert concept.concept_id is not None
        assert concept.created_at is not None
        assert concept.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_concept(self, db_session):
        """Test retrieving a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create a concept first
            concept_data = ConceptCreate(name="Deep Learning")
            created_concept = await kg_service.create_concept(concept_data)

            # Retrieve it
            retrieved_concept = await kg_service.get_concept(created_concept.concept_id)

            assert retrieved_concept is not None
            assert retrieved_concept.concept_id == created_concept.concept_id
            assert retrieved_concept.name == "Deep Learning"

    @pytest.mark.asyncio
    async def test_update_concept(self, db_session):
        """Test updating a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create a concept
            concept_data = ConceptCreate(name="Supervised Learning")
            created_concept = await kg_service.create_concept(concept_data)

            # Update it
            updated_concept = await kg_service.update_concept(
                created_concept.concept_id,
                name="Supervised Machine Learning",
                description="A type of machine learning using labeled data",
            )

            assert updated_concept is not None
            assert updated_concept.name == "Supervised Machine Learning"
            assert (
                updated_concept.description
                == "A type of machine learning using labeled data"
            )

    @pytest.mark.asyncio
    async def test_delete_concept(self, db_session):
        """Test deleting a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create a concept
            concept_data = ConceptCreate(name="Unsupervised Learning")
            created_concept = await kg_service.create_concept(concept_data)

            # Delete it
            deleted = await kg_service.delete_concept(created_concept.concept_id)
            assert deleted is True

            # Verify it's gone
            retrieved = await kg_service.get_concept(created_concept.concept_id)
            assert retrieved is None

    @pytest.mark.asyncio
    async def test_create_relationship(self, db_session):
        """Test creating a relationship between concepts"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create two concepts
            concept1_data = ConceptCreate(name="Python")
            concept2_data = ConceptCreate(name="Programming")

            concept1 = await kg_service.create_concept(concept1_data)
            concept2 = await kg_service.create_concept(concept2_data)

            # Create relationship
            relationship_data = RelationshipCreate(
                source_concept_id=concept1.concept_id,
                target_concept_id=concept2.concept_id,
                type="relates_to",
                strength=0.9,
            )

            relationship = await kg_service.create_relationship(relationship_data)

            assert relationship.source_concept_id == concept1.concept_id
            assert relationship.target_concept_id == concept2.concept_id
            assert relationship.type == "relates_to"
            assert relationship.strength == 0.9

    @pytest.mark.asyncio
    async def test_get_relationships_for_concept(self, db_session):
        """Test getting relationships for a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create concepts and relationships
            concept1_data = ConceptCreate(name="JavaScript")
            concept2_data = ConceptCreate(name="Web Development")
            concept3_data = ConceptCreate(name="React")

            concept1 = await kg_service.create_concept(concept1_data)
            concept2 = await kg_service.create_concept(concept2_data)
            concept3 = await kg_service.create_concept(concept3_data)

            # Create relationships
            rel1_data = RelationshipCreate(
                source_concept_id=concept1.concept_id,
                target_concept_id=concept2.concept_id,
                type="relates_to",
            )
            rel2_data = RelationshipCreate(
                source_concept_id=concept3.concept_id,
                target_concept_id=concept2.concept_id,
                type="dives_deep_to",
            )

            await kg_service.create_relationship(rel1_data)
            await kg_service.create_relationship(rel2_data)

            # Get relationships for concept2
            relationships = await kg_service.get_relationships_for_concept(
                concept2.concept_id
            )

            assert len(relationships) == 2
            relationship_types = {rel.type for rel in relationships}
            assert "relates_to" in relationship_types
            assert "dives_deep_to" in relationship_types

    @pytest.mark.asyncio
    async def test_create_concept_file_link(self, db_session):
        """Test creating a concept-file link"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create a concept
            concept_data = ConceptCreate(name="Database")
            concept = await kg_service.create_concept(concept_data)

            # Create concept-file link
            link_data = ConceptFileCreate(
                concept_id=concept.concept_id,
                file_id=1,
                workspace_id=1,
                snippet="A database is an organized collection of data",
                relevance_score=0.95,
            )

            link = await kg_service.create_concept_file_link(link_data)

            assert link.concept_id == concept.concept_id
            assert link.file_id == 1
            assert link.workspace_id == 1
            assert link.snippet == "A database is an organized collection of data"
            assert link.relevance_score == 0.95

    @pytest.mark.asyncio
    async def test_get_files_for_concept(self, db_session):
        """Test getting files linked to a concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create a concept
            concept_data = ConceptCreate(name="SQL")
            concept = await kg_service.create_concept(concept_data)

            # Create multiple file links
            link1_data = ConceptFileCreate(
                concept_id=concept.concept_id,
                file_id=1,
                workspace_id=1,
                snippet="SQL is a domain-specific language",
                relevance_score=0.9,
            )
            link2_data = ConceptFileCreate(
                concept_id=concept.concept_id,
                file_id=2,
                workspace_id=1,
                snippet="Structured Query Language",
                relevance_score=0.85,
            )

            await kg_service.create_concept_file_link(link1_data)
            await kg_service.create_concept_file_link(link2_data)

            # Get files for concept
            files = await kg_service.get_files_for_concept(concept.concept_id)

            assert len(files) == 2
            file_ids = {f.file_id for f in files}
            assert 1 in file_ids
            assert 2 in file_ids

    @pytest.mark.asyncio
    async def test_search_concepts(self, db_session):
        """Test searching concepts by name or description"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create concepts
            concepts_data = [
                ConceptCreate(
                    name="Python Programming",
                    description="High-level programming language",
                ),
                ConceptCreate(
                    name="Java Programming",
                    description="Object-oriented programming language",
                ),
                ConceptCreate(
                    name="Web Development",
                    description="Building websites and web applications",
                ),
            ]

            for concept_data in concepts_data:
                await kg_service.create_concept(concept_data)

            # Search for "Programming"
            results = await kg_service.search_concepts("Programming", limit=10)

            assert len(results) >= 2
            concept_names = {c.name for c in results}
            assert "Python Programming" in concept_names
            assert "Java Programming" in concept_names

    @pytest.mark.asyncio
    async def test_get_workspace_concepts(self, db_session):
        """Test getting concepts for a workspace"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create concepts and link them to files in workspace 1
            concept1_data = ConceptCreate(name="Frontend")
            concept2_data = ConceptCreate(name="Backend")

            concept1 = await kg_service.create_concept(concept1_data)
            concept2 = await kg_service.create_concept(concept2_data)

            # Create file links
            link1_data = ConceptFileCreate(
                concept_id=concept1.concept_id,
                file_id=1,
                workspace_id=1,
                relevance_score=0.9,
            )
            link2_data = ConceptFileCreate(
                concept_id=concept2.concept_id,
                file_id=2,
                workspace_id=1,
                relevance_score=0.85,
            )

            await kg_service.create_concept_file_link(link1_data)
            await kg_service.create_concept_file_link(link2_data)

            # Get workspace concepts
            workspace_concepts = await kg_service.get_workspace_concepts(1)

            assert len(workspace_concepts) == 2
            concept_names = {wc["concept"].name for wc in workspace_concepts}
            assert "Frontend" in concept_names
            assert "Backend" in concept_names

    @pytest.mark.asyncio
    async def test_get_concepts_for_file(self, db_session):
        """Test getting concepts linked to a file"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create concepts
            concept1_data = ConceptCreate(name="Database Design")
            concept2_data = ConceptCreate(name="SQL Queries")

            concept1 = await kg_service.create_concept(concept1_data)
            concept2 = await kg_service.create_concept(concept2_data)

            # Create concept-file links
            link1_data = ConceptFileCreate(
                concept_id=concept1.concept_id,
                file_id=1,
                workspace_id=1,
                snippet="Database design principles",
                relevance_score=0.9,
            )
            link2_data = ConceptFileCreate(
                concept_id=concept2.concept_id,
                file_id=1,
                workspace_id=1,
                snippet="SQL query examples",
                relevance_score=0.85,
            )
            link3_data = ConceptFileCreate(
                concept_id=concept1.concept_id,
                file_id=2,
                workspace_id=1,
                snippet="Normalization concepts",
                relevance_score=0.8,
            )

            await kg_service.create_concept_file_link(link1_data)
            await kg_service.create_concept_file_link(link2_data)
            await kg_service.create_concept_file_link(link3_data)

            # Get concepts for file 1
            concepts_file1 = await kg_service.get_concepts_for_file(1)

            assert len(concepts_file1) == 2
            concept_names = {cf.concept_id for cf in concepts_file1}
            assert concept1.concept_id in concept_names
            assert concept2.concept_id in concept_names

            # Get concepts for file 2
            concepts_file2 = await kg_service.get_concepts_for_file(2)

            assert len(concepts_file2) == 1
            assert concepts_file2[0].concept_id == concept1.concept_id

            # Get concepts for non-existent file
            concepts_none = await kg_service.get_concepts_for_file(999)
            assert concepts_none == []

    @pytest.mark.asyncio
    async def test_get_concept_graph(self, db_session):
        """Test getting a concept graph around a central concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            # Create central concept
            central_concept_data = ConceptCreate(name="Machine Learning")
            central_concept = await kg_service.create_concept(central_concept_data)

            # Create related concepts
            related1_data = ConceptCreate(name="Supervised Learning")
            related2_data = ConceptCreate(name="Unsupervised Learning")
            related3_data = ConceptCreate(name="Neural Networks")

            related1 = await kg_service.create_concept(related1_data)
            related2 = await kg_service.create_concept(related2_data)
            related3 = await kg_service.create_concept(related3_data)

            # Create relationships
            rel1_data = RelationshipCreate(
                source_concept_id=central_concept.concept_id,
                target_concept_id=related1.concept_id,
                type="has_type",
                strength=0.9,
            )
            rel2_data = RelationshipCreate(
                source_concept_id=central_concept.concept_id,
                target_concept_id=related2.concept_id,
                type="has_type",
                strength=0.8,
            )
            rel3_data = RelationshipCreate(
                source_concept_id=related1.concept_id,
                target_concept_id=related3.concept_id,
                type="uses",
                strength=0.7,
            )

            await kg_service.create_relationship(rel1_data)
            await kg_service.create_relationship(rel2_data)
            await kg_service.create_relationship(rel3_data)

            # Get concept graph
            graph = await kg_service.get_concept_graph(
                central_concept.concept_id, depth=2
            )

            # Should include central concept and directly related concepts
            assert len(graph["nodes"]) >= 3  # Central + 2 related
            assert len(graph["edges"]) >= 2  # Relationships from central

            # Check central node
            central_node = next(
                n for n in graph["nodes"] if n["id"] == central_concept.concept_id
            )
            assert central_node["name"] == "Machine Learning"

            # Check edges
            edge_types = {e["type"] for e in graph["edges"]}
            assert "has_type" in edge_types

    @pytest.mark.asyncio
    async def test_get_concept_graph_nonexistent(self, db_session):
        """Test getting concept graph for non-existent concept"""
        async with db_session() as session:
            kg_service = KnowledgeGraphService(session)

            graph = await kg_service.get_concept_graph("nonexistent-concept-id")

            assert graph["nodes"] == []
            assert graph["edges"] == []
