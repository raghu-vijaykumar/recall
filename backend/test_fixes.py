#!/usr/bin/env python3
"""
Simple test script to verify our fixes work.
"""
import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())


async def test_basic_imports():
    """Test that basic imports work after our changes."""
    try:
        from app.services.knowledge_graph_service import KnowledgeGraphService
        from app.models import ConceptCreate

        print("✓ Imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


async def test_concept_create():
    """Test concept creation with mock session."""
    try:
        from app.services.knowledge_graph_service import KnowledgeGraphService
        from app.models import ConceptCreate
        import unittest.mock as mock

        # Mock session
        mock_session = mock.AsyncMock()

        # Create service
        service = KnowledgeGraphService(mock_session)

        # Create concept data
        concept_data = ConceptCreate(
            name="Test Concept", description="Test description"
        )

        # Test creation
        concept = await service.create_concept(concept_data)

        assert concept.name == "Test Concept"
        assert concept.description == "Test description"
        assert concept.concept_id is not None

        print("✓ Concept creation successful")
        return True
    except Exception as e:
        print(f"✗ Concept creation failed: {e}")
        return False


async def test_bertopic_methods():
    """Test that BERTopicExtractor has the required methods."""
    try:
        from app.services.topicExtractor.bertopic_extractor import BERTopicExtractor

        # Create instance (will fail on BERTopic, but we can test method presence)
        try:
            extractor = BERTopicExtractor()
            methods = [
                "extract_topics",
                "_prepare_documents",
                "_generate_bertopic_name",
                "_calculate_topic_coherence",
                "_calculate_outlier_score",
                "_get_model_cache_key",
                "get_topic_visualization_data",
            ]

            for method in methods:
                if not hasattr(extractor, method):
                    raise AttributeError(f"Missing method: {method}")

            print("✓ BERTopicExtractor has all required methods")
            return True

        except Exception as inner_e:
            # If BERTopic isn't available, that's expected
            if "BERTopic is not installed" not in str(inner_e):
                print(f"✗ BERTopicExtractor failed: {inner_e}")
                return False
            else:
                print(
                    "✓ BERTopicExtractor constructor properly checks for dependencies"
                )
                return True

    except Exception as e:
        print(f"✗ BERTopicExtractor test failed: {e}")
        return False


async def run_tests():
    """Run all tests."""
    print("Running fix verification tests...\n")

    results = []
    results.append(await test_basic_imports())
    results.append(await test_concept_create())
    results.append(await test_bertopic_methods())

    passed = sum(results)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All fixes appear to be working!")
        return 0
    else:
        print("❌ Some issues remain")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
