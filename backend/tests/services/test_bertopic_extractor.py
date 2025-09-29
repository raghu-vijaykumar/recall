"""
Tests for BERTopicExtractor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import unittest
import numpy as np

from app.services.topicExtractor.bertopic_extractor import BERTopicExtractor
from app.services.topicExtractor.base import BaseTopicExtractor
from app.services.embedding_service import EmbeddingService


class TestBERTopicExtractor:
    """Test cases for BERTopicExtractor"""

    @pytest.fixture
    def sample_concepts_data(self):
        """Sample concept data for testing"""
        return [
            {
                "id": "concept_1",
                "name": "machine learning",
                "description": "A type of artificial intelligence",
                "relevance_score": 0.9,
            },
            {
                "id": "concept_2",
                "name": "neural networks",
                "description": "Interconnected nodes that process information",
                "relevance_score": 0.8,
            },
            {
                "id": "concept_3",
                "name": "deep learning",
                "description": "A subset of machine learning with multiple layers",
                "relevance_score": 0.85,
            },
            {
                "id": "concept_4",
                "name": "computer vision",
                "description": "Teaching computers to interpret visual information",
                "relevance_score": 0.7,
            },
            {
                "id": "concept_5",
                "name": "natural language processing",
                "description": "AI that understands human language",
                "relevance_score": 0.75,
            },
        ]

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing"""
        mock_service = MagicMock(spec=EmbeddingService)
        mock_service.get_instance.return_value = mock_service
        return mock_service

    def test_bertopic_extractor_initialization(self, mock_embedding_service):
        """Test BERTopicExtractor can be initialized with default parameters"""
        extractor = BERTopicExtractor(
            embedding_service=mock_embedding_service,
            model_name="all-MiniLM-L6-v2",
            min_topic_size=2,
            max_topic_areas=10,
        )

        assert extractor.model_name == "all-MiniLM-L6-v2"
        assert extractor.min_topic_size == 2
        assert extractor.max_topic_areas == 10
        assert extractor.embedding_service == mock_embedding_service

    def test_bertopic_extractor_custom_config(self, mock_embedding_service):
        """Test BERTopicExtractor with custom configuration"""
        config = {
            "model_name": "all-mpnet-base-v2",
            "min_topic_size": 5,
            "nr_topics": 10,
            "diversity": 0.5,
            "coherence_threshold": 0.2,
        }

        extractor = BERTopicExtractor(
            embedding_service=mock_embedding_service, **config
        )

        assert extractor.model_name == "all-mpnet-base-v2"
        assert extractor.min_topic_size == 5
        assert extractor.nr_topics == 10
        assert extractor.diversity == 0.5
        assert extractor.coherence_threshold == 0.2

    def test_bertopic_extractor_inheritance(self, mock_embedding_service):
        """Test that BERTopicExtractor properly inherits from BaseTopicExtractor"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)
        assert isinstance(extractor, BaseTopicExtractor)

    def test_document_preparation(self, mock_embedding_service):
        """Test document preparation for BERTopic"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        concepts_data = [
            {
                "id": "concept_1",
                "name": "test concept",
                "description": "A test description",
                "relevance_score": 0.8,
            }
        ]

        # Test the async method
        import asyncio

        async def test_prep():
            documents = await extractor._prepare_documents(concepts_data)
            assert len(documents) == 1
            assert "test concept" in documents[0]
            assert "test description" in documents[0]

        asyncio.run(test_prep())

    def test_topic_name_generation(self, mock_embedding_service):
        """Test topic name generation from BERTopic words"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        # Test with topic words
        topic_data = {
            "words": [("machine", 0.8), ("learning", 0.7), ("algorithm", 0.6)]
        }
        name = extractor._generate_topic_name(topic_data)

        assert "Machine Learning" in name
        assert len(name) <= 50  # Should not be too long

    def test_empty_topic_words(self, mock_embedding_service):
        """Test topic name generation with empty topic words"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        name = extractor._generate_topic_name({"words": []})
        assert name == "General Topic"

    def test_coherence_calculation(self, mock_embedding_service):
        """Test topic coherence calculation"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        # Test with high coherence words
        coherence = extractor._calculate_topic_coherence()

        assert 0.0 <= coherence <= 1.0

    def test_outlier_score_calculation(self, mock_embedding_service):
        """Test outlier score calculation"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        # Test with high relevance concepts (should be low outlier score)
        outlier_score = extractor._calculate_outlier_score()
        assert 0.0 <= outlier_score <= 1.0
        assert outlier_score < 0.5  # Should be low for high-relevance concepts

    def test_model_cache_key_generation(self, mock_embedding_service):
        """Test model cache key generation"""
        extractor = BERTopicExtractor(
            embedding_service=mock_embedding_service,
            model_name="test-model",
            min_topic_size=3,
            nr_topics=10,
            diversity=0.5,
        )

        cache_key = extractor._get_model_cache_key()
        expected_key = "bertopic_test-model_3_10_0.5"
        assert cache_key == expected_key

    @pytest.mark.asyncio
    async def test_extract_topics_insufficient_concepts(self, mock_embedding_service):
        """Test extraction with insufficient concepts"""
        extractor = BERTopicExtractor(
            embedding_service=mock_embedding_service, min_topic_size=5
        )

        # Only 2 concepts, but minimum is 5
        concepts_data = [
            {"id": "c1", "name": "concept 1", "relevance_score": 0.8},
            {"id": "c2", "name": "concept 2", "relevance_score": 0.7},
        ]

        topic_areas = await extractor.extract_topics(1, concepts_data)

        assert len(topic_areas) == 0

    def test_bertopic_model_creation(self, mock_embedding_service):
        """Test BERTopic model creation and caching"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        # Mock the BERTopic model to avoid actual initialization
        with unittest.mock.patch(
            "app.services.topicExtractor.bertopic_extractor.BERTopic"
        ) as mock_bertopic:
            mock_model = MagicMock()
            mock_bertopic.return_value = mock_model

            # Test model creation
            model = extractor._create_bertopic_processor(
                extractor._create_bertopic_config()
            )
            assert model == mock_model
            mock_bertopic.assert_called_once()

    def test_visualization_data_generation(self, mock_embedding_service):
        """Test visualization data generation"""
        extractor = BERTopicExtractor(embedding_service=mock_embedding_service)

        # Mock BERTopic model
        mock_model = MagicMock()
        mock_topic_info = MagicMock()
        mock_topic_info.to_dict.return_value = [
            {"Topic": 0, "Count": 10, "Name": "Topic 0"},
            {"Topic": 1, "Count": 8, "Name": "Topic 1"},
        ]
        mock_model.get_topic_info.return_value = mock_topic_info
        mock_model.get_topics.return_value = [0, 1]
        mock_model.get_topic.return_value = [("word1", 0.8), ("word2", 0.7)]

        viz_data = extractor.get_topic_visualization_data()

        assert "nodes" in viz_data
        assert "edges" in viz_data


if __name__ == "__main__":
    pytest.main([__file__])
