"""
Tests for the core BERTopic processor functionality.

Tests basic topic modeling, configuration, and integration.
"""

import pytest
import tempfile
import os
from pathlib import Path

# Import from the new package structure
from bertopic.core import BERTopicProcessor, TopicModelingConfig


class TestTopicModelingConfig:
    """Test configuration class functionality."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = TopicModelingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.min_topic_size == 15
        assert config.max_topics is None
        assert config.verbose is False

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = TopicModelingConfig(
            model_name="all-mpnet-base-v2",
            min_topic_size=10,
            max_topics=20,
            verbose=True,
        )
        assert config.model_name == "all-mpnet-base-v2"
        assert config.min_topic_size == 10
        assert config.max_topics == 20
        assert config.verbose is True


class TestBERTopicProcessor:
    """Test the main BERTopic processor."""

    def test_initialization(self):
        """Test processor initialization."""
        config = TopicModelingConfig()
        processor = BERTopicProcessor(config)
        assert processor.config == config

    def test_process_documents_basic(self):
        """Test basic document processing with simple test data."""
        # Create test documents
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing helps computers understand text",
            "Topic modeling discovers hidden themes in documents",
            "Clustering algorithms group similar data points together",
            "Neural networks are inspired by biological brains",
            "Deep learning uses multiple layers of neural networks",
        ]

        config = TopicModelingConfig(
            model_name="all-MiniLM-L6-v2",  # Use lightweight model for testing
            min_topic_size=3,
            verbose=False,
        )
        processor = BERTopicProcessor(config)

        topics, probabilities = processor.process_documents(documents)

        # Basic sanity checks
        assert len(topics) == len(documents)
        assert probabilities.shape[0] == len(documents)
        assert isinstance(topics, list)
        assert all(isinstance(t, int) for t in topics)

        # Get statistics
        stats = processor.get_statistics()
        assert "total_topics" in stats
        assert "total_documents" in stats
        assert isinstance(stats["total_topics"], int)
        assert isinstance(stats["total_documents"], int)

    def test_get_topics(self):
        """Test topic information retrieval."""
        documents = [
            "Data science combines statistics and programming",
            "Python is popular for data analysis",
            "R is also used for statistical computing",
            "Machine learning predicts outcomes from data",
            "AI systems learn from examples",
        ]

        processor = BERTopicProcessor()
        processor.process_documents(documents)

        topics = processor.get_topics()
        assert isinstance(topics, dict)

        # Should have at least one topic
        if topics:
            topic_id = next(iter(topics.keys()))
            topic_info = topics[topic_id]
            assert "name" in topic_info
            assert "count" in topic_info
            assert "words" in topic_info

    def test_save_results(self):
        """Test result saving functionality."""
        documents = [
            "This is about artificial intelligence",
            "Machine learning is part of AI",
            "Deep learning uses neural networks",
        ]

        processor = BERTopicProcessor()
        processor.process_documents(documents)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results"
            processor.save_results(output_path)

            # Check if files were created
            assert (output_path / "topics.json").exists()
            assert (output_path / "document_assignments.json").exists()

    def test_statistics_after_processing(self):
        """Test statistics generation after processing."""
        documents = ["Test document one", "Test document two", "Different topic here"]

        processor = BERTopicProcessor()
        processor.process_documents(documents)

        stats = processor.get_statistics()

        expected_keys = [
            "total_topics",
            "total_documents",
            "outlier_documents",
            "documents_per_topic",
            "average_docs_per_topic",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"

        assert stats["total_documents"] == len(documents)
        assert isinstance(stats["average_docs_per_topic"], float)


class TestErrorHandling:
    """Test error handling in the processor."""

    def test_empty_documents(self):
        """Test handling of empty document list."""
        processor = BERTopicProcessor()

        with pytest.raises(ValueError, match="Cannot process empty document list"):
            processor.process_documents([])

    def test_save_before_processing(self):
        """Test error when trying to save before processing."""
        processor = BERTopicProcessor()

        with pytest.raises(RuntimeError, match="No exporter available"):
            processor.save_results(Path("/tmp"))

        with pytest.raises(RuntimeError, match="No visualizer available"):
            processor.create_visualizations(Path("/tmp"))

        with pytest.raises(RuntimeError, match="No summarizer available"):
            processor.generate_summary()


if __name__ == "__main__":
    pytest.main([__file__])
