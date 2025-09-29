"""
BERTopic Knowledge Graph Package - Production Ready

This package provides enhanced BERTopic topic modeling with:
- Document preprocessing pipeline
- Knowledge graph generation
- Incremental indexing
- Multi-model comparison
- Interactive visualizations
"""

from .core import BERTopicProcessor, TopicModelingConfig
from .preprocessing import DocumentPreprocessor, PreprocessingConfig
from .knowledge_graph import TopicKnowledgeGraphBuilder, KnowledgeGraphNavigator
from .incremental import IncrementalIndexManager
from .utils import DocumentLoader

__all__ = [
    "BERTopicProcessor",
    "TopicModelingConfig",
    "DocumentPreprocessor",
    "PreprocessingConfig",
    "TopicKnowledgeGraphBuilder",
    "KnowledgeGraphNavigator",
    "IncrementalIndexManager",
    "DocumentLoader",
]
