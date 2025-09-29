#!/usr/bin/env python3
"""
Knowledge Graph Builder Module for BERTopic Enhancement
Extracts context windows and builds knowledge graphs from topic clusters
"""

import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

# Configure package logger
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGraphNode:
    """Represents a node in the knowledge graph"""

    id: str
    type: str  # 'topic', 'concept', 'document', 'subtopic'
    name: str
    description: str
    metadata: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class KnowledgeGraphEdge:
    """Represents an edge in the knowledge graph"""

    id: str
    source_id: str
    target_id: str
    type: str  # 'contains', 'relates_to', 'subtopic_of', 'document_of'
    weight: float
    metadata: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ContextWindowExtractor:
    """Extracts context windows around topic mentions in documents"""

    def __init__(self, window_size: int = 3, max_context_length: int = 200):
        """
        Initialize context window extractor.

        Args:
            window_size: Number of sentences to include around topic mentions
            max_context_length: Maximum characters for context snippet
        """
        self.window_size = window_size
        self.max_context_length = max_context_length

    def extract_context_windows(
        self, documents: List[str], topic_keywords: Dict[int, List[str]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract context windows for each topic from documents.

        Args:
            documents: List of document texts
            topic_keywords: Dict mapping topic_id -> list of keywords

        Returns:
            Dict mapping topic_id -> list of context windows
        """
        context_windows = {topic_id: [] for topic_id in topic_keywords.keys()}

        for doc_idx, document in enumerate(documents):
            # Split document into sentences (simple split by period)
            sentences = [s.strip() for s in document.split(".") if s.strip()]
            if not sentences:
                continue

            # Find mentions of topic keywords in each sentence
            for topic_id, keywords in topic_keywords.items():
                topic_mentions = []

                for i, sentence in enumerate(sentences):
                    # Check if any keyword appears in this sentence
                    sentence_lower = sentence.lower()
                    keyword_matches = [
                        keyword
                        for keyword in keywords
                        if keyword.lower() in sentence_lower
                    ]

                    if keyword_matches:
                        # Extract context window around this sentence
                        start_idx = max(0, i - self.window_size)
                        end_idx = min(len(sentences), i + self.window_size + 1)

                        context_sentences = sentences[start_idx:end_idx]
                        context_text = ". ".join(context_sentences)

                        # Truncate if too long
                        if len(context_text) > self.max_context_length:
                            context_text = (
                                context_text[: self.max_context_length] + "..."
                            )

                        topic_mentions.append(
                            {
                                "document_index": doc_idx,
                                "sentence_index": i,
                                "matched_keywords": keyword_matches,
                                "context": context_text,
                                "full_sentence": sentence,
                            }
                        )

                context_windows[topic_id].extend(topic_mentions)

        return context_windows


class TopicKnowledgeGraphBuilder:
    """Builds knowledge graph from BERTopic clusters"""

    def __init__(self, context_window_size: int = 3):
        """
        Initialize the knowledge graph builder.

        Args:
            context_window_size: Number of sentences for context windows
        """
        self.context_extractor = ContextWindowExtractor(window_size=context_window_size)
        self.nodes = {}
        self.edges = {}
        self.topic_hierarchy = {}

    def build_from_bertopic(
        self,
        modeler,
        documents: List[str],
        min_relationship_strength: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from BERTopic model results.

        Args:
            modeler: Fitted DocumentTopicModeler instance
            documents: Original documents list
            min_relationship_strength: Minimum strength for relationships

        Returns:
            Knowledge graph data structure
        """
        print("🔗 Building Knowledge Graph from BERTopic clusters...")
        print("=" * 60)

        # Get all topics
        all_topics = modeler.get_all_topics()
        if not all_topics:
            return {"nodes": [], "edges": [], "hierarchy": {}}

        print(f"📊 Processing {len(all_topics)} topics...")

        # Step 1: Create topic nodes
        topic_nodes = self._create_topic_nodes(all_topics)

        # Step 2: Extract context windows for each topic
        topic_keywords = {
            topic_id: [word for word, score in topic_info.get("words", [])[:5]]
            for topic_id, topic_info in all_topics.items()
        }

        context_windows = self.context_extractor.extract_context_windows(
            documents, topic_keywords
        )

        # Step 3: Create concept nodes from context windows
        concept_nodes = self._create_concept_nodes(context_windows, all_topics)

        # Step 4: Create document nodes
        document_nodes = self._create_document_nodes(documents, modeler)

        # Step 5: Build hierarchical relationships
        hierarchy_edges = self._build_topic_hierarchy(all_topics)

        # Step 6: Build concept-to-topic relationships
        concept_edges = self._build_concept_relationships(
            concept_nodes, topic_nodes, context_windows, min_relationship_strength
        )

        # Step 7: Build document-to-topic relationships
        document_edges = self._build_document_relationships(
            document_nodes, topic_nodes, modeler
        )

        # Combine all nodes and edges
        all_nodes = {**topic_nodes, **concept_nodes, **document_nodes}
        all_edges = {**hierarchy_edges, **concept_edges, **document_edges}

        # Build topic hierarchy for navigation
        self.topic_hierarchy = self._build_navigation_hierarchy(all_topics)

        print(
            f"✅ Knowledge Graph built: {len(all_nodes)} nodes, {len(all_edges)} edges"
        )

        return {
            "nodes": [node.to_dict() for node in all_nodes.values()],
            "edges": [edge.to_dict() for edge in all_edges.values()],
            "hierarchy": self.topic_hierarchy,
            "metadata": {
                "total_topics": len(topic_nodes),
                "total_concepts": len(concept_nodes),
                "total_documents": len(document_nodes),
                "context_windows_extracted": sum(
                    len(windows) for windows in context_windows.values()
                ),
            },
        }

    def _create_topic_nodes(
        self, all_topics: Dict[int, Dict[str, Any]]
    ) -> Dict[str, KnowledgeGraphNode]:
        """Create nodes for each topic"""
        topic_nodes = {}

        for topic_id, topic_info in all_topics.items():
            node_id = f"topic_{topic_id}"

            # Create topic description from words and sample docs
            words = topic_info.get("words", [])
            top_words = [word for word, score in words[:5]]
            word_str = ", ".join(top_words)

            sample_docs = topic_info.get("representative_docs", [])
            sample_text = sample_docs[0][:100] + "..." if sample_docs else ""

            description = f"Topic covering {topic_info['count']} documents. Key themes: {word_str}"
            if sample_text:
                description += f" Example: {sample_text}"

            topic_nodes[node_id] = KnowledgeGraphNode(
                id=node_id,
                type="topic",
                name=topic_info["name"],
                description=description,
                metadata={
                    "topic_id": topic_id,
                    "document_count": topic_info["count"],
                    "top_words": top_words,
                    "word_scores": dict(words[:10]),
                    "coherence_score": self._calculate_topic_coherence(words),
                },
                created_at=datetime.now(),
            )

        return topic_nodes

    def _create_concept_nodes(
        self,
        context_windows: Dict[int, List[Dict[str, Any]]],
        all_topics: Dict[int, Dict[str, Any]],
    ) -> Dict[str, KnowledgeGraphNode]:
        """Create concept nodes from context windows"""
        concept_nodes = {}
        concept_counter = 0

        for topic_id, windows in context_windows.items():
            topic_name = all_topics[topic_id]["name"]

            # Group context windows by concept/theme
            concepts = self._extract_concepts_from_windows(windows, topic_name)

            for concept_name, concept_data in concepts.items():
                concept_id = f"concept_{concept_counter}"
                concept_counter += 1

                concept_nodes[concept_id] = KnowledgeGraphNode(
                    id=concept_id,
                    type="concept",
                    name=concept_name,
                    description=f"Concept extracted from {concept_data['window_count']} context windows in topic '{topic_name}'",
                    metadata={
                        "topic_id": topic_id,
                        "window_count": concept_data["window_count"],
                        "total_mentions": concept_data["total_mentions"],
                        "sample_contexts": concept_data["sample_contexts"][:3],
                        "related_keywords": concept_data["related_keywords"],
                    },
                    created_at=datetime.now(),
                )

        return concept_nodes

    def _create_document_nodes(
        self, documents: List[str], modeler
    ) -> Dict[str, KnowledgeGraphNode]:
        """Create document nodes"""
        document_nodes = {}

        for doc_idx, doc_info in modeler.document_topics.items():
            node_id = f"doc_{doc_idx}"

            # Create document summary
            doc_text = documents[doc_idx]
            summary = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text

            document_nodes[node_id] = KnowledgeGraphNode(
                id=node_id,
                type="document",
                name=f"Document {doc_idx + 1}",
                description=summary,
                metadata={
                    "document_index": doc_idx,
                    "topic_id": doc_info["topic_id"],
                    "topic_name": doc_info["topic_name"],
                    "confidence": doc_info["probability"],
                    "word_count": len(doc_text.split()),
                    "character_count": len(doc_text),
                },
                created_at=datetime.now(),
            )

        return document_nodes

    def _build_topic_hierarchy(
        self, all_topics: Dict[int, Dict[str, Any]]
    ) -> Dict[str, KnowledgeGraphEdge]:
        """Build hierarchical relationships between topics"""
        edges = {}

        # Sort topics by document count (size)
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )

        # Create subtopic relationships (larger topics contain smaller ones)
        for i, (topic_id, topic_info) in enumerate(sorted_topics):
            for j, (other_topic_id, other_topic_info) in enumerate(sorted_topics):
                if i != j and topic_id != -1 and other_topic_id != -1:
                    # Calculate similarity based on word overlap
                    similarity = self._calculate_topic_similarity(
                        topic_info.get("words", []), other_topic_info.get("words", [])
                    )

                    if similarity > 0.3:  # Minimum similarity threshold
                        edge_id = f"hierarchy_{topic_id}_{other_topic_id}"
                        source_id = f"topic_{topic_id}"
                        target_id = f"topic_{other_topic_id}"

                        edges[edge_id] = KnowledgeGraphEdge(
                            id=edge_id,
                            source_id=source_id,
                            target_id=target_id,
                            type=(
                                "subtopic_of"
                                if topic_info["count"] > other_topic_info["count"]
                                else "relates_to"
                            ),
                            weight=similarity,
                            metadata={
                                "similarity_type": "word_overlap",
                                "topic_size_source": topic_info["count"],
                                "topic_size_target": other_topic_info["count"],
                            },
                            created_at=datetime.now(),
                        )

        return edges

    def _build_concept_relationships(
        self,
        concept_nodes: Dict[str, KnowledgeGraphNode],
        topic_nodes: Dict[str, KnowledgeGraphNode],
        context_windows: Dict[int, List[Dict[str, Any]]],
        min_strength: float,
    ) -> Dict[str, KnowledgeGraphEdge]:
        """Build relationships between concepts and topics"""
        edges = {}

        for concept_id, concept_node in concept_nodes.items():
            topic_id = concept_node.metadata["topic_id"]
            topic_node_id = f"topic_{topic_id}"

            if topic_node_id in topic_nodes:
                edge_id = f"concept_topic_{concept_id}_{topic_node_id}"

                edges[edge_id] = KnowledgeGraphEdge(
                    id=edge_id,
                    source_id=concept_id,
                    target_id=topic_node_id,
                    type="belongs_to_topic",
                    weight=concept_node.metadata["window_count"] / 10.0,  # Normalize
                    metadata={
                        "relationship_type": "concept_to_topic",
                        "context_window_count": concept_node.metadata["window_count"],
                        "mentions": concept_node.metadata["total_mentions"],
                    },
                    created_at=datetime.now(),
                )

        return edges

    def _build_document_relationships(
        self,
        document_nodes: Dict[str, KnowledgeGraphNode],
        topic_nodes: Dict[str, KnowledgeGraphNode],
        modeler,
    ) -> Dict[str, KnowledgeGraphEdge]:
        """Build relationships between documents and topics"""
        edges = {}

        for doc_id, doc_node in document_nodes.items():
            topic_id = doc_node.metadata["topic_id"]
            topic_node_id = f"topic_{topic_id}"

            if topic_node_id in topic_nodes:
                edge_id = f"doc_topic_{doc_id}_{topic_node_id}"

                edges[edge_id] = KnowledgeGraphEdge(
                    id=edge_id,
                    source_id=doc_id,
                    target_id=topic_node_id,
                    type="belongs_to_topic",
                    weight=doc_node.metadata["confidence"],
                    metadata={
                        "relationship_type": "document_to_topic",
                        "confidence": doc_node.metadata["confidence"],
                        "document_index": doc_node.metadata["document_index"],
                    },
                    created_at=datetime.now(),
                )

        return edges

    def _extract_concepts_from_windows(
        self, windows: List[Dict[str, Any]], topic_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract key concepts from context windows"""
        from collections import Counter

        concepts = {}

        for window in windows:
            context = window["context"]
            keywords = window["matched_keywords"]

            # Simple concept extraction: look for noun phrases and important terms
            # In practice, you might use NLP libraries like spaCy
            words = context.split()
            important_words = [w for w in words if len(w) > 4 and w.isalpha()]

            for keyword in keywords:
                if keyword not in concepts:
                    concepts[keyword] = {
                        "window_count": 0,
                        "total_mentions": 0,
                        "sample_contexts": [],
                        "related_keywords": set(),
                    }

                concepts[keyword]["window_count"] += 1
                concepts[keyword]["total_mentions"] += len(keywords)
                concepts[keyword]["sample_contexts"].append(window["context"][:100])

                # Add related important words
                for word in important_words[:3]:
                    concepts[keyword]["related_keywords"].add(word)

        # Convert sets to lists for JSON serialization
        for concept_data in concepts.values():
            concept_data["related_keywords"] = list(concept_data["related_keywords"])

        return concepts

    def _calculate_topic_similarity(
        self, words1: List[Tuple[str, float]], words2: List[Tuple[str, float]]
    ) -> float:
        """Calculate similarity between two topics based on word overlap"""
        if not words1 or not words2:
            return 0.0

        # Create word sets
        words_set1 = {word for word, score in words1[:10]}
        words_set2 = {word for word, score in words2[:10]}

        # Calculate Jaccard similarity
        intersection = len(words_set1.intersection(words_set2))
        union = len(words_set1.union(words_set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_topic_coherence(self, words: List[Tuple[str, float]]) -> float:
        """Calculate coherence score for a topic"""
        if len(words) < 2:
            return 0.0

        scores = [score for word, score in words[:5]]
        if not scores:
            return 0.0

        # Simple coherence measure based on score variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Lower variance = higher coherence
        coherence = 1.0 / (1.0 + variance)
        return min(1.0, coherence)

    def _build_navigation_hierarchy(
        self, all_topics: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build hierarchical structure for navigation"""
        hierarchy = {"main_topics": [], "topic_details": {}, "navigation_paths": []}

        # Sort topics by size for main navigation
        sorted_topics = sorted(
            all_topics.items(), key=lambda x: x[1]["count"], reverse=True
        )

        # Create main topic entries
        for topic_id, topic_info in sorted_topics[:10]:  # Top 10 topics
            hierarchy["main_topics"].append(
                {
                    "id": f"topic_{topic_id}",
                    "name": topic_info["name"],
                    "document_count": topic_info["count"],
                    "top_words": [
                        word for word, score in topic_info.get("words", [])[:3]
                    ],
                }
            )

            # Store detailed info
            hierarchy["topic_details"][f"topic_{topic_id}"] = {
                "topic_id": topic_id,
                "name": topic_info["name"],
                "description": f"Topic with {topic_info['count']} documents",
                "document_count": topic_info["count"],
                "words": topic_info.get("words", []),
                "sample_documents": topic_info.get("representative_docs", [])[:2],
            }

        return hierarchy


class KnowledgeGraphNavigator:
    """
    Provides reader-friendly navigation through knowledge graph.

    Enables intuitive exploration of topics, concepts, and relationships
    with search and similarity capabilities.
    """

    def __init__(self, knowledge_graph: Dict[str, Any]):
        """
        Initialize navigator with knowledge graph data.

        Args:
            knowledge_graph: Knowledge graph data structure from TopicKnowledgeGraphBuilder
        """
        self.graph = knowledge_graph
        self.nodes = {node["id"]: node for node in knowledge_graph["nodes"]}
        self.edges = {edge["id"]: edge for edge in knowledge_graph["edges"]}
        self.hierarchy = knowledge_graph.get("hierarchy", {})

    def get_topic_overview(self) -> Dict[str, Any]:
        """
        Get high-level overview of all topics.

        Returns:
            Dictionary with topic statistics and hierarchy
        """
        topic_nodes = [node for node in self.graph["nodes"] if node["type"] == "topic"]

        return {
            "total_topics": len(topic_nodes),
            "topics": sorted(
                topic_nodes, key=lambda x: x["metadata"]["document_count"], reverse=True
            ),
            "hierarchy": self.hierarchy.get("main_topics", []),
        }

    def explore_topic(self, topic_id: str) -> Dict[str, Any]:
        """
        Explore a specific topic in detail.

        Args:
            topic_id: Topic identifier (e.g., "topic_0")

        Returns:
            Detailed information about the topic and its relationships
        """
        if topic_id not in self.nodes:
            return {"error": f"Topic '{topic_id}' not found"}

        topic_node = self.nodes[topic_id]

        # Get related concepts
        concept_edges = [
            edge
            for edge in self.graph["edges"]
            if edge["target_id"] == topic_id and edge["type"] == "belongs_to_topic"
        ]
        concept_nodes = [
            self.nodes[edge["source_id"]]
            for edge in concept_edges
            if edge["source_id"] in self.nodes
        ]

        # Get related documents
        document_edges = [
            edge
            for edge in self.graph["edges"]
            if edge["target_id"] == topic_id and edge["type"] == "belongs_to_topic"
        ]
        document_nodes = [
            self.nodes[edge["source_id"]]
            for edge in document_edges
            if edge["source_id"] in self.nodes
        ]

        # Get related topics
        related_topic_edges = [
            edge
            for edge in self.graph["edges"]
            if edge["source_id"] == topic_id
            and edge["type"] in ["subtopic_of", "relates_to"]
        ]
        related_topics = [
            self.nodes[edge["target_id"]]
            for edge in related_topic_edges
            if edge["target_id"] in self.nodes
        ]

        return {
            "topic": topic_node,
            "concepts": concept_nodes,
            "documents": document_nodes,
            "related_topics": related_topics,
            "navigation_path": self._get_navigation_path(topic_id),
        }

    def _get_navigation_path(self, topic_id: str) -> List[Dict[str, Any]]:
        """
        Get breadcrumb navigation path for a topic.

        Args:
            topic_id: Topic identifier

        Returns:
            List of navigation path elements
        """
        path = []

        # Simple path: topic -> main topic area
        topic_node = self.nodes.get(topic_id)
        if topic_node:
            path.append({"id": topic_id, "name": topic_node["name"], "type": "topic"})

        return path

    def search_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph for relevant nodes.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of search results with relevance scores
        """
        query_lower = query.lower()
        results = []

        for node in self.graph["nodes"]:
            # Search in name and description
            name_lower = node["name"].lower()
            desc_lower = node["description"].lower()

            if query_lower in name_lower or query_lower in desc_lower:
                # Calculate relevance score
                name_matches = query_lower in name_lower
                desc_matches = query_lower in desc_lower

                relevance = 1.0 if name_matches else 0.5 if desc_matches else 0.3

                results.append(
                    {
                        "node": node,
                        "relevance": relevance,
                        "match_type": "name" if name_matches else "description",
                    }
                )

        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]

    def get_similar_topics(self, topic_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get topics similar to the given topic.

        Args:
            topic_id: Topic identifier to find similar topics for
            limit: Maximum number of similar topics to return

        Returns:
            List of similar topics with similarity scores
        """
        if topic_id not in self.nodes:
            return []

        # Find edges of type "relates_to" or "subtopic_of"
        related_edges = [
            edge
            for edge in self.graph["edges"]
            if edge["source_id"] == topic_id
            and edge["type"] in ["relates_to", "subtopic_of"]
        ]

        similar_topics = []
        for edge in related_edges:
            if edge["target_id"] in self.nodes:
                target_node = self.nodes[edge["target_id"]]
                similar_topics.append(
                    {
                        "topic": target_node,
                        "similarity": edge["weight"],
                        "relationship_type": edge["type"],
                    }
                )

        # Sort by similarity
        similar_topics.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_topics[:limit]

    def export_for_visualization(self) -> Dict[str, Any]:
        """
        Export graph data in format suitable for visualization.

        Returns:
            Dictionary with nodes and edges formatted for D3.js or similar
        """
        return {
            "nodes": [
                {
                    "id": node["id"],
                    "label": node["name"],
                    "type": node["type"],
                    "metadata": node["metadata"],
                }
                for node in self.graph["nodes"]
            ],
            "edges": [
                {
                    "id": edge["id"],
                    "source": edge["source_id"],
                    "target": edge["target_id"],
                    "type": edge["type"],
                    "weight": edge["weight"],
                }
                for edge in self.graph["edges"]
            ],
        }
