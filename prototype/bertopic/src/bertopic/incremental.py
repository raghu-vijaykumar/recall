#!/usr/bin/env python3
"""
Incremental Index Manager for BERTopic Knowledge Graph
Manages incremental updates without full re-processing
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Configure package logger
logger = logging.getLogger(__name__)


class IncrementalIndexManager:
    """Manages incremental updates to topic index without full regeneration"""

    def __init__(self, index_file: str = "topic_index.json"):
        """
        Initialize incremental index manager.

        Args:
            index_file: File to store index metadata
        """
        self.index_file = index_file
        self.index_metadata = self._load_index_metadata()

    def _load_index_metadata(self) -> Dict[str, Any]:
        """Load existing index metadata"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, "r") as f:
                    return json.load(f)
        except:
            pass
        return {"file_hashes": {}, "last_updated": None, "version": "1.0"}

    def _save_index_metadata(self):
        """Save index metadata to file"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, "w") as f:
            json.dump(self.index_metadata, f, indent=2)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for change detection"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def detect_file_changes(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Detect which files have changed since last indexing.

        Args:
            file_paths: List of file paths to check

        Returns:
            Dict with 'new_files', 'modified_files', 'deleted_files'
        """
        current_hashes = {}
        for file_path in file_paths:
            current_hashes[file_path] = self.calculate_file_hash(file_path)

        previous_hashes = self.index_metadata.get("file_hashes", {})

        new_files = []
        modified_files = []
        unchanged_files = []

        for file_path, current_hash in current_hashes.items():
            if file_path not in previous_hashes:
                new_files.append(file_path)
            elif previous_hashes[file_path] != current_hash:
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        # Find deleted files (in previous but not in current)
        deleted_files = [
            file_path
            for file_path in previous_hashes.keys()
            if file_path not in current_hashes
        ]

        changes = {
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "unchanged_files": unchanged_files,
            "total_new_modified": len(new_files) + len(modified_files),
        }

        print(f"📊 Change Detection: {changes['total_new_modified']} files changed")
        if changes["new_files"]:
            print(f"   🆕 New files: {len(changes['new_files'])}")
        if changes["modified_files"]:
            print(f"   🔄 Modified files: {len(changes['modified_files'])}")
        if changes["deleted_files"]:
            print(f"   🗑️  Deleted files: {len(changes['deleted_files'])}")

        return changes

    def update_index_metadata(self, file_paths: List[str], model_version: str = "1.0"):
        """Update index metadata with current file hashes"""
        current_hashes = {}
        for file_path in file_paths:
            current_hashes[file_path] = self.calculate_file_hash(file_path)

        self.index_metadata["file_hashes"] = current_hashes
        self.index_metadata["last_updated"] = datetime.now().isoformat()
        self.index_metadata["version"] = model_version
        self._save_index_metadata()

    def should_reindex(self, file_paths: List[str], force: bool = False) -> bool:
        """
        Determine if full reindexing is needed.

        Args:
            file_paths: List of file paths to check
            force: Force reindexing regardless of changes

        Returns:
            True if reindexing is needed
        """
        if force:
            return True

        changes = self.detect_file_changes(file_paths)
        return changes["total_new_modified"] > 0


class KnowledgeGraphNavigator:
    """Provides reader-friendly navigation through knowledge graph"""

    def __init__(self, knowledge_graph: Dict[str, Any]):
        """
        Initialize navigator with knowledge graph data.

        Args:
            knowledge_graph: Knowledge graph data structure
        """
        self.graph = knowledge_graph
        self.nodes = {node["id"]: node for node in knowledge_graph["nodes"]}
        self.edges = {edge["id"]: edge for edge in knowledge_graph["edges"]}
        self.hierarchy = knowledge_graph.get("hierarchy", {})

    def get_topic_overview(self) -> Dict[str, Any]:
        """Get high-level overview of all topics"""
        topic_nodes = [node for node in self.graph["nodes"] if node["type"] == "topic"]

        return {
            "total_topics": len(topic_nodes),
            "topics": sorted(
                topic_nodes, key=lambda x: x["metadata"]["document_count"], reverse=True
            ),
            "hierarchy": self.hierarchy.get("main_topics", []),
        }

    def explore_topic(self, topic_id: str) -> Dict[str, Any]:
        """Explore a specific topic in detail"""
        if topic_id not in self.nodes:
            return {"error": "Topic not found"}

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
        """Get breadcrumb navigation path for a topic"""
        path = []

        # Simple path: topic -> main topic area
        topic_node = self.nodes.get(topic_id)
        if topic_node:
            path.append({"id": topic_id, "name": topic_node["name"], "type": "topic"})

        return path

    def search_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph for relevant nodes"""
        query_lower = query.lower()
        results = []

        for node in self.graph["nodes"]:
            # Search in name and description
            if (
                query_lower in node["name"].lower()
                or query_lower in node["description"].lower()
            ):

                # Calculate relevance score
                name_matches = query_lower in node["name"].lower()
                desc_matches = query_lower in node["description"].lower()

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
        """Get topics similar to the given topic"""
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
        """Export graph data in format suitable for visualization"""
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
