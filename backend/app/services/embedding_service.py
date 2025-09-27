"""
Embedding Service for managing text embeddings and vector storage with ChromaDB
"""

from logging import log
import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import hashlib
import torch


class EmbeddingService:
    """
    Singleton service for managing text embeddings with configurable models and ChromaDB storage
    """

    # Singleton instance
    _instance = None

    # Available embedding models with their configurations
    AVAILABLE_MODELS = {
        "sentence-transformers/all-MiniLM-L12-v2": {
            "dimensions": 384,
            "size_mb": 33,
            "description": "Improved version of L6-v2 with better performance (33MB)",
            "recommended": True,
        },
        "BAAI/bge-large-en-v1.5": {
            "dimensions": 1024,
            "size_mb": 334,
            "description": "High-performance model with excellent retrieval accuracy (334MB)",
            "recommended": False,
        },
        "nomic-ai/nomic-embed-text-v1.5": {
            "dimensions": 768,
            "size_mb": 274,
            "description": "Modern model with strong performance and efficiency (274MB)",
            "recommended": False,
        },
        "intfloat/e5-large-v2": {
            "dimensions": 1024,
            "size_mb": 336,
            "description": "Excellent general-purpose model with high accuracy (336MB)",
            "recommended": False,
        },
        "jinaai/jina-embeddings-v2-base-en": {
            "dimensions": 768,
            "size_mb": 554,
            "description": "Modern transformer-based model with strong multilingual support (554MB)",
            "recommended": False,
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "dimensions": 768,
            "size_mb": 110,
            "description": "Established high-quality model with GPU acceleration (110MB)",
            "recommended": False,
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimensions": 384,
            "size_mb": 23,
            "description": "Fast, lightweight general-purpose model (23MB)",
            "recommended": False,
        },
    }

    def __new__(cls, persist_directory: str = None):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls, persist_directory: str = None) -> "EmbeddingService":
        """
        Get the singleton instance of EmbeddingService

        Args:
            persist_directory: Directory to store ChromaDB data (only used on first instantiation)

        Returns:
            The singleton EmbeddingService instance
        """
        if cls._instance is None:
            cls._instance = cls(persist_directory)
        return cls._instance

    def __init__(self, persist_directory: str = None):
        """
        Initialize the embedding service (called only once due to singleton pattern)

        Args:
            persist_directory: Directory to store ChromaDB data (defaults to user data dir)
        """
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        """
        Initialize the embedding service

        Args:
            persist_directory: Directory to store ChromaDB data (defaults to user data dir)
        """
        if persist_directory is None:
            # Use user data directory by default (same as database)
            import os

            home_dir = os.path.expanduser("~")
            persist_directory = os.path.join(home_dir, ".recall", "embeddings")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Current model state
        self.current_model_name = None
        self.current_model = None
        self.chroma_client = None
        self.collection = None

        # Model switching state
        self.model_switch_in_progress = False

    async def initialize(
        self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ) -> bool:
        """
        Initialize with a specific embedding model

        Args:
            model_name: Name of the embedding model to use

        Returns:
            True if initialization successful
        """
        try:
            # Validate model
            if model_name not in self.AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {model_name}")

            # Detect available device (GPU if available, otherwise CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Initializing embedding model '{model_name}' on device: {device}")

            # Load model (this can take time)
            self.current_model = SentenceTransformer(model_name, device=device)
            self.current_model_name = model_name

            # Initialize ChromaDB client
            chroma_settings = Settings(
                persist_directory=str(self.persist_directory / model_name),
                is_persistent=True,
            )
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory / model_name), settings=chroma_settings
            )

            # Get or create collection for this model
            # Replace all non-alphanumeric characters with underscores for valid collection names
            safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
            collection_name = f"workspace_concepts_{safe_model_name}"
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "model": model_name,
                    "dimensions": self.AVAILABLE_MODELS[model_name]["dimensions"],
                },
            )

            return True

        except Exception as e:
            print(f"Failed to initialize embedding service: {e}")
            return False

    async def switch_model(
        self, new_model_name: str, reembed_all: bool = False
    ) -> Dict[str, Any]:
        """
        Switch to a different embedding model

        Args:
            new_model_name: Name of the new model
            reembed_all: Whether to re-embed all existing content

        Returns:
            Status dictionary with results
        """
        if self.model_switch_in_progress:
            return {"success": False, "error": "Model switch already in progress"}

        if new_model_name == self.current_model_name:
            return {"success": True, "message": "Model already active"}

        self.model_switch_in_progress = True

        try:
            # Count existing embeddings if re-embedding
            existing_count = 0
            if reembed_all and self.collection:
                existing_count = self.collection.count()

            # Initialize new model
            success = await self.initialize(new_model_name)
            if not success:
                return {"success": False, "error": "Failed to initialize new model"}

            result = {
                "success": True,
                "old_model": self.current_model_name,
                "new_model": new_model_name,
                "reembedded": False,
            }

            # Re-embed all content if requested
            if reembed_all and existing_count > 0:
                # Note: In a full implementation, we'd need to:
                # 1. Get all existing concept-file links from database
                # 2. Re-extract text content
                # 3. Generate new embeddings
                # 4. Update ChromaDB collection
                # For now, we'll just clear the old collection
                # Use same safe naming convention as in initialize
                old_safe_name = "".join(
                    c if c.isalnum() else "_" for c in self.current_model_name
                )
                old_collection_name = f"workspace_concepts_{old_safe_name}"
                try:
                    old_client = chromadb.PersistentClient(
                        path=str(self.persist_directory / self.current_model_name)
                    )
                    old_client.delete_collection(old_collection_name)
                except:
                    pass  # Old collection might not exist

                result["reembedded"] = True
                result["message"] = (
                    f"Switched to {new_model_name}. Re-embedding of {existing_count} items needed."
                )

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

        finally:
            self.model_switch_in_progress = False

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available embedding models"""
        return self.AVAILABLE_MODELS.copy()

    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active model"""
        if not self.current_model_name:
            return None

        info = self.AVAILABLE_MODELS[self.current_model_name].copy()
        info["name"] = self.current_model_name
        info["active"] = True

        # Add device information
        if self.current_model:
            device = next(self.current_model.parameters()).device
            info["device"] = str(device)
            info["gpu_available"] = torch.cuda.is_available()
        else:
            info["device"] = "unknown"
            info["gpu_available"] = torch.cuda.is_available()

        # Add collection stats if available
        if self.collection:
            try:
                info["collection_count"] = self.collection.count()
            except:
                info["collection_count"] = 0

        return info

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self.current_model:
            raise RuntimeError("Embedding service not initialized")

        # Run embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.current_model.encode, texts)

        # Convert to list of lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        return embeddings

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def add_concept_embedding(
        self, concept_id: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a concept embedding to the vector database

        Args:
            concept_id: Unique concept identifier
            text: Text content to embed
            metadata: Additional metadata

        Returns:
            True if successful
        """
        if not self.collection:
            return False

        try:
            # Generate embedding
            embedding = await self.embed_text(text)

            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update(
                {
                    "concept_id": concept_id,
                    "model": self.current_model_name,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

            # Add to collection
            self.collection.add(
                ids=[concept_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata],
            )

            return True

        except Exception as e:
            print(f"Failed to add concept embedding: {e}")
            return False

    async def search_similar_concepts(
        self, query_text: str, limit: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for concepts similar to the query text

        Args:
            query_text: Text to search for
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar concepts with scores
        """
        if not self.collection:
            return []

        try:
            # Generate query embedding
            query_embedding = await self.embed_text(query_text)

            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            similar_concepts = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, concept_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity

                    if similarity >= threshold:
                        similar_concepts.append(
                            {
                                "concept_id": concept_id,
                                "similarity": similarity,
                                "text": (
                                    results["documents"][0][i]
                                    if results["documents"]
                                    else ""
                                ),
                                "metadata": (
                                    results["metadatas"][0][i]
                                    if results["metadatas"]
                                    else {}
                                ),
                            }
                        )

            return similar_concepts

        except Exception as e:
            print(f"Failed to search similar concepts: {e}")
            return []

    async def update_concept_embedding(
        self, concept_id: str, new_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing concept embedding

        Args:
            concept_id: Concept to update
            new_text: New text content
            metadata: Updated metadata

        Returns:
            True if successful
        """
        if not self.collection:
            return False

        try:
            # Remove old embedding
            self.collection.delete(ids=[concept_id])

            # Add new embedding
            return await self.add_concept_embedding(concept_id, new_text, metadata)

        except Exception as e:
            print(f"Failed to update concept embedding: {e}")
            return False

    async def remove_concept_embedding(self, concept_id: str) -> bool:
        """
        Remove a concept embedding from the database

        Args:
            concept_id: Concept to remove

        Returns:
            True if successful
        """
        if not self.collection:
            return False

        try:
            self.collection.delete(ids=[concept_id])
            return True
        except Exception as e:
            print(f"Failed to remove concept embedding: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        if not self.collection:
            return {"error": "No collection available"}

        try:
            count = self.collection.count()
            return {
                "model": self.current_model_name,
                "total_concepts": count,
                "dimensions": self.AVAILABLE_MODELS.get(
                    self.current_model_name, {}
                ).get("dimensions", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_concept_id(self, text: str, workspace_id: int) -> str:
        """
        Generate a deterministic concept ID from text and workspace

        Args:
            text: Concept text
            workspace_id: Workspace identifier

        Returns:
            Unique concept ID
        """
        # Create hash of text + workspace for deterministic ID
        content = f"{workspace_id}:{text.lower().strip()}"
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:32]  # First 32 chars of hash
