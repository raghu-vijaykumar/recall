"""
Knowledge Graph API routes
"""

print("[KG] MODULE LOADED", file=__import__("sys").stderr)
__import__("sys").stderr.flush()

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..database import get_db
from ..services import WorkspaceAnalysisService
from ..services.embedding_service import EmbeddingService
import os
from pathlib import Path


def get_user_data_directory() -> str:
    """Get the user data directory for storing embeddings (same as database location)"""
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, ".recall", "embeddings")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


router = APIRouter(prefix="/api/knowledge-graph", tags=["knowledge-graph"])


# Pydantic models for API requests/responses
class KnowledgeGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class SuggestedTopicsResponse(BaseModel):
    topics: List[Dict[str, Any]]


@router.post("/workspaces/{workspace_id}/analyze")
async def analyze_workspace(
    workspace_id: int,
    force_reanalysis: bool = False,
    file_paths: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger workspace analysis to extract concepts and build knowledge graph
    """
    # Get workspace path from database
    workspace_query = text(
        "SELECT folder_path FROM workspaces WHERE id = :workspace_id"
    )
    result = await db.execute(workspace_query, {"workspace_id": workspace_id})
    workspace_record = result.fetchone()

    if not workspace_record:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace_path = workspace_record.folder_path

    # Initialize analysis service with embedding support
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)

    # Auto-initialize embedding model if not already done
    if not embedding_service.current_model:
        try:
            # Initialize with recommended model
            success = await embedding_service.initialize()
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to initialize embedding model"
                )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error initializing embedding model: {str(e)}"
            )

    analysis_service = WorkspaceAnalysisService(db=db, extractor_type="heuristic")

    try:
        # Run analysis
        results = await analysis_service.analyze_workspace(
            workspace_id=workspace_id,
            workspace_path=workspace_path,
            force_reanalysis=force_reanalysis,
            file_paths=file_paths,
        )

        return {
            "status": "analysis_completed",
            "task_id": f"analysis_{workspace_id}_{hash(str(file_paths))}",
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Concept-related endpoints removed in topics-only architecture
# All concept CRUD operations have been removed


# Embedding management endpoints
@router.get("/embeddings/models")
async def get_available_embedding_models():
    """Get information about available embedding models"""
    # Create a temporary service instance to get model info
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)
    models = embedding_service.get_available_models()
    current = embedding_service.get_current_model_info()

    return {
        "available_models": models,
        "current_model": current,
    }


@router.post("/embeddings/initialize")
async def initialize_embedding_model(model_name: str):
    """Initialize the embedding service with a specific model"""
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)

    try:
        success = await embedding_service.initialize(model_name)
        if success:
            return {
                "success": True,
                "model": model_name,
                "message": f"Successfully initialized {model_name}",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize model")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/embeddings/switch-model")
async def switch_embedding_model(new_model_name: str, reembed_all: bool = False):
    """Switch to a different embedding model"""
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)

    # Initialize with current model first if not already done
    if not embedding_service.current_model:
        # Try to initialize with any available model
        available = embedding_service.get_available_models()
        current_name = None
        for name, info in available.items():
            if info.get("recommended"):
                current_name = name
                break
        if not current_name:
            current_name = list(available.keys())[0]

        success = await embedding_service.initialize(current_name)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to initialize current model"
            )

    # Switch to new model
    result = await embedding_service.switch_model(new_model_name, reembed_all)

    if result["success"]:
        return result
    else:
        raise HTTPException(
            status_code=500, detail=result.get("error", "Switch failed")
        )


@router.get("/embeddings/stats")
async def get_embedding_stats():
    """Get statistics about the current embedding collection"""
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)

    if not embedding_service.current_model:
        return {"error": "Embedding service not initialized"}

    stats = await embedding_service.get_collection_stats()
    return stats


@router.post("/embeddings/search")
async def search_similar_concepts(query: str, limit: int = 10, threshold: float = 0.0):
    """Search for concepts similar to the query text"""
    user_data_dir = get_user_data_directory()
    embedding_service = EmbeddingService(persist_directory=user_data_dir)

    if not embedding_service.current_model:
        raise HTTPException(status_code=500, detail="Embedding service not initialized")

    results = await embedding_service.search_similar_concepts(query, limit, threshold)
    return {"results": results}
