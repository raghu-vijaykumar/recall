"""
Workspace API routes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.models.workspace import (
    Workspace,
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceStats,
)
from app.services.workspace_service import WorkspaceService
from app.services.database import DatabaseService

# Create router
router = APIRouter()


# Dependency to get workspace service
def get_workspace_service() -> WorkspaceService:
    db_service = DatabaseService("database/recall.db")
    return WorkspaceService(db_service)


@router.post("/", response_model=Workspace)
async def create_workspace(
    workspace: WorkspaceCreate,
    service: WorkspaceService = Depends(get_workspace_service),
):
    """Create a new workspace"""
    try:
        return service.create_workspace(workspace)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create workspace: {str(e)}"
        )


@router.get("/", response_model=List[Workspace])
async def get_workspaces(service: WorkspaceService = Depends(get_workspace_service)):
    """Get all workspaces"""
    try:
        return service.get_all_workspaces()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workspaces: {str(e)}"
        )


@router.get("/{workspace_id}", response_model=Workspace)
async def get_workspace(
    workspace_id: int, service: WorkspaceService = Depends(get_workspace_service)
):
    """Get a specific workspace by ID"""
    try:
        workspace = service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return workspace
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workspace: {str(e)}"
        )


@router.put("/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: int,
    workspace_update: WorkspaceUpdate,
    service: WorkspaceService = Depends(get_workspace_service),
):
    """Update a workspace"""
    try:
        workspace = service.update_workspace(workspace_id, workspace_update)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return workspace
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update workspace: {str(e)}"
        )


@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: int, service: WorkspaceService = Depends(get_workspace_service)
):
    """Delete a workspace"""
    try:
        success = service.delete_workspace(workspace_id)
        if not success:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"message": "Workspace deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete workspace: {str(e)}"
        )


@router.get("/{workspace_id}/stats", response_model=WorkspaceStats)
async def get_workspace_stats(
    workspace_id: int, service: WorkspaceService = Depends(get_workspace_service)
):
    """Get detailed statistics for a workspace"""
    try:
        # Check if workspace exists
        workspace = service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return service.get_workspace_stats(workspace_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workspace stats: {str(e)}"
        )
