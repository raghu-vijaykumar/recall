"""
File API routes
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional
from pathlib import Path

from app.models.file import (
    FileItem,
    FileCreate,
    FileUpdate,
    FileTreeNode,
)
from app.services.file_service import FileService
from app.services.database import DatabaseService

# Create router
router = APIRouter()


# Dependency to get file service
def get_file_service() -> FileService:
    db_service = DatabaseService()  # Use environment variable DATABASE_PATH
    return FileService(db_service)


@router.post("/", response_model=FileItem)
async def create_file(
    file_data: FileCreate, service: FileService = Depends(get_file_service)
):
    """Create a new file"""
    try:
        return service.create_file(file_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create file: {str(e)}")


@router.get("/workspace/{workspace_id}", response_model=List[FileItem])
async def get_workspace_files(
    workspace_id: int, service: FileService = Depends(get_file_service)
):
    """Get all files in a workspace"""
    try:
        return service.get_files_by_workspace(workspace_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workspace files: {str(e)}"
        )


@router.get("/{file_id}", response_model=FileItem)
async def get_file(file_id: int, service: FileService = Depends(get_file_service)):
    """Get file by ID"""
    try:
        file = service.get_file(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return file
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file: {str(e)}")


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: int, service: FileService = Depends(get_file_service)
):
    """Get file content"""
    try:
        content = service.get_file_content(file_id)
        if content is None:
            raise HTTPException(status_code=404, detail="File not found")
        return {"content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get file content: {str(e)}"
        )


@router.put("/{file_id}", response_model=FileItem)
async def update_file(
    file_id: int,
    update_data: FileUpdate,
    service: FileService = Depends(get_file_service),
):
    """Update file information"""
    try:
        file = service.update_file(file_id, update_data)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return file
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update file: {str(e)}")


@router.put("/{file_id}/content")
async def update_file_content(
    file_id: int, content: str, service: FileService = Depends(get_file_service)
):
    """Update file content"""
    try:
        success = service.save_file_content(file_id, content)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"message": "File content updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update file content: {str(e)}"
        )


@router.delete("/{file_id}")
async def delete_file(file_id: int, service: FileService = Depends(get_file_service)):
    """Delete a file"""
    try:
        success = service.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        return {"message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/tree/{workspace_id}", response_model=List[FileTreeNode])
async def get_file_tree(
    workspace_id: int, service: FileService = Depends(get_file_service)
):
    """Get file tree structure for workspace"""
    try:
        return service.get_file_tree(workspace_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get file tree: {str(e)}"
        )


@router.post("/upload/{workspace_id}")
async def upload_file(
    workspace_id: int,
    file: UploadFile = File(...),
    service: FileService = Depends(get_file_service),
):
    """Upload a file to workspace"""
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode("utf-8")

        # Create file data
        file_data = FileCreate(
            name=file.filename,
            path=file.filename,  # Simple case - filename as path
            file_type=service._get_file_type(file.filename),
            size=len(content),
            workspace_id=workspace_id,
            content=content_str,
        )

        created_file = service.create_file(file_data)
        return created_file
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@router.post("/scan-folder/{workspace_id}")
async def scan_folder(
    workspace_id: int, service: FileService = Depends(get_file_service)
):
    """Scan folder associated with workspace and create files"""
    try:
        created_files = service.scan_workspace_folder(workspace_id)
        return {
            "message": f"Successfully scanned folder and created {len(created_files)} files",
            "files_created": len(created_files),
            "files": created_files,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan folder: {str(e)}")
