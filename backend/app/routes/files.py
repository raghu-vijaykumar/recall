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
from app.services.workspace_analysis_service import WorkspaceAnalysisService
from sqlalchemy.ext.asyncio import AsyncSession

# Import get_db from the database module
from app.services.database import get_db

# Create router
router = APIRouter()


# Dependency to get file service
def get_file_service() -> FileService:
    # Use singleton instance
    db_service = DatabaseService()
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
    file_id: int,
    content: str,
    service: FileService = Depends(get_file_service),
    db: AsyncSession = Depends(get_db),
):
    """Update file content"""
    try:
        success = service.save_file_content(file_id, content)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        # Trigger incremental topic analysis update
        await _trigger_incremental_analysis(file_id, db)

        return {"message": "File content updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update file content: {str(e)}"
        )


@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    service: FileService = Depends(get_file_service),
    db: AsyncSession = Depends(get_db),
):
    """Delete a file"""
    try:
        # Get file info before deletion for incremental analysis
        file_info = service.get_file(file_id)
        success = service.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        # Trigger incremental analysis cleanup
        if file_info:
            await _trigger_file_deletion_cleanup(
                file_info.path, file_info.workspace_id, db
            )

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


# Helper functions for incremental analysis
async def _trigger_incremental_analysis(file_id: int, db: AsyncSession):
    """Trigger incremental embedding analysis for a modified file"""
    try:
        # Get file information
        from sqlalchemy import text

        query = text(
            """
            SELECT f.path, f.workspace_id, w.path as workspace_path
            FROM files f
            JOIN workspaces w ON f.workspace_id = w.id
            WHERE f.id = :file_id
        """
        )

        result = await db.execute(query, {"file_id": file_id})
        file_record = result.fetchone()

        if file_record:
            # Note: Incremental file analysis not yet supported in topics-only architecture
            # Full workspace analysis should be triggered instead
            # TODO: Implement incremental topic analysis if needed
            pass

    except Exception as e:
        # Log error but don't fail the file operation
        print(f"Incremental analysis failed for file {file_id}: {str(e)}")


async def _trigger_file_deletion_cleanup(
    file_path: str, workspace_id: int, db: AsyncSession
):
    """Clean up embeddings and relationships when a file is deleted"""
    try:
        # Note: File-specific cleanup not yet implemented in topics-only architecture
        # TODO: Implement topic cleanup for deleted files if needed
        pass

    except Exception as e:
        # Log error but don't fail the file deletion
        print(f"File deletion cleanup failed for {file_path}: {str(e)}")
