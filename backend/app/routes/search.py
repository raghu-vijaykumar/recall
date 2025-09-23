from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import re

router = APIRouter()


class SearchRequest(BaseModel):
    workspace_id: int
    query: str
    folder_path: str


class SearchMatch(BaseModel):
    line: int
    text: str


class SearchResult(BaseModel):
    path: str
    name: str
    matches: List[SearchMatch]


def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file by examining its content using byte translation"""
    try:
        # Read first 1024 bytes and check if all bytes are text characters
        with open(file_path, "rb") as f:
            sample = f.read(1024)

        # Define valid text characters (control chars + printable ASCII)
        textchars = bytearray(
            {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F}
        )

        # If translate removes all bytes, it's pure text; if bytes remain, it's binary
        return not bool(sample.translate(None, textchars))

    except (IOError, OSError):
        return False


def search_file_content(file_path: str, query: str) -> List[SearchMatch]:
    """Search for query in file content and return matches with line numbers"""
    matches = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                if query.lower() in line.lower():
                    # Get context around the match (up to 100 chars)
                    start = max(0, line.find(query) - 50)
                    end = min(len(line), line.find(query) + len(query) + 50)
                    context = line[start:end].strip()
                    if start > 0:
                        context = "..." + context
                    if end < len(line):
                        context = context + "..."

                    matches.append(SearchMatch(line=line_num, text=context))
    except (IOError, OSError) as e:
        print(f"Error reading file {file_path}: {e}")

    return matches


def search_directory(folder_path: str, query: str) -> List[SearchResult]:
    """Recursively search directory for files containing the query"""
    results = []

    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                # Skip certain directories and files
                if any(
                    skip in file_path
                    for skip in [
                        "node_modules",
                        ".git",
                        "__pycache__",
                        ".next",
                        "dist",
                        "build",
                    ]
                ):
                    continue

                if is_text_file(file_path):
                    matches = search_file_content(file_path, query)
                    if matches:
                        # Get relative path from folder_path
                        rel_path = os.path.relpath(file_path, folder_path)
                        results.append(
                            SearchResult(path=rel_path, name=file, matches=matches)
                        )
    except (OSError, IOError) as e:
        print(f"Error searching directory {folder_path}: {e}")

    return results


@router.post("/content", response_model=List[SearchResult])
async def search_content(request: SearchRequest):
    """Search for content within files in the workspace"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail="Workspace folder not found")

    try:
        results = search_directory(request.folder_path, request.query.strip())
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
