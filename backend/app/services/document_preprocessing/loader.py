import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


class DocumentLoader:
    """
    Handles loading documents from various file formats and folders.

    Provides unified interface for loading text documents from files and directories,
    with support for multiple formats and intelligent filtering.
    """

    @staticmethod
    def load_from_file(file_path: str) -> List[str]:
        """
        Load documents from various file formats.

        Args:
            file_path: Path to the file containing documents

        Returns:
            List of document strings

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".txt":
                return DocumentLoader._load_txt_file(file_path)
            elif file_extension == ".csv":
                return DocumentLoader._load_csv_file(file_path)
            elif file_extension == ".json":
                return DocumentLoader._load_json_file(file_path)
            else:
                # Default to text file parsing
                return DocumentLoader._load_txt_file(file_path)

        except Exception as e:
            logging.error(f"Error loading file {file_path}: {e}")
            raise

    @staticmethod
    def load_from_folder(
        folder_path: str, max_files: int = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load documents from all text files in a folder and its subfolders.

        Args:
            folder_path: Path to the folder containing documents
            max_files: Maximum number of files to process (None for all)

        Returns:
            Tuple of (documents_list, files_info_list)
            files_info_list contains metadata about each processed file

        Raises:
            FileNotFoundError: If folder does not exist
            ValueError: If path is not a directory
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        documents = []
        files_info = []
        processed_count = 0

        logging.info(f"Scanning folder: {folder_path}")

        # Walk through all files in the folder recursively
        for root, dirs, files in os.walk(folder_path):
            # Skip common directories that shouldn't be processed
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in {
                    "node_modules",
                    "__pycache__",
                    ".git",
                    "dist",
                    "build",
                    "target",
                }
            ]

            for file in files:
                file_path = os.path.join(root, file)

                # Skip hidden files
                if file.startswith("."):
                    continue

                # Check if it's a text file
                if not DocumentLoader._is_text_file(file_path):
                    continue

                # Get file size
                try:
                    file_size = os.path.getsize(file_path)
                    file_size_mb = file_size / (1024 * 1024)

                    # Skip very large files (>50MB) to avoid memory issues
                    if file_size_mb > 50:
                        logging.warning(
                            f"Skipping large file ({file_size_mb:.1f}MB): {file_path}"
                        )
                        continue

                    # Load document from file
                    try:
                        file_documents = DocumentLoader.load_from_file(file_path)

                        # Add metadata for each document
                        for doc in file_documents:
                            if doc.strip():  # Only add non-empty documents
                                documents.append(doc.strip())
                                files_info.append(
                                    {
                                        "file_path": file_path,
                                        "file_name": file,
                                        "file_size_bytes": file_size,
                                        "file_size_mb": round(file_size_mb, 2),
                                        "relative_path": os.path.relpath(
                                            file_path, folder_path
                                        ),
                                        "document_length": len(doc),
                                        "document_word_count": len(doc.split()),
                                    }
                                )

                        processed_count += 1

                        # Check if we've reached the max files limit
                        if max_files and processed_count >= max_files:
                            logging.info(f"Reached maximum file limit: {max_files}")
                            break

                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")

                except Exception as e:
                    logging.error(f"Error checking file {file_path}: {e}")

            # Break if we've reached the max files limit
            if max_files and processed_count >= max_files:
                break

        # Log summary
        if files_info:
            # Show largest files
            largest_files = sorted(
                files_info, key=lambda x: x["file_size_mb"], reverse=True
            )[:5]
            logging.info("Largest files processed:")
            for file_info in largest_files:
                logging.info(
                    f"  • {file_info['file_name']} ({file_info['file_size_mb']:.1f}MB)"
                )

        logging.info(
            f"Folder processing complete: {len(documents)} documents from {processed_count} files "
            f"({sum(f['file_size_mb'] for f in files_info):.1f}MB total)"
        )

        return documents, files_info

    @staticmethod
    def _load_txt_file(file_path: str) -> List[str]:
        """Load documents from a text file"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try different splitting strategies
        documents = []

        # First try splitting by double newlines (paragraphs)
        paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
        if len(paragraphs) > 1:
            documents = paragraphs
        else:
            # Try splitting by single newlines (lines)
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if len(lines) > 1:
                documents = lines
            else:
                # Treat entire file as one document
                documents = [content.strip()]

        return documents

    @staticmethod
    def _load_csv_file(file_path: str) -> List[str]:
        """Load documents from a CSV file"""
        try:
            import pandas as pd

            df = pd.read_csv(file_path)

            # Look for common text columns
            text_columns = [
                "text",
                "content",
                "description",
                "body",
                "article",
                "document",
            ]

            documents = []
            for col in text_columns:
                if col in df.columns:
                    documents = df[col].dropna().astype(str).tolist()
                    if documents:
                        break

            # If no standard column found, use the first text-like column
            if not documents:
                for col in df.columns:
                    if df[col].dtype == "object":  # Text column
                        documents = df[col].dropna().astype(str).tolist()
                        if documents:
                            break

            if not documents:
                raise ValueError("No text columns found in CSV file")

            return documents

        except ImportError:
            raise ImportError(
                "pandas is required for CSV file loading. Install with: pip install pandas"
            )

    @staticmethod
    def _load_json_file(file_path: str) -> List[str]:
        """Load documents from a JSON file"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []

        # Handle different JSON structures
        if isinstance(data, list):
            # List of documents
            for item in data:
                if isinstance(item, str):
                    documents.append(item)
                elif isinstance(item, dict):
                    # Look for text fields in dictionaries
                    text_fields = [
                        "text",
                        "content",
                        "description",
                        "body",
                        "article",
                        "document",
                    ]
                    for field in text_fields:
                        if field in item and isinstance(item[field], str):
                            documents.append(item[field])
                            break
                    else:
                        # If no standard field, use the whole dict as string
                        documents.append(str(item))
        elif isinstance(data, dict):
            # Single document or dict with documents
            if "documents" in data and isinstance(data["documents"], list):
                documents = data["documents"]
            elif any(key in data for key in ["text", "content", "description", "body"]):
                # Single document in dict
                text_fields = [
                    "text",
                    "content",
                    "description",
                    "body",
                    "article",
                    "document",
                ]
                for field in text_fields:
                    if field in data and isinstance(data[field], str):
                        documents = [data[field]]
                        break
                else:
                    documents = [str(data)]
            else:
                documents = [str(data)]

        if not documents:
            raise ValueError("No documents found in JSON file")

        return documents

    @staticmethod
    def _is_text_file(file_path: str) -> bool:
        """
        Determine if a file is a text file that should be processed for topic modeling.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file should be processed, False if it's binary or should be excluded
        """
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Define binary file extensions to exclude
        binary_extensions = {
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".ico",
            ".svg",
            ".webp",
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
            ".wma",
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".zip",
            ".rar",
            ".7z",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".pyc",
            ".pyo",
            ".class",
            ".jar",
            ".war",
            ".ear",
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            ".eot",
        }

        # Exclude binary extensions
        if ext in binary_extensions:
            return False

        # Check MIME type
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                # Exclude binary MIME types
                if mime_type.startswith(
                    ("image/", "audio/", "video/", "application/")
                ) and not mime_type.startswith(
                    (
                        "text/",
                        "application/json",
                        "application/xml",
                        "application/javascript",
                    )
                ):
                    return False
        except:
            pass

        # Check if file can be decoded as text
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                # Check for null bytes (indicates binary file)
                if b"\x00" in chunk:
                    return False
        except:
            return False

        return True
