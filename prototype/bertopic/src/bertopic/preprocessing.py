#!/usr/bin/env python3
"""
Document Preprocessing Pipeline for BERTopic
Class-based preprocessing with folder-level operations and incremental tracking
"""

import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import mimetypes
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    import langdetect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("langdetect not available, language filtering disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sklearn not available, deduplication disabled")

# Configure package logger
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for document preprocessing"""

    # Text cleaning options
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    normalize_unicode: bool = True
    lowercase: bool = True

    # Quality filtering
    min_words_per_doc: int = 10
    max_words_per_doc: int = 1000
    max_word_length: int = 50
    max_number_ratio: float = 0.5  # Max ratio of numeric tokens

    # Language filtering
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    language_detection: bool = True

    # Deduplication
    remove_duplicates: bool = True
    duplicate_threshold: float = 0.95
    max_documents: Optional[int] = None

    # Performance
    chunk_size: int = 1000
    parallel_processing: bool = False


class TextCleaner:
    """Handles basic text cleaning operations"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def clean_text(self, text: str) -> str:
        """Apply all configured text cleaning operations"""
        if not text:
            return ""

        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        if self.config.remove_emails:
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text
            )

        # Remove HTML tags
        if self.config.remove_html:
            text = re.sub(r"<[^>]+>", "", text)

        # Normalize unicode
        if self.config.normalize_unicode:
            text = text.replace("\u00a0", " ")  # Non-breaking spaces
            text = re.sub(r"[\u2000-\u206f]", " ", text)  # Remove unicode separators

        # Convert to lowercase
        if self.config.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text


class QualityFilter:
    """Filters documents based on quality criteria"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def is_quality_document(self, text: str) -> Tuple[bool, str]:
        """
        Check if document meets quality criteria
        Returns: (is_quality, reason)
        """
        words = text.split()

        # Check minimum word count
        if len(words) < self.config.min_words_per_doc:
            return False, f"too_short_{len(words)}_words"

        # Check maximum word count
        if len(words) > self.config.max_words_per_doc:
            return False, f"too_long_{len(words)}_words"

        # Check for excessively long words (likely gibberish)
        if any(len(word) > self.config.max_word_length for word in words):
            return False, "long_words_found"

        # Check numeric token ratio
        if self.config.max_number_ratio < 1.0:
            numeric_ratio = len([w for w in words if w.isdigit()]) / len(words)
            if numeric_ratio > self.config.max_number_ratio:
                return False, f"too_many_numbers_{numeric_ratio:.2f}"

        return True, "quality"

    def filter_documents(
        self, documents: List[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter list of documents and return statistics"""
        filtered_docs = []
        filter_stats = {"total": len(documents), "kept": 0, "filtered": {}}

        for doc in documents:
            is_quality, reason = self.is_quality_document(doc)
            if is_quality:
                filtered_docs.append(doc)
                filter_stats["kept"] += 1
            else:
                filter_stats["filtered"][reason] = (
                    filter_stats["filtered"].get(reason, 0) + 1
                )

        return filtered_docs, filter_stats


class LanguageFilter:
    """Handles language detection and filtering"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.lang_detector_available = LANGDETECT_AVAILABLE

    def detect_language(self, text: str) -> str:
        """Detect the language of a text document"""
        if not self.lang_detector_available:
            return "unknown"

        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def is_allowed_language(self, text: str) -> Tuple[bool, str]:
        """Check if document is in an allowed language"""
        if not self.config.language_detection:
            return True, "detection_disabled"

        if not self.config.allowed_languages:
            return True, "no_language_filter"

        detected_lang = self.detect_language(text)
        is_allowed = detected_lang in self.config.allowed_languages

        return is_allowed, detected_lang

    def filter_by_language(
        self, documents: List[str]
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter documents by language and return statistics"""
        filtered_docs = []
        lang_stats = {}

        for doc in documents:
            is_allowed, detected_lang = self.is_allowed_language(doc)
            if is_allowed:
                filtered_docs.append(doc)
            lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1

        return filtered_docs, lang_stats


class DuplicateDetector:
    """Detects and removes near-duplicate documents"""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.sklearn_available = SKLEARN_AVAILABLE

    def find_duplicates(self, documents: List[str]) -> List[bool]:
        """
        Find near-duplicate documents
        Returns boolean mask: True = keep, False = duplicate
        """
        if not self.config.remove_duplicates or not self.sklearn_available:
            return [True] * len(documents)

        if len(documents) <= 1:
            return [True] * len(documents)

        try:
            # Create TF-IDF vectors for similarity comparison
            max_features = min(1000, len(documents) // 2)
            vectorizer = TfidfVectorizer(
                max_features=max_features, stop_words="english", ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Keep track of documents to keep
            keep_mask = [True] * len(documents)

            # For each document, check if it's too similar to any previous kept document
            for i in range(len(documents)):
                for j in range(i):
                    if (
                        keep_mask[j]
                        and similarity_matrix[i, j] > self.config.duplicate_threshold
                    ):
                        keep_mask[i] = (
                            False  # This is a duplicate of an earlier document
                        )
                        break

            return keep_mask

        except Exception as e:
            print(f"Duplicate detection failed: {e}")
            return [True] * len(documents)

    def remove_duplicates(self, documents: List[str]) -> Tuple[List[str], int]:
        """Remove duplicates and return count of removed documents"""
        if len(documents) <= 1:
            return documents, 0

        keep_mask = self.find_duplicates(documents)
        filtered_docs = [doc for doc, keep in zip(documents, keep_mask) if keep]
        duplicates_removed = len(documents) - len(filtered_docs)

        return filtered_docs, duplicates_removed


class DocumentPreprocessor:
    """
    Main document preprocessing pipeline
    Combines all preprocessing steps with proper error handling
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.language_filter = LanguageFilter(self.config)
        self.duplicate_detector = DuplicateDetector(self.config)

    def preprocess_single_document(self, text: str) -> Optional[str]:
        """Preprocess a single document through the full pipeline"""
        try:
            # Step 1: Basic text cleaning
            cleaned_text = self.cleaner.clean_text(text)

            # Step 2: Quality filtering
            is_quality, _ = self.quality_filter.is_quality_document(cleaned_text)
            if not is_quality:
                return None

            # Step 3: Language filtering
            is_allowed, _ = self.language_filter.is_allowed_language(cleaned_text)
            if not is_allowed:
                return None

            return cleaned_text

        except Exception as e:
            print(f"Error preprocessing document: {e}")
            return None

    def preprocess_documents(
        self, documents: List[str], verbose: bool = True
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Preprocess multiple documents through the full pipeline
        Returns: (processed_docs, statistics_dict)
        """
        if verbose:
            print(f"🔄 Starting preprocessing of {len(documents)} documents")

        stats = {"input_count": len(documents), "stages": {}}

        current_docs = documents

        # Step 1: Basic text cleaning
        if verbose:
            print("🧹 Cleaning text (URLs, emails, HTML, whitespace)...")
        cleaned_docs = [self.cleaner.clean_text(doc) for doc in current_docs]
        stats["stages"]["cleaning"] = {"output_count": len(cleaned_docs)}

        # Step 2: Quality filtering
        if verbose:
            print("🔍 Quality filtering (length, numeric ratio, etc.)...")
        current_docs, quality_stats = self.quality_filter.filter_documents(cleaned_docs)
        stats["stages"]["quality_filter"] = quality_stats

        # Step 3: Language filtering
        if verbose:
            print("🌍 Language filtering...")
        current_docs, lang_stats = self.language_filter.filter_by_language(current_docs)
        stats["stages"]["language_filter"] = lang_stats

        # Step 4: Deduplication
        if verbose:
            print("🔄 Removing duplicates...")
        current_docs, duplicates_removed = self.duplicate_detector.remove_duplicates(
            current_docs
        )
        stats["stages"]["deduplication"] = {"duplicates_removed": duplicates_removed}

        # Final statistics
        stats["final_count"] = len(current_docs)
        stats["total_removed"] = stats["input_count"] - stats["final_count"]

        if verbose:
            print(f"✅ Preprocessing complete: {stats['final_count']} documents kept")

        return current_docs, stats


class FolderPreprocessor:
    """
    Handles folder-level preprocessing operations with incremental tracking
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.preprocessor = DocumentPreprocessor(self.config)

    def _is_text_file(self, file_path: str) -> bool:
        """Check if file should be processed for text"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        binary_extensions = {
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            ".dat",
            ".db",
            ".sqlite",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".ico",
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".mp4",
            ".avi",
            ".mkv",
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
        }

        if ext in binary_extensions:
            return False

        # Additional check: try to read first 1024 bytes
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:  # Binary file indicator
                    return False
        except:
            return False

        return True

    def _analyze_files_by_extension(self, folder_path: str) -> Dict[str, Any]:
        """
        Analyze all files in folder and group by extension
        Returns: Dictionary with extension analysis statistics
        """
        extension_stats = {
            "total_files": 0,
            "extensions": {},
            "files_by_extension": {},
            "text_files_count": 0,
            "binary_files_count": 0,
        }

        # Collect all files recursively
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {".git"}]

            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        extension_stats["total_files"] = len(all_files)

        # Group files by extension
        for file_path in all_files:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower() if ext else "no_extension"

            # Count by extension
            extension_stats["extensions"][ext] = (
                extension_stats["extensions"].get(ext, 0) + 1
            )

            # Track individual files
            if ext not in extension_stats["files_by_extension"]:
                extension_stats["files_by_extension"][ext] = []
            extension_stats["files_by_extension"][ext].append(
                os.path.basename(file_path)
            )

            # Count text vs binary files
            if self._is_text_file(file_path):
                extension_stats["text_files_count"] += 1
            else:
                extension_stats["binary_files_count"] += 1

        return extension_stats

    def _load_file_documents(self, file_path: str) -> List[str]:
        """Load documents from a single file"""
        extension = os.path.splitext(file_path)[1].lower()

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        documents = []

        if extension in [".txt", ".md"]:
            # Split on double newlines (paragraphs) first, then single newlines
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                documents = paragraphs
            else:
                # Fallback to line-based splitting
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                documents = lines if len(lines) > 1 else [content.strip()]

        elif extension == ".json":
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    documents = [
                        str(item) if not isinstance(item, str) else item
                        for item in data
                    ]
                else:
                    documents = [str(data)]
            except:
                documents = [content]

        elif extension == ".csv":
            # Simple CSV parsing - assume first column or look for text columns
            lines = content.split("\n")
            if len(lines) > 1:
                # Try to parse as simple CSV
                try:
                    import csv
                    from io import StringIO

                    reader = csv.DictReader(StringIO(content))
                    for row in reader:
                        # Use first non-empty text field
                        text_content = None
                        for value in row.values():
                            if isinstance(value, str) and value.strip():
                                text_content = value
                                break
                        if text_content:
                            documents.append(text_content)
                except:
                    # Fallback: treat each line as a document
                    documents = [line.strip() for line in lines if line.strip()]

        else:
            # Default: treat as text file
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            documents = paragraphs if len(paragraphs) > 1 else [content.strip()]

        return documents

    def preprocess_folder(
        self, input_folder: str, output_folder: str, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess all documents in an input folder and save to output folder
        Returns statistics about the preprocessing operations
        """
        os.makedirs(output_folder, exist_ok=True)

        all_input_docs = []
        file_info = []

        # Analyze files by extension first
        print(f"📊 Analyzing files by extension in: {input_folder}")
        extension_analysis = self._analyze_files_by_extension(input_folder)

        print(f"📈 Found {extension_analysis['total_files']} total files")
        print(
            f"📁 File extensions: {list(extension_analysis['extensions'].keys())}"
        )
        print(f"📄 Text files: {extension_analysis['text_files_count']}")
        print(f"📦 Binary files: {extension_analysis['binary_files_count']}")

        # Show top extensions
        sorted_extensions = sorted(
            extension_analysis["extensions"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        print("📊 Top file extensions:")
        for ext, count in sorted_extensions[:10]:  # Show top 10
            print(f"   {ext}: {count} files")

        # Collect all documents from input folder
        if verbose:
            print(f"📂 Scanning input folder: {input_folder}")

        for root, dirs, files in os.walk(input_folder):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {".git"}]

            for file in files:
                file_path = os.path.join(root, file)

                if not self._is_text_file(file_path):
                    continue

                try:
                    file_docs = self._load_file_documents(file_path)
                    if file_docs:
                        all_input_docs.extend(file_docs)

                        # Track file information
                        rel_path = os.path.relpath(file_path, input_folder)
                        file_info.append(
                            {
                                "input_path": file_path,
                                "output_path": os.path.join(output_folder, rel_path),
                                "document_count": len(file_docs),
                                "file_size": os.path.getsize(file_path),
                            }
                        )

                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
                    continue

        if not all_input_docs:
            return {"error": "No documents found in input folder"}

        if verbose:
            print(
                f"📝 Found {len(all_input_docs)} raw documents from {len(file_info)} files"
            )

        # Run preprocessing pipeline
        processed_docs, preprocess_stats = self.preprocessor.preprocess_documents(
            all_input_docs, verbose
        )

        # Save processed documents
        os.makedirs(output_folder, exist_ok=True)

        # For now, save all processed documents as a single JSON file
        # In a production system, you might want to maintain folder structure
        output_file = os.path.join(output_folder, "processed_documents.json")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "documents": processed_docs,
                        "metadata": {
                            "input_folder": input_folder,
                            "output_folder": output_folder,
                            "processed_at": datetime.now().isoformat(),
                            "config": {
                                "remove_urls": self.config.remove_urls,
                                "remove_emails": self.config.remove_emails,
                                "min_words_per_doc": self.config.min_words_per_doc,
                                "max_words_per_doc": self.config.max_words_per_doc,
                                "allowed_languages": self.config.allowed_languages,
                                "duplicate_threshold": self.config.duplicate_threshold,
                            },
                        },
                        "statistics": preprocess_stats,
                        "file_info": file_info,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            if verbose:
                print(
                    f"💾 Saved {len(processed_docs)} processed documents to: {output_file}"
                )

            return {
                "success": True,
                "output_file": output_file,
                "files_processed": len(file_info),
                "extension_analysis": extension_analysis,
                **preprocess_stats,
            }

        except Exception as e:
            error_msg = f"Failed to save processed documents: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg, "success": False}


class PreprocessingPipeline:
    """
    High-level pipeline that combines preprocessing with BERTopic processing
    """

    def __init__(self, preprocessing_config: Optional[PreprocessingConfig] = None):
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.preprocessor = FolderPreprocessor(self.preprocessing_config)

    def run_complete_pipeline(
        self,
        input_folder: str,
        output_folder: str,
        bertopic_processor,
        **bertopic_kwargs,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: preprocessing → BERTopic processing
        bertopic_processor should be a callable that takes documents and kwargs
        """
        from datetime import datetime

        results = {
            "pipeline_started": datetime.now().isoformat(),
            "phases": {},
            "final_results": None,
        }

        try:
            # Phase 1: Preprocessing
            print("🔄 Phase 1: Document Preprocessing")
            preprocess_stats = self.preprocessor.preprocess_folder(
                input_folder, output_folder, verbose=True
            )
            results["phases"]["preprocessing"] = preprocess_stats

            if "error" in preprocess_stats:
                results["error"] = preprocess_stats["error"]
                return results

            # Load processed documents for BERTopic
            output_file = preprocess_stats["output_file"]
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_docs = data["documents"]

            # Phase 2: BERTopic Processing
            print("🤖 Phase 2: BERTopic Processing")
            print(f"📄 Processing {len(processed_docs)} preprocessed documents")

            bertopic_results = bertopic_processor(processed_docs, **bertopic_kwargs)
            results["phases"]["bertopic"] = {
                "documents_processed": len(processed_docs),
                "bertopic_results": bertopic_results,
            }
            results["final_results"] = bertopic_results

            results["pipeline_completed"] = datetime.now().isoformat()

        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Pipeline failed: {e}")

        return results


def create_default_config() -> PreprocessingConfig:
    """Create a sensible default preprocessing configuration"""
    return PreprocessingConfig(
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        lowercase=True,
        min_words_per_doc=10,
        max_words_per_doc=2000,
        allowed_languages=["en"],
        remove_duplicates=True,
        duplicate_threshold=0.95,
    )


# CLI interface for standalone preprocessing
def main():
    """Command line interface for document preprocessing"""
    import argparse

    parser = argparse.ArgumentParser(description="Document Preprocessing Pipeline")
    parser.add_argument(
        "--input-folder", required=True, help="Input folder with raw documents"
    )
    parser.add_argument(
        "--output-folder", required=True, help="Output folder for processed documents"
    )
    parser.add_argument(
        "--config-file", help="JSON config file for preprocessing settings"
    )
    parser.add_argument(
        "--min-words", type=int, default=10, help="Minimum words per document"
    )
    parser.add_argument(
        "--max-words", type=int, default=2000, help="Maximum words per document"
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        default=True,
        help="Remove similar documents",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Load configuration
    config = create_default_config()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
            # Update config attributes
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Override with command line args
    config.min_words_per_doc = args.min_words
    config.max_words_per_doc = args.max_words
    config.remove_duplicates = args.remove_duplicates

    # Run preprocessing
    processor = FolderPreprocessor(config)
    results = processor.preprocess_folder(
        args.input_folder, args.output_folder, verbose=not args.quiet
    )

    print(
        f"✅ Preprocessing complete. Results saved to {results.get('output_file', args.output_folder)}"
    )


if __name__ == "__main__":
    main()
