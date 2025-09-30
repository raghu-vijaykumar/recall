import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from .config import PreprocessingConfig
from .preprocessor import DocumentPreprocessor
from .loader import DocumentLoader


class DocumentPreprocessingService:
    """
    Generic service for document preprocessing before topic extraction.

    Integrates DocumentLoader for file/folder processing and PreprocessingPipeline
    for text cleaning, filtering, and quality improvement. Follows OOP design.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing service with configuration.

        Args:
            config: Preprocessing configuration, uses defaults if None
        """
        self.config = config or self._create_default_config()
        self.document_loader = DocumentLoader()
        self.preprocessor = DocumentPreprocessor(self.config)

        logging.info("DocumentPreprocessingService initialized with config")
        logging.info(
            f"Config: min_words_per_doc={self.config.min_words_per_doc}, "
            f"max_words_per_doc={self.config.max_words_per_doc}, "
            f"remove_duplicates={self.config.remove_duplicates}"
        )

    def _create_default_config(self) -> PreprocessingConfig:
        """Create sensible default preprocessing configuration."""
        return PreprocessingConfig(
            # Text cleaning - moderate settings
            remove_urls=True,
            remove_emails=True,
            remove_html=True,
            normalize_unicode=True,
            lowercase=True,
            # Quality filtering - balanced settings
            min_words_per_doc=10,  # At least 10 words
            max_words_per_doc=50000,  # Max 50000 words to prevent memory issues
            max_word_length=50,
            max_number_ratio=0.5,
            # Language filtering
            allowed_languages=["en"],
            language_detection=True,
            # Deduplication
            remove_duplicates=True,
            duplicate_threshold=0.95,
            # Performance
            chunk_size=1000,
            parallel_processing=False,
        )

    async def preprocess_workspace_files(
        self,
        workspace_path: str,
        max_files: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Preprocess all text files in a workspace directory.

        Uses DocumentLoader to load files from workspace, then applies
        PreprocessingPipeline for cleaning and filtering.

        Args:
            workspace_path: Path to the workspace directory
            max_files: Maximum number of files to process (None for all)

        Returns:
            Tuple of (processed_documents, file_metadata_list)
        """
        start_time = datetime.utcnow()
        logging.info(f"Starting document preprocessing for workspace: {workspace_path}")

        try:
            # Step 1: Load documents from workspace using DocumentLoader
            logging.info("Loading documents from workspace folder...")
            raw_documents, file_metadata = self.document_loader.load_from_folder(
                workspace_path, max_files=max_files
            )

            if not raw_documents:
                logging.warning("No documents found in workspace")
                return [], []

            logging.info(
                f"Loaded {len(raw_documents)} raw documents from {len(file_metadata)} files"
            )

            # Step 2: Apply preprocessing pipeline
            logging.info("Applying preprocessing pipeline...")
            processed_documents, preprocessing_stats = (
                self.preprocessor.preprocess_documents(raw_documents, verbose=True)
            )

            # Step 3: Create final processed documents with metadata
            processed_docs_with_metadata = self._create_processed_docs_with_metadata(
                processed_documents, file_metadata, preprocessing_stats
            )

            duration = (datetime.utcnow() - start_time).total_seconds()

            logging.info(
                f"Document preprocessing completed - "
                f"Input: {preprocessing_stats['input_count']} documents, "
                f"Output: {preprocessing_stats['final_count']} documents in {duration:.2f}s"
            )

            return processed_documents, processed_docs_with_metadata

        except Exception as e:
            logging.error(f"Error in document preprocessing: {e}")
            raise

    def _create_processed_docs_with_metadata(
        self,
        processed_documents: List[str],
        original_file_metadata: List[Dict[str, Any]],
        preprocessing_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Create processed documents with associated metadata for topic extraction.

        Args:
            processed_documents: Preprocessed document texts
            original_file_metadata: Original file metadata from DocumentLoader
            preprocessing_stats: Statistics from preprocessing pipeline

        Returns:
            List of document dictionaries with metadata
        """
        processed_docs_metadata = []

        # For simplicity, since preprocessing might filter/change document count,
        # we'll create metadata based on available info
        for i, doc_text in enumerate(processed_documents):
            doc_metadata = {
                "id": f"processed_doc_{i}",
                "content": doc_text,
                "text": doc_text,  # Alias for compatibility
                "word_count": len(doc_text.split()),
                "preprocessing_stats": preprocessing_stats,
                "processed_at": datetime.utcnow().isoformat(),
            }

            # If we have original file metadata, try to associate
            if i < len(original_file_metadata):
                original_meta = original_file_metadata[i]
                doc_metadata.update(
                    {
                        "file_path": original_meta.get("file_path", ""),
                        "file_name": original_meta.get("file_name", ""),
                        "file_size_bytes": original_meta.get("file_size_bytes", 0),
                        "file_size_mb": original_meta.get("file_size_mb", 0.0),
                    }
                )

            processed_docs_metadata.append(doc_metadata)

        return processed_docs_metadata

    def preprocess_single_file(
        self, file_path: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Preprocess a single file.

        Args:
            file_path: Path to the file to preprocess

        Returns:
            Tuple of (processed_documents, metadata_list)
        """
        logging.info(f"Preprocessing single file: {file_path}")

        try:
            # Load document using DocumentLoader
            raw_docs = self.document_loader.load_from_file(file_path)

            if not raw_docs:
                logging.warning(f"No content extracted from {file_path}")
                return [], []

            # Apply preprocessing
            processed_docs, stats = self.preprocessor.preprocess_documents(
                raw_docs, verbose=False
            )

            # Create metadata
            metadata = [
                {
                    "id": f"file_{Path(file_path).name}_{i}",
                    "content": doc,
                    "text": doc,
                    "word_count": len(doc.split()),
                    "file_path": file_path,
                    "file_name": Path(file_path).name,
                    "preprocessing_stats": stats,
                    "processed_at": datetime.utcnow().isoformat(),
                }
                for i, doc in enumerate(processed_docs)
            ]

            logging.info(f"Processed {len(processed_docs)} documents from {file_path}")

            return processed_docs, metadata

        except Exception as e:
            logging.error(f"Error preprocessing file {file_path}: {e}")
            return [], []

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get current preprocessing configuration and statistics.

        Returns:
            Dictionary with preprocessing configuration and current state
        """
        return {
            "config": {
                "min_words_per_doc": self.config.min_words_per_doc,
                "max_words_per_doc": self.config.max_words_per_doc,
                "remove_duplicates": self.config.remove_duplicates,
                "allowed_languages": self.config.allowed_languages,
                "remove_urls": self.config.remove_urls,
                "remove_emails": self.config.remove_emails,
            },
            "service_status": "active",
        }


# Factory function for easy service creation
def create_document_preprocessing_service(
    config: Optional[PreprocessingConfig] = None,
) -> DocumentPreprocessingService:
    """
    Factory function to create DocumentPreprocessingService with default config.

    Args:
        config: Optional preprocessing configuration

    Returns:
        Initialized DocumentPreprocessingService
    """
    return DocumentPreprocessingService(config=config)
