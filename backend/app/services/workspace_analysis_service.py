"""
Optimized Workspace Analysis Service for extracting concepts and relationships from user files
Supports batched processing, embeddings, and incremental updates for large workspaces
"""

import os
import uuid
import asyncio
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..models import ConceptCreate, RelationshipCreate, ConceptFileCreate
from . import KnowledgeGraphService
from .embedding_service import EmbeddingService

# NLP imports
try:
    import spacy
    from keybert import KeyBERT
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("spaCy, NLTK, or KeyBERT not available. Using fallback methods.")

# Import concept extraction classes
from .concept_extraction.extractors import HeuristicExtractor


class WorkspaceAnalysisService:
    """
    Optimized service for analyzing workspace files to extract concepts and build knowledge graphs.
    Supports batched processing, embeddings, and incremental updates for large workspaces.
    """

    def __init__(
        self,
        db: AsyncSession,
        kg_service: Optional[KnowledgeGraphService] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.db = db
        self.kg_service = kg_service or KnowledgeGraphService(db)
        self.embedding_service = embedding_service

        # Processing configuration
        self.batch_size = 20  # Files per batch
        self.max_concepts_per_file = 50
        self.similarity_threshold = 0.8  # Minimum strength for relationships

        # File types to analyze
        self.supported_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
        }

        # Load non-concept words from external file (stop words handled by NLP libraries)
        self.non_concept_words = self._load_non_concept_words()

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not text:
            return ""

        # Remove markdown formatting
        text = re.sub(r"#+\s*", "", text)  # Headers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # Links

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove extra whitespace
        text = text.strip()

        return text

    def _load_non_concept_words(self) -> Set[str]:
        """Load non-concept words from external file"""
        non_concept_words = set()

        # Load from file
        non_concept_file = (
            Path(__file__).parent.parent / "resources" / "non_concept_words.txt"
        )
        if non_concept_file.exists():
            try:
                with open(non_concept_file, "r", encoding="utf-8") as f:
                    non_concept_words = {
                        line.strip().lower() for line in f if line.strip()
                    }
            except Exception as e:
                logging.warning(f"Failed to load non-concept words from file: {e}")
                # Fallback to empty set if file loading fails
                non_concept_words = set()

        return non_concept_words

    async def analyze_workspace(
        self,
        workspace_id: int,
        workspace_path: str,
        force_reanalysis: bool = False,
        file_paths: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
        use_flattened_file: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze a workspace using optimized batched processing with embeddings.

        Args:
            workspace_id: ID of the workspace to analyze
            workspace_path: Path to the workspace directory
            force_reanalysis: If True, re-analyze all files
            file_paths: Specific files to analyze (for incremental updates)
            progress_callback: Optional callback for progress updates

        Returns:
            Analysis results with statistics
        """
        start_time = datetime.utcnow()
        logging.info(
            f"[ANALYSIS] Starting workspace analysis for workspace {workspace_id}"
        )
        logging.info(f"[ANALYSIS] Workspace path: {workspace_path}")
        logging.info(f"[ANALYSIS] Force reanalysis: {force_reanalysis}")

        # Get files to analyze
        if file_paths:
            all_files = [Path(workspace_path) / fp for fp in file_paths]
            logging.info(f"[ANALYSIS] Analyzing {len(all_files)} specific files")
        else:
            all_files = self._discover_files(workspace_path)
            logging.info(f"[ANALYSIS] Discovered {len(all_files)} files in workspace")

        # Filter out files that don't need analysis
        files_to_analyze = []
        skipped_files = 0
        for file_path in all_files:
            if not file_path.exists() or not file_path.is_file():
                logging.debug(f"[ANALYSIS] Skipping non-existent file: {file_path}")
                continue
            if not force_reanalysis and await self._file_already_analyzed(
                file_path, workspace_id
            ):
                skipped_files += 1
                continue
            files_to_analyze.append(file_path)

        logging.info(f"[ANALYSIS] Skipped {skipped_files} already analyzed files")
        logging.info(f"[ANALYSIS] Will analyze {len(files_to_analyze)} files")

        total_files = len(files_to_analyze)
        if total_files == 0:
            logging.info("[ANALYSIS] No files need analysis")
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": [],
                "duration_seconds": 0.0,
                "message": "No files need analysis",
            }

        # Process in batches
        analyzed_files = 0
        concepts_created = 0
        relationships_created = 0
        errors = []

        logging.info(f"[ANALYSIS] Processing files in batches of {self.batch_size}")
        for batch_start in range(0, total_files, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = files_to_analyze[batch_start:batch_end]

            batch_num = batch_start // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            logging.info(
                f"[ANALYSIS] Processing batch {batch_num}/{total_batches} ({batch_start}-{batch_end-1})"
            )

            try:
                # Process batch
                batch_results = await self._process_batch(batch_files, workspace_id)
                analyzed_files += batch_results["files_processed"]
                concepts_created += batch_results["concepts_created"]
                relationships_created += batch_results["relationships_created"]

                logging.info(
                    f"[ANALYSIS] Batch {batch_num} completed: {batch_results['files_processed']} files, {batch_results['concepts_created']} concepts, {batch_results['relationships_created']} relationships"
                )

                # Progress callback
                if progress_callback:
                    progress = (batch_end) / total_files
                    await progress_callback(
                        workspace_id,
                        progress,
                        f"Processed {batch_end}/{total_files} files",
                    )

            except Exception as e:
                error_msg = (
                    f"Error processing batch {batch_start}-{batch_end}: {str(e)}"
                )
                errors.append(error_msg)
                logging.error(f"[ANALYSIS] {error_msg}")

        logging.info(
            f"[ANALYSIS] File processing completed. Total: {analyzed_files} files, {concepts_created} concepts, {relationships_created} relationships"
        )

        # Build relationships across all concepts using embeddings
        if self.embedding_service and concepts_created > 0:
            logging.info("[ANALYSIS] Building embedding-based relationships")
            try:
                await self._build_embedding_relationships(workspace_id)
                logging.info("[ANALYSIS] Embedding relationships built successfully")
            except Exception as e:
                error_msg = f"Error building embedding relationships: {str(e)}"
                errors.append(error_msg)
                logging.error(f"[ANALYSIS] {error_msg}")
        else:
            logging.info(
                "[ANALYSIS] Skipping embedding relationships (no embedding service or no concepts)"
            )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logging.info(
            f"[ANALYSIS] Workspace analysis completed in {duration:.2f} seconds"
        )

        return {
            "workspace_id": workspace_id,
            "files_analyzed": analyzed_files,
            "concepts_created": concepts_created,
            "relationships_created": relationships_created,
            "errors": errors,
            "duration_seconds": duration,
        }

    def _discover_files(self, workspace_path: str) -> List[Path]:
        """Discover all analyzable files in the workspace"""
        workspace_dir = Path(workspace_path)
        if not workspace_dir.exists() or not workspace_dir.is_dir():
            return []

        files = []
        for ext in self.supported_extensions:
            files.extend(workspace_dir.rglob(f"*{ext}"))

        return files

    async def _file_already_analyzed(self, file_path: Path, workspace_id: int) -> bool:
        """Check if file has already been analyzed"""
        query = text(
            """
            SELECT COUNT(*) FROM concept_files cf
            JOIN files f ON cf.file_id = f.id
            WHERE f.path = :file_path AND cf.workspace_id = :workspace_id
            """
        )

        result = await self.db.execute(
            query, {"file_path": str(file_path), "workspace_id": workspace_id}
        )

        count = result.scalar()
        return count > 0

    async def _analyze_file(
        self, file_path: Path, workspace_id: int
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Analyze a single file to extract concepts and relationships

        Returns:
            Tuple of (concepts_data, relationships_data)
        """
        logging.info(f"[FILE_ANALYSIS] Starting analysis of file: {file_path.name}")

        # Read and preprocess content
        content = self._read_file_content(file_path)
        if not content:
            logging.info(
                f"[FILE_ANALYSIS] File {file_path.name} is empty or unreadable"
            )
            return [], []

        content_length = len(content)
        logging.info(
            f"[FILE_ANALYSIS] Read {content_length} characters from {file_path.name}"
        )

        cleaned_content = self._preprocess_text(content)
        cleaned_length = len(cleaned_content)
        logging.info(
            f"[FILE_ANALYSIS] Preprocessed content: {cleaned_length} characters after cleaning"
        )

        # Extract concepts with line tracking (use original content for line numbers)
        logging.info(f"[FILE_ANALYSIS] Extracting concepts from {file_path.name}")
        concepts = self._extract_concepts_with_lines(content)
        logging.info(
            f"[FILE_ANALYSIS] Extracted {len(concepts)} concepts from {file_path.name}"
        )

        if not concepts:
            logging.info(f"[FILE_ANALYSIS] No concepts found in {file_path.name}")
            return [], []

        # Get or create file record
        logging.info(
            f"[FILE_ANALYSIS] Creating/updating file record for {file_path.name}"
        )
        file_id = await self._get_or_create_file_record(file_path, workspace_id)

        # Store concepts and relationships
        stored_concepts = []
        stored_relationships = []

        logging.info(
            f"[FILE_ANALYSIS] Storing {len(concepts)} concepts for {file_path.name}"
        )
        for i, concept_data in enumerate(concepts):
            if (i + 1) % 10 == 0:  # Log progress every 10 concepts
                logging.info(
                    f"[FILE_ANALYSIS] Processed {i + 1}/{len(concepts)} concepts for {file_path.name}"
                )

            # Create concept
            concept = await self.kg_service.create_concept(
                ConceptCreate(
                    name=concept_data["name"],
                    description=concept_data.get("description", ""),
                )
            )
            stored_concepts.append(concept)

            # Create concept-file link with line pointers
            await self.kg_service.create_concept_file_link(
                ConceptFileCreate(
                    concept_id=concept.concept_id,
                    file_id=file_id,
                    workspace_id=workspace_id,
                    snippet=concept_data.get("snippet", ""),
                    relevance_score=concept_data.get("score", 0.5),
                    start_line=concept_data.get("start_line"),
                    end_line=concept_data.get("end_line"),
                )
            )

        logging.info(
            f"[FILE_ANALYSIS] Stored {len(stored_concepts)} concepts for {file_path.name}"
        )

        # Infer relationships between concepts
        logging.info(
            f"[FILE_ANALYSIS] Inferring relationships between {len(stored_concepts)} concepts"
        )
        relationships = self._infer_relationships(stored_concepts, cleaned_content)
        logging.info(
            f"[FILE_ANALYSIS] Found {len(relationships)} potential relationships"
        )

        for rel_data in relationships:
            try:
                relationship = await self.kg_service.create_relationship(
                    RelationshipCreate(
                        source_concept_id=rel_data["source_id"],
                        target_concept_id=rel_data["target_id"],
                        type=rel_data["type"],
                        strength=rel_data.get("strength", 0.5),
                    )
                )
                stored_relationships.append(relationship)
            except Exception:
                # Skip duplicate relationships
                pass

        logging.info(
            f"[FILE_ANALYSIS] Created {len(stored_relationships)} relationships for {file_path.name}"
        )
        logging.info(
            f"[FILE_ANALYSIS] Completed analysis of {file_path.name}: {len(stored_concepts)} concepts, {len(stored_relationships)} relationships"
        )

        return stored_concepts, stored_relationships

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding handling"""
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Try with errors='ignore'
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""
        except Exception:
            return ""

    def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from text using simple heuristics
        In production, this would use spaCy, KeyBERT, or other NLP libraries
        """
        concepts = []

        if not text:
            return concepts

        # Simple noun phrase extraction (basic implementation)
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Extract potential concepts (capitalized words, technical terms)
            words = re.findall(r"\b[A-Z][a-z]+\b|\b[a-z]{4,}\b", sentence)

        for word in words:
            word_lower = word.lower()
            if len(word) > 3:  # Let NLP libraries handle stop words
                # Find context snippet
                snippet_start = max(0, sentence.find(word) - 50)
                snippet_end = min(len(sentence), sentence.find(word) + len(word) + 50)
                snippet = sentence[snippet_start:snippet_end].strip()

                # Calculate relevance score based on word characteristics
                score = 0.6  # Base score

                # Boost score for capitalized words (likely proper nouns or important terms)
                if word[0].isupper():
                    score += 0.2

                # Boost score for longer words (likely more specific terms)
                if len(word) > 6:
                    score += 0.1

                # Boost score for words that appear multiple times in the text
                word_count = text.lower().count(word.lower())
                if word_count > 1:
                    score += min(0.2, word_count * 0.05)

                concepts.append(
                    {
                        "name": word,
                        "description": f"Concept mentioned in context: {snippet[:100]}...",
                        "snippet": snippet,
                        "score": score,
                    }
                )

        # Remove duplicates and apply quality threshold
        seen = set()
        unique_concepts = []
        for concept in concepts:
            # Only keep high-quality concepts (score > 0.8)
            if concept["score"] > 0.8 and concept["name"].lower() not in seen:
                seen.add(concept["name"].lower())
                unique_concepts.append(concept)

        return unique_concepts[:50]  # Limit concepts per file

    def _extract_concepts_with_lines(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract concepts from text while tracking line numbers for precise navigation
        Uses spaCy for entity recognition and KeyBERT for keyword extraction
        """
        concepts = []

        if not content:
            return concepts

        # Use NLP libraries if available, otherwise fallback to heuristics
        if NLP_AVAILABLE:
            try:
                concepts = self._extract_concepts_with_nlp(content)
            except Exception as e:
                logging.warning(f"NLP extraction failed, using fallback: {e}")
                concepts = self._extract_concepts_fallback(content)
        else:
            concepts = self._extract_concepts_fallback(content)

        # Remove duplicates and apply quality threshold
        seen = set()
        unique_concepts = []
        for concept in concepts:
            # Only keep high-quality concepts (score > 0.7 for NLP, 0.8 for fallback)
            threshold = 0.7 if NLP_AVAILABLE else 0.8
            if concept["score"] > threshold and concept["name"].lower() not in seen:
                seen.add(concept["name"].lower())
                unique_concepts.append(concept)

        return unique_concepts[: self.max_concepts_per_file]

    def _extract_concepts_with_nlp(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract concepts using spaCy and KeyBERT with line number tracking
        """
        concepts = []

        # Initialize NLP models (lazy loading)
        if not hasattr(self, "_nlp_model"):
            try:
                # Download required NLTK data
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)

                # Load spaCy model
                self._nlp_model = spacy.load("en_core_web_sm")

                # Initialize KeyBERT
                self._kw_model = KeyBERT()

            except Exception as e:
                logging.error(f"Failed to initialize NLP models: {e}")
                raise e

        # Split content into lines for line tracking
        lines = content.split("\n")

        # Process content with spaCy for named entities
        doc = self._nlp_model(content)

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in [
                "PERSON",
                "ORG",
                "GPE",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
            ]:
                # Find which line this entity appears on
                line_num = self._find_line_for_position(lines, ent.start_char)

                # Get context around the entity
                context_lines = self._get_context_lines(lines, line_num, context_size=2)
                context_text = "\n".join(context_lines)

                score = self._calculate_concept_score(ent.text, content, context_text)

                concepts.append(
                    {
                        "name": ent.text,
                        "description": f"Named entity ({ent.label_}) mentioned in context: {context_text[:120]}...",
                        "snippet": context_text,
                        "score": score,
                        "start_line": line_num,
                        "end_line": line_num,
                    }
                )

        # Use KeyBERT for keyword extraction
        try:
            # Extract keywords with scores
            keywords = self._kw_model.extract_keywords(
                content,
                keyphrase_ngram_range=(1, 3),  # Single words to 3-word phrases
                stop_words="english",
                top_n=50,
                use_mmr=True,  # Maximal Marginal Relevance for diversity
                diversity=0.3,
            )

            for keyword, score in keywords:
                # Skip if already extracted as named entity
                if any(c["name"].lower() == keyword.lower() for c in concepts):
                    continue

                # Find line number for this keyword
                line_num = self._find_line_for_keyword(lines, keyword)

                # Get context
                context_lines = self._get_context_lines(lines, line_num, context_size=2)
                context_text = "\n".join(context_lines)

                # Boost score for technical terms
                adjusted_score = min(
                    1.0,
                    score
                    + self._calculate_concept_score(keyword, content, context_text)
                    * 0.3,
                )

                concepts.append(
                    {
                        "name": keyword,
                        "description": f"Key term extracted from context: {context_text[:120]}...",
                        "snippet": context_text,
                        "score": adjusted_score,
                        "start_line": line_num,
                        "end_line": line_num,
                    }
                )

        except Exception as e:
            logging.warning(f"KeyBERT extraction failed: {e}")

        return concepts

    def _find_line_for_position(self, lines: List[str], char_position: int) -> int:
        """
        Find which line number contains a specific character position
        """
        current_pos = 0
        for line_num, line in enumerate(lines):
            line_length = len(line) + 1  # +1 for newline
            if current_pos <= char_position < current_pos + line_length:
                return line_num
            current_pos += line_length
        return 0

    def _find_line_for_keyword(self, lines: List[str], keyword: str) -> int:
        """
        Find the first line that contains a keyword
        """
        for line_num, line in enumerate(lines):
            if keyword.lower() in line.lower():
                return line_num
        return 0

    def _extract_concepts_fallback(self, content: str) -> List[Dict[str, Any]]:
        """
        Fallback concept extraction using HeuristicExtractor when NLP libraries are unavailable
        """
        # Use HeuristicExtractor for concept extraction
        extractor = HeuristicExtractor(stop_words=self.non_concept_words)

        # Extract concepts
        concepts = extractor.extract_concepts(content)

        # Convert to the expected format with additional metadata
        formatted_concepts = []
        lines = content.split("\n")

        for concept in concepts:
            # Find line number for this concept
            line_num = self._find_line_for_concept(concept, lines)

            # Get context around the concept (using HeuristicExtractor's method)
            context_lines = extractor._get_context_lines(
                lines, line_num, context_size=2
            )
            context_text = "\n".join(context_lines)

            formatted_concept = {
                "name": concept["name"],
                "description": f"Concept extracted from context: {context_text[:120]}...",
                "snippet": context_text,
                "score": concept.get("score", 0.8),
                "start_line": line_num,
                "end_line": line_num,
            }
            formatted_concepts.append(formatted_concept)

        return formatted_concepts

    def _extract_concepts_fallback_for_flattened(
        self, content: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from flattened files using the HeuristicExtractor.
        """
        # Use HeuristicExtractor for concept extraction
        extractor = HeuristicExtractor(stop_words=self.non_concept_words)

        # Extract concepts
        concepts = extractor.extract_concepts(content)

        # Convert to the expected format with additional metadata
        formatted_concepts = []
        lines = content.split("\n")

        for concept in concepts:
            # Find line number for this concept
            line_num = self._find_line_for_concept(concept, lines)

            # Get context around the concept (using HeuristicExtractor's method)
            context_lines = extractor._get_context_lines(
                lines, line_num, context_size=2
            )
            context_text = "\n".join(context_lines)

            formatted_concept = {
                "name": concept["name"],
                "description": f"Concept extracted from context: {context_text[:120]}...",
                "snippet": context_text,
                "score": concept.get("score", 0.8),
                "start_line": line_num,
                "end_line": line_num,
            }
            formatted_concepts.append(formatted_concept)

        # Log top 100 ranked concepts
        logging.info(
            f"[FLATTENED_ANALYSIS] Total concepts extracted with HeuristicExtractor: {len(formatted_concepts)}"
        )
        logging.info("[FLATTENED_ANALYSIS] Top 100 ranked concepts:")
        for i, concept in enumerate(formatted_concepts[:100]):
            logging.info(
                f"[FLATTENED_ANALYSIS] #{i+1:2d}: {concept['name']} (score: {concept['score']:.3f})"
            )

        # Return top 100 concepts
        return formatted_concepts[:100]

    def _find_line_for_concept(self, concept: Dict[str, Any], lines: List[str]) -> int:
        """
        Find the line number where a concept appears in the text
        """
        concept_name = concept.get("name", "")
        if not concept_name:
            return 0

        # Look for the concept name in the lines
        for line_num, line in enumerate(lines):
            if concept_name.lower() in line.lower():
                return line_num

        return 0

    def _extract_concepts_heuristic_fallback(
        self, content: str
    ) -> List[Dict[str, Any]]:
        """
        Fallback concept extraction using HeuristicExtractor
        """
        # Use HeuristicExtractor for concept extraction
        extractor = HeuristicExtractor(stop_words=self.non_concept_words)

        # Extract concepts
        concepts = extractor.extract_concepts(content)

        # Convert to the expected format with additional metadata
        formatted_concepts = []
        lines = content.split("\n")

        for concept in concepts:
            # Find line number for this concept
            line_num = self._find_line_for_concept(concept, lines)

            # Get context around the concept (using HeuristicExtractor's method)
            context_lines = extractor._get_context_lines(
                lines, line_num, context_size=2
            )
            context_text = "\n".join(context_lines)

            formatted_concept = {
                "name": concept["name"],
                "description": f"Concept extracted from context: {context_text[:120]}...",
                "snippet": context_text,
                "score": concept.get("score", 0.8),
                "start_line": line_num,
                "end_line": line_num,
            }
            formatted_concepts.append(formatted_concept)

        # Rank concepts by score and log top 100
        ranked_concepts = sorted(
            formatted_concepts, key=lambda x: x["score"], reverse=True
        )

        # Log top 100 ranked concepts
        logging.info(
            f"[FLATTENED_ANALYSIS] Total concepts extracted with HeuristicExtractor fallback: {len(ranked_concepts)}"
        )
        logging.info("[FLATTENED_ANALYSIS] Top 100 ranked concepts (fallback):")
        for i, concept in enumerate(ranked_concepts[:100]):
            logging.info(
                f"[FLATTENED_ANALYSIS] #{i+1:2d}: {concept['name']} (score: {concept['score']:.3f})"
            )

        # Return top 100 concepts
        return ranked_concepts[:100]

    def _infer_relationships(
        self, concepts: List[Any], text: str
    ) -> List[Dict[str, Any]]:
        """
        Infer relationships between concepts based on co-occurrence and context
        """
        relationships = []

        # Simple co-occurrence based relationships
        concept_names = [c.name.lower() for c in concepts]

        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue

                name1 = concept1.name.lower()
                name2 = concept2.name.lower()

                # Check if concepts appear in same sentences
                sentences = re.split(r"[.!?]+", text)
                co_occurrences = 0

                for sentence in sentences:
                    if name1 in sentence.lower() and name2 in sentence.lower():
                        co_occurrences += 1

                if co_occurrences > 0:
                    # Calculate relationship strength
                    strength = min(0.9, co_occurrences * 0.3)

                    # Only create relationships with sufficient strength
                    if strength >= 0.8:
                        # Determine relationship type based on context
                        rel_type = "relates_to"

                        # Check for hierarchical indicators
                        if any(
                            word in text.lower()
                            for word in ["type of", "kind of", "subtype", "extends"]
                        ):
                            rel_type = "dives_deep_to"

                        relationships.append(
                            {
                                "source_id": concept1.concept_id,
                                "target_id": concept2.concept_id,
                                "type": rel_type,
                                "strength": strength,
                            }
                        )

        return relationships

    async def _get_or_create_file_record(
        self, file_path: Path, workspace_id: int
    ) -> int:
        """Get or create file record in database"""
        # Check if file exists
        query = text(
            "SELECT id FROM files WHERE path = :path AND workspace_id = :workspace_id"
        )

        result = await self.db.execute(
            query, {"path": str(file_path), "workspace_id": workspace_id}
        )

        file_record = result.fetchone()

        if file_record:
            return file_record.id

        # Create new file record
        insert_query = text(
            """
            INSERT INTO files (workspace_id, name, path, file_type, size, created_at, updated_at)
            VALUES (:workspace_id, :name, :path, :file_type, :size, :created_at, :updated_at)
            """
        )

        file_size = file_path.stat().st_size if file_path.exists() else 0
        file_type = file_path.suffix.lstrip(".") if file_path.suffix else "unknown"

        await self.db.execute(
            insert_query,
            {
                "workspace_id": workspace_id,
                "name": file_path.name,
                "path": str(file_path),
                "file_type": file_type,
                "size": file_size,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )

        # Get the new file ID
        result = await self.db.execute(text("SELECT last_insert_rowid()"))
        return result.scalar()

    async def _process_batch(
        self, file_batch: List[Path], workspace_id: int
    ) -> Dict[str, Any]:
        """Process a batch of files sequentially to avoid session conflicts"""
        logging.info(f"[BATCH] Processing batch of {len(file_batch)} files")

        # Process files sequentially to avoid database session conflicts
        concepts_created = 0
        relationships_created = 0
        files_processed = 0
        errors = 0

        for i, file_path in enumerate(file_batch):
            try:
                logging.info(
                    f"[BATCH] Processing file {i+1}/{len(file_batch)}: {file_path.name}"
                )
                file_concepts, file_relationships = await self._analyze_file_async(
                    file_path, workspace_id
                )
                concepts_created += len(file_concepts)
                relationships_created += len(file_relationships)
                files_processed += 1

                logging.info(
                    f"[BATCH] File {file_path.name} completed: {len(file_concepts)} concepts, {len(file_relationships)} relationships"
                )

            except Exception as e:
                logging.error(f"[BATCH] Error processing file {file_path}: {e}")
                errors += 1
                continue

        logging.info(
            f"[BATCH] Batch completed: {files_processed} files processed, {errors} errors, {concepts_created} total concepts, {relationships_created} total relationships"
        )

        return {
            "files_processed": files_processed,
            "concepts_created": concepts_created,
            "relationships_created": relationships_created,
        }

    async def _analyze_file_async(
        self, file_path: Path, workspace_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Analyze a single file asynchronously"""
        return await self._analyze_file(file_path, workspace_id)

    def _analyze_file_sync(
        self, file_path: Path, workspace_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Synchronous file analysis for thread pool execution"""
        # Read and preprocess content
        content = self._read_file_content(file_path)
        if not content:
            return [], []

        cleaned_content = self._preprocess_text(content)

        # Extract concepts using embeddings if available
        if self.embedding_service:
            concepts = self._extract_concepts_with_embeddings(
                cleaned_content, file_path
            )
        else:
            concepts = self._extract_concepts(cleaned_content)

        return concepts, []  # Relationships handled separately with embeddings

    def _extract_concepts_with_embeddings(
        self, text: str, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Extract concepts using embeddings for better semantic understanding"""
        if not self.embedding_service:
            return self._extract_concepts(text)

        concepts = []

        # Split text into chunks for better embedding
        chunks = self._split_text_into_chunks(text, max_length=500)

        for chunk in chunks:
            if not chunk.strip():
                continue

            # Extract candidate concepts from chunk
            candidates = self._extract_concept_candidates(chunk)

            for candidate in candidates:
                concepts.append(
                    {
                        "name": candidate["name"],
                        "description": f"Found in {file_path.name}: {candidate['context'][:100]}...",
                        "snippet": candidate["context"],
                        "score": candidate["score"],
                        "chunk": chunk[:200],  # Store chunk for embedding
                    }
                )

        # Remove duplicates and limit
        seen = set()
        unique_concepts = []
        for concept in concepts:
            key = concept["name"].lower()
            if key not in seen:
                seen.add(key)
                unique_concepts.append(concept)

        return unique_concepts[: self.max_concepts_per_file]

    def _extract_concept_candidates(self, text: str) -> List[Dict[str, Any]]:
        """Extract concept candidates with context"""
        candidates = []

        # Use regex to find potential concepts
        words = re.findall(r"\b[A-Z][a-z]{2,}\b|\b[a-z]{4,}\b", text)

        for word in words:
            word_lower = word.lower()
            if len(word) > 3:  # Let NLP libraries handle stop words
                # Get context around the word
                start = max(0, text.find(word) - 40)
                end = min(len(text), text.find(word) + len(word) + 40)
                context = text[start:end].strip()

                candidates.append(
                    {
                        "name": word,
                        "context": context,
                        "score": (
                            0.8 if word[0].isupper() else 0.6
                        ),  # Capitalized words get higher score
                    }
                )

        return candidates

    def _split_text_into_chunks(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into manageable chunks for embedding"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = re.split(r"[.!?]+", text)

        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue

            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _build_embedding_relationships(self, workspace_id: int):
        """Build relationships between concepts using embeddings"""
        if not self.embedding_service:
            return

        # Get all concepts for this workspace
        query = text(
            """
            SELECT DISTINCT c.concept_id, c.name
            FROM concepts c
            JOIN concept_files cf ON c.concept_id = cf.concept_id
            WHERE cf.workspace_id = :workspace_id
        """
        )

        result = await self.db.execute(query, {"workspace_id": workspace_id})
        concept_rows = result.fetchall()

        if len(concept_rows) < 2:
            return  # Need at least 2 concepts for relationships

        concepts = [{"id": row.concept_id, "name": row.name} for row in concept_rows]

        # Generate embeddings for all concept names
        concept_names = [c["name"] for c in concepts]
        embeddings = await self.embedding_service.embed_texts(concept_names)

        # Calculate similarity matrix
        relationships_created = 0
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i >= j:
                    continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= self.similarity_threshold:
                    try:
                        # Create relationship
                        await self.kg_service.create_relationship(
                            RelationshipCreate(
                                source_concept_id=concept1["id"],
                                target_concept_id=concept2["id"],
                                type="semantically_related",
                                strength=float(similarity),
                            )
                        )
                        relationships_created += 1
                    except Exception:
                        # Skip duplicate relationships
                        pass

        logging.info(
            f"Created {relationships_created} embedding-based relationships for workspace {workspace_id}"
        )

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def analyze_flattened_file(
        self, flattened_file_path: str, workspace_id: int
    ) -> Dict[str, Any]:
        """
        Analyze a flattened workspace file containing all workspace content.

        Args:
            flattened_file_path: Path to the flattened file
            workspace_id: ID of the workspace

        Returns:
            Analysis results with statistics
        """
        start_time = datetime.utcnow()
        logging.info(
            f"[FLATTENED_ANALYSIS] Starting flattened file analysis for workspace {workspace_id}"
        )
        logging.info(f"[FLATTENED_ANALYSIS] Flattened file path: {flattened_file_path}")

        flattened_path = Path(flattened_file_path)
        if not flattened_path.exists():
            logging.error(
                f"[FLATTENED_ANALYSIS] Flattened file does not exist: {flattened_file_path}"
            )
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": [f"Flattened file not found: {flattened_file_path}"],
                "duration_seconds": 0.0,
                "message": "Flattened file not found",
            }

        # Read the flattened file content
        content = self._read_file_content(flattened_path)
        if not content:
            logging.error(f"[FLATTENED_ANALYSIS] Flattened file is empty or unreadable")
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 0,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": ["Flattened file is empty or unreadable"],
                "duration_seconds": 0.0,
                "message": "Flattened file is empty",
            }

        content_length = len(content)
        logging.info(
            f"[FLATTENED_ANALYSIS] Read {content_length} characters from flattened file"
        )

        # Extract concepts from the entire flattened content (skip NLP to avoid model loading issues)
        logging.info(f"[FLATTENED_ANALYSIS] Extracting concepts from flattened content")
        concepts_data = self._extract_concepts_fallback_for_flattened(content)
        logging.info(
            f"[FLATTENED_ANALYSIS] Extracted {len(concepts_data)} concepts from flattened file"
        )

        if not concepts_data:
            logging.info(f"[FLATTENED_ANALYSIS] No concepts found in flattened file")
            return {
                "workspace_id": workspace_id,
                "files_analyzed": 1,
                "concepts_created": 0,
                "relationships_created": 0,
                "errors": [],
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "message": "No concepts found in flattened file",
            }

        # Create or get file record for the flattened file
        logging.info(
            f"[FLATTENED_ANALYSIS] Creating/updating file record for flattened file"
        )
        file_id = await self._get_or_create_file_record(flattened_path, workspace_id)

        # Store concepts
        stored_concepts = []
        logging.info(f"[FLATTENED_ANALYSIS] Storing {len(concepts_data)} concepts")

        for i, concept_data in enumerate(concepts_data):
            if (i + 1) % 10 == 0:  # Log progress every 10 concepts
                logging.info(
                    f"[FLATTENED_ANALYSIS] Processed {i + 1}/{len(concepts_data)} concepts"
                )

            try:
                # Create concept
                concept = await self.kg_service.create_concept(
                    ConceptCreate(
                        name=concept_data["name"],
                        description=concept_data.get("description", ""),
                    )
                )
                stored_concepts.append(concept)

                # Create concept-file link with line pointers
                await self.kg_service.create_concept_file_link(
                    ConceptFileCreate(
                        concept_id=concept.concept_id,
                        file_id=file_id,
                        workspace_id=workspace_id,
                        snippet=concept_data.get("snippet", ""),
                        relevance_score=concept_data.get("score", 0.5),
                        start_line=concept_data.get("start_line"),
                        end_line=concept_data.get("end_line"),
                    )
                )
            except Exception as e:
                logging.warning(
                    f"[FLATTENED_ANALYSIS] Failed to store concept {concept_data['name']}: {e}"
                )
                continue

        logging.info(f"[FLATTENED_ANALYSIS] Stored {len(stored_concepts)} concepts")

        # Infer relationships between concepts
        logging.info(
            f"[FLATTENED_ANALYSIS] Inferring relationships between {len(stored_concepts)} concepts"
        )
        relationships_data = self._infer_relationships(stored_concepts, content)
        logging.info(
            f"[FLATTENED_ANALYSIS] Found {len(relationships_data)} potential relationships"
        )

        stored_relationships = []
        for rel_data in relationships_data:
            try:
                relationship = await self.kg_service.create_relationship(
                    RelationshipCreate(
                        source_concept_id=rel_data["source_id"],
                        target_concept_id=rel_data["target_id"],
                        type=rel_data["type"],
                        strength=rel_data.get("strength", 0.5),
                    )
                )
                stored_relationships.append(relationship)
            except Exception:
                # Skip duplicate relationships
                pass

        logging.info(
            f"[FLATTENED_ANALYSIS] Created {len(stored_relationships)} relationships"
        )

        # Build embedding-based relationships if available (with timeout)
        if self.embedding_service and len(stored_concepts) > 1:
            logging.info("[FLATTENED_ANALYSIS] Building embedding-based relationships")
            try:
                # Add timeout to prevent hanging
                import asyncio

                await asyncio.wait_for(
                    self._build_embedding_relationships(workspace_id),
                    timeout=30.0,  # 30 second timeout
                )
                logging.info(
                    "[FLATTENED_ANALYSIS] Embedding relationships built successfully"
                )
            except asyncio.TimeoutError:
                logging.warning(
                    "[FLATTENED_ANALYSIS] Embedding relationships timed out, skipping"
                )
            except Exception as e:
                logging.error(
                    f"[FLATTENED_ANALYSIS] Error building embedding relationships: {e}"
                )

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        logging.info(
            f"[FLATTENED_ANALYSIS] Flattened file analysis completed in {duration:.2f} seconds"
        )

        return {
            "workspace_id": workspace_id,
            "files_analyzed": 1,
            "concepts_created": len(stored_concepts),
            "relationships_created": len(stored_relationships),
            "errors": [],
            "duration_seconds": duration,
        }

    async def analyze_file_incremental(
        self, file_path: str, workspace_id: int
    ) -> Dict[str, Any]:
        """Analyze a single file for incremental updates"""
        file_path_obj = Path(file_path)

        # Remove existing concepts for this file
        await self._remove_file_concepts(file_path, workspace_id)

        # Analyze the file
        concepts, relationships = await self._analyze_file(file_path_obj, workspace_id)

        return {
            "file_path": file_path,
            "concepts_created": len(concepts),
            "relationships_created": len(relationships),
        }

    async def _remove_file_concepts(self, file_path: str, workspace_id: int):
        """Remove all concepts associated with a file"""
        # This is a simplified version - in production you'd want to be more careful
        # about not removing concepts that appear in other files
        query = text(
            """
            DELETE FROM concept_files
            WHERE file_id IN (
                SELECT id FROM files WHERE path = :file_path AND workspace_id = :workspace_id
            )
        """
        )

        await self.db.execute(
            query, {"file_path": file_path, "workspace_id": workspace_id}
        )
        await self.db.commit()

    async def get_workspace_stats(self, workspace_id: int) -> Dict[str, Any]:
        """Get analysis statistics for a workspace"""
        # File count
        file_query = text(
            "SELECT COUNT(*) FROM files WHERE workspace_id = :workspace_id"
        )
        file_result = await self.db.execute(file_query, {"workspace_id": workspace_id})
        file_count = file_result.scalar()

        # Concept count
        concept_query = text(
            """
            SELECT COUNT(DISTINCT c.concept_id)
            FROM concepts c
            JOIN concept_files cf ON c.concept_id = cf.concept_id
            WHERE cf.workspace_id = :workspace_id
        """
        )
        concept_result = await self.db.execute(
            concept_query, {"workspace_id": workspace_id}
        )
        concept_count = concept_result.scalar()

        # Relationship count
        rel_query = text(
            """
            SELECT COUNT(*)
            FROM relationships r
            WHERE r.source_concept_id IN (
                SELECT DISTINCT cf.concept_id
                FROM concept_files cf
                WHERE cf.workspace_id = :workspace_id
            )
        """
        )
        rel_result = await self.db.execute(rel_query, {"workspace_id": workspace_id})
        relationship_count = rel_result.scalar()

        return {
            "workspace_id": workspace_id,
            "total_files": file_count,
            "total_concepts": concept_count,
            "total_relationships": relationship_count,
        }
