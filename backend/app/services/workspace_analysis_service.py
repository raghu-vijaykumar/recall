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
        Fallback concept extraction using heuristics when NLP libraries are unavailable
        """
        concepts = []

        # Split content into lines for line tracking
        lines = content.split("\n")

        # Process each line to find concepts
        for line_num, line in enumerate(lines):
            if not line.strip():
                continue

            # Clean the line for concept extraction
            cleaned_line = self._preprocess_text(line)
            if len(cleaned_line) < 5:  # Skip very short lines
                continue

            # Extract potential concepts from this line
            line_concepts = self._extract_concepts_from_line(
                cleaned_line, line_num, lines
            )

            concepts.extend(line_concepts)

        return concepts

    def _extract_concepts_from_line(
        self, line: str, line_num: int, all_lines: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from a single line with line number tracking
        Improved filtering to avoid gibberish and lorem ipsum content
        """
        concepts = []

        # Skip lines that look like lorem ipsum or template content
        if self._is_lorem_ipsum_or_gibberish(line):
            return concepts

        # Extract potential multi-word concepts first
        # Look for patterns like: "Machine Learning", "Design Pattern", "Abstract Factory"
        multi_word_patterns = [
            # Technical terms with multiple words
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Title Case phrases
            r"\b[a-z]+(?:\s+[a-z]+){1,3}\b",  # lowercase phrases (2-4 words)
            # Domain-specific patterns
            r"\b[A-Z][a-z]+\s+(?:Pattern|Algorithm|Method|Class|Interface|Framework|Library|Protocol|Database|System|Network|Security|Authentication|Authorization)\b",
            r"\b(?:Design|Creational|Structural|Behavioral|Factory|Singleton|Observer|Strategy|Command|Adapter|Bridge|Composite|Decorator|Facade|Proxy|Template|Iterator|State|Memento|Visitor)\s+[A-Z][a-z]+\b",
            # Programming concepts
            r"\b(?:Object|Class|Method|Function|Variable|Constant|Interface|Abstract|Concrete|Static|Dynamic|Virtual|Override|Implement|Inherit|Constructor|Destructor|Exception|Thread|Process|Memory|Cache|Database|Query|Transaction)\s+[A-Z][a-z]+\b",
            # Data structures and algorithms
            r"\b(?:Linked|Binary|Search|Sort|Tree|Graph|Hash|Array|List|Stack|Queue|Heap|Trie|Bloom|Filter)\s+[A-Z][a-z]+\b",
        ]

        for pattern in multi_word_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if self._is_meaningful_concept(match) and not self._is_gibberish_term(
                    match
                ):
                    # Get broader context (surrounding lines)
                    context_lines = self._get_context_lines(
                        all_lines, line_num, context_size=2
                    )
                    context_text = "\n".join(context_lines)

                    score = self._calculate_concept_score(
                        match, "\n".join(all_lines), context_text
                    )

                    if score > 0.85:  # Higher threshold for multi-word concepts
                        concepts.append(
                            {
                                "name": match,
                                "description": f"Concept mentioned in context: {context_text[:120]}...",
                                "snippet": context_text,
                                "score": score,
                                "start_line": line_num,
                                "end_line": line_num,
                            }
                        )

        # Also extract high-quality single words (technical terms)
        single_word_patterns = [
            r"\b[A-Z][a-z]{6,}\b",  # Long capitalized words (likely specific terms)
            r"\b[a-z]{7,}\b",  # Long lowercase words (technical terms)
        ]

        for pattern in single_word_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if self._is_meaningful_concept(match) and not self._is_gibberish_term(
                    match
                ):
                    # Get broader context
                    context_lines = self._get_context_lines(
                        all_lines, line_num, context_size=1
                    )
                    context_text = "\n".join(context_lines)

                    score = self._calculate_concept_score(
                        match, "\n".join(all_lines), context_text
                    )

                    if score > 0.85:  # Higher threshold for single words
                        concepts.append(
                            {
                                "name": match,
                                "description": f"Concept mentioned in context: {context_text[:120]}...",
                                "snippet": context_text,
                                "score": score,
                                "start_line": line_num,
                                "end_line": line_num,
                            }
                        )

        return concepts

    def _get_context_lines(
        self, all_lines: List[str], center_line: int, context_size: int = 2
    ) -> List[str]:
        """
        Get context lines around a center line
        """
        start = max(0, center_line - context_size)
        end = min(len(all_lines), center_line + context_size + 1)
        return all_lines[start:end]

    def _is_meaningful_concept(self, term: str) -> bool:
        """
        Check if a term is likely to be a meaningful concept
        """
        if not term or len(term) < 4:
            return False

        term_lower = term.lower()

        # Skip common non-concept words
        if term_lower in self.non_concept_words:
            return False

        # Must contain at least one letter
        if not re.search(r"[a-zA-Z]", term):
            return False

        # Skip if it's just numbers or symbols
        if re.match(r"^[^a-zA-Z]*$", term):
            return False

        return True

    def _is_lorem_ipsum_or_gibberish(self, line: str) -> bool:
        """
        Generic gibberish detection using statistical analysis
        """
        if not line or len(line.strip()) < 10:
            return False

        # Use combined quality assessment
        quality = self._assess_content_quality(line)
        return quality["is_gibberish"]

    def _is_gibberish_term(self, term: str) -> bool:
        """
        Generic term gibberish detection using statistical analysis
        """
        if not term or len(term) < 3:
            return False

        # Use combined quality assessment for the term
        quality = self._assess_content_quality(term)
        return quality["is_gibberish"]

    def _assess_content_quality(self, text: str) -> dict:
        """
        Combined quality assessment using multiple statistical measures
        """
        if not text or len(text.strip()) < 5:
            return {"quality_score": 0.0, "is_gibberish": True}

        # Run all statistical analyses
        char_entropy = self._calculate_text_entropy(text)
        word_entropy = self._calculate_word_entropy(text)
        repetition_score = self._detect_repetitive_patterns(text)
        word_dist = self._analyze_word_distribution(text)
        word_len = self._analyze_word_lengths(text)
        gini_coefficient = self._analyze_stop_word_ratio(text)
        template_score = self._detect_template_patterns(text)

        # Weighted quality score (0.0 = gibberish, 1.0 = high quality)
        quality_score = (
            (char_entropy / 5.0) * 0.2  # Character entropy
            + (word_entropy / 8.0) * 0.2  # Word entropy
            + (1.0 - repetition_score) * 0.2  # Low repetition = good
            + (1.0 - word_dist["gibberish_score"]) * 0.15  # Word distribution
            + word_len["coherence_score"] * 0.15  # Word length coherence
            + (1.0 - gini_coefficient) * 0.05  # Even word distribution
            + (1.0 - template_score) * 0.05  # Low template similarity
        )

        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        return {
            "quality_score": quality_score,
            "is_gibberish": quality_score < 0.3,  # Threshold for gibberish
            "char_entropy": char_entropy,
            "word_entropy": word_entropy,
            "repetition_score": repetition_score,
            "gini_coefficient": gini_coefficient,
            "template_score": template_score,
        }

    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text characters"""
        if not text:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_word_entropy(self, text: str) -> float:
        """Calculate entropy based on word distribution"""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < 3:
            return 0.0

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        entropy = 0.0
        total_words = len(words)
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)

        return entropy

    def _detect_repetitive_patterns(self, text: str, n: int = 3) -> float:
        """Detect repetitive n-gram patterns"""
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < n * 2:
            return 0.0

        # Generate n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))

        # Count n-gram frequencies
        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Calculate repetition ratio
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)

        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def _analyze_word_distribution(self, text: str) -> dict:
        """Analyze word frequency distribution for gibberish detection"""
        words = re.findall(r"\b\w+\b", text.lower())

        if len(words) < 5:
            return {"gibberish_score": 1.0}  # Too short to analyze

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Calculate distribution metrics
        unique_words = len(word_counts)
        total_words = len(words)
        uniqueness_ratio = unique_words / total_words

        # Check for over-repetition of few words
        top_word_count = max(word_counts.values()) if word_counts else 0
        repetition_ratio = top_word_count / total_words

        # Very low uniqueness or high repetition indicates gibberish
        gibberish_score = 0.0
        if uniqueness_ratio < 0.3:  # Less than 30% unique words
            gibberish_score += 0.5
        if repetition_ratio > 0.4:  # One word used more than 40% of the time
            gibberish_score += 0.5

        return {
            "uniqueness_ratio": uniqueness_ratio,
            "repetition_ratio": repetition_ratio,
            "gibberish_score": gibberish_score,
        }

    def _analyze_word_lengths(self, text: str) -> dict:
        """Analyze word length distribution"""
        words = re.findall(r"\b\w+\b", text)

        if not words:
            return {"avg_length": 0, "coherence_score": 0.0}

        lengths = [len(word) for word in words]
        avg_length = sum(lengths) / len(lengths)

        # Very short average word length indicates gibberish
        coherence_score = min(1.0, avg_length / 5.0)  # Normalize to 5 chars

        return {"avg_length": avg_length, "coherence_score": coherence_score}

    def _analyze_stop_word_ratio(self, text: str) -> float:
        """Analyze ratio of common stop words"""
        # Dynamic stop word detection based on frequency
        words = re.findall(r"\b\w+\b", text.lower())
        word_counts = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Most frequent words are likely stop words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Top 10 most frequent words
        top_words = [word for word, count in sorted_words[:10]]

        # Calculate how evenly distributed the content is
        # High concentration on few words = likely gibberish
        if len(sorted_words) < 3:
            return 1.0  # Too few words to analyze

        # Gini coefficient for word distribution
        total_count = sum(count for word, count in sorted_words)
        cumulative = 0
        gini = 0.0

        for i, (word, count) in enumerate(sorted_words):
            cumulative += count
            gini += (i + 1) * count

        gini = 1 - (2 * gini) / (len(sorted_words) * total_count)

        return gini  # High gini = uneven distribution = gibberish

    def _detect_template_patterns(self, text: str) -> float:
        """Detect template/placeholder patterns dynamically"""
        lines = text.split("\n")

        # Look for similar line structures
        line_patterns = []
        for line in lines:
            # Extract pattern: word types, punctuation, brackets
            pattern = re.sub(r"\w+", "W", line)  # Replace words with W
            pattern = re.sub(r"\d+", "N", pattern)  # Replace numbers with N
            line_patterns.append(pattern.strip())

        # Count pattern frequencies
        pattern_counts = {}
        for pattern in line_patterns:
            if pattern:  # Skip empty patterns
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Calculate template score
        total_lines = len([p for p in line_patterns if p])
        if total_lines == 0:
            return 0.0

        # High similarity in line structures indicates template content
        max_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
        template_score = max_pattern_count / total_lines

        return template_score

    def _calculate_concept_score(
        self, term: str, full_text: str, context: str
    ) -> float:
        """
        Calculate a relevance score for a concept based on various factors
        """
        score = 0.6  # Base score

        term_lower = term.lower()

        # Boost score for capitalized words (likely proper nouns or important terms)
        if term[0].isupper():
            score += 0.2

        # Boost score for longer words (likely more specific terms)
        if len(term) > 6:
            score += 0.1

        # Boost score for words that appear multiple times in the text
        word_count = full_text.lower().count(term_lower)
        if word_count > 1:
            score += min(0.2, word_count * 0.05)

        # Boost score for technical terms (contains numbers, underscores, etc.)
        if re.search(r"[0-9_]", term):
            score += 0.1

        # Boost score for terms that appear in headings or important positions
        lines = full_text.split("\n")
        for i, line in enumerate(lines):
            if term_lower in line.lower():
                # Boost for lines that look like headings
                if line.strip().startswith(
                    ("#", "##", "###", "-", "*", "1.", "2.", "3.")
                ):
                    score += 0.15
                    break
                # Boost for early lines (likely more important)
                if i < 10:
                    score += 0.05
                    break

        # Cap the score at 1.0
        return min(1.0, score)

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
