import re
from typing import List, Optional
from dataclasses import dataclass, field


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
