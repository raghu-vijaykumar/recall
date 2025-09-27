"""
Utility functions for concept extraction.
"""

import os
from typing import Set

try:
    import nltk
    from nltk.corpus import stopwords

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


def load_stop_words(non_concept_file: str) -> Set[str]:
    """Load stop words from file and NLTK/spaCy"""
    stop_words = set()

    # Load from non-concept words file
    try:
        with open(non_concept_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stop_words.add(word)
    except FileNotFoundError:
        print(f"Warning: {non_concept_file} not found")

    # Add NLTK stop words
    if NLTK_AVAILABLE:
        try:
            nltk_stop_words = set(stopwords.words("english"))
            stop_words.update(nltk_stop_words)
        except LookupError:
            print("Warning: NLTK stopwords not available")

    # Add spaCy stop words
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.blank("en")
            spacy_stop_words = nlp.Defaults.stop_words
            stop_words.update(spacy_stop_words)
        except Exception as e:
            print(f"Warning: Could not load spaCy stop words: {e}")

    return stop_words
