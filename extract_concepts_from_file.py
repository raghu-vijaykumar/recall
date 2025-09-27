#!/usr/bin/env python3
"""
Script to extract concepts from a flattened workspace file using simplified analysis logic.
"""

import os
import sys
import re
import math
from pathlib import Path
from typing import List, Dict, Any


def is_text_file(file_path: str) -> bool:
    """Check if file is likely text-based by reading first few bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            # Check for null bytes (indicates binary)
            if b"\x00" in chunk:
                return False
            # Try to decode as UTF-8
            chunk.decode("utf-8")
            return True
    except (UnicodeDecodeError, OSError):
        return False


def _calculate_text_entropy(text: str) -> float:
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


def _calculate_word_entropy(text: str) -> float:
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


def _assess_content_quality(text: str) -> dict:
    """
    Combined quality assessment using multiple statistical measures
    """
    if not text or len(text.strip()) < 5:
        return {"quality_score": 0.0, "is_gibberish": True}

    # Run all statistical analyses
    char_entropy = _calculate_text_entropy(text)
    word_entropy = _calculate_word_entropy(text)

    # Very short average word length indicates gibberish
    words = re.findall(r"\b\w+\b", text)
    avg_length = sum(len(word) for word in words) / len(words) if words else 0
    coherence_score = min(1.0, avg_length / 5.0)  # Normalize to 5 chars

    # Weighted quality score (0.0 = gibberish, 1.0 = high quality)
    quality_score = (
        (char_entropy / 5.0) * 0.4  # Character entropy
        + (word_entropy / 8.0) * 0.4  # Word entropy
        + coherence_score * 0.2  # Word length coherence
    )

    # Clamp to [0, 1]
    quality_score = max(0.0, min(1.0, quality_score))

    return {
        "quality_score": quality_score,
        "is_gibberish": quality_score < 0.3,  # Threshold for gibberish
    }


def _is_gibberish_term(term: str) -> bool:
    """
    Generic term gibberish detection using statistical analysis
    """
    if not term or len(term) < 3:
        return False

    # Use combined quality assessment for the term
    quality = _assess_content_quality(term)
    return quality["is_gibberish"]


def _calculate_concept_score(term: str, full_text: str, context: str) -> float:
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

    # Cap the score at 1.0
    return min(1.0, score)


def extract_concepts_from_content(content: str) -> List[Dict[str, Any]]:
    """
    Extract concepts from text content using simplified analysis logic
    """
    concepts = []

    if not content:
        return concepts

    # Split content into lines for line tracking
    lines = content.split("\n")

    # Process each line to find concepts
    for line_num, line in enumerate(lines):
        if not line.strip():
            continue

        # Clean the line for concept extraction
        cleaned_line = re.sub(r"#+\s*", "", line)  # Headers
        cleaned_line = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned_line)  # Bold
        cleaned_line = re.sub(r"\*([^*]+)\*", r"\1", cleaned_line)  # Italic
        cleaned_line = re.sub(r"`([^`]+)`", r"\1", cleaned_line)  # Code
        cleaned_line = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned_line)  # Links

        if len(cleaned_line) < 5:  # Skip very short lines
            continue

        # Skip lines that look like gibberish
        if _assess_content_quality(cleaned_line)["is_gibberish"]:
            continue

        # Extract potential multi-word concepts first
        multi_word_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Title Case phrases
            r"\b[a-z]+(?:\s+[a-z]+){1,3}\b",  # lowercase phrases (2-4 words)
            r"\b[A-Z][a-z]+\s+(?:Pattern|Algorithm|Method|Class|Interface|Framework|Library|Protocol|Database|System|Network|Security|Authentication|Authorization)\b",
            r"\b(?:Design|Creational|Structural|Behavioral|Factory|Singleton|Observer|Strategy|Command|Adapter|Bridge|Composite|Decorator|Facade|Proxy|Template|Iterator|State|Memento|Visitor)\s+[A-Z][a-z]+\b",
            r"\b(?:Object|Class|Method|Function|Variable|Constant|Interface|Abstract|Concrete|Static|Dynamic|Virtual|Override|Implement|Inherit|Constructor|Destructor|Exception|Thread|Process|Memory|Cache|Database|Query|Transaction)\s+[A-Z][a-z]+\b",
        ]

        for pattern in multi_word_patterns:
            matches = re.findall(pattern, cleaned_line)
            for match in matches:
                if len(match) > 3 and not _is_gibberish_term(match):
                    # Get broader context (surrounding lines)
                    start = max(0, line_num - 2)
                    end = min(len(lines), line_num + 3)
                    context_lines = lines[start:end]
                    context_text = "\n".join(context_lines)

                    score = _calculate_concept_score(match, content, context_text)

                    if score > 0.8:  # Higher threshold for multi-word concepts
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
            matches = re.findall(pattern, cleaned_line)
            for match in matches:
                if len(match) > 3 and not _is_gibberish_term(match):
                    # Get broader context
                    start = max(0, line_num - 1)
                    end = min(len(lines), line_num + 2)
                    context_lines = lines[start:end]
                    context_text = "\n".join(context_lines)

                    score = _calculate_concept_score(match, content, context_text)

                    if score > 0.8:  # Higher threshold for single words
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

    # Remove duplicates and apply final quality threshold
    seen = set()
    unique_concepts = []
    for concept in concepts:
        key = concept["name"].lower()
        if key not in seen and concept["score"] > 0.7:
            seen.add(key)
            unique_concepts.append(concept)

    return unique_concepts[:200]  # Limit concepts


def extract_concepts_from_file(flattened_file_path: str, output_file: str):
    """Extract concepts from a flattened workspace file"""

    # Read the flattened file
    with open(flattened_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Read {len(content)} characters from flattened file")

    # Extract concepts using simplified analysis logic
    concepts = extract_concepts_from_content(content)

    print(f"Extracted {len(concepts)} concepts")

    # Write concepts to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("EXTRACTED CONCEPTS FROM FLATTENED WORKSPACE\n")
        f.write("=" * 50 + "\n\n")

        # Sort concepts by score
        concepts.sort(key=lambda x: x["score"], reverse=True)

        for i, concept in enumerate(concepts, 1):
            f.write(f"{i}. {concept['name']}\n")
            f.write(f"   Score: {concept['score']:.3f}\n")
            f.write(f"   Description: {concept['description']}\n")
            f.write(f"   Line: {concept.get('start_line', 'N/A')}\n")
            f.write(f"   Snippet: {concept['snippet'][:200]}...\n\n")

    print(f"Concepts written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python extract_concepts_from_file.py <flattened_file> <output_file>"
        )
        sys.exit(1)

    flattened_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_concepts_from_file(flattened_file, output_file)
