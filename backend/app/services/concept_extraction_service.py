"""
Backward compatibility wrapper for the refactored concept extraction service.

This module now imports from the new concept_extraction package structure.
"""

# Import everything from the new package for backward compatibility
try:
    # Try relative import first (when used as part of the package)
    from .concept_extraction import (
        ConceptExtractor,
        RankingAlgorithm,
        EntityRecognitionExtractor,
        FrequencyBasedRanking,
        TFIDFRanking,
        ConceptExtractionService,
        load_stop_words,
    )
except ImportError:
    # Fall back to absolute import (when run as standalone script)
    from concept_extraction import (
        ConceptExtractor,
        RankingAlgorithm,
        EntityRecognitionExtractor,
        FrequencyBasedRanking,
        TFIDFRanking,
        ConceptExtractionService,
        load_stop_words,
    )


# Keep the old function for backward compatibility
def extract_concepts_from_file(
    flattened_file_path: str, output_file: str, use_spacy: bool = False
):
    """Extract concepts from a flattened workspace file using the new service"""

    import os

    # Load stop words
    stop_words_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "non_concept_words.txt"
    )
    stop_words = load_stop_words(stop_words_file)

    # Create extractor and ranker
    extractor = EntityRecognitionExtractor(stop_words=stop_words, use_spacy=use_spacy)
    ranker = FrequencyBasedRanking()  # Can be changed to TFIDFRanking()

    # Create service
    service = ConceptExtractionService(extractor, ranker)

    # Read content
    with open(flattened_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"Read {len(content)} characters from flattened file")

    # Extract and rank concepts
    concepts = service.extract_and_rank_concepts(content)

    print(f"Extracted {len(concepts)} concepts")

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("EXTRACTED CONCEPTS FROM FLATTENED WORKSPACE\n")
        f.write("=" * 50 + "\n\n")

        for i, concept in enumerate(concepts[:200], 1):  # Limit to top 200
            f.write(f"{i}. {concept['name']}\n")
            f.write(f"   Score: {concept.get('score', 0):.3f}\n")
            f.write(f"   Type: {concept.get('type', 'unknown')}\n")
            if "line" in concept:
                f.write(f"   Line: {concept['line']}\n")
            if "context" in concept:
                f.write(f"   Context: {concept['context'][:200]}...\n")
            f.write("\n")

    print(f"Concepts written to {output_file}")
