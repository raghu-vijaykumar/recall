import os
import tempfile
import pytest
from backend.app.services.concept_extraction_service import extract_concepts_from_file


def test_extract_concepts_from_file():
    """Test the concept extraction from a flattened workspace file"""

    # Get the path to the test resource
    test_resource_dir = os.path.join(os.path.dirname(__file__), "resources")
    flattened_file = os.path.join(test_resource_dir, "flattened_workspace.txt")

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_output:
        output_file = temp_output.name

    try:
        # Call the function
        extract_concepts_from_file(flattened_file, output_file)

        # Verify the output file was created and has content
        assert os.path.exists(output_file), "Output file should be created"

        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic checks
        assert "EXTRACTED CONCEPTS FROM FLATTENED WORKSPACE" in content
        assert len(content) > 0, "Output file should not be empty"

        # Check that concepts are extracted (should have numbered list)
        lines = content.split("\n")
        concept_lines = [
            line
            for line in lines
            if line.strip() and line[0].isdigit() and ". " in line
        ]
        assert len(concept_lines) > 0, "Should have extracted some concepts"

    finally:
        # Clean up
        if os.path.exists(output_file):
            os.unlink(output_file)


if __name__ == "__main__":
    import sys

    # Add the parent directory to Python path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    if len(sys.argv) != 3:
        print(
            "Usage: python test_concept_extraction_service.py <flattened_file> <output_file>"
        )
        sys.exit(1)

    flattened_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_concepts_from_file(flattened_file, output_file)
