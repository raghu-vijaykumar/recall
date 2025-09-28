import os
import tempfile
import shutil
from pathlib import Path
import pytest
from backend.app.services.workspace_flattener import WorkspaceFlattener


class TestWorkspaceFlattener:
    """Test cases for the WorkspaceFlattener class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_files = [
            "file1.txt",
            "file2.py",
            "subdir/file3.md",
            "binary_file.jpg",
        ]

        # Create test directory structure
        for file_path in self.test_files:
            full_path = self.temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Create text files with content
            if file_path.endswith((".txt", ".py", ".md")):
                full_path.write_text(f"Content of {file_path}\n")
            else:
                # Create a simple binary file
                full_path.write_bytes(b"\x00\x01\x02\x03")

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.temp_dir)

        # Clean up any created userdata directory
        userdata_dir = Path("userdata")
        if userdata_dir.exists():
            shutil.rmtree(userdata_dir)

    def test_initialization(self):
        """Test WorkspaceFlattener initialization."""
        flattener = WorkspaceFlattener(str(self.temp_dir))
        assert flattener.root_dir == self.temp_dir
        assert isinstance(flattener.git_files, list)

    def test_is_text_file(self):
        """Test text file detection."""
        flattener = WorkspaceFlattener(str(self.temp_dir))

        # Test text files
        assert flattener._is_text_file(self.temp_dir / "file1.txt")
        assert flattener._is_text_file(self.temp_dir / "file2.py")
        assert flattener._is_text_file(self.temp_dir / "subdir" / "file3.md")

        # Test binary file
        assert not flattener._is_text_file(self.temp_dir / "binary_file.jpg")

        # Test non-existent file
        assert not flattener._is_text_file(self.temp_dir / "nonexistent.txt")

    def test_get_all_files(self):
        """Test getting all files when git is not available."""
        flattener = WorkspaceFlattener(str(self.temp_dir))
        # Force empty git files to test fallback
        flattener.git_files = []

        all_files = flattener._get_all_files()
        expected_files = [
            str(Path(f))
            for f in ["binary_file.jpg", "file1.txt", "file2.py", "subdir/file3.md"]
        ]

        # Sort both lists for comparison
        assert sorted(all_files) == sorted(expected_files)

    def test_flatten_workspace_custom_output(self):
        """Test flattening workspace with custom output file."""
        flattener = WorkspaceFlattener(str(self.temp_dir))

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as temp_file:
            output_file = temp_file.name

        try:
            result_path = flattener.flatten_workspace(output_file)

            # Check that the method returns the output file path
            assert result_path == output_file

            # Check that the file was created and has content
            assert os.path.exists(output_file)

            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify header
            assert "FLATTENED WORKSPACE FILE" in content
            assert f"Generated from: {self.temp_dir}" in content

            # Verify file contents are included (should include text files only)
            assert "Content of file1.txt" in content
            assert "Content of file2.py" in content
            assert "Content of subdir/file3.md" in content

            # Binary file should not be included
            assert "binary_file.jpg" not in content

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_flatten_workspace_default_output(self):
        """Test flattening workspace with default output file."""
        flattener = WorkspaceFlattener(str(self.temp_dir))

        result_path = flattener.flatten_workspace()

        # Check that a file was created in the expected location (user's home .recall directory)
        home_dir = Path.home()
        expected_prefix = str(
            home_dir / ".recall" / "workspace" / "flattened_workspace1_"
        )
        assert result_path.startswith(expected_prefix)
        assert result_path.endswith(".txt")
        assert os.path.exists(result_path)

        # Verify the directory was created
        recall_workspace_dir = home_dir / ".recall" / "workspace"
        assert recall_workspace_dir.exists()

        # Check file content
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "FLATTENED WORKSPACE FILE" in content
        assert "Content of file1.txt" in content

    def test_flatten_workspace_empty_directory(self):
        """Test flattening an empty directory."""
        empty_dir = Path(tempfile.mkdtemp())
        try:
            flattener = WorkspaceFlattener(str(empty_dir))

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as temp_file:
                output_file = temp_file.name

            try:
                result_path = flattener.flatten_workspace(output_file)

                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Should still have header even with no files
                assert "FLATTENED WORKSPACE FILE" in content
                assert f"Generated from: {empty_dir}" in content

            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)

        finally:
            shutil.rmtree(empty_dir)

    def test_git_tracked_files_fallback(self):
        """Test that git file detection falls back gracefully."""
        # Create a directory without git
        no_git_dir = Path(tempfile.mkdtemp())
        try:
            flattener = WorkspaceFlattener(str(no_git_dir))

            # Should have empty git_files list but not crash
            assert flattener.git_files == []

            # Should still work with fallback to all files
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as temp_file:
                output_file = temp_file.name

            try:
                result_path = flattener.flatten_workspace(output_file)

                with open(output_file, "r", encoding="utf-8") as f:
                    content = f.read()

                assert "FLATTENED WORKSPACE FILE" in content

            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)

        finally:
            shutil.rmtree(no_git_dir)
