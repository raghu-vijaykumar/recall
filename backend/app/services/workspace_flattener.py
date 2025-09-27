"""
Workspace flattening utility for consolidating files into a single document.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class WorkspaceFlattener:
    """Class for flattening workspace files into a single consolidated file."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.git_files = self._get_git_tracked_files()

    def _is_text_file(self, file_path: Path) -> bool:
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

    def _get_git_tracked_files(self) -> List[str]:
        """Get list of files tracked by git."""
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split("\n")
        except subprocess.CalledProcessError:
            # If git command fails, fall back to all files
            print("Warning: Could not get git tracked files, falling back to all files")
            return []

    def _get_all_files(self) -> List[str]:
        """Fallback method to get all files if git is not available."""
        all_files = []
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file():
                all_files.append(str(file_path.relative_to(self.root_dir)))
        return all_files

    def flatten_workspace(self, output_file: Optional[str] = None) -> str:
        """Flatten all git-tracked files in workspace to a single file.

        Args:
            output_file: Optional path to output file. If not provided, saves to
                        userdata/workspace/flattened_workspace1_<timestamp>.txt

        Returns:
            The path to the output file that was created.
        """
        # Generate default output file if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use user's home directory where the database is stored
            home_dir = Path.home()
            recall_dir = home_dir / ".recall"
            workspace_dir = recall_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(workspace_dir / f"flattened_workspace1_{timestamp}.txt")

        # Use git files if available, otherwise fall back to all files
        files_to_process = self.git_files if self.git_files else self._get_all_files()

        with open(output_file, "w", encoding="utf-8") as outfile:
            # Write header
            outfile.write("=" * 80 + "\n")
            outfile.write("FLATTENED WORKSPACE FILE\n")
            outfile.write(f"Generated from: {self.root_dir}\n")
            outfile.write(f"Generated at: {os.path.abspath(output_file)}\n")
            outfile.write("=" * 80 + "\n\n")

            # Process files
            for file_rel_path in sorted(files_to_process):
                if not file_rel_path.strip():
                    continue

                file_path = self.root_dir / file_rel_path

                if file_path.is_file() and self._is_text_file(file_path):
                    # Write file header
                    outfile.write("=" * 80 + "\n")
                    outfile.write(f"FILE: {file_rel_path}\n")
                    outfile.write("=" * 80 + "\n\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            content = infile.read()
                            outfile.write(content)
                            outfile.write("\n\n")
                    except Exception as e:
                        outfile.write(f"ERROR reading file: {e}\n\n")

        return output_file
