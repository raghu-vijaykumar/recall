#!/usr/bin/env python3
"""
Script to flatten all workspace files tracked by git into a single file with path headers.
Only includes non-binary files.
"""

import os
import sys
import subprocess
from pathlib import Path


def is_text_file(file_path):
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


def get_git_tracked_files(root_dir):
    """Get list of files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        # If git command fails, fall back to all files
        print("Warning: Could not get git tracked files, falling back to all files")
        return []


def flatten_workspace(root_dir, output_file):
    """Flatten all git-tracked files in workspace to a single file."""
    root_path = Path(root_dir)

    # Get git tracked files
    git_files = get_git_tracked_files(root_dir)
    if not git_files:
        # Fallback: walk directory if git not available
        git_files = []
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                git_files.append(str(file_path.relative_to(root_path)))

    with open(output_file, "w", encoding="utf-8") as outfile:
        # Write header
        outfile.write("=" * 80 + "\n")
        outfile.write("FLATTENED WORKSPACE FILE\n")
        outfile.write(f"Generated from: {root_dir}\n")
        outfile.write(f"Generated at: {os.path.abspath(output_file)}\n")
        outfile.write("=" * 80 + "\n\n")

        # Process git tracked files
        for file_rel_path in sorted(git_files):
            if not file_rel_path.strip():
                continue

            file_path = root_path / file_rel_path

            if file_path.is_file() and is_text_file(file_path):
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flatten_workspace.py <root_dir> <output_file>")
        sys.exit(1)

    root_dir = sys.argv[1]
    output_file = sys.argv[2]

    flatten_workspace(root_dir, output_file)
    print(f"Flattened workspace written to: {output_file}")
