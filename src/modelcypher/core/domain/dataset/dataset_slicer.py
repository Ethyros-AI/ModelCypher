# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""Dataset slicer.

Splits JSONL datasets into smaller files or extracts the first N rows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class SliceMode(str, Enum):
    """Slicing mode."""

    HEAD = "head"
    """Extract first N rows."""

    CHUNK = "chunk"
    """Split into chunks of N rows."""


@dataclass(frozen=True)
class DatasetSliceRecipe:
    """Recipe for slicing a dataset."""

    mode: SliceMode
    """Slicing mode (head or chunk)."""

    count: int
    """Number of rows per slice (or total for head mode)."""

    source_path: Path
    """Path to source JSONL file."""

    output_directory: Path
    """Directory for output files."""

    base_filename: str
    """Base filename for output files."""

    @property
    def is_empty(self) -> bool:
        """Whether recipe is invalid."""
        return self.count < 1 or not self.base_filename.strip()


class DatasetSlicingError(Exception):
    """Error during dataset slicing."""

    pass


class DatasetSlicer:
    """Splits JSONL datasets into smaller files."""

    # Allowed filename characters
    ALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9._-]")

    def __init__(self):
        """Initialize slicer."""
        pass

    def slice(self, recipe: DatasetSliceRecipe) -> list[Path]:
        """Slice a dataset according to recipe.

        Args:
            recipe: Slicing recipe.

        Returns:
            List of output file paths.

        Raises:
            DatasetSlicingError: If slicing fails.
        """
        if recipe.is_empty:
            raise DatasetSlicingError("Slice size must be greater than zero")

        # Create output directory
        recipe.output_directory.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        sanitized_base = self._sanitize_filename(recipe.base_filename)
        if not sanitized_base:
            raise DatasetSlicingError(
                "Base filename contained no valid characters after sanitization"
            )

        results: list[Path] = []
        chunk_index = 0
        rows_in_chunk = 0
        total_rows = 0
        current_file: Path | None = None
        current_handle = None

        def open_next_chunk():
            nonlocal chunk_index, rows_in_chunk, current_file, current_handle

            if current_handle:
                current_handle.close()

            chunk_index += 1
            rows_in_chunk = 0

            if recipe.mode == SliceMode.HEAD:
                filename = f"{sanitized_base}-head-{recipe.count}.jsonl"
            else:
                filename = f"{sanitized_base}-part-{chunk_index}.jsonl"

            current_file = recipe.output_directory / filename

            # Remove existing file
            if current_file.exists():
                current_file.unlink()

            current_handle = open(current_file, "w", encoding="utf-8")
            results.append(current_file)

        def close_current():
            nonlocal current_handle
            if current_handle:
                current_handle.close()
                current_handle = None

        try:
            with open(recipe.source_path, "r", encoding="utf-8") as reader:
                for line in reader:
                    # Check if we've reached limit for head mode
                    if recipe.mode == SliceMode.HEAD and total_rows >= recipe.count:
                        break

                    # Open new chunk if needed
                    if current_handle is None:
                        open_next_chunk()
                    elif recipe.mode == SliceMode.CHUNK and rows_in_chunk >= recipe.count:
                        open_next_chunk()

                    # Write line
                    current_handle.write(line)
                    if not line.endswith("\n"):
                        current_handle.write("\n")

                    rows_in_chunk += 1
                    total_rows += 1

                    # Check head limit again
                    if recipe.mode == SliceMode.HEAD and total_rows >= recipe.count:
                        break

        except FileNotFoundError:
            raise DatasetSlicingError(f"Source file not found: {recipe.source_path}")
        except OSError as e:
            raise DatasetSlicingError(f"I/O error: {e}")
        finally:
            close_current()

        if total_rows == 0:
            # Clean up empty files
            for path in results:
                path.unlink(missing_ok=True)
            raise DatasetSlicingError("No rows were written during slicing")

        if not results:
            raise DatasetSlicingError("No rows were written during slicing")

        logger.info(
            f"Dataset slicing complete: {total_rows} rows processed into {len(results)} file(s)"
        )
        return results

    def _sanitize_filename(self, raw: str, max_length: int = 80) -> str:
        """Sanitize filename for safe filesystem use.

        Args:
            raw: Raw filename.
            max_length: Maximum length.

        Returns:
            Sanitized filename.
        """
        # Keep only allowed characters
        sanitized = self.ALLOWED_CHARS.sub("", raw)

        # Truncate if needed
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        # Trim trailing punctuation
        sanitized = sanitized.rstrip(".-_")

        # Trim leading dots
        sanitized = sanitized.lstrip(".")

        return sanitized

    def iterate_chunks(
        self, path: Path, chunk_size: int
    ) -> Iterator[list[str]]:
        """Iterate over dataset in chunks.

        Args:
            path: Path to JSONL file.
            chunk_size: Lines per chunk.

        Yields:
            Lists of lines (each list is one chunk).
        """
        current_chunk: list[str] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                current_chunk.append(line)

                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []

        # Yield remaining
        if current_chunk:
            yield current_chunk
