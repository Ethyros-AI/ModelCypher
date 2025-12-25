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

"""Dataset file enumerator.

Streaming JSONL/CSV parsing with compression support.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class DatasetFileFormat(str, Enum):
    """Supported file formats."""

    JSONL = "jsonl"
    """JSON Lines format."""

    CSV = "csv"
    """Comma-separated values."""

    TSV = "tsv"
    """Tab-separated values."""

    JSON = "json"
    """Single JSON array."""


class CompressionType(str, Enum):
    """Supported compression types."""

    NONE = "none"
    """No compression."""

    GZIP = "gzip"
    """Gzip compression."""


@dataclass(frozen=True)
class FileMetadata:
    """Metadata about a dataset file."""

    path: Path
    """File path."""

    file_format: DatasetFileFormat
    """Detected file format."""

    compression: CompressionType
    """Compression type."""

    size_bytes: int
    """File size in bytes."""

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / 1_000_000


@dataclass
class EnumeratedSample:
    """A sample from file enumeration."""

    line_number: int
    """Line number in source file (1-based)."""

    sample_index: int
    """Sample index (0-based)."""

    data: dict[str, Any]
    """Parsed sample data."""

    raw_line: str | None = None
    """Raw line text if preserved."""


@dataclass
class EnumerationError:
    """Error during enumeration."""

    line_number: int
    """Line number where error occurred."""

    message: str
    """Error message."""

    raw_line: str | None = None
    """Raw line that caused error."""


class DatasetFileEnumerator:
    """Enumerates samples from dataset files.

    Supports:
    - JSONL (line-delimited JSON)
    - CSV/TSV (with header row)
    - JSON (single array)
    - Gzip compression
    """

    # File extension mappings
    FORMAT_EXTENSIONS = {
        ".jsonl": DatasetFileFormat.JSONL,
        ".jsonlines": DatasetFileFormat.JSONL,
        ".ndjson": DatasetFileFormat.JSONL,
        ".csv": DatasetFileFormat.CSV,
        ".tsv": DatasetFileFormat.TSV,
        ".json": DatasetFileFormat.JSON,
    }

    def __init__(
        self,
        preserve_raw_lines: bool = False,
        max_line_length: int = 10_000_000,
    ):
        """Initialize enumerator.

        Args:
            preserve_raw_lines: Whether to keep raw line text.
            max_line_length: Maximum allowed line length.
        """
        self._preserve_raw = preserve_raw_lines
        self._max_line_length = max_line_length

    def detect_format(self, path: Path) -> tuple[DatasetFileFormat, CompressionType]:
        """Detect file format and compression.

        Args:
            path: File path.

        Returns:
            Tuple of (format, compression).
        """
        name = path.name.lower()
        compression = CompressionType.NONE

        # Check for gzip
        if name.endswith(".gz"):
            compression = CompressionType.GZIP
            name = name[:-3]  # Remove .gz

        # Check format extension
        for ext, fmt in self.FORMAT_EXTENSIONS.items():
            if name.endswith(ext):
                return fmt, compression

        # Default to JSONL
        return DatasetFileFormat.JSONL, compression

    def get_metadata(self, path: Path) -> FileMetadata:
        """Get file metadata.

        Args:
            path: File path.

        Returns:
            File metadata.
        """
        file_format, compression = self.detect_format(path)

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        return FileMetadata(
            path=path,
            file_format=file_format,
            compression=compression,
            size_bytes=size,
        )

    def enumerate(
        self, path: Path, limit: int | None = None
    ) -> Iterator[EnumeratedSample | EnumerationError]:
        """Enumerate samples from a file.

        Args:
            path: File path.
            limit: Maximum samples to yield.

        Yields:
            EnumeratedSample or EnumerationError for each row.
        """
        metadata = self.get_metadata(path)

        if metadata.file_format == DatasetFileFormat.JSONL:
            yield from self._enumerate_jsonl(path, metadata.compression, limit)
        elif metadata.file_format in (DatasetFileFormat.CSV, DatasetFileFormat.TSV):
            delimiter = "\t" if metadata.file_format == DatasetFileFormat.TSV else ","
            yield from self._enumerate_csv(path, metadata.compression, delimiter, limit)
        elif metadata.file_format == DatasetFileFormat.JSON:
            yield from self._enumerate_json_array(path, metadata.compression, limit)
        else:
            yield EnumerationError(
                line_number=0,
                message=f"Unsupported file format: {metadata.file_format}",
            )

    def _open_file(self, path: Path, compression: CompressionType) -> TextIO | Any:
        """Open file with appropriate decompression.

        Args:
            path: File path.
            compression: Compression type.

        Returns:
            File handle.
        """
        if compression == CompressionType.GZIP:
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    def _enumerate_jsonl(
        self,
        path: Path,
        compression: CompressionType,
        limit: int | None,
    ) -> Iterator[EnumeratedSample | EnumerationError]:
        """Enumerate JSONL file."""
        sample_index = 0
        line_number = 0

        try:
            with self._open_file(path, compression) as f:
                for line in f:
                    line_number += 1
                    line = line.strip()

                    if not line:
                        continue

                    if len(line) > self._max_line_length:
                        yield EnumerationError(
                            line_number=line_number,
                            message=f"Line exceeds maximum length: {len(line)}",
                            raw_line=line[:100] + "..." if self._preserve_raw else None,
                        )
                        continue

                    try:
                        data = json.loads(line)
                        yield EnumeratedSample(
                            line_number=line_number,
                            sample_index=sample_index,
                            data=data,
                            raw_line=line if self._preserve_raw else None,
                        )
                        sample_index += 1

                        if limit is not None and sample_index >= limit:
                            break

                    except json.JSONDecodeError as e:
                        yield EnumerationError(
                            line_number=line_number,
                            message=f"Invalid JSON: {e}",
                            raw_line=line[:100] + "..." if self._preserve_raw else None,
                        )

        except FileNotFoundError:
            yield EnumerationError(
                line_number=0,
                message=f"File not found: {path}",
            )
        except OSError as e:
            yield EnumerationError(
                line_number=0,
                message=f"I/O error: {e}",
            )

    def _enumerate_csv(
        self,
        path: Path,
        compression: CompressionType,
        delimiter: str,
        limit: int | None,
    ) -> Iterator[EnumeratedSample | EnumerationError]:
        """Enumerate CSV/TSV file."""
        import csv

        sample_index = 0
        line_number = 0
        headers: list[str] | None = None

        try:
            with self._open_file(path, compression) as f:
                reader = csv.reader(f, delimiter=delimiter)

                for row in reader:
                    line_number += 1

                    if headers is None:
                        # First row is headers
                        headers = row
                        continue

                    if len(row) != len(headers):
                        yield EnumerationError(
                            line_number=line_number,
                            message=f"Column count mismatch: expected {len(headers)}, got {len(row)}",
                        )
                        continue

                    # Convert to dict
                    data = dict(zip(headers, row))

                    yield EnumeratedSample(
                        line_number=line_number,
                        sample_index=sample_index,
                        data=data,
                    )
                    sample_index += 1

                    if limit is not None and sample_index >= limit:
                        break

        except FileNotFoundError:
            yield EnumerationError(
                line_number=0,
                message=f"File not found: {path}",
            )
        except OSError as e:
            yield EnumerationError(
                line_number=0,
                message=f"I/O error: {e}",
            )

    def _enumerate_json_array(
        self,
        path: Path,
        compression: CompressionType,
        limit: int | None,
    ) -> Iterator[EnumeratedSample | EnumerationError]:
        """Enumerate JSON array file."""
        try:
            with self._open_file(path, compression) as f:
                content = f.read()

            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                yield EnumerationError(
                    line_number=0,
                    message=f"Invalid JSON: {e}",
                )
                return

            if not isinstance(data, list):
                yield EnumerationError(
                    line_number=0,
                    message="JSON file must contain an array",
                )
                return

            for i, item in enumerate(data):
                if limit is not None and i >= limit:
                    break

                if not isinstance(item, dict):
                    yield EnumerationError(
                        line_number=0,
                        message=f"Array element {i} is not an object",
                    )
                    continue

                yield EnumeratedSample(
                    line_number=i + 1,  # Use 1-based index
                    sample_index=i,
                    data=item,
                )

        except FileNotFoundError:
            yield EnumerationError(
                line_number=0,
                message=f"File not found: {path}",
            )
        except OSError as e:
            yield EnumerationError(
                line_number=0,
                message=f"I/O error: {e}",
            )

    def count_samples(self, path: Path) -> int:
        """Count samples in a file.

        Args:
            path: File path.

        Returns:
            Number of valid samples.
        """
        count = 0
        for result in self.enumerate(path):
            if isinstance(result, EnumeratedSample):
                count += 1
        return count
