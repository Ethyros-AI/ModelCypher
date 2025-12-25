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

"""JSONL parser with normalization.

Parses and normalizes JSONL (JSON Lines) data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ParsedJSONLRow:
    """Result of parsing a JSONL row."""

    line_number: int
    """Line number in source file (1-indexed)."""

    raw_line: str
    """Raw line content."""

    fields: dict[str, str]
    """Normalized field dictionary (all values as strings)."""

    errors: tuple[str, ...]
    """Validation errors (empty if valid)."""

    pretty_json: str
    """Formatted JSON for display."""

    is_valid: bool
    """Whether the row is valid."""


class JSONLParser:
    """Parses and normalizes JSONL data.

    Provides utilities for parsing JSONL files line-by-line, normalizing
    field types, and validating JSON structure.
    """

    DEFAULT_MAX_LINE_LENGTH = 10_000_000  # 10MB

    @classmethod
    def parse_row(
        cls,
        line: str,
        line_number: int,
        max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
    ) -> ParsedJSONLRow:
        """Parse a single JSONL line.

        Args:
            line: Raw JSONL line (single JSON object).
            line_number: Line number in file (for error reporting).
            max_line_length: Maximum allowed line length.

        Returns:
            Parsed row with fields, validation errors, and formatted JSON.
        """
        # Security: Reject excessively long lines
        if len(line) > max_line_length:
            return ParsedJSONLRow(
                line_number=line_number,
                raw_line=line[:100] + "... (truncated)",
                fields={},
                errors=(f"Line exceeds {max_line_length} characters",),
                pretty_json="",
                is_valid=False,
            )

        trimmed = line.strip()

        # Empty line
        if not trimmed:
            return ParsedJSONLRow(
                line_number=line_number,
                raw_line=line,
                fields={},
                errors=(),
                pretty_json="",
                is_valid=True,
            )

        # Parse JSON
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError as e:
            return ParsedJSONLRow(
                line_number=line_number,
                raw_line=line,
                fields={},
                errors=(f"Invalid JSON: {e}",),
                pretty_json="",
                is_valid=False,
            )

        # Must be an object
        if not isinstance(parsed, dict):
            return ParsedJSONLRow(
                line_number=line_number,
                raw_line=line,
                fields={},
                errors=("Invalid JSON - expected object with key-value pairs",),
                pretty_json="",
                is_valid=False,
            )

        # Normalize fields
        normalized = cls.normalize_fields(parsed)

        # Validate
        errors: list[str] = []
        if not normalized:
            errors.append("Empty object - at least one field required")

        # Generate pretty JSON
        pretty_json = cls.compact_json(normalized)

        return ParsedJSONLRow(
            line_number=line_number,
            raw_line=line,
            fields=normalized,
            errors=tuple(errors),
            pretty_json=pretty_json,
            is_valid=len(errors) == 0,
        )

    @classmethod
    def normalize_fields(cls, dictionary: dict[str, Any]) -> dict[str, str]:
        """Normalize JSON fields to string representations.

        Converts all JSON values to strings for consistent handling.

        Args:
            dictionary: Raw JSON dictionary.

        Returns:
            Dictionary with all values converted to strings.
        """
        result: dict[str, str] = {}

        for key, value in dictionary.items():
            if isinstance(value, str):
                result[key] = value
            elif isinstance(value, bool):
                result[key] = "true" if value else "false"
            elif isinstance(value, (int, float)):
                result[key] = str(value)
            elif isinstance(value, (list, dict)):
                try:
                    result[key] = json.dumps(value)
                except (TypeError, ValueError):
                    result[key] = str(value)
            elif value is None:
                result[key] = "null"
            else:
                result[key] = str(value)

        return result

    @classmethod
    def compact_json(cls, fields: dict[str, str]) -> str:
        """Convert fields to compact JSON.

        Args:
            fields: Normalized field dictionary.

        Returns:
            Compact JSON string (single line, sorted keys).
        """
        if not fields:
            return "{}"

        try:
            return json.dumps(fields, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    @classmethod
    def pretty_json(cls, fields: dict[str, str]) -> str:
        """Convert fields to pretty-printed JSON.

        Args:
            fields: Normalized field dictionary.

        Returns:
            Pretty-printed JSON with indentation.
        """
        if not fields:
            return "{}"

        try:
            return json.dumps(fields, sort_keys=True, indent=2)
        except (TypeError, ValueError):
            return "{}"

    @classmethod
    def parse_file(
        cls,
        lines: Iterator[str],
        max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
    ) -> Iterator[ParsedJSONLRow]:
        """Parse JSONL lines from an iterator.

        Args:
            lines: Iterator of lines.
            max_line_length: Maximum allowed line length.

        Yields:
            Parsed rows.
        """
        for line_number, line in enumerate(lines, start=1):
            yield cls.parse_row(line, line_number, max_line_length)

    @classmethod
    def denormalize_fields(cls, fields: dict[str, str]) -> dict[str, Any]:
        """Convert normalized fields back to native types.

        Attempts to parse JSON arrays/objects and convert numbers/booleans.

        Args:
            fields: Normalized field dictionary.

        Returns:
            Dictionary with parsed values.
        """
        result: dict[str, Any] = {}

        for key, value in fields.items():
            # Try to parse as JSON (for arrays/objects)
            if value.startswith(("[", "{")):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass

            # Check for null
            if value == "null":
                result[key] = None
                continue

            # Check for boolean
            if value == "true":
                result[key] = True
                continue
            if value == "false":
                result[key] = False
                continue

            # Try to parse as number
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
                continue
            except ValueError:
                pass

            # Keep as string
            result[key] = value

        return result
