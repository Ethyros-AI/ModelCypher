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

"""Pure-domain data preparation: format detection, validation, statistics.

No framework imports. No IO (IO lives in the service layer).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LengthStats:
    """Distribution statistics for sequence lengths."""

    min: int
    max: int
    mean: float
    median: float
    p95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "min": self.min,
            "max": self.max,
            "mean": round(self.mean, 1),
            "median": round(self.median, 1),
            "p95": round(self.p95, 1),
        }


def compute_length_stats(lengths: Sequence[int]) -> LengthStats | None:
    """Compute distribution statistics from a list of lengths."""
    if not lengths:
        return None
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
    else:
        median = sorted_lengths[mid]
    return LengthStats(
        min=sorted_lengths[0],
        max=sorted_lengths[-1],
        mean=sum(sorted_lengths) / n,
        median=median,
        p95=sorted_lengths[p95_idx],
    )


# ---------------------------------------------------------------------------
# Dataset statistics result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStatistics:
    """Result of data validation and statistics."""

    format_detected: str
    n_samples: int
    n_valid: int
    n_removed: int
    char_length_stats: LengthStats | None
    token_length_stats: LengthStats | None
    warnings: list[str]
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "format_detected": self.format_detected,
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_removed": self.n_removed,
            "warnings": list(self.warnings),
            "output_path": self.output_path,
        }
        if self.char_length_stats is not None:
            d["char_length_stats"] = self.char_length_stats.to_dict()
        if self.token_length_stats is not None:
            d["token_length_stats"] = self.token_length_stats.to_dict()
        return d


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# HF dataset pattern: a name like "gsm8k" or "org/dataset" with no file extension
_HF_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+(/[a-zA-Z0-9_.-]+)?$")


def detect_source_format(source: str) -> str:
    """Detect the format of a data source.

    Returns one of: 'jsonl', 'csv', 'txt', 'huggingface', 'conversation_json', 'parquet'.
    """
    path = Path(source)

    # If it doesn't look like a file path and matches HF pattern, assume HF
    if not path.exists() and _HF_PATTERN.match(source):
        return "huggingface"

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".jsonlines"):
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".json":
        # Could be conversation JSON or generic JSON
        return "conversation_json"
    if suffix in (".txt", ".md", ".text"):
        return "txt"

    # Fallback: if it exists, treat as text; if not, assume HF
    if path.exists():
        return "txt"
    return "huggingface"


# ---------------------------------------------------------------------------
# JSONL validation
# ---------------------------------------------------------------------------


def validate_jsonl_lines(
    lines: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate JSONL lines and return (valid_samples, warnings).

    Checks:
    - Each line is valid JSON
    - Each sample has "text" or "messages" field
    - Skips empty lines
    - Detects duplicates
    """
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_texts: set[str] = set()
    n_empty = 0
    n_invalid_json = 0
    n_missing_field = 0
    n_duplicate = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            n_empty += 1
            continue
        try:
            sample = json.loads(stripped)
        except json.JSONDecodeError:
            n_invalid_json += 1
            continue
        if not isinstance(sample, dict):
            n_invalid_json += 1
            continue

        has_text = "text" in sample
        has_messages = "messages" in sample
        if not has_text and not has_messages:
            n_missing_field += 1
            continue

        # Duplicate detection (on text content)
        text_key = sample.get("text", "") or json.dumps(sample.get("messages", []))
        if text_key in seen_texts:
            n_duplicate += 1
            continue
        seen_texts.add(text_key)

        valid.append(sample)

    if n_empty:
        warnings.append(f"{n_empty} empty lines skipped")
    if n_invalid_json:
        warnings.append(f"{n_invalid_json} lines with invalid JSON skipped")
    if n_missing_field:
        warnings.append(
            f"{n_missing_field} samples missing 'text' or 'messages' field skipped"
        )
    if n_duplicate:
        warnings.append(f"{n_duplicate} duplicate samples removed")

    return valid, warnings


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv_to_samples(
    rows: list[dict[str, str]],
    text_column: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert CSV rows to training samples.

    If text_column is specified, use that column for the "text" field.
    Otherwise, auto-detect: look for 'text', 'content', 'question', 'input'.
    """
    warnings: list[str] = []

    if not rows:
        return [], ["No rows in CSV"]

    headers = list(rows[0].keys())

    # Resolve text column
    if text_column and text_column in headers:
        col = text_column
    elif text_column:
        warnings.append(
            f"Column '{text_column}' not found. Available: {', '.join(headers)}"
        )
        return [], warnings
    else:
        # Auto-detect
        candidates = ["text", "content", "question", "input", "prompt"]
        col = None
        for c in candidates:
            if c in headers:
                col = c
                break
        if col is None:
            # Fall back to first column
            col = headers[0]
            warnings.append(
                f"No standard text column found. Using first column: '{col}'"
            )

    samples: list[dict[str, Any]] = []
    n_empty = 0
    for row in rows:
        text = row.get(col, "").strip()
        if not text:
            n_empty += 1
            continue
        samples.append({"text": text})

    if n_empty:
        warnings.append(f"{n_empty} rows with empty text skipped")

    return samples, warnings


# ---------------------------------------------------------------------------
# Conversation JSON parsing
# ---------------------------------------------------------------------------


def parse_conversation_json(
    data: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse conversation-format JSON into training samples.

    Accepts:
    - List of message dicts: [{"role": "user", "content": "..."}, ...]
    - List of conversations: [[{"role": ..., "content": ...}, ...], ...]
    - Dict with "messages" key
    - Dict with "conversations" key
    """
    warnings: list[str] = []

    if isinstance(data, dict):
        if "messages" in data:
            data = [data["messages"]]
        elif "conversations" in data:
            data = data["conversations"]
        else:
            return [], ["JSON object has no 'messages' or 'conversations' key"]

    if not isinstance(data, list):
        return [], ["Expected a JSON array"]

    if not data:
        return [], ["Empty JSON array"]

    # Check if it's a single conversation (list of message dicts)
    # or multiple conversations (list of lists)
    first = data[0]
    if isinstance(first, dict) and "role" in first:
        # Single conversation
        conversations = [data]
    elif isinstance(first, list):
        conversations = data
    elif isinstance(first, dict) and "messages" in first:
        conversations = [c["messages"] for c in data if "messages" in c]
    else:
        return [], ["Unrecognized conversation format"]

    samples: list[dict[str, Any]] = []
    for conv in conversations:
        if not isinstance(conv, list):
            continue
        messages = []
        for msg in conv:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        if messages:
            samples.append({"messages": messages})

    if not samples:
        warnings.append("No valid conversations found")

    return samples, warnings
