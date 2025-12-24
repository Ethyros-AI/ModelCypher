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

"""Normalized representation of a dataset line for safety validation.

TrainingSample parses common JSON structures (text, completion, messages array)
and falls back to the raw line if no structured content is found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingSample:
    """Normalized representation of a dataset line that can be validated for safety.

    Training datasets may be stored as JSONL or plain text. TrainingSample
    parses the common JSON structures we expect (text, completion, or a
    messages array) and falls back to the raw line if no structured content
    is found. The normalized text is trimmed so regex filters do not need to
    duplicate whitespace handling.
    """

    raw: str
    """Original line as it appears in the dataset file."""

    text: str
    """Normalized textual content extracted from the raw line."""

    source_path: Path
    """Source file for traceability when reporting issues."""

    @classmethod
    def from_line(cls, raw: str, source_path: Path) -> TrainingSample:
        """Create a TrainingSample from a raw line and source path."""
        text = cls._extract_text(raw)
        return cls(raw=raw, text=text, source_path=source_path)

    @staticmethod
    def _extract_text(raw: str) -> str:
        """Extract normalized text from a raw line.

        Attempts to parse as JSON and extract text from common fields.
        Falls back to the raw line if parsing fails.
        """
        stripped = raw.strip()
        if not stripped:
            return ""

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        if not isinstance(data, dict):
            return stripped

        # Try "text" field
        if "text" in data and isinstance(data["text"], str):
            return data["text"].strip()

        # Try "completion" field alone
        if "completion" in data and isinstance(data["completion"], str):
            completion = data["completion"].strip()
            # If there's also a prompt, combine them
            if "prompt" in data and isinstance(data["prompt"], str):
                prompt = data["prompt"].strip()
                return f"{prompt}\n{completion}"
            return completion

        # Try "messages" array (chat format)
        if "messages" in data and isinstance(data["messages"], list):
            contents = []
            for message in data["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content = message.get("content")
                    if isinstance(content, str):
                        contents.append(content.strip())
            if contents:
                return "\n".join(contents)

        # Fallback to raw
        return stripped

    def __str__(self) -> str:
        """Return the normalized text."""
        return self.text
