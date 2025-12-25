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

"""Dataset text extractor.

Extracts and normalizes text from various dataset formats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from modelcypher.core.domain.validation.dataset_validation_models import (
    DatasetContentFormat,
)


@dataclass(frozen=True)
class ExtractedText:
    """Extracted text from a sample."""

    text: str
    """Extracted text content."""

    source_format: DatasetContentFormat
    """Original format."""

    field_count: int
    """Number of fields extracted from."""

    total_chars: int
    """Total character count."""

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return self.total_chars // 4


class DatasetTextExtractor:
    """Extracts text content from dataset samples.

    Handles all supported formats and normalizes output.
    """

    # Characters to normalize
    WHITESPACE_RE = re.compile(r"\s+")

    def extract(
        self,
        sample: dict[str, Any],
        detected_format: DatasetContentFormat | None = None,
        normalize_whitespace: bool = True,
        include_roles: bool = False,
    ) -> ExtractedText:
        """Extract text from a sample.

        Args:
            sample: Parsed sample.
            detected_format: Known format (auto-detects if None).
            normalize_whitespace: Whether to normalize whitespace.
            include_roles: Whether to include role markers in chat extraction.

        Returns:
            Extracted text.
        """
        if detected_format is None:
            detected_format = self._detect_format(sample)

        if detected_format == DatasetContentFormat.TEXT:
            text, field_count = self._extract_text_format(sample)
        elif detected_format in (DatasetContentFormat.CHAT, DatasetContentFormat.TOOLS):
            text, field_count = self._extract_chat_format(sample, include_roles)
        elif detected_format == DatasetContentFormat.INSTRUCTION:
            text, field_count = self._extract_instruction_format(sample)
        elif detected_format == DatasetContentFormat.COMPLETION:
            text, field_count = self._extract_completion_format(sample)
        else:
            text, field_count = self._extract_unknown_format(sample)

        if normalize_whitespace:
            text = self._normalize_whitespace(text)

        return ExtractedText(
            text=text,
            source_format=detected_format,
            field_count=field_count,
            total_chars=len(text),
        )

    def _detect_format(self, sample: dict[str, Any]) -> DatasetContentFormat:
        """Detect format from sample structure."""
        if "text" in sample:
            return DatasetContentFormat.TEXT
        if "messages" in sample:
            if "tools" in sample:
                return DatasetContentFormat.TOOLS
            return DatasetContentFormat.CHAT
        if "instruction" in sample or "input" in sample:
            return DatasetContentFormat.INSTRUCTION
        if "prompt" in sample and "completion" in sample:
            return DatasetContentFormat.COMPLETION
        return DatasetContentFormat.UNKNOWN

    def _extract_text_format(self, sample: dict[str, Any]) -> tuple[str, int]:
        """Extract from text format."""
        text = sample.get("text", "")
        if isinstance(text, str):
            return text, 1
        return "", 0

    def _extract_chat_format(self, sample: dict[str, Any], include_roles: bool) -> tuple[str, int]:
        """Extract from chat format."""
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            return "", 0

        parts: list[str] = []
        field_count = 0

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str) and content:
                if include_roles:
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(content)
                field_count += 1

        return "\n\n".join(parts), field_count

    def _extract_instruction_format(self, sample: dict[str, Any]) -> tuple[str, int]:
        """Extract from instruction format."""
        parts: list[str] = []
        field_count = 0

        # System prompt if present
        system = sample.get("system", "")
        if isinstance(system, str) and system:
            parts.append(system)
            field_count += 1

        # Instruction or input
        instruction = sample.get("instruction") or sample.get("input", "")
        if isinstance(instruction, str) and instruction:
            parts.append(instruction)
            field_count += 1

        # Output
        output = sample.get("output", "")
        if isinstance(output, str) and output:
            parts.append(output)
            field_count += 1

        return "\n\n".join(parts), field_count

    def _extract_completion_format(self, sample: dict[str, Any]) -> tuple[str, int]:
        """Extract from completion format."""
        parts: list[str] = []
        field_count = 0

        prompt = sample.get("prompt", "")
        if isinstance(prompt, str) and prompt:
            parts.append(prompt)
            field_count += 1

        completion = sample.get("completion", "")
        if isinstance(completion, str) and completion:
            parts.append(completion)
            field_count += 1

        return "\n\n".join(parts), field_count

    def _extract_unknown_format(self, sample: dict[str, Any]) -> tuple[str, int]:
        """Extract from unknown format by concatenating string values."""
        parts: list[str] = []

        for key, value in sample.items():
            if isinstance(value, str) and value:
                parts.append(value)

        return "\n\n".join(parts), len(parts)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        text = self.WHITESPACE_RE.sub(" ", text)
        # Strip leading/trailing
        return text.strip()

    def extract_for_tokenization(
        self,
        sample: dict[str, Any],
        detected_format: DatasetContentFormat | None = None,
    ) -> str:
        """Extract text optimized for tokenization.

        Preserves structure for chat templates.

        Args:
            sample: Parsed sample.
            detected_format: Known format.

        Returns:
            Text suitable for tokenization.
        """
        if detected_format is None:
            detected_format = self._detect_format(sample)

        # For chat/tools, preserve structure
        if detected_format in (DatasetContentFormat.CHAT, DatasetContentFormat.TOOLS):
            return self._extract_chat_for_tokenization(sample)

        # For others, just extract text
        extracted = self.extract(
            sample,
            detected_format=detected_format,
            normalize_whitespace=False,
            include_roles=False,
        )
        return extracted.text

    def _extract_chat_for_tokenization(self, sample: dict[str, Any]) -> str:
        """Extract chat for tokenization with markers."""
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            return ""

        parts: list[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Use special markers for tokenization
                if role == "system":
                    parts.append(f"<|system|>\n{content}")
                elif role == "user":
                    parts.append(f"<|user|>\n{content}")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}")
                elif role == "tool":
                    parts.append(f"<|tool|>\n{content}")

        return "\n".join(parts)
