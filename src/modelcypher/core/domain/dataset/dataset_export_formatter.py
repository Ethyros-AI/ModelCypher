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

"""Dataset export formatter.

Normalizes dataset rows to specific formats for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from modelcypher.core.domain.dataset.chat_message import ChatMessage
from modelcypher.core.domain.dataset.chat_template_library import ChatTemplate


class DatasetExportFormat(str, Enum):
    """Export format types."""

    TEXT = "text"
    """Plain text format."""

    CHAT = "chat"
    """Chat/conversation format."""

    TOOLS = "tools"
    """Tool-augmented chat format."""

    INSTRUCTION = "instruction"
    """Instruction format."""

    COMPLETION = "completion"
    """Completion format."""

    UNKNOWN = "unknown"
    """Unknown format."""


class DatasetExportFormatterError(Exception):
    """Error during dataset export formatting."""

    pass


class DatasetExportFormatter:
    """Utility for normalizing dataset rows to specific formats.

    Detects the input format (text, chat, tools, instruction, completion)
    and converts to the desired format.
    """

    def __init__(self, chat_template: ChatTemplate = ChatTemplate.CHATML):
        """Initialize formatter.

        Args:
            chat_template: Template to use for chat formatting.
        """
        self._chat_template = chat_template

    def normalize_line(
        self,
        raw_line: str,
        format_hint: DatasetExportFormat = DatasetExportFormat.UNKNOWN,
        target_format: DatasetExportFormat | None = None,
    ) -> str:
        """Normalize a raw dataset line to the requested format.

        Args:
            raw_line: Raw JSON string from dataset file.
            format_hint: Hint about expected format.
            target_format: Desired output format (defaults to detected format).

        Returns:
            Normalized JSON string.

        Raises:
            DatasetExportFormatterError: For invalid or malformed input.
        """
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            raise DatasetExportFormatterError("Dataset row is not valid JSON")

        if not isinstance(obj, dict):
            raise DatasetExportFormatterError("Dataset row must be a JSON object")

        detected_format = self._detect_format(obj, format_hint)
        desired_format = target_format or detected_format

        payload = self._normalize_payload(obj, detected_format, desired_format, raw_line)

        if not payload:
            raise DatasetExportFormatterError("Dataset row is empty after normalization")

        return json.dumps(payload, sort_keys=True)

    def _detect_format(
        self, obj: dict[str, Any], hint: DatasetExportFormat
    ) -> DatasetExportFormat:
        """Detect format from object structure."""
        if hint != DatasetExportFormat.UNKNOWN:
            return hint

        # Detection logic
        if "text" in obj:
            return DatasetExportFormat.TEXT
        if "messages" in obj:
            if "tools" in obj:
                return DatasetExportFormat.TOOLS
            return DatasetExportFormat.CHAT
        if "prompt" in obj and "completion" in obj:
            return DatasetExportFormat.COMPLETION
        if "instruction" in obj:
            return DatasetExportFormat.INSTRUCTION

        return DatasetExportFormat.TEXT  # Default fallback

    def _normalize_payload(
        self,
        obj: dict[str, Any],
        source_format: DatasetExportFormat,
        target_format: DatasetExportFormat,
        original_line: str,
    ) -> dict[str, Any]:
        """Normalize object to target format."""
        if target_format == DatasetExportFormat.TEXT:
            text = self._render_text(obj, source_format, original_line)
            return {"text": text}

        if target_format == DatasetExportFormat.COMPLETION:
            prompt, completion = self._render_completion(obj, source_format)
            return {"prompt": prompt, "completion": completion}

        if target_format == DatasetExportFormat.CHAT:
            messages = self._render_messages(obj, source_format)
            return {"messages": messages}

        if target_format == DatasetExportFormat.TOOLS:
            messages, tools = self._render_tools(obj, source_format)
            return {"messages": messages, "tools": tools}

        if target_format == DatasetExportFormat.INSTRUCTION:
            return self._render_instruction_payload(obj, source_format)

        # Unknown - extract text
        text = self._extract_text_fallback(obj)
        if not text:
            raise DatasetExportFormatterError("Dataset row is empty after normalization")
        return {"text": text}

    def _render_text(
        self,
        obj: dict[str, Any],
        source_format: DatasetExportFormat,
        original_line: str,
    ) -> str:
        """Render as text format."""
        if source_format == DatasetExportFormat.TEXT:
            text = obj.get("text", "")
            if not isinstance(text, str):
                raise DatasetExportFormatterError("Missing required field 'text'")
            trimmed = text.strip()
            if not trimmed:
                raise DatasetExportFormatterError("Dataset row is empty after normalization")
            return trimmed

        if source_format == DatasetExportFormat.INSTRUCTION:
            instruction = obj.get("instruction", "")
            output = obj.get("output", "")
            if not instruction:
                raise DatasetExportFormatterError("Missing required field 'instruction'")
            if not output:
                raise DatasetExportFormatterError("Missing required field 'output'")

            sections = ["### Instruction", instruction]
            input_text = obj.get("input", "")
            if input_text and input_text.strip():
                sections.extend(["### Input", input_text])
            sections.extend(["### Output", output])
            return "\n\n".join(sections)

        if source_format == DatasetExportFormat.COMPLETION:
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            if not prompt:
                raise DatasetExportFormatterError("Missing required field 'prompt'")
            if not completion:
                raise DatasetExportFormatterError("Missing required field 'completion'")
            return f"{prompt}\n\n### Response\n{completion}"

        if source_format in (DatasetExportFormat.CHAT, DatasetExportFormat.TOOLS):
            messages = self._parse_messages(obj)
            chat_messages = [
                ChatMessage(role=m.get("role", ""), content=m.get("content", ""))
                for m in messages
            ]
            formatted = self._chat_template.format_messages(chat_messages)
            if not formatted.strip():
                raise DatasetExportFormatterError("Dataset row is empty after normalization")
            return formatted.strip()

        # Fallback
        text = self._extract_text_fallback(obj)
        if not text:
            raise DatasetExportFormatterError("Dataset row is empty after normalization")
        return text

    def _render_completion(
        self, obj: dict[str, Any], source_format: DatasetExportFormat
    ) -> tuple[str, str]:
        """Render as completion format."""
        if source_format == DatasetExportFormat.COMPLETION:
            prompt = obj.get("prompt", "").strip()
            completion = obj.get("completion", "").strip()
            if not prompt or not completion:
                raise DatasetExportFormatterError(
                    "Missing required field 'prompt' or 'completion'"
                )
            return prompt, completion

        if source_format == DatasetExportFormat.INSTRUCTION:
            instruction = obj.get("instruction", "")
            output = obj.get("output", "").strip()
            input_text = obj.get("input", "").strip()

            if not instruction:
                raise DatasetExportFormatterError("Missing required field 'instruction'")
            if not output:
                raise DatasetExportFormatterError("Missing required field 'output'")

            prompt = f"{instruction}\n\n{input_text}" if input_text else instruction
            return prompt.strip(), output

        raise DatasetExportFormatterError(
            f"Cannot convert {source_format.value} format to completion"
        )

    def _render_messages(
        self, obj: dict[str, Any], source_format: DatasetExportFormat
    ) -> list[dict[str, Any]]:
        """Render as messages format."""
        if source_format in (DatasetExportFormat.CHAT, DatasetExportFormat.TOOLS):
            return self._parse_messages(obj)

        if source_format == DatasetExportFormat.COMPLETION:
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            if not prompt or not completion:
                raise DatasetExportFormatterError(
                    "Missing required field 'prompt' or 'completion'"
                )
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]

        if source_format == DatasetExportFormat.INSTRUCTION:
            instruction = obj.get("instruction", "")
            output = obj.get("output", "")
            input_text = obj.get("input", "")

            if not instruction or not output:
                raise DatasetExportFormatterError(
                    "Missing required field 'instruction' or 'output'"
                )

            user_message = (
                f"{instruction}\n\n{input_text}"
                if input_text and input_text.strip()
                else instruction
            )
            return [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output},
            ]

        if source_format == DatasetExportFormat.TEXT:
            text = obj.get("text", "")
            if not text:
                raise DatasetExportFormatterError("Missing required field 'text'")
            return [{"role": "user", "content": text}]

        raise DatasetExportFormatterError(
            f"Cannot convert {source_format.value} format to chat"
        )

    def _render_tools(
        self, obj: dict[str, Any], source_format: DatasetExportFormat
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Render as tools format."""
        if source_format != DatasetExportFormat.TOOLS:
            raise DatasetExportFormatterError(
                f"Cannot convert {source_format.value} format to tools"
            )

        messages = self._parse_messages(obj)
        tools = obj.get("tools", [])

        if not isinstance(tools, list) or not tools:
            raise DatasetExportFormatterError("Missing required field 'tools'")

        return messages, tools

    def _render_instruction_payload(
        self, obj: dict[str, Any], source_format: DatasetExportFormat
    ) -> dict[str, Any]:
        """Render as instruction format."""
        if source_format != DatasetExportFormat.INSTRUCTION:
            raise DatasetExportFormatterError(
                f"Cannot convert {source_format.value} format to instruction"
            )

        instruction = obj.get("instruction", "")
        output = obj.get("output", "")

        if not instruction:
            raise DatasetExportFormatterError("Missing required field 'instruction'")
        if not output:
            raise DatasetExportFormatterError("Missing required field 'output'")

        payload: dict[str, Any] = {
            "instruction": instruction,
            "output": output,
        }

        if "input" in obj:
            payload["input"] = obj["input"]

        return payload

    def _parse_messages(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse messages array from object."""
        messages = obj.get("messages", [])
        if not isinstance(messages, list):
            raise DatasetExportFormatterError("Missing required field 'messages'")
        if not messages:
            raise DatasetExportFormatterError("Messages array cannot be empty")
        return messages

    def _extract_text_fallback(self, obj: dict[str, Any]) -> str:
        """Extract text using fallback methods."""
        # Try common text fields
        for field in ["text", "content", "prompt", "instruction"]:
            if field in obj and isinstance(obj[field], str):
                return obj[field].strip()

        # Concatenate all string values
        texts = [str(v) for v in obj.values() if isinstance(v, str)]
        return " ".join(texts).strip()


def convert_format(
    rows: list[str],
    target_format: DatasetExportFormat,
    chat_template: ChatTemplate = ChatTemplate.CHATML,
) -> list[str]:
    """Convert multiple rows to target format.

    Args:
        rows: List of JSON lines.
        target_format: Target format.
        chat_template: Template for chat formatting.

    Returns:
        List of converted JSON lines.

    Raises:
        DatasetExportFormatterError: If conversion fails.
    """
    formatter = DatasetExportFormatter(chat_template=chat_template)
    results: list[str] = []

    for row in rows:
        try:
            converted = formatter.normalize_line(
                row, target_format=target_format
            )
            results.append(converted)
        except DatasetExportFormatterError:
            # Skip invalid rows
            continue

    return results
