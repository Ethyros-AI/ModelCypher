"""Automatic dataset repair engine for JSONL training datasets.

Converts chat, instruction, completion, and markdown formats to MLX-compatible
`{"text": "..."}` format. Creates timestamped backups before changes.

**Supported conversions:** Chat messages, instruction/output pairs, prompt/completion,
markdown headers, plain text.

**Safety:** Atomic writes with automatic backup. Unfixable lines reported for manual review.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class FixType(str, Enum):
    """Types of fixes that can be applied."""

    FORMAT_CONVERSION = "format_conversion"
    """chat/instruction/completion → {"text": "..."}"""

    MARKDOWN_CONVERSION = "markdown_conversion"
    """# Markdown → {"text": "..."}"""

    WRAPPED_IN_JSON = "wrapped_in_json"
    """plain text → {"text": "..."}"""

    SYNTAX_FIX = "syntax_fix"
    """invalid JSON → valid JSON"""

    REMOVED_INVALID_TEXT = "removed_invalid_text"
    """dropped empty or whitespace-only {"text": "..."} entries"""


@dataclass(frozen=True)
class Fix:
    """A single fix applied to a line."""

    line_number: int
    """Line number where fix was applied."""

    type: FixType
    """Type of fix applied."""

    before: str
    """Original line content."""

    after: str
    """Fixed line content."""

    description: str
    """Human-readable description of the fix."""


@dataclass(frozen=True)
class UnfixableLine:
    """A line that could not be automatically fixed."""

    line_number: int
    """Line number of the unfixable line."""

    content: str
    """Content of the unfixable line."""


@dataclass(frozen=True)
class AutoFixResult:
    """Result of an auto-fix operation."""

    fixed_count: int
    """Number of lines that were fixed."""

    unfixable_count: int
    """Number of lines that could not be fixed."""

    fixes: list[Fix]
    """List of fixes applied."""

    unfixable_lines: list[UnfixableLine]
    """List of lines that could not be fixed."""

    backup_path: Optional[Path]
    """Path to the backup file."""

    @property
    def is_fully_fixed(self) -> bool:
        """Whether all issues were fixed."""
        return self.unfixable_count == 0

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_fully_fixed:
            return f"Fixed all {self.fixed_count} issues"
        else:
            return f"Fixed {self.fixed_count} issues, {self.unfixable_count} still need manual attention"


class AutoFixEngine:
    """Automatic dataset repair engine for JSONL training datasets."""

    def __init__(self, chat_template: Optional[str] = None):
        """Initialize the auto-fix engine.

        Args:
            chat_template: Optional chat template for converting chat format.
        """
        self.chat_template = chat_template or self._default_chat_template()

    def auto_fix(self, file_path: Path) -> AutoFixResult:
        """Auto-fixes a JSONL file and returns the result.

        Args:
            file_path: Path to the JSONL file to fix.

        Returns:
            AutoFixResult with details of fixes applied.
        """
        file_path = file_path.resolve()

        fixes: list[Fix] = []
        unfixable_lines: list[UnfixableLine] = []

        # Create backup first
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        backup_filename = f"{file_path.stem}.{timestamp}.backup.jsonl"
        backup_path = file_path.parent / backup_filename
        shutil.copy2(file_path, backup_path)

        # Create temp file for output
        temp_path = file_path.parent / f"{file_path.stem}.auto-fix.tmp"

        # Process file line by line
        with open(file_path, "r", encoding="utf-8") as read_file:
            with open(temp_path, "w", encoding="utf-8") as write_file:
                for line_number, line in enumerate(read_file, start=1):
                    trimmed = line.strip()

                    if not trimmed:
                        continue

                    result = self._fix_line(trimmed, line, line_number)

                    if result.action == "keep":
                        write_file.write(result.output_line + "\n")
                        if result.fix:
                            fixes.append(result.fix)
                    elif result.action == "remove":
                        if result.fix:
                            fixes.append(result.fix)
                        # Don't write removed lines
                    elif result.action == "unfixable":
                        write_file.write(line)  # Keep original
                        unfixable_lines.append(
                            UnfixableLine(line_number=line_number, content=line)
                        )

        # Replace original with fixed file
        temp_path.replace(file_path)

        return AutoFixResult(
            fixed_count=len(fixes),
            unfixable_count=len(unfixable_lines),
            fixes=fixes,
            unfixable_lines=unfixable_lines,
            backup_path=backup_path,
        )

    def _fix_line(
        self, trimmed_line: str, original_line: str, line_number: int
    ) -> "_LineFixResult":
        """Fix a single line."""
        # Try to parse as JSON first
        try:
            data = json.loads(trimmed_line)
            if isinstance(data, dict):
                return self._fix_format(data, trimmed_line, original_line, line_number)
        except json.JSONDecodeError:
            pass

        # Not valid JSON - try to fix syntax
        return self._fix_json_syntax(trimmed_line, original_line, line_number)

    def _fix_format(
        self,
        data: dict,
        trimmed_line: str,
        original_line: str,
        line_number: int,
    ) -> "_LineFixResult":
        """Fix format issues in valid JSON."""
        # Check if already has "text" field
        if "text" in data:
            text = data["text"]
            if isinstance(text, str):
                normalized = text.strip()
                if not normalized:
                    # Empty text - remove line
                    fix = Fix(
                        line_number=line_number,
                        type=FixType.REMOVED_INVALID_TEXT,
                        before=original_line,
                        after="",
                        description="Removed empty or whitespace-only text example",
                    )
                    return _LineFixResult(action="remove", output_line="", fix=fix)

            # Already correct format
            return _LineFixResult(action="keep", output_line=trimmed_line, fix=None)

        # Convert from other formats to {"text": "..."}
        converted_text = self._convert_to_text_format(data)
        if converted_text:
            fixed_json = {"text": converted_text}
            fixed_line = json.dumps(fixed_json, ensure_ascii=False)

            fix = Fix(
                line_number=line_number,
                type=FixType.FORMAT_CONVERSION,
                before=original_line,
                after=fixed_line,
                description='Converted to MLX format: {"text": "..."}',
            )

            return _LineFixResult(action="keep", output_line=fixed_line, fix=fix)

        # Unknown format - cannot fix
        return _LineFixResult(action="unfixable", output_line=original_line, fix=None)

    def _convert_to_text_format(self, data: dict) -> Optional[str]:
        """Convert various formats to plain text."""
        # Chat format: {"messages": [...]}
        if "messages" in data and isinstance(data["messages"], list):
            return self._convert_chat_format(data["messages"])

        # Instruction format: {"instruction": "...", "output": "..."}
        if "instruction" in data and "output" in data:
            instruction = data.get("instruction", "")
            output = data.get("output", "")
            return f"{instruction}\n\n{output}"

        # Completion format: {"prompt": "...", "completion": "..."}
        if "prompt" in data and "completion" in data:
            prompt = data.get("prompt", "")
            completion = data.get("completion", "")
            return f"{prompt}\n\n{completion}"

        # Generic: concatenate all string values
        string_values = [v for v in data.values() if isinstance(v, str)]
        if string_values:
            return "\n\n".join(string_values)

        return None

    def _convert_chat_format(self, messages: list) -> Optional[str]:
        """Convert chat message format to plain text."""
        try:
            formatted_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    formatted_parts.append(f"<|{role}|>\n{content}")
            return "\n".join(formatted_parts) if formatted_parts else None
        except Exception:
            return None

    def _fix_json_syntax(
        self, trimmed_line: str, original_line: str, line_number: int
    ) -> "_LineFixResult":
        """Fix JSON syntax errors."""
        # 1. Markdown header detected
        if trimmed_line.startswith("#"):
            text = trimmed_line.lstrip("#").strip()
            fixed_json = {"text": text}
            fixed_line = json.dumps(fixed_json, ensure_ascii=False)

            fix = Fix(
                line_number=line_number,
                type=FixType.MARKDOWN_CONVERSION,
                before=original_line,
                after=fixed_line,
                description="Converted markdown to JSONL",
            )

            return _LineFixResult(action="keep", output_line=fixed_line, fix=fix)

        # 2. Plain text (not starting with {)
        if not trimmed_line.startswith("{"):
            fixed_json = {"text": trimmed_line}
            fixed_line = json.dumps(fixed_json, ensure_ascii=False)

            fix = Fix(
                line_number=line_number,
                type=FixType.WRAPPED_IN_JSON,
                before=original_line,
                after=fixed_line,
                description="Wrapped plain text in JSON",
            )

            return _LineFixResult(action="keep", output_line=fixed_line, fix=fix)

        # Fallback: malformed JSON that resembles an object should be reviewed manually
        return _LineFixResult(action="unfixable", output_line=original_line, fix=None)

    def _default_chat_template(self) -> str:
        """Default chat template."""
        return "<|{role}|>\n{content}"


@dataclass
class _LineFixResult:
    """Internal result of fixing a single line."""

    action: str  # "keep", "remove", or "unfixable"
    output_line: str
    fix: Optional[Fix]
