"""Dataset format analyzer.

Detects content format and validates structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from modelcypher.core.domain.validation.dataset_validation_models import (
    DatasetContentFormat,
    ValidationError,
    ValidationErrorKind,
    ValidationWarning,
    ValidationWarningKind,
)


@dataclass
class FormatAnalysisResult:
    """Result of format analysis."""

    format: DatasetContentFormat
    """Detected format."""

    confidence: float
    """Confidence in detection (0.0-1.0)."""

    errors: list[ValidationError]
    """Validation errors."""

    warnings: list[ValidationWarning]
    """Validation warnings."""

    @property
    def is_valid(self) -> bool:
        """Whether the sample is valid for the detected format."""
        return len(self.errors) == 0


class DatasetFormatAnalyzer:
    """Analyzes dataset samples to detect format and validate structure.

    Detection priority: text → tools → chat → completion → instruction
    """

    VALID_ROLES = {"system", "user", "assistant", "tool"}

    def detect_format(self, sample: dict[str, Any]) -> DatasetContentFormat:
        """Detect the format of a single sample.

        Args:
            sample: Parsed JSON sample.

        Returns:
            Detected content format.
        """
        # Text format: has 'text' field
        if "text" in sample and isinstance(sample["text"], str):
            return DatasetContentFormat.TEXT

        # Tools format: has 'messages' and 'tools'
        if "messages" in sample and "tools" in sample:
            if isinstance(sample["messages"], list) and isinstance(
                sample["tools"], list
            ):
                return DatasetContentFormat.TOOLS

        # Chat format: has 'messages' array
        if "messages" in sample and isinstance(sample["messages"], list):
            return DatasetContentFormat.CHAT

        # Completion format: has 'prompt' and 'completion'
        if "prompt" in sample and "completion" in sample:
            return DatasetContentFormat.COMPLETION

        # Instruction format: has 'instruction' and 'output'
        if "instruction" in sample and "output" in sample:
            return DatasetContentFormat.INSTRUCTION

        # Also check for 'input' + 'output' as instruction variant
        if "input" in sample and "output" in sample:
            return DatasetContentFormat.INSTRUCTION

        return DatasetContentFormat.UNKNOWN

    def analyze(
        self,
        sample: dict[str, Any],
        line_number: Optional[int] = None,
        sample_index: Optional[int] = None,
    ) -> FormatAnalysisResult:
        """Analyze a sample for format and validity.

        Args:
            sample: Parsed JSON sample.
            line_number: Line number in source file (1-based).
            sample_index: Sample index (0-based).

        Returns:
            Analysis result with format, errors, and warnings.
        """
        detected_format = self.detect_format(sample)

        if detected_format == DatasetContentFormat.UNKNOWN:
            return FormatAnalysisResult(
                format=detected_format,
                confidence=0.0,
                errors=[
                    ValidationError(
                        kind=ValidationErrorKind.MISSING_REQUIRED_FIELD,
                        message="Could not detect format - no recognized fields",
                        line_number=line_number,
                        sample_index=sample_index,
                    )
                ],
                warnings=[],
            )

        # Validate based on detected format
        errors: list[ValidationError] = []
        warnings: list[ValidationWarning] = []

        if detected_format == DatasetContentFormat.TEXT:
            self._validate_text(sample, line_number, sample_index, errors, warnings)
        elif detected_format == DatasetContentFormat.CHAT:
            self._validate_chat(sample, line_number, sample_index, errors, warnings)
        elif detected_format == DatasetContentFormat.TOOLS:
            self._validate_tools(sample, line_number, sample_index, errors, warnings)
        elif detected_format == DatasetContentFormat.INSTRUCTION:
            self._validate_instruction(
                sample, line_number, sample_index, errors, warnings
            )
        elif detected_format == DatasetContentFormat.COMPLETION:
            self._validate_completion(
                sample, line_number, sample_index, errors, warnings
            )

        # Confidence based on validation results
        confidence = 1.0 if not errors else 0.5

        return FormatAnalysisResult(
            format=detected_format,
            confidence=confidence,
            errors=errors,
            warnings=warnings,
        )

    def _validate_text(
        self,
        sample: dict[str, Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate text format sample."""
        text = sample.get("text", "")
        if not isinstance(text, str):
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.MISSING_REQUIRED_FIELD,
                    message="'text' field must be a string",
                    line_number=line_number,
                    field_name="text",
                    sample_index=sample_index,
                )
            )
            return

        if not text.strip():
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.EMPTY_CONTENT,
                    message="'text' field is empty",
                    line_number=line_number,
                    field_name="text",
                    sample_index=sample_index,
                )
            )

        # Check length
        if len(text) < 10:
            warnings.append(
                ValidationWarning(
                    kind=ValidationWarningKind.SHORT_CONTENT,
                    message=f"Text is very short ({len(text)} chars)",
                    line_number=line_number,
                    field_name="text",
                    sample_index=sample_index,
                )
            )

    def _validate_chat(
        self,
        sample: dict[str, Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate chat format sample."""
        messages = sample.get("messages", [])
        if not isinstance(messages, list):
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.INVALID_MESSAGES_STRUCTURE,
                    message="'messages' must be an array",
                    line_number=line_number,
                    field_name="messages",
                    sample_index=sample_index,
                )
            )
            return

        if not messages:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.EMPTY_CONTENT,
                    message="'messages' array is empty",
                    line_number=line_number,
                    field_name="messages",
                    sample_index=sample_index,
                )
            )
            return

        # Validate each message and check semantics
        self._validate_chat_semantics(
            messages, line_number, sample_index, errors, warnings
        )

    def _validate_chat_semantics(
        self,
        messages: list[Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate chat message semantics."""
        has_system = False
        has_user = False
        has_assistant = False
        system_count = 0
        prev_role: Optional[str] = None

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(
                    ValidationError(
                        kind=ValidationErrorKind.INVALID_MESSAGES_STRUCTURE,
                        message=f"Message {i} is not an object",
                        line_number=line_number,
                        sample_index=sample_index,
                    )
                )
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            # Check role
            if role not in self.VALID_ROLES:
                errors.append(
                    ValidationError(
                        kind=ValidationErrorKind.INVALID_ROLE,
                        message=f"Invalid role '{role}' in message {i}",
                        line_number=line_number,
                        sample_index=sample_index,
                    )
                )
                continue

            # Track roles
            if role == "system":
                has_system = True
                system_count += 1
            elif role == "user":
                has_user = True
            elif role == "assistant":
                has_assistant = True

            # Check content (allow empty for tool calls)
            if role != "tool" and not content and "tool_calls" not in msg:
                errors.append(
                    ValidationError(
                        kind=ValidationErrorKind.EMPTY_MESSAGE_CONTENT,
                        message=f"Empty content in {role} message {i}",
                        line_number=line_number,
                        sample_index=sample_index,
                    )
                )

            # Check role alternation (user/assistant should alternate)
            if prev_role == "user" and role == "user":
                warnings.append(
                    ValidationWarning(
                        kind=ValidationWarningKind.UNUSUAL_ROLE_PATTERN,
                        message=f"Consecutive user messages at index {i}",
                        line_number=line_number,
                        sample_index=sample_index,
                    )
                )
            elif prev_role == "assistant" and role == "assistant":
                # Allow consecutive assistant if tool_calls involved
                if "tool_calls" not in messages[i - 1]:
                    warnings.append(
                        ValidationWarning(
                            kind=ValidationWarningKind.UNUSUAL_ROLE_PATTERN,
                            message=f"Consecutive assistant messages at index {i}",
                            line_number=line_number,
                            sample_index=sample_index,
                        )
                    )

            if role != "system":
                prev_role = role

        # Check for duplicate system messages
        if system_count > 1:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.DUPLICATE_SYSTEM_MESSAGE,
                    message=f"Found {system_count} system messages (expected 0 or 1)",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

        # Check for required roles
        if not has_user:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.MISSING_USER_MESSAGE,
                    message="No user message in conversation",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

        if not has_assistant:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.MISSING_ASSISTANT_MESSAGE,
                    message="No assistant message in conversation",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

        # Warn if no system prompt
        if not has_system:
            warnings.append(
                ValidationWarning(
                    kind=ValidationWarningKind.MISSING_SYSTEM_PROMPT,
                    message="No system prompt (optional but recommended)",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

    def _validate_tools(
        self,
        sample: dict[str, Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate tools format sample."""
        # First validate as chat
        self._validate_chat(sample, line_number, sample_index, errors, warnings)

        # Then validate tools array
        tools = sample.get("tools", [])
        if not isinstance(tools, list):
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.MISSING_REQUIRED_FIELD,
                    message="'tools' must be an array",
                    line_number=line_number,
                    field_name="tools",
                    sample_index=sample_index,
                )
            )
            return

        # Validate tool call/response pairing
        messages = sample.get("messages", [])
        if isinstance(messages, list):
            self._validate_tool_calls(
                messages, line_number, sample_index, errors, warnings
            )

    def _validate_tool_calls(
        self,
        messages: list[Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate tool call/response pairing."""
        pending_tool_calls: set[str] = set()

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")

            # Track tool calls
            if "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict) and "id" in tc:
                            pending_tool_calls.add(tc["id"])

            # Track tool responses
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id in pending_tool_calls:
                    pending_tool_calls.remove(tool_call_id)
                else:
                    errors.append(
                        ValidationError(
                            kind=ValidationErrorKind.ORPHAN_TOOL_RESPONSE,
                            message=f"Tool response at index {i} has no matching call",
                            line_number=line_number,
                            sample_index=sample_index,
                        )
                    )

        # Check for unanswered tool calls
        if pending_tool_calls:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.TOOL_CALL_WITHOUT_RESPONSE,
                    message=f"{len(pending_tool_calls)} tool call(s) without response",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

    def _validate_instruction(
        self,
        sample: dict[str, Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate instruction format sample."""
        # Check for instruction or input field
        instruction = sample.get("instruction") or sample.get("input", "")
        output = sample.get("output", "")

        if not instruction:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.MISSING_REQUIRED_FIELD,
                    message="Missing 'instruction' or 'input' field",
                    line_number=line_number,
                    sample_index=sample_index,
                )
            )

        if not output:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.EMPTY_CONTENT,
                    message="'output' field is empty",
                    line_number=line_number,
                    field_name="output",
                    sample_index=sample_index,
                )
            )

    def _validate_completion(
        self,
        sample: dict[str, Any],
        line_number: Optional[int],
        sample_index: Optional[int],
        errors: list[ValidationError],
        warnings: list[ValidationWarning],
    ) -> None:
        """Validate completion format sample."""
        prompt = sample.get("prompt", "")
        completion = sample.get("completion", "")

        if not prompt:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.EMPTY_CONTENT,
                    message="'prompt' field is empty",
                    line_number=line_number,
                    field_name="prompt",
                    sample_index=sample_index,
                )
            )

        if not completion:
            errors.append(
                ValidationError(
                    kind=ValidationErrorKind.EMPTY_CONTENT,
                    message="'completion' field is empty",
                    line_number=line_number,
                    field_name="completion",
                    sample_index=sample_index,
                )
            )

    def detect_format_from_samples(
        self, samples: list[dict[str, Any]]
    ) -> tuple[DatasetContentFormat, float]:
        """Detect format from multiple samples with voting.

        Args:
            samples: List of parsed samples.

        Returns:
            Tuple of (detected format, confidence).
        """
        if not samples:
            return DatasetContentFormat.UNKNOWN, 0.0

        format_counts: dict[DatasetContentFormat, int] = {}

        for sample in samples:
            fmt = self.detect_format(sample)
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        # Find majority format
        best_format = DatasetContentFormat.UNKNOWN
        best_count = 0
        for fmt, count in format_counts.items():
            if count > best_count:
                best_format = fmt
                best_count = count

        confidence = best_count / len(samples) if samples else 0.0

        return best_format, confidence
