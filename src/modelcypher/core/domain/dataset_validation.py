from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DatasetContentFormat(str, Enum):
    text = "text"
    chat = "chat"
    tools = "tools"
    instruction = "instruction"
    completion = "completion"
    unknown = "unknown"


@dataclass(frozen=True)
class ValidationError:
    kind: str
    line: int | None = None
    field: str | None = None
    sample: str | None = None
    count: int | None = None
    minimum: int | None = None
    length: int | None = None
    limit: int | None = None
    message_index: int | None = None
    role: str | None = None
    expected: str | None = None
    got: str | None = None

    @property
    def message(self) -> str:
        if self.kind == "notValidJSON":
            prefix = (self.sample or "")[:50]
            return f"Line {self.line}: Not valid JSON - {prefix}..."
        if self.kind == "missingRequiredField":
            return f"Line {self.line}: Missing required field '{self.field}'"
        if self.kind == "emptyContent":
            return f"Line {self.line}: Field '{self.field}' is empty"
        if self.kind == "invalidEncoding":
            return f"Line {self.line}: Invalid UTF-8 encoding"
        if self.kind == "markdownDetected":
            prefix = (self.sample or "")[:30]
            return f"Line {self.line}: Markdown detected (expected JSONL) - {prefix}..."
        if self.kind == "tooFewSamples":
            return f"Only {self.count} samples found, need at least {self.minimum} for training"
        if self.kind == "emptyFile":
            return "File is empty or contains no valid samples"
        if self.kind == "lineTooLarge":
            return (
                f"Line {self.line} exceeds safety limit ({self.length} bytes, limit {self.limit} bytes)"
            )
        if self.kind == "emptyMessageContent":
            return (
                f"Line {self.line}: Message #{self.message_index} ({self.role}) has empty content"
            )
        if self.kind == "missingSystemPrompt":
            return f"Line {self.line}: Missing system prompt in messages"
        if self.kind == "invalidRoleAlternation":
            return (
                f"Line {self.line}: Message #{self.message_index} expected role '{self.expected}' "
                f"but got '{self.got}'"
            )
        if self.kind == "firstMessageNotSystem":
            return f"Line {self.line}: First message must use the 'system' role"
        if self.kind == "lastMessageNotAssistant":
            return f"Line {self.line}: Last message must be from the assistant role"
        return "Validation error"


class DatasetFormatAnalyzer:
    def detect_format(self, json_obj: dict[str, Any]) -> DatasetContentFormat:
        if "text" in json_obj:
            return DatasetContentFormat.text
        if "tools" in json_obj:
            return DatasetContentFormat.tools
        messages = json_obj.get("messages")
        if isinstance(messages, list) and any(isinstance(m, dict) and "tool_calls" in m for m in messages):
            return DatasetContentFormat.tools
        if "messages" in json_obj:
            return DatasetContentFormat.chat
        if "prompt" in json_obj and "completion" in json_obj:
            return DatasetContentFormat.completion
        if "instruction" in json_obj and "output" in json_obj:
            return DatasetContentFormat.instruction
        return DatasetContentFormat.unknown

    def validate_format(
        self,
        json_obj: dict[str, Any],
        expected_format: DatasetContentFormat,
        line_number: int,
    ) -> list[ValidationError]:
        errors: list[ValidationError] = []

        if expected_format == DatasetContentFormat.text:
            if "text" not in json_obj:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="text"))
            else:
                text = json_obj.get("text")
                if isinstance(text, str) and not text.strip():
                    errors.append(ValidationError("emptyContent", line=line_number, field="text"))
        elif expected_format == DatasetContentFormat.chat:
            messages = json_obj.get("messages")
            if messages is None:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="messages"))
            elif isinstance(messages, list) and not messages:
                errors.append(
                    ValidationError("missingRequiredField", line=line_number, field="messages (at least one message)")
                )
            elif isinstance(messages, list):
                errors.extend(self._validate_chat_semantics(messages, line_number))
        elif expected_format == DatasetContentFormat.tools:
            messages = json_obj.get("messages")
            if messages is None:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="messages"))
            elif isinstance(messages, list) and not messages:
                errors.append(
                    ValidationError("missingRequiredField", line=line_number, field="messages (at least one message)")
                )
            elif isinstance(messages, list):
                errors.extend(self._validate_chat_semantics(messages, line_number))
            tools = json_obj.get("tools")
            if tools is None:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="tools"))
            elif isinstance(tools, list) and not tools:
                errors.append(
                    ValidationError("missingRequiredField", line=line_number, field="tools (at least one tool)")
                )
        elif expected_format == DatasetContentFormat.instruction:
            if "instruction" not in json_obj:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="instruction"))
            else:
                instruction = json_obj.get("instruction")
                if isinstance(instruction, str) and not instruction.strip():
                    errors.append(ValidationError("emptyContent", line=line_number, field="instruction"))
            if "output" not in json_obj:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="output"))
            else:
                output = json_obj.get("output")
                if isinstance(output, str) and not output.strip():
                    errors.append(ValidationError("emptyContent", line=line_number, field="output"))
        elif expected_format == DatasetContentFormat.completion:
            if "prompt" not in json_obj:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="prompt"))
            else:
                prompt = json_obj.get("prompt")
                if isinstance(prompt, str) and not prompt.strip():
                    errors.append(ValidationError("emptyContent", line=line_number, field="prompt"))
            if "completion" not in json_obj:
                errors.append(ValidationError("missingRequiredField", line=line_number, field="completion"))
            else:
                completion = json_obj.get("completion")
                if isinstance(completion, str) and not completion.strip():
                    errors.append(ValidationError("emptyContent", line=line_number, field="completion"))
        else:
            errors.append(
                ValidationError(
                    "missingRequiredField",
                    line=line_number,
                    field='unknown format - expected {"text": "..."}',
                )
            )

        return errors

    def required_string_fields(self, dataset_format: DatasetContentFormat) -> list[str]:
        if dataset_format == DatasetContentFormat.text:
            return ["text"]
        if dataset_format == DatasetContentFormat.instruction:
            return ["instruction", "output"]
        if dataset_format == DatasetContentFormat.completion:
            return ["prompt", "completion"]
        return []

    def _validate_chat_semantics(self, messages: list[Any], line_number: int) -> list[ValidationError]:
        normalized: list[tuple[str | None, str | None]] = []
        for message in messages:
            if not isinstance(message, dict):
                normalized.append((None, None))
                continue
            role_raw = message.get("role")
            role = role_raw.strip().lower() if isinstance(role_raw, str) else None
            content = message.get("content")
            if content is None:
                content = message.get("text")
            if content is None:
                content = message.get("value")
            content = content.strip() if isinstance(content, str) else None
            role = role if role else None
            content = content if content else None
            normalized.append((role, content))

        errors: list[ValidationError] = []
        for index, (role, content) in enumerate(normalized, start=1):
            role_label = role or "unknown"
            if not content:
                errors.append(
                    ValidationError(
                        "emptyMessageContent",
                        line=line_number,
                        message_index=index,
                        role=role_label,
                    )
                )

        roles = [role for role, _ in normalized]
        if not any(role == "system" for role in roles):
            errors.append(ValidationError("missingSystemPrompt", line=line_number))

        if roles and roles[0] != "system":
            errors.append(ValidationError("firstMessageNotSystem", line=line_number))

        if roles and roles[-1] != "assistant":
            errors.append(ValidationError("lastMessageNotAssistant", line=line_number))

        start_index = 1 if roles and roles[0] == "system" else 0
        expected_role = "user"
        for index in range(start_index, len(roles)):
            actual_role = roles[index] or "unknown"
            if actual_role != expected_role:
                errors.append(
                    ValidationError(
                        "invalidRoleAlternation",
                        line=line_number,
                        message_index=index + 1,
                        expected=expected_role,
                        got=actual_role,
                    )
                )
                expected_role = "assistant" if expected_role == "user" else "user"
                continue
            expected_role = "assistant" if expected_role == "user" else "user"

        return errors
