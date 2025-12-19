from __future__ import annotations

import json
from typing import Any

from modelcypher.core.domain.chat_template import ChatMessage, ChatTemplateEngine
from modelcypher.core.domain.dataset_validation import DatasetContentFormat, DatasetFormatAnalyzer


class DatasetExportFormatterError(ValueError):
    pass


def normalized_line(
    raw_line: str,
    format_hint: DatasetContentFormat,
    target_format: DatasetContentFormat | None = None,
) -> str:
    try:
        json_obj = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise DatasetExportFormatterError("Dataset row is not valid JSON.") from exc
    if not isinstance(json_obj, dict):
        raise DatasetExportFormatterError("Dataset row is not valid JSON.")

    detected = _detect_format(json_obj, format_hint)
    desired = target_format or detected
    payload = _normalized_payload(
        json_obj,
        source_format=detected,
        target_format=desired,
        original_line=raw_line,
    )
    if not payload:
        raise DatasetExportFormatterError("Dataset row is empty after normalization.")

    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise DatasetExportFormatterError("Unable to encode normalized row.") from exc
    return encoded


def _detect_format(obj: dict[str, Any], hint: DatasetContentFormat) -> DatasetContentFormat:
    if hint != DatasetContentFormat.unknown:
        return hint
    analyzer = DatasetFormatAnalyzer()
    detected = analyzer.detect_format(obj)
    return DatasetContentFormat.text if detected == DatasetContentFormat.unknown else detected


def _normalized_payload(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
    target_format: DatasetContentFormat,
    original_line: str,
) -> dict[str, Any]:
    if target_format == DatasetContentFormat.text:
        text = _render_text(obj, source_format=source_format, original_line=original_line)
        return {"text": text}
    if target_format == DatasetContentFormat.completion:
        prompt, completion = _render_completion(obj, source_format=source_format)
        return {"completion": completion, "prompt": prompt}
    if target_format == DatasetContentFormat.chat:
        messages = _render_messages(obj, source_format=source_format)
        return {"messages": messages}
    if target_format == DatasetContentFormat.tools:
        messages, tools = _render_tools(obj, source_format=source_format)
        return {"messages": messages, "tools": tools}
    if target_format == DatasetContentFormat.instruction:
        return _render_instruction_payload(obj, source_format=source_format)
    if target_format == DatasetContentFormat.unknown:
        text = _extract_text(original_line)
        if not text:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return {"text": text}
    raise DatasetExportFormatterError(
        f"Cannot convert {source_format.value} format to {target_format.value}."
    )


def _render_text(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
    original_line: str,
) -> str:
    if source_format == DatasetContentFormat.text:
        if "text" not in obj:
            raise DatasetExportFormatterError("Dataset row is missing required field 'text'.")
        text = obj.get("text")
        if not isinstance(text, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'text'.")
        trimmed = text.strip()
        if not trimmed:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return trimmed
    if source_format == DatasetContentFormat.instruction:
        instruction = obj.get("instruction")
        output = obj.get("output")
        if not isinstance(instruction, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'instruction'.")
        if not isinstance(output, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'output'.")
        sections = ["### Instruction", instruction]
        input_text = obj.get("input")
        if isinstance(input_text, str) and input_text.strip():
            sections.extend(["### Input", input_text])
        sections.extend(["### Output", output])
        joined = "\n\n".join(sections)
        if not joined.strip():
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return joined
    if source_format == DatasetContentFormat.completion:
        prompt = obj.get("prompt")
        completion = obj.get("completion")
        if not isinstance(prompt, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'prompt'.")
        if not isinstance(completion, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'completion'.")
        merged = f"{prompt}\n\n### Response\n{completion}"
        if not merged.strip():
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return merged
    if source_format in {DatasetContentFormat.chat, DatasetContentFormat.tools}:
        messages = _messages_array(obj)
        chat_messages: list[ChatMessage] = []
        for message in messages:
            role = message.get("role")
            if not isinstance(role, str):
                continue
            content = message.get("content")
            if content is None:
                content = message.get("text")
            if content is None:
                content = message.get("value")
            if not isinstance(content, str):
                content = ""
            chat_messages.append(ChatMessage(role=role, content=content))
        formatted = ChatTemplateEngine.apply_template(chat_messages)
        trimmed = formatted.strip()
        if not trimmed:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return trimmed
    if source_format == DatasetContentFormat.unknown:
        text = _extract_text(original_line)
        if not text:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return text
    raise DatasetExportFormatterError(
        f"Cannot convert {source_format.value} format to {DatasetContentFormat.text.value}."
    )


def _render_completion(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
) -> tuple[str, str]:
    if source_format == DatasetContentFormat.completion:
        prompt = obj.get("prompt")
        completion = obj.get("completion")
        if not isinstance(prompt, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'prompt'.")
        if not isinstance(completion, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'completion'.")
        trimmed_prompt = prompt.strip()
        trimmed_completion = completion.strip()
        if not trimmed_prompt or not trimmed_completion:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return trimmed_prompt, trimmed_completion
    if source_format == DatasetContentFormat.instruction:
        instruction = obj.get("instruction")
        output = obj.get("output")
        if not isinstance(instruction, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'instruction'.")
        if not isinstance(output, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'output'.")
        input_text = obj.get("input")
        if isinstance(input_text, str) and input_text.strip():
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        trimmed_prompt = prompt.strip()
        trimmed_completion = output.strip()
        if not trimmed_prompt or not trimmed_completion:
            raise DatasetExportFormatterError("Dataset row is empty after normalization.")
        return trimmed_prompt, trimmed_completion
    raise DatasetExportFormatterError(
        f"Cannot convert {source_format.value} format to {DatasetContentFormat.completion.value}."
    )


def _render_messages(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
) -> list[dict[str, Any]]:
    if source_format in {DatasetContentFormat.chat, DatasetContentFormat.tools}:
        return _messages_array(obj)
    if source_format == DatasetContentFormat.completion:
        prompt = obj.get("prompt")
        completion = obj.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'prompt/completion'.")
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    if source_format == DatasetContentFormat.instruction:
        instruction = obj.get("instruction")
        output = obj.get("output")
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'instruction/output'.")
        input_text = obj.get("input")
        if isinstance(input_text, str) and input_text.strip():
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
        return [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output},
        ]
    if source_format == DatasetContentFormat.text:
        text = obj.get("text")
        if not isinstance(text, str):
            raise DatasetExportFormatterError("Dataset row is missing required field 'text'.")
        return [{"role": "user", "content": text}]
    raise DatasetExportFormatterError(
        f"Cannot convert {source_format.value} format to {DatasetContentFormat.chat.value}."
    )


def _render_tools(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
) -> tuple[list[dict[str, Any]], list[Any]]:
    if source_format != DatasetContentFormat.tools:
        raise DatasetExportFormatterError(
            f"Cannot convert {source_format.value} format to {DatasetContentFormat.tools.value}."
        )
    messages = _messages_array(obj)
    tools = obj.get("tools")
    if tools is None:
        raise DatasetExportFormatterError("Dataset row is missing required field 'tools'.")
    if isinstance(tools, list) and not tools:
        raise DatasetExportFormatterError("Dataset row is missing required field 'tools (at least one tool)'.")
    if not isinstance(tools, list):
        raise DatasetExportFormatterError("Dataset row is missing required field 'tools'.")
    return messages, tools


def _render_instruction_payload(
    obj: dict[str, Any],
    source_format: DatasetContentFormat,
) -> dict[str, Any]:
    if source_format != DatasetContentFormat.instruction:
        raise DatasetExportFormatterError(
            f"Cannot convert {source_format.value} format to {DatasetContentFormat.instruction.value}."
        )
    instruction = obj.get("instruction")
    output = obj.get("output")
    if not isinstance(instruction, str):
        raise DatasetExportFormatterError("Dataset row is missing required field 'instruction'.")
    if not isinstance(output, str):
        raise DatasetExportFormatterError("Dataset row is missing required field 'output'.")
    payload: dict[str, Any] = {"instruction": instruction, "output": output}
    if "input" in obj:
        payload["input"] = obj.get("input")
    return payload


def _messages_array(obj: dict[str, Any]) -> list[dict[str, Any]]:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        raise DatasetExportFormatterError("Dataset row is missing required field 'messages'.")
    if not messages:
        raise DatasetExportFormatterError("Dataset row is missing required field 'messages (at least one message)'.")
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append(message)
    return normalized


def _extract_text(raw: str) -> str:
    try:
        json_obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if not isinstance(json_obj, dict):
        return raw.strip()

    if isinstance(json_obj.get("text"), str):
        return json_obj["text"].strip()
    if isinstance(json_obj.get("completion"), str):
        return json_obj["completion"].strip()
    prompt = json_obj.get("prompt")
    completion = json_obj.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        return f"{prompt.strip()}\n{completion.strip()}"
    messages = json_obj.get("messages")
    if isinstance(messages, list):
        joined = "\n".join(
            message.get("content", "").strip()
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
        if joined:
            return joined
    return raw.strip()
