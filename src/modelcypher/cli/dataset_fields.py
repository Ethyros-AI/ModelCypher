from __future__ import annotations

import json

import typer

from modelcypher.core.domain.dataset_validation import DatasetContentFormat


def parse_format(raw: str) -> DatasetContentFormat:
    value = raw.lower()
    if value == "text":
        return DatasetContentFormat.text
    if value == "chat":
        return DatasetContentFormat.chat
    if value == "completion":
        return DatasetContentFormat.completion
    if value == "tools":
        return DatasetContentFormat.tools
    if value == "instruction":
        return DatasetContentFormat.instruction
    raise typer.BadParameter(
        "Unsupported format. Use text, chat, completion, tools, or instruction."
    )


def parse_fields(raw: str, argument: str) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Invalid JSON for {argument}. Example: {argument} '{{\"text\": \"hello\"}}'"
        ) from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"Invalid JSON for {argument}. Value must be a JSON object.")
    return payload


def pretty_fields(fields: dict) -> str:
    return json.dumps(fields, indent=2, sort_keys=True, ensure_ascii=True)


def preview_line(raw: str, limit: int = 120) -> str:
    single_line = raw.replace("\n", " ")
    if len(single_line) <= limit:
        return single_line
    return single_line[:limit] + "..."
