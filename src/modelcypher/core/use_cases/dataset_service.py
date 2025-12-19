from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from modelcypher.adapters.filesystem_storage import FileSystemStore
from modelcypher.core.domain.models import DatasetInfo
from modelcypher.utils.paths import expand_path


class DatasetService:
    def __init__(self, store: FileSystemStore | None = None) -> None:
        self.store = store or FileSystemStore()

    def validate_dataset(self, path: str) -> dict:
        resolved = expand_path(path)
        if not resolved.exists():
            return {
                "valid": False,
                "totalExamples": 0,
                "averageTokens": 0.0,
                "maxTokens": 0,
                "minTokens": 0,
                "errors": [f"File not found: {path}"],
                "warnings": [],
            }

        total = 0
        token_counts: list[int] = []
        errors: list[str] = []
        warnings: list[str] = []

        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"Invalid JSON on line {line_number}")
                    continue
                text = payload.get("text")
                if text is None and "messages" in payload:
                    text = " ".join(msg.get("content", "") for msg in payload["messages"])
                if not text:
                    warnings.append(f"Empty text on line {line_number}")
                    continue
                total += 1
                token_counts.append(len(str(text).split()))

        if not token_counts:
            return {
                "valid": False,
                "totalExamples": 0,
                "averageTokens": 0.0,
                "maxTokens": 0,
                "minTokens": 0,
                "errors": errors or ["No valid samples found"],
                "warnings": warnings,
            }

        average = sum(token_counts) / len(token_counts)
        return {
            "valid": len(errors) == 0,
            "totalExamples": total,
            "averageTokens": average,
            "maxTokens": max(token_counts),
            "minTokens": min(token_counts),
            "errors": errors,
            "warnings": warnings,
        }

    def preprocess_dataset(self, input_path: str, output_path: str, tokenizer_model: str) -> dict:
        resolved_input = expand_path(input_path)
        resolved_output = expand_path(output_path)
        processed = 0
        skipped = 0
        total_tokens = 0

        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        with resolved_input.open("r", encoding="utf-8") as handle, resolved_output.open(
            "w", encoding="utf-8"
        ) as out:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                text = payload.get("text")
                if text is None and "messages" in payload:
                    text = " ".join(msg.get("content", "") for msg in payload["messages"])
                if not text:
                    skipped += 1
                    continue
                tokens = len(str(text).split())
                total_tokens += tokens
                processed += 1
                out.write(json.dumps({"text": text, "tokens": tokens}) + "\n")

        return {
            "processedExamples": processed,
            "skippedExamples": skipped,
            "outputPath": str(resolved_output),
            "totalTokens": total_tokens,
            "tokenizer": tokenizer_model,
        }

    def register_dataset(self, path: str) -> DatasetInfo:
        resolved = expand_path(path)
        info = DatasetInfo(
            id=str(uuid.uuid4()),
            name=resolved.stem,
            path=str(resolved),
            size_bytes=resolved.stat().st_size,
            example_count=self._count_lines(resolved),
            created_at=datetime.utcnow(),
        )
        self.store.register_dataset(info)
        return info

    def list_datasets(self) -> list[DatasetInfo]:
        return self.store.list_datasets()

    def delete_dataset(self, path: str) -> None:
        resolved = expand_path(path)
        if resolved.exists():
            resolved.unlink()
        datasets = self.store.list_datasets()
        for dataset in datasets:
            if dataset.path == str(resolved) or dataset.id == path or dataset.name == path:
                self.store.delete_dataset(dataset.id)

    @staticmethod
    def _count_lines(path: Path) -> int:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
