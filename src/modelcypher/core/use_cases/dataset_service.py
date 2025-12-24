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

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.core.domain.dataset_validator import DatasetValidator
from modelcypher.core.domain.dataset_validation import DatasetContentFormat
from modelcypher.core.domain.models import DatasetInfo
from modelcypher.utils.paths import expand_path

if TYPE_CHECKING:
    from modelcypher.ports.storage import DatasetStore


class DatasetService:
    def __init__(self, store: "DatasetStore") -> None:
        self.store = store

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
        if not resolved.is_file():
            return {
                "valid": False,
                "totalExamples": 0,
                "averageTokens": 0.0,
                "maxTokens": 0,
                "minTokens": 0,
                "errors": ["Path is not a regular file (directories not supported)"],
                "warnings": [],
            }

        validator = DatasetValidator()
        result = validator.validate(resolved)

        def round_half_away_from_zero(value: float) -> int:
            if value >= 0:
                return int(math.floor(value + 0.5))
            return int(math.ceil(value - 0.5))

        def token_estimate(length: int) -> int:
            if length <= 0:
                return 0
            return max(1, round_half_away_from_zero(length / 4.0))

        has_samples = result.sample_count > 0
        avg_tokens = token_estimate(result.stats.avg_sample_length) if has_samples else 0
        max_tokens = token_estimate(result.stats.max_length) if has_samples else 0
        min_tokens = token_estimate(result.stats.min_length) if has_samples else 0

        errors = [err.message for err in result.errors]
        warnings = result.warnings
        is_valid = result.is_valid

        if is_valid and result.format != DatasetContentFormat.text:
            errors.append("Missing required field 'text'")
            is_valid = False

        if is_valid:
            try:
                self.register_dataset(path)
            except Exception:
                pass

        return {
            "valid": is_valid,
            "totalExamples": result.sample_count,
            "averageTokens": float(avg_tokens),
            "maxTokens": max_tokens,
            "minTokens": min_tokens,
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
