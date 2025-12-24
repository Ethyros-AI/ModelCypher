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

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from modelcypher.core.domain.dataset_file_enumerator import DatasetFileEnumerator, DatasetLineTooLargeError
from modelcypher.core.domain.dataset_validation import DatasetContentFormat, DatasetFormatAnalyzer, ValidationError


@dataclass(frozen=True)
class DatasetStats:
    avg_sample_length: int
    min_length: int
    max_length: int
    schema_consistency: float
    duplicate_count: int


@dataclass(frozen=True)
class DatasetValidationResult:
    is_valid: bool
    format: DatasetContentFormat
    sample_count: int
    errors: list[ValidationError]
    warnings: list[str]
    stats: DatasetStats


class DatasetValidator:
    def __init__(self, enumerator: DatasetFileEnumerator | None = None) -> None:
        self.enumerator = enumerator or DatasetFileEnumerator(max_line_bytes=2 * 1024 * 1024)
        self.format_analyzer = DatasetFormatAnalyzer()

    def validate(self, path: Path) -> DatasetValidationResult:
        errors: list[ValidationError] = []
        warnings: list[str] = []
        detected_format = DatasetContentFormat.unknown
        total_length = 0
        min_length = None
        max_length = 0
        schema_reference: list[str] | None = None
        schema_match_count = 0
        schema_sample_count = 0
        seen_hashes: set[bytes] = set()
        duplicate_count = 0
        invalid_json_reports = 0
        format_error_reports = 0
        sample_count = 0

        def record_invalid_json(line_number: int, sample: str) -> None:
            nonlocal invalid_json_reports
            if invalid_json_reports >= 10:
                return
            errors.append(ValidationError("notValidJSON", line=line_number, sample=sample))
            invalid_json_reports += 1

        def record_format_errors(new_errors: list[ValidationError]) -> None:
            nonlocal format_error_reports
            if format_error_reports >= 10:
                return
            remaining = max(0, 10 - format_error_reports)
            if remaining == 0:
                return
            limited = new_errors[:remaining]
            errors.extend(limited)
            format_error_reports += len(limited)

        def record_length_metrics(length: int) -> None:
            nonlocal total_length, min_length, max_length
            total_length += length
            if min_length is None or length < min_length:
                min_length = length
            if length > max_length:
                max_length = length

        def record_schema(keys: list[str]) -> None:
            nonlocal schema_reference, schema_match_count, schema_sample_count
            schema_sample_count += 1
            if schema_reference is None:
                schema_reference = keys
                schema_match_count += 1
                return
            if keys == schema_reference:
                schema_match_count += 1

        def record_duplicate(line: str) -> None:
            nonlocal duplicate_count
            digest = hashlib.blake2b(line.encode("utf-8"), digest_size=16).digest()
            if digest in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(digest)

        def process(record) -> bool:
            nonlocal detected_format, sample_count
            sample_count += 1
            try:
                line = record.data.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(ValidationError("invalidEncoding", line=record.line_number))
                return True

            trimmed = line.strip()
            if trimmed.startswith("#"):
                errors.append(
                    ValidationError("markdownDetected", line=record.line_number, sample=trimmed[:50])
                )
                return True

            record_duplicate(trimmed)
            record_length_metrics(len(trimmed))

            try:
                json_obj = json.loads(trimmed)
            except json.JSONDecodeError:
                record_invalid_json(record.line_number, trimmed[:100])
                return True

            if not isinstance(json_obj, dict):
                record_invalid_json(record.line_number, trimmed[:100])
                return True

            sample_format = self.format_analyzer.detect_format(json_obj)
            if detected_format == DatasetContentFormat.unknown:
                detected_format = sample_format

            record_schema(sorted(json_obj.keys()))
            format_errors = self.format_analyzer.validate_format(
                json_obj,
                expected_format=detected_format,
                line_number=record.line_number,
            )
            record_format_errors(format_errors)

            for field in self.format_analyzer.required_string_fields(detected_format):
                value = json_obj.get(field)
                if isinstance(value, str) and not value.strip():
                    errors.append(ValidationError("emptyContent", line=record.line_number, field=field))

            return True

        try:
            self.enumerator.enumerate_lines(str(path), process)
        except DatasetLineTooLargeError as exc:
            stats = DatasetStats(
                avg_sample_length=0,
                min_length=0,
                max_length=0,
                schema_consistency=0.0,
                duplicate_count=0,
            )
            return DatasetValidationResult(
                is_valid=False,
                format=DatasetContentFormat.unknown,
                sample_count=0,
                errors=[
                    ValidationError(
                        "lineTooLarge",
                        line=exc.line_number,
                        length=exc.length,
                        limit=exc.limit,
                    )
                ],
                warnings=[],
                stats=stats,
            )
        except ValueError:
            errors.append(ValidationError("notValidJSON", line=0, sample="Enumeration failed"))

        if sample_count == 0:
            stats = DatasetStats(
                avg_sample_length=0,
                min_length=0,
                max_length=0,
                schema_consistency=0.0,
                duplicate_count=0,
            )
            return DatasetValidationResult(
                is_valid=False,
                format=DatasetContentFormat.unknown,
                sample_count=0,
                errors=[ValidationError("emptyFile")],
                warnings=[],
                stats=stats,
            )

        avg_length = total_length // sample_count if sample_count > 0 else 0
        resolved_min = min_length if min_length is not None else 0
        schema_consistency = (
            float(schema_match_count) / float(schema_sample_count)
            if schema_sample_count > 0
            else 0.0
        )

        if sample_count < 50:
            warnings.append(
                f"{sample_count} samples - recommended minimum is 50+ for meaningful training"
            )
        if avg_length < 50:
            warnings.append(f"Average length {avg_length} chars is short (recommended 100+)")
        if schema_consistency < 0.95:
            warnings.append(
                f"Schema consistency {int(schema_consistency * 100)}% - some samples have different fields"
            )
        duplicate_percentage = float(duplicate_count) / float(sample_count) if sample_count else 0.0
        if duplicate_percentage > 0.1:
            warnings.append(
                f"{duplicate_count} duplicate samples ({int(duplicate_percentage * 100)}%) detected"
            )

        stats = DatasetStats(
            avg_sample_length=avg_length,
            min_length=resolved_min,
            max_length=max_length,
            schema_consistency=schema_consistency,
            duplicate_count=duplicate_count,
        )

        return DatasetValidationResult(
            is_valid=len(errors) == 0,
            format=detected_format,
            sample_count=sample_count,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
