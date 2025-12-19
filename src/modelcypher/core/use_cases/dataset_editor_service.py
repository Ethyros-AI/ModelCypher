from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcypher.core.domain.dataset_export_formatter import DatasetExportFormatterError, normalized_line
from modelcypher.core.domain.dataset_file_enumerator import DatasetFileEnumerator
from modelcypher.core.domain.dataset_validation import DatasetContentFormat, DatasetFormatAnalyzer
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.utils.limits import MAX_FIELD_BYTES, MAX_PREVIEW_LINES, MAX_RAW_BYTES


@dataclass(frozen=True)
class DatasetRowSnapshot:
    line_number: int
    raw: str
    format: DatasetContentFormat
    fields: dict[str, Any]
    validation_messages: list[str]
    raw_truncated: bool
    raw_full_bytes: int
    fields_truncated: list[str]


@dataclass(frozen=True)
class DatasetEditResult:
    status: str
    line_number: int | None
    row: DatasetRowSnapshot | None
    warnings: list[str]


@dataclass(frozen=True)
class DatasetConversionResult:
    source_path: str
    output_path: str
    target_format: DatasetContentFormat
    line_count: int
    warnings: list[str]


@dataclass(frozen=True)
class DatasetPreviewResult:
    path: str
    rows: list[DatasetRowSnapshot]


class DatasetEditorError(ValueError):
    pass


class DatasetEditorService:
    def __init__(
        self,
        format_analyzer: DatasetFormatAnalyzer | None = None,
        file_enumerator: DatasetFileEnumerator | None = None,
        job_service: JobService | None = None,
    ) -> None:
        self.format_analyzer = format_analyzer or DatasetFormatAnalyzer()
        self.file_enumerator = file_enumerator or DatasetFileEnumerator()
        self.job_service = job_service or JobService()
        self._check_active_jobs = os.environ.get("TC_DISABLE_ACTIVE_JOB_CHECK") != "1"

    def get_row(self, path: str, line_number: int) -> DatasetRowSnapshot:
        url = self._existing_path(path)
        if line_number <= 0:
            raise DatasetEditorError(f"Invalid line number: {line_number}")

        result: DatasetRowSnapshot | None = None

        def process(record):
            nonlocal result
            if record.line_number == line_number:
                raw_line = record.data.decode("utf-8")
                result = self._snapshot(raw_line, line_number)
                return False
            return True

        try:
            self.file_enumerator.enumerate_lines(url, process)
        except UnicodeDecodeError as exc:
            raise DatasetEditorError(f"Line {line_number} is not UTF-8 encoded.") from exc

        if result is None:
            raise DatasetEditorError(f"Invalid line number: {line_number}")
        return result

    def update_row(self, path: str, line_number: int, content: dict[str, Any]) -> DatasetEditResult:
        url = self._existing_path(path)
        if line_number <= 0:
            raise DatasetEditorError(f"Invalid line number: {line_number}")

        format_hint = self._detect_format(content)
        normalized = self._normalized_line_from_fields(content, format_hint, target_format=None)
        self._replace_line(url, line_number, normalized["line"])

        snapshot = self._snapshot(normalized["line"], line_number)
        warnings = self._active_job_warnings(url)
        return DatasetEditResult(status="updated", line_number=line_number, row=snapshot, warnings=warnings)

    def add_row(
        self,
        path: str,
        format_hint: DatasetContentFormat,
        fields: dict[str, Any],
    ) -> DatasetEditResult:
        url = self._normalized_path(path)
        normalized = self._normalized_line_from_fields(fields, format_hint, target_format=format_hint)
        new_line = self._append_line(url, normalized["line"])
        snapshot = self._snapshot(normalized["line"], new_line)
        warnings = self._active_job_warnings(url)
        return DatasetEditResult(status="added", line_number=new_line, row=snapshot, warnings=warnings)

    def delete_row(self, path: str, line_number: int) -> DatasetEditResult:
        url = self._existing_path(path)
        if line_number <= 0:
            raise DatasetEditorError(f"Invalid line number: {line_number}")
        self._remove_line(url, line_number)
        warnings = self._active_job_warnings(url)
        return DatasetEditResult(status="deleted", line_number=line_number, row=None, warnings=warnings)

    def convert_dataset(
        self,
        path: str,
        target_format: DatasetContentFormat,
        output_path: str,
    ) -> DatasetConversionResult:
        input_path = self._existing_path(path)
        output = self._normalized_path(output_path)
        temp_path = self._temporary_path(output)
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        lines_written = 0
        try:
            with temp_path.open("w", encoding="utf-8") as writer:
                def process(record):
                    nonlocal lines_written
                    raw = record.data.decode("utf-8")
                    normalized = self._normalized_line_from_raw(
                        raw,
                        format_hint=DatasetContentFormat.unknown,
                        target_format=target_format,
                    )
                    writer.write(normalized["line"] + "\n")
                    lines_written += 1
                    return True

                self.file_enumerator.enumerate_lines(input_path, process)
        except (UnicodeDecodeError, DatasetExportFormatterError, DatasetEditorError, ValueError) as exc:
            if temp_path.exists():
                temp_path.unlink()
            raise DatasetEditorError(str(exc)) from exc

        self._replace_file(output, temp_path)
        warnings = self._active_job_warnings(input_path)
        return DatasetConversionResult(
            source_path=str(input_path),
            output_path=str(output),
            target_format=target_format,
            line_count=lines_written,
            warnings=warnings,
        )

    def preview(self, path: str, limit: int) -> DatasetPreviewResult:
        url = self._existing_path(path)
        row_limit = max(1, min(limit, MAX_PREVIEW_LINES))
        rows: list[DatasetRowSnapshot] = []

        def process(record):
            try:
                raw = record.data.decode("utf-8")
            except UnicodeDecodeError:
                return True
            rows.append(self._snapshot(raw, record.line_number))
            return len(rows) < row_limit

        self.file_enumerator.enumerate_lines(url, process)
        return DatasetPreviewResult(path=str(url), rows=rows)

    def _existing_path(self, path: str) -> Path:
        resolved = self._normalized_path(path)
        if not resolved.exists():
            raise DatasetEditorError(f"Dataset not found at {resolved}")
        return resolved

    def _normalized_path(self, path: str) -> Path:
        return Path(path).expanduser().resolve()

    def _snapshot(self, raw_line: str, line_number: int) -> DatasetRowSnapshot:
        display_raw, raw_truncated, raw_bytes = self._truncate_raw(raw_line)

        try:
            json_obj = json.loads(raw_line)
        except json.JSONDecodeError:
            return DatasetRowSnapshot(
                line_number=line_number,
                raw=display_raw,
                format=DatasetContentFormat.unknown,
                fields={},
                validation_messages=["Invalid JSON"],
                raw_truncated=raw_truncated,
                raw_full_bytes=raw_bytes,
                fields_truncated=[],
            )
        if not isinstance(json_obj, dict):
            return DatasetRowSnapshot(
                line_number=line_number,
                raw=display_raw,
                format=DatasetContentFormat.unknown,
                fields={},
                validation_messages=["Invalid JSON"],
                raw_truncated=raw_truncated,
                raw_full_bytes=raw_bytes,
                fields_truncated=[],
            )

        parsed = self._truncated_fields(json_obj)
        if parsed is None:
            return DatasetRowSnapshot(
                line_number=line_number,
                raw=display_raw,
                format=DatasetContentFormat.unknown,
                fields={},
                validation_messages=["Invalid JSON"],
                raw_truncated=raw_truncated,
                raw_full_bytes=raw_bytes,
                fields_truncated=[],
            )

        detected_format = self.format_analyzer.detect_format(json_obj)
        validation = self.format_analyzer.validate_format(
            json_obj=json_obj,
            expected_format=detected_format,
            line_number=line_number,
        )
        messages = [error.message for error in validation]

        return DatasetRowSnapshot(
            line_number=line_number,
            raw=display_raw,
            format=detected_format,
            fields=parsed["fields"],
            validation_messages=messages,
            raw_truncated=raw_truncated,
            raw_full_bytes=raw_bytes,
            fields_truncated=parsed["truncated_fields"],
        )

    def _normalized_line_from_fields(
        self,
        fields: dict[str, Any],
        format_hint: DatasetContentFormat,
        target_format: DatasetContentFormat | None,
    ) -> dict[str, Any]:
        try:
            encoded = json.dumps(fields, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise DatasetEditorError("Content is not valid JSON.") from exc

        return self._normalized_line_from_raw(encoded, format_hint, target_format)

    def _normalized_line_from_raw(
        self,
        raw_line: str,
        format_hint: DatasetContentFormat,
        target_format: DatasetContentFormat | None,
    ) -> dict[str, Any]:
        try:
            line = normalized_line(raw_line, format_hint=format_hint, target_format=target_format)
        except DatasetExportFormatterError as exc:
            raise DatasetEditorError(str(exc)) from exc

        try:
            json_obj = json.loads(line)
        except json.JSONDecodeError:
            json_obj = {}
        detected = self.format_analyzer.detect_format(json_obj) if isinstance(json_obj, dict) else DatasetContentFormat.unknown
        resolved_format = target_format or (detected if detected != DatasetContentFormat.unknown else format_hint)
        return {"line": line, "format": resolved_format}

    def _truncated_fields(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        truncated_fields: list[str] = []
        for key, value in obj.items():
            converted = self._convert_and_truncate(value)
            if converted is None:
                return None
            result[key] = converted["value"]
            if converted["truncated"]:
                truncated_fields.append(key)
        return {"fields": result, "truncated_fields": truncated_fields}

    def _detect_format(self, fields: dict[str, Any]) -> DatasetContentFormat:
        return self.format_analyzer.detect_format(fields)

    def _truncate_raw(self, content: str) -> tuple[str, bool, int]:
        return self._truncate_string(content, limit=MAX_RAW_BYTES, suffix_separator="\n")

    def _truncate_field_string(self, content: str) -> tuple[str, bool, int]:
        return self._truncate_string(content, limit=MAX_FIELD_BYTES, suffix_separator=" ")

    def _truncate_string(self, content: str, limit: int, suffix_separator: str) -> tuple[str, bool, int]:
        raw_bytes = content.encode("utf-8")
        full_bytes = len(raw_bytes)
        if full_bytes <= limit:
            return content, False, full_bytes

        truncated = raw_bytes[:limit]
        while truncated:
            try:
                safe = truncated.decode("utf-8")
                break
            except UnicodeDecodeError:
                truncated = truncated[:-1]
        else:
            safe = ""

        annotated = f"{safe}{suffix_separator}... [truncated, {full_bytes} bytes total]"
        return annotated, True, full_bytes

    def _convert_and_truncate(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            truncated, was_truncated, _ = self._truncate_field_string(value)
            return {"value": truncated, "truncated": was_truncated}
        if isinstance(value, bool):
            return {"value": value, "truncated": False}
        if isinstance(value, int):
            return {"value": float(value), "truncated": False}
        if isinstance(value, float):
            return {"value": value, "truncated": False}
        if value is None:
            return {"value": None, "truncated": False}
        if isinstance(value, list):
            items: list[Any] = []
            was_truncated = False
            for element in value:
                converted = self._convert_and_truncate(element)
                if converted is None:
                    return None
                items.append(converted["value"])
                if converted["truncated"]:
                    was_truncated = True
            return {"value": items, "truncated": was_truncated}
        if isinstance(value, dict):
            obj: dict[str, Any] = {}
            was_truncated = False
            for key, nested in value.items():
                converted = self._convert_and_truncate(nested)
                if converted is None:
                    return None
                obj[str(key)] = converted["value"]
                if converted["truncated"]:
                    was_truncated = True
            return {"value": obj, "truncated": was_truncated}
        return None

    def _active_job_warnings(self, dataset_path: Path) -> list[str]:
        if not self._check_active_jobs:
            return []
        normalized_target = self._normalized_path(str(dataset_path)).as_posix()
        try:
            jobs = self.job_service.list_job_records(active_only=True)
        except Exception:
            return []
        active = [job for job in jobs if self._normalized_path(job.dataset_path).as_posix() == normalized_target]
        if not active:
            return []
        ids = ", ".join(job.job_id for job in active)
        return [f"Dataset is referenced by active training jobs: {ids}"]

    def _temporary_path(self, path: Path) -> Path:
        filename = path.name
        return path.parent / f".{filename}.tmp-{os.urandom(4).hex()}"

    def _replace_file(self, path: Path, temp_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            temp_path.replace(path)
        else:
            temp_path.rename(path)

    def _replace_line(self, path: Path, line_number: int, new_line: str) -> None:
        temp_path = self._temporary_path(path)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        found = False

        with temp_path.open("w", encoding="utf-8") as writer:
            def process(record):
                nonlocal found
                raw = record.data.decode("utf-8")
                if record.line_number == line_number:
                    writer.write(new_line + "\n")
                    found = True
                else:
                    writer.write(raw + "\n")
                return True

            self.file_enumerator.enumerate_lines(path, process)

        if not found:
            temp_path.unlink(missing_ok=True)
            raise DatasetEditorError(f"Invalid line number: {line_number}")

        self._replace_file(path, temp_path)

    def _remove_line(self, path: Path, line_number: int) -> None:
        temp_path = self._temporary_path(path)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        found = False

        with temp_path.open("w", encoding="utf-8") as writer:
            def process(record):
                nonlocal found
                raw = record.data.decode("utf-8")
                if record.line_number == line_number:
                    found = True
                else:
                    writer.write(raw + "\n")
                return True

            self.file_enumerator.enumerate_lines(path, process)

        if not found:
            temp_path.unlink(missing_ok=True)
            raise DatasetEditorError(f"Invalid line number: {line_number}")

        self._replace_file(path, temp_path)

    def _append_line(self, path: Path, line: str) -> int:
        temp_path = self._temporary_path(path)
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        last_line = 0

        with temp_path.open("w", encoding="utf-8") as writer:
            if path.exists():
                def process(record):
                    nonlocal last_line
                    raw = record.data.decode("utf-8")
                    writer.write(raw + "\n")
                    last_line = record.line_number
                    return True

                self.file_enumerator.enumerate_lines(path, process)
            writer.write(line + "\n")

        self._replace_file(path, temp_path)
        return last_line + 1
