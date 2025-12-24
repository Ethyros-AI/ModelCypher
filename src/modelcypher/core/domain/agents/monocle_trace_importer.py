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

"""Monocle/OpenTelemetry trace importer.

Imports agent traces from Monocle/OpenTelemetry JSON format into
ModelCypher's AgentTrace format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from modelcypher.core.domain.agents.agent_trace import (
    AgentTrace,
    PayloadDigest,
    TraceKind,
    TraceSource,
    TraceSpan,
    TraceStatus,
)
from modelcypher.core.domain.agents.agent_trace_value import (
    AgentTraceValue,
    ImportOptions,
)


class ImportError(Exception):
    """Error during trace import."""

    pass


class ImportErrorKind(str, Enum):
    """Kind of import error."""

    INVALID_JSON = "invalid_json"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    MISSING_SPANS = "missing_spans"


@dataclass(frozen=True)
class ImportResult:
    """Result of trace import."""

    traces: list[AgentTrace]
    """Successfully imported traces."""

    warnings: list[str] = field(default_factory=list)
    """Warnings encountered during import."""


@dataclass(frozen=True)
class ParsedFileName:
    """Parsed Monocle trace file name."""

    workflow_name: str | None
    trace_id: str | None
    timestamp: str | None


@dataclass(frozen=True)
class DecodedSpan:
    """Decoded span with trace ID."""

    trace_id: str
    span: TraceSpan


class MonocleTraceImporter:
    """Imports agent traces from Monocle/OpenTelemetry JSON format."""

    @staticmethod
    def import_file(
        data: bytes,
        file_name: str | None = None,
        imported_at: datetime | None = None,
        value_options: ImportOptions | None = None,
    ) -> ImportResult:
        """Import traces from JSON data.

        Args:
            data: JSON data bytes.
            file_name: Original file name (for metadata extraction).
            imported_at: Import timestamp.
            value_options: Options for trace value import.

        Returns:
            Import result with traces and warnings.

        Raises:
            ImportError: If import fails.
        """
        if imported_at is None:
            imported_at = datetime.now()
        if value_options is None:
            value_options = ImportOptions.safe_default()

        # Parse JSON
        try:
            json_data = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ImportError(ImportErrorKind.INVALID_JSON.value)

        # Extract span objects
        span_objects: list[Any]
        if isinstance(json_data, list):
            span_objects = json_data
        elif isinstance(json_data, dict):
            if "spans" in json_data and isinstance(json_data["spans"], list):
                span_objects = json_data["spans"]
            else:
                raise ImportError(ImportErrorKind.UNSUPPORTED_SHAPE.value)
        else:
            raise ImportError(ImportErrorKind.UNSUPPORTED_SHAPE.value)

        if not span_objects:
            raise ImportError(ImportErrorKind.MISSING_SPANS.value)

        # Parse file name for metadata
        metadata = MonocleTraceImporter._parse_file_name(file_name)

        # Decode spans
        warnings: list[str] = []
        spans_by_trace_id: dict[str, list[TraceSpan]] = {}

        for index, any_span in enumerate(span_objects):
            if not isinstance(any_span, dict):
                warnings.append(f"span[{index}]: expected object")
                continue

            decoded = MonocleTraceImporter._decode_span(any_span, value_options)
            if decoded is None:
                warnings.append(f"span[{index}]: missing required fields")
                continue

            if decoded.trace_id not in spans_by_trace_id:
                spans_by_trace_id[decoded.trace_id] = []
            spans_by_trace_id[decoded.trace_id].append(decoded.span)

        if not spans_by_trace_id:
            raise ImportError(ImportErrorKind.MISSING_SPANS.value)

        # Build traces
        traces: list[AgentTrace] = []
        for trace_id, spans in spans_by_trace_id.items():
            # Sort spans by start time
            sorted_spans = sorted(
                spans,
                key=lambda s: (s.start_time or datetime.min, s.operation_name),
            )

            # Determine times
            start_times = [s.start_time for s in sorted_spans if s.start_time]
            end_times = [s.end_time for s in sorted_spans if s.end_time]
            started_at = min(start_times) if start_times else imported_at
            completed_at = max(end_times) if end_times else None

            # Determine status
            has_error = any(
                s.status == TraceStatus.failed
                for s in sorted_spans
            )
            trace_status = TraceStatus.failed if has_error else TraceStatus.success

            # Infer model ID
            base_model_id = MonocleTraceImporter._infer_model_id(sorted_spans)

            # Create trace
            trace = AgentTrace(
                id=MonocleTraceImporter._deterministic_uuid(trace_id) or UUID(int=0),
                kind=TraceKind.agent_pipeline,
                started_at=started_at,
                completed_at=completed_at,
                status=trace_status,
                input_digest=PayloadDigest.hashing_with_preview(trace_id, "Imported trace"),
                base_model_id=base_model_id,
                source=TraceSource(
                    provider="monocle",
                    trace_id=metadata.trace_id if metadata else trace_id,
                    imported_at=imported_at,
                    original_format="otel",
                ),
                spans=sorted_spans,
            )
            traces.append(trace)

        # Sort by start time descending (newest first)
        traces.sort(key=lambda t: t.started_at, reverse=True)

        return ImportResult(traces=traces, warnings=warnings)

    @staticmethod
    def _parse_file_name(file_name: str | None) -> ParsedFileName | None:
        """Parse Monocle trace file name for metadata."""
        if not file_name:
            return None

        # Remove extension
        base = file_name.rsplit(".", 1)[0] if "." in file_name else file_name

        if not base.startswith("monocle_trace_"):
            return None

        remainder = base[len("monocle_trace_"):]
        parts = remainder.split("_")
        if len(parts) < 3:
            return None

        trace_id = parts[-2]
        timestamp = parts[-1]
        workflow_name = "_".join(parts[:-2])

        return ParsedFileName(
            workflow_name=workflow_name if workflow_name else None,
            trace_id=trace_id if trace_id else None,
            timestamp=timestamp if timestamp else None,
        )

    @staticmethod
    def _decode_span(
        obj: dict[str, Any],
        value_options: ImportOptions,
    ) -> DecodedSpan | None:
        """Decode a span from JSON object."""
        name = obj.get("name") or obj.get("span_name")
        trace_id, span_id = MonocleTraceImporter._resolve_ids(obj)

        if not trace_id or not span_id or not name:
            return None

        parent_span_id = (
            obj.get("parent_id")
            or obj.get("parentSpanId")
            or obj.get("parent_span_id")
        )

        start_time = MonocleTraceImporter._resolve_timestamp(
            obj.get("start_time") or obj.get("startTime"),
            obj.get("startTimeUnixNano") or obj.get("start_time_unix_nano"),
        )
        end_time = MonocleTraceImporter._resolve_timestamp(
            obj.get("end_time") or obj.get("endTime"),
            obj.get("endTimeUnixNano") or obj.get("end_time_unix_nano"),
        )

        status = MonocleTraceImporter._resolve_status(obj.get("status"))
        attributes = MonocleTraceImporter._resolve_attributes(
            obj.get("attributes"), value_options
        )
        events = MonocleTraceImporter._resolve_events(
            obj.get("events"), value_options
        )

        # Convert status to TraceStatus enum
        span_status = TraceStatus.success
        if status is not None and isinstance(status, dict) and status.get("code") == "error":
            span_status = TraceStatus.failed

        # Convert events to simple dicts
        events_dicts = [
            {"name": e["name"], "timestamp": e.get("timestamp"), "attributes": e.get("attributes", {})}
            if isinstance(e, dict) else e
            for e in events
        ] if events else []

        # Convert attributes to simple dicts
        attributes_dicts = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in attributes.items()
        } if attributes else {}

        span = TraceSpan(
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=name,
            start_time=start_time or datetime.now(),
            end_time=end_time,
            status=span_status,
            attributes=attributes_dicts,
            events=events_dicts,
        )

        return DecodedSpan(trace_id=trace_id, span=span)

    @staticmethod
    def _resolve_ids(obj: dict[str, Any]) -> tuple[str | None, str | None]:
        """Resolve trace and span IDs from object."""
        context = obj.get("context")
        if isinstance(context, dict):
            trace_id = (
                context.get("trace_id")
                or context.get("traceId")
                or context.get("traceID")
            )
            span_id = (
                context.get("span_id")
                or context.get("spanId")
                or context.get("spanID")
            )
            if trace_id or span_id:
                return trace_id, span_id

        trace_id = obj.get("trace_id") or obj.get("traceId") or obj.get("traceID")
        span_id = obj.get("span_id") or obj.get("spanId") or obj.get("spanID")
        return trace_id, span_id

    @staticmethod
    def _resolve_timestamp(
        iso: Any,
        unix_nano: Any,
    ) -> datetime | None:
        """Resolve timestamp from ISO string or unix nano."""
        if isinstance(iso, str):
            # Try ISO 8601
            try:
                return datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except ValueError:
                pass

            # Try unix seconds
            try:
                return datetime.fromtimestamp(float(iso))
            except ValueError:
                pass

        if unix_nano is not None:
            return MonocleTraceImporter._resolve_unix_nano_timestamp(unix_nano)

        return None

    @staticmethod
    def _resolve_unix_nano_timestamp(value: Any) -> datetime | None:
        """Resolve timestamp from unix nanoseconds."""
        if isinstance(value, str):
            try:
                nanos = int(value)
                return datetime.fromtimestamp(nanos / 1_000_000_000)
            except ValueError:
                try:
                    seconds = float(value)
                    return datetime.fromtimestamp(seconds)
                except ValueError:
                    pass

        if isinstance(value, (int, float)):
            nanos = int(value)
            return datetime.fromtimestamp(nanos / 1_000_000_000)

        return None

    @staticmethod
    def _resolve_status(any_status: Any) -> dict[str, Any] | None:
        """Resolve span status from object."""
        if not isinstance(any_status, dict):
            return None

        message = any_status.get("message") or any_status.get("description")
        raw_code = (
            any_status.get("code")
            or any_status.get("status_code")
            or any_status.get("statusCode")
        )

        if isinstance(raw_code, str):
            upper = raw_code.upper()
            if "ERROR" in upper:
                return {"code": "error", "message": message}
            if "OK" in upper:
                return {"code": "ok", "message": message}
            return {"code": "unset", "message": message}

        if isinstance(raw_code, int):
            if raw_code == 2:
                return {"code": "error", "message": message}
            if raw_code == 1:
                return {"code": "ok", "message": message}
            return {"code": "unset", "message": message}

        return None

    @staticmethod
    def _resolve_attributes(
        any_attributes: Any,
        value_options: ImportOptions,
    ) -> dict[str, AgentTraceValue]:
        """Resolve attributes from object."""
        if any_attributes is None:
            return {}

        result: dict[str, AgentTraceValue] = {}

        if isinstance(any_attributes, dict):
            for key, value in any_attributes.items():
                unwrapped = MonocleTraceImporter._unwrap_otlp_any_value(value)
                trace_value = AgentTraceValue.from_any(unwrapped, value_options)
                if trace_value is not None:
                    result[key] = trace_value
            return result

        if isinstance(any_attributes, list):
            for entry in any_attributes:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("key")
                if not isinstance(key, str):
                    continue
                raw = entry.get("value") or entry.get("val") or entry.get("anyValue")
                if raw is None:
                    continue
                unwrapped = MonocleTraceImporter._unwrap_otlp_any_value(raw)
                trace_value = AgentTraceValue.from_any(unwrapped, value_options)
                if trace_value is not None:
                    result[key] = trace_value
            return result

        return {}

    @staticmethod
    def _resolve_events(
        any_events: Any,
        value_options: ImportOptions,
    ) -> list[dict[str, Any]]:
        """Resolve events from array."""
        if not isinstance(any_events, list):
            return []

        events: list[dict[str, Any]] = []
        for entry in any_events:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue

            timestamp = MonocleTraceImporter._resolve_timestamp(
                entry.get("timestamp") or entry.get("time"),
                entry.get("timeUnixNano") or entry.get("time_unix_nano"),
            )

            attributes = MonocleTraceImporter._resolve_attributes(
                entry.get("attributes"), value_options
            )

            # Convert AgentTraceValue to dict for storage
            attr_dicts = {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in attributes.items()
            }

            events.append({
                "name": name,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "attributes": attr_dicts,
            })

        return events

    @staticmethod
    def _unwrap_otlp_any_value(value: Any) -> Any:
        """Unwrap OTLP anyValue wrapper."""
        if not isinstance(value, dict):
            return value

        if "stringValue" in value:
            return value["stringValue"]
        if "boolValue" in value:
            return value["boolValue"]
        if "intValue" in value:
            int_val = value["intValue"]
            if isinstance(int_val, str):
                try:
                    return int(int_val)
                except ValueError:
                    return int_val
            return int_val
        if "doubleValue" in value:
            double_val = value["doubleValue"]
            if isinstance(double_val, str):
                try:
                    return float(double_val)
                except ValueError:
                    return double_val
            return double_val
        if "bytesValue" in value:
            return value["bytesValue"]

        if "arrayValue" in value:
            array_value = value["arrayValue"]
            if isinstance(array_value, dict) and "values" in array_value:
                return [
                    MonocleTraceImporter._unwrap_otlp_any_value(v)
                    for v in array_value["values"]
                ]

        if "kvlistValue" in value:
            kvlist = value["kvlistValue"]
            if isinstance(kvlist, dict) and "values" in kvlist:
                obj: dict[str, Any] = {}
                for entry in kvlist["values"]:
                    if not isinstance(entry, dict):
                        continue
                    key = entry.get("key")
                    if not isinstance(key, str):
                        continue
                    raw = entry.get("value") or entry.get("anyValue")
                    obj[key] = MonocleTraceImporter._unwrap_otlp_any_value(raw)
                return obj

        return value

    @staticmethod
    def _deterministic_uuid(trace_id: str) -> UUID | None:
        """Create deterministic UUID from 32-char hex trace ID."""
        trimmed = trace_id.strip()
        if len(trimmed) != 32:
            return None
        if not re.match(r"^[0-9a-fA-F]{32}$", trimmed):
            return None

        # Convert to UUID format
        return UUID(trimmed)

    @staticmethod
    def _infer_model_id(spans: list[TraceSpan]) -> str | None:
        """Infer model ID from span attributes."""
        allowlist = [
            "model",
            "llm.model",
            "genai.model",
            "openai.model",
            "anthropic.model",
            "google.model",
        ]

        for span in spans:
            for key in allowlist:
                value = span.attributes.get(key)
                if value is None:
                    continue

                # Handle AgentTraceValue (serialized as dict)
                if isinstance(value, dict):
                    type_id = value.get("_type")
                    if type_id == "tc.trace.digest.v1":
                        preview = value.get("preview")
                        if preview:
                            return preview
                # Handle raw string values
                elif isinstance(value, str):
                    return value

        return None
