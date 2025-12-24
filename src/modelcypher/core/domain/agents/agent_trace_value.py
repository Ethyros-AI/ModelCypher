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

"""Privacy-preserving JSON value for imported traces.

String values are stored as digests (optional short preview) so that external
traces can be retained locally without persisting raw prompts, tool outputs,
or other sensitive content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from modelcypher.core.domain.agents.agent_trace import PayloadDigest
from modelcypher.core.domain.agents.agent_trace_sanitizer import AgentTraceSanitizer


@dataclass(frozen=True)
class ImportOptions:
    """Options for importing trace values."""

    max_string_preview_length: int = 64
    """Maximum characters allowed for persisting a string preview.

    Previews are only included when the string is short and has no whitespace/newlines.
    """

    @classmethod
    def safe_default(cls) -> ImportOptions:
        """Default import options with short previews."""
        return cls(max_string_preview_length=64)

    @classmethod
    def no_previews(cls) -> ImportOptions:
        """Import options that disable previews entirely."""
        return cls(max_string_preview_length=0)


class AgentTraceValueKind(str, Enum):
    """Kind of trace value."""

    DIGEST = "digest"
    NUMBER = "number"
    BOOL = "bool"
    OBJECT = "object"
    ARRAY = "array"
    NULL = "null"


@dataclass
class AgentTraceValue:
    """A privacy-preserving JSON value used in imported traces.

    String values are stored as digests (optional short preview) so that
    external traces can be retained locally without persisting raw prompts,
    tool outputs, or other sensitive content.
    """

    kind: AgentTraceValueKind
    """The kind of value."""

    digest: PayloadDigest | None = None
    """Digest value (if kind is DIGEST)."""

    number: float | None = None
    """Number value (if kind is NUMBER)."""

    boolean: bool | None = None
    """Boolean value (if kind is BOOL)."""

    object: dict[str, "AgentTraceValue"] | None = None
    """Object value (if kind is OBJECT)."""

    array: list["AgentTraceValue"] | None = None
    """Array value (if kind is ARRAY)."""

    @classmethod
    def from_any(
        cls,
        value: Any,
        options: ImportOptions | None = None,
    ) -> AgentTraceValue | None:
        """Create a trace value from any Python value.

        Args:
            value: The value to convert.
            options: Import options controlling preview behavior.

        Returns:
            AgentTraceValue, or None if conversion fails.
        """
        if options is None:
            options = ImportOptions.safe_default()

        if value is None:
            return cls(kind=AgentTraceValueKind.NULL)

        if isinstance(value, bool):
            return cls(kind=AgentTraceValueKind.BOOL, boolean=value)

        if isinstance(value, (int, float)):
            return cls(kind=AgentTraceValueKind.NUMBER, number=float(value))

        if isinstance(value, str):
            sanitized = AgentTraceSanitizer.sanitize(value)

            # Include preview only if short and no whitespace
            preview: str | None = None
            if (
                options.max_string_preview_length > 0
                and len(sanitized) <= options.max_string_preview_length
                and not any(c.isspace() for c in sanitized)
            ):
                preview = sanitized

            digest = PayloadDigest.hashing_with_preview(value, preview)
            return cls(kind=AgentTraceValueKind.DIGEST, digest=digest)

        if isinstance(value, list):
            mapped = []
            for item in value:
                converted = cls.from_any(item, options)
                if converted is None:
                    return None
                mapped.append(converted)
            return cls(kind=AgentTraceValueKind.ARRAY, array=mapped)

        if isinstance(value, dict):
            mapped = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    return None
                converted = cls.from_any(item, options)
                if converted is None:
                    return None
                mapped[key] = converted
            return cls(kind=AgentTraceValueKind.OBJECT, object=mapped)

        # Unsupported type
        return None

    @classmethod
    def null(cls) -> AgentTraceValue:
        """Create a null value."""
        return cls(kind=AgentTraceValueKind.NULL)

    @classmethod
    def from_string(
        cls, text: str, options: ImportOptions | None = None
    ) -> AgentTraceValue:
        """Create a digest value from a string."""
        result = cls.from_any(text, options)
        if result is None:
            raise ValueError("Failed to create trace value from string")
        return result

    @classmethod
    def from_number(cls, value: float) -> AgentTraceValue:
        """Create a number value."""
        return cls(kind=AgentTraceValueKind.NUMBER, number=value)

    @classmethod
    def from_bool(cls, value: bool) -> AgentTraceValue:
        """Create a boolean value."""
        return cls(kind=AgentTraceValueKind.BOOL, boolean=value)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        if self.kind == AgentTraceValueKind.DIGEST and self.digest is not None:
            return {
                "_type": "tc.trace.digest.v1",
                "sha256": self.digest.sha256,
                "character_count": self.digest.character_count,
                "byte_count": self.digest.byte_count,
                "preview": self.digest.preview,
            }
        elif self.kind == AgentTraceValueKind.NUMBER:
            return {"_type": "number", "value": self.number}
        elif self.kind == AgentTraceValueKind.BOOL:
            return {"_type": "bool", "value": self.boolean}
        elif self.kind == AgentTraceValueKind.OBJECT and self.object is not None:
            return {
                "_type": "object",
                "value": {k: v.to_dict() for k, v in self.object.items()},
            }
        elif self.kind == AgentTraceValueKind.ARRAY and self.array is not None:
            return {"_type": "array", "value": [v.to_dict() for v in self.array]}
        else:
            return {"_type": "null"}

    @classmethod
    def from_dict(cls, data: dict) -> AgentTraceValue | None:
        """Create from dictionary."""
        type_id = data.get("_type")

        if type_id == "tc.trace.digest.v1":
            digest = PayloadDigest(
                sha256=data["sha256"],
                character_count=data["character_count"],
                byte_count=data["byte_count"],
                preview=data.get("preview"),
            )
            return cls(kind=AgentTraceValueKind.DIGEST, digest=digest)

        if type_id == "number":
            return cls(kind=AgentTraceValueKind.NUMBER, number=data.get("value"))

        if type_id == "bool":
            return cls(kind=AgentTraceValueKind.BOOL, boolean=data.get("value"))

        if type_id == "object":
            obj_data = data.get("value", {})
            obj = {}
            for k, v in obj_data.items():
                converted = cls.from_dict(v)
                if converted is None:
                    return None
                obj[k] = converted
            return cls(kind=AgentTraceValueKind.OBJECT, object=obj)

        if type_id == "array":
            arr_data = data.get("value", [])
            arr = []
            for v in arr_data:
                converted = cls.from_dict(v)
                if converted is None:
                    return None
                arr.append(converted)
            return cls(kind=AgentTraceValueKind.ARRAY, array=arr)

        if type_id == "null" or type_id is None:
            return cls(kind=AgentTraceValueKind.NULL)

        return None
