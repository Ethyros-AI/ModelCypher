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

"""Canonical action envelope emitted by Agent Cypher skills (micro-adapters).

This is the preferred "policy" representation for behavior traces:
- Structured (Codable) and easy to validate
- Designed to be sanitized deterministically
- Suitable for distillation datasets (observation → action)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from modelcypher.core.domain.agents.agent_json_extractor import (
    AgentJSONSnippetExtractor,
)
from modelcypher.core.domain.agents.agent_trace_sanitizer import AgentTraceSanitizer


class ActionKind(str, Enum):
    """Kind of action an agent can take."""

    TOOL_CALL = "tool_call"
    RESPOND = "respond"
    ASK_CLARIFICATION = "ask_clarification"
    REFUSE = "refuse"
    DEFERRAL = "defer"


class ResponseFormat(str, Enum):
    """Format of a response."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass(frozen=True)
class ActionToolCall:
    """Tool call payload."""

    name: str
    """Tool name."""

    arguments: dict[str, Any] = field(default_factory=dict)
    """Tool arguments."""


@dataclass(frozen=True)
class ActionResponse:
    """Response payload."""

    text: str
    """Response text."""

    format: ResponseFormat | None = None
    """Response format."""


@dataclass(frozen=True)
class ActionClarification:
    """Clarification payload."""

    question: str
    """Clarifying question."""

    options: list[str] | None = None
    """Suggested options."""


@dataclass(frozen=True)
class ActionRefusal:
    """Refusal payload."""

    reason: str
    """Reason for refusal."""


@dataclass(frozen=True)
class ActionDeferral:
    """Deferral payload."""

    reason: str
    """Reason for deferral."""


@dataclass(frozen=True)
class ActionExtraction:
    """Result of extracting action from output."""

    json: str
    """Extracted JSON string."""

    action: AgentActionEnvelope | None
    """Decoded action, if schema matched."""


@dataclass
class AgentActionEnvelope:
    """Canonical action envelope emitted by Agent Cypher skills (micro-adapters).

    This is the preferred "policy" representation for behavior traces:
    - Structured (Codable) and easy to validate
    - Designed to be sanitized deterministically
    - Suitable for distillation datasets (observation → action)
    """

    SCHEMA_ID = "tc.agent.action.v1"
    SCHEMA_VERSION = 1

    schema: str
    """Schema identifier."""

    version: int
    """Schema version."""

    kind: ActionKind
    """Kind of action."""

    action_id: UUID | None = None
    """Optional action identifier."""

    confidence: float | None = None
    """Confidence score (0-1)."""

    notes: str | None = None
    """Optional notes."""

    tool: ActionToolCall | None = None
    """Tool call payload (if kind is TOOL_CALL)."""

    response: ActionResponse | None = None
    """Response payload (if kind is RESPOND)."""

    clarification: ActionClarification | None = None
    """Clarification payload (if kind is ASK_CLARIFICATION)."""

    refusal: ActionRefusal | None = None
    """Refusal payload (if kind is REFUSE)."""

    deferral: ActionDeferral | None = None
    """Deferral payload (if kind is DEFERRAL)."""

    @classmethod
    def create(
        cls,
        kind: ActionKind,
        action_id: UUID | None = None,
        confidence: float | None = None,
        notes: str | None = None,
        tool: ActionToolCall | None = None,
        response: ActionResponse | None = None,
        clarification: ActionClarification | None = None,
        refusal: ActionRefusal | None = None,
        deferral: ActionDeferral | None = None,
    ) -> AgentActionEnvelope:
        """Create an action with default schema and version."""
        return cls(
            schema=cls.SCHEMA_ID,
            version=cls.SCHEMA_VERSION,
            kind=kind,
            action_id=action_id,
            confidence=confidence,
            notes=notes,
            tool=tool,
            response=response,
            clarification=clarification,
            refusal=refusal,
            deferral=deferral,
        )

    def sanitized(self) -> AgentActionEnvelope:
        """Create a sanitized copy of this action."""
        return AgentActionEnvelope(
            schema=self.schema,
            version=self.version,
            kind=self.kind,
            action_id=self.action_id,
            confidence=self.confidence,
            notes=(
                AgentTraceSanitizer.sanitize(self.notes)
                if self.notes
                else None
            ),
            tool=(
                ActionToolCall(
                    name=AgentTraceSanitizer.sanitize(self.tool.name),
                    arguments=AgentTraceSanitizer.sanitize_json_value(
                        self.tool.arguments
                    ),
                )
                if self.tool
                else None
            ),
            response=(
                ActionResponse(
                    text=AgentTraceSanitizer.sanitize(self.response.text),
                    format=self.response.format,
                )
                if self.response
                else None
            ),
            clarification=(
                ActionClarification(
                    question=AgentTraceSanitizer.sanitize(self.clarification.question),
                    options=(
                        [AgentTraceSanitizer.sanitize(o) for o in self.clarification.options]
                        if self.clarification.options
                        else None
                    ),
                )
                if self.clarification
                else None
            ),
            refusal=(
                ActionRefusal(reason=AgentTraceSanitizer.sanitize(self.refusal.reason))
                if self.refusal
                else None
            ),
            deferral=(
                ActionDeferral(reason=AgentTraceSanitizer.sanitize(self.deferral.reason))
                if self.deferral
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "_schema": self.schema,
            "_version": self.version,
            "kind": self.kind.value,
        }

        if self.action_id is not None:
            result["action_id"] = str(self.action_id)
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.notes is not None:
            result["notes"] = self.notes

        if self.tool is not None:
            result["tool"] = {
                "name": self.tool.name,
                "arguments": self.tool.arguments,
            }

        if self.response is not None:
            resp: dict[str, Any] = {"text": self.response.text}
            if self.response.format is not None:
                resp["format"] = self.response.format.value
            result["response"] = resp

        if self.clarification is not None:
            clar: dict[str, Any] = {"question": self.clarification.question}
            if self.clarification.options is not None:
                clar["options"] = self.clarification.options
            result["clarification"] = clar

        if self.refusal is not None:
            result["refusal"] = {"reason": self.refusal.reason}

        if self.deferral is not None:
            result["defer"] = {"reason": self.deferral.reason}

        return result

    def to_json(self, pretty: bool = True) -> str:
        """Encode to JSON string."""
        return json.dumps(self.to_dict(), indent=2 if pretty else None, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentActionEnvelope | None:
        """Create from dictionary."""
        schema = data.get("_schema")
        version = data.get("_version")

        if schema != cls.SCHEMA_ID or version != cls.SCHEMA_VERSION:
            return None

        kind_str = data.get("kind")
        if not kind_str:
            return None

        try:
            kind = ActionKind(kind_str)
        except ValueError:
            return None

        action_id = None
        if "action_id" in data:
            try:
                action_id = UUID(data["action_id"])
            except (ValueError, TypeError):
                pass

        tool = None
        if "tool" in data:
            tool_data = data["tool"]
            tool = ActionToolCall(
                name=tool_data.get("name", ""),
                arguments=tool_data.get("arguments", {}),
            )

        response = None
        if "response" in data:
            resp_data = data["response"]
            format_val = None
            if "format" in resp_data:
                try:
                    format_val = ResponseFormat(resp_data["format"])
                except ValueError:
                    pass
            response = ActionResponse(
                text=resp_data.get("text", ""),
                format=format_val,
            )

        clarification = None
        if "clarification" in data:
            clar_data = data["clarification"]
            clarification = ActionClarification(
                question=clar_data.get("question", ""),
                options=clar_data.get("options"),
            )

        refusal = None
        if "refusal" in data:
            refusal = ActionRefusal(reason=data["refusal"].get("reason", ""))

        deferral = None
        if "defer" in data:
            deferral = ActionDeferral(reason=data["defer"].get("reason", ""))

        return cls(
            schema=schema,
            version=version,
            kind=kind,
            action_id=action_id,
            confidence=data.get("confidence"),
            notes=data.get("notes"),
            tool=tool,
            response=response,
            clarification=clarification,
            refusal=refusal,
            deferral=deferral,
        )

    @classmethod
    def decode(cls, json_str: str) -> AgentActionEnvelope | None:
        """Decode from JSON string if schema and version match."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        return cls.from_dict(data)

    @classmethod
    def extract(cls, output: str) -> ActionExtraction | None:
        """Extract and decode the first JSON object from output.

        Returns both the extracted JSON and a decoded AgentActionEnvelope
        (if it matches the canonical schema ID + version).
        """
        json_str = AgentJSONSnippetExtractor.extract_first_json_object(output)
        if json_str is None:
            return None

        return ActionExtraction(
            json=json_str,
            action=cls.decode(json_str),
        )
