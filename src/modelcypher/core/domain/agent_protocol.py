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

"""Agent-ready output protocol.

Structured envelope for every CLI command, designed for frontier AI agents
to interpret results, advise humans, and chain commands autonomously.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AgentRecommendation:
    """A concrete next step the agent can take."""

    action: str
    reason: str
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"action": self.action, "reason": self.reason}
        if self.command is not None:
            d["command"] = self.command
        return d


@dataclass(frozen=True)
class AgentDiagnostics:
    """Agent-readable interpretation of command results."""

    summary: str
    observations: list[str] = field(default_factory=list)
    recommendations: list[AgentRecommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "observations": list(self.observations),
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


@dataclass(frozen=True)
class AgentMetadata:
    """Command execution metadata."""

    timestamp: str
    model: str | None = None
    adapter_path: str | None = None
    duration_seconds: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"timestamp": self.timestamp}
        if self.model is not None:
            d["model"] = self.model
        if self.adapter_path is not None:
            d["adapter_path"] = self.adapter_path
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass(frozen=True)
class AgentEnvelope:
    """Structured output envelope for all CLI commands.

    The ``result`` field contains the command-specific payload (same data
    as the existing flat JSON output). The ``diagnostics`` field adds
    agent-readable interpretation. This lets existing parsers read
    ``envelope["result"]`` while agents use the full envelope.
    """

    command: str
    status: str  # "success" | "failure" | "partial"
    result: dict[str, Any]
    diagnostics: AgentDiagnostics
    metadata: AgentMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "result": self.result,
            "diagnostics": self.diagnostics.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


def make_metadata(
    *,
    model: str | None = None,
    adapter_path: str | None = None,
    duration_seconds: float | None = None,
    seed: int | None = None,
) -> AgentMetadata:
    """Create metadata with current UTC timestamp."""
    return AgentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        adapter_path=adapter_path,
        duration_seconds=duration_seconds,
        seed=seed,
    )
