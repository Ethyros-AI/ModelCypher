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

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NextAction:
    """An exact follow-up command or artifact reference."""

    name: str
    reason: str
    command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reason": self.reason,
            "command": self.command,
        }


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
    # Commensurability identity fields
    model_id: str | None = None
    data_hash: str | None = None
    eval_data_hash: str | None = None
    benchmark_suite: str | None = None

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
        if self.model_id is not None:
            d["model_id"] = self.model_id
        if self.data_hash is not None:
            d["data_hash"] = self.data_hash
        if self.eval_data_hash is not None:
            d["eval_data_hash"] = self.eval_data_hash
        if self.benchmark_suite is not None:
            d["benchmark_suite"] = self.benchmark_suite
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
    next_actions: list[NextAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "result": self.result,
            "diagnostics": self.diagnostics.to_dict(),
            "metadata": self.metadata.to_dict(),
            "next_actions": [action.to_dict() for action in self.next_actions],
        }


def file_hash(path: str | Path) -> str | None:
    """Compute SHA-256 hex digest of a file's contents.

    Returns None if the file doesn't exist or can't be read.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def model_id(model_path: str | Path) -> str | None:
    """Compute architecture identity hash from config.json.

    Hashes (architecture, hidden_size, num_layers, vocab_size) to produce
    a stable identifier for the model architecture. Returns None if
    config.json is missing or unreadable.
    """
    try:
        config_path = Path(model_path) / "config.json"
        if not config_path.is_file():
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Extract architecture identity fields
        identity = (
            config.get("architectures", config.get("model_type", "")),
            config.get("hidden_size", config.get("d_model", "")),
            config.get("num_hidden_layers", config.get("num_layers", "")),
            config.get("vocab_size", ""),
        )
        return hashlib.sha256(str(identity).encode()).hexdigest()[:16]
    except (OSError, json.JSONDecodeError):
        return None


def derived_eval_hash(
    data_hash: str,
    seed: int,
    n_eval: int,
) -> str:
    """Compute a stable identity hash for an auto-derived eval split.

    When ``--eval-data`` is not provided, the training pipeline derives
    a held-out split deterministically from the training data, seed, and
    pilot-variance-measured split size.  This function produces a stable
    SHA-256 identity from those three inputs so that commensurability
    checks can compare auto-derived splits.
    """
    identity = f"{data_hash}:{seed}:{n_eval}"
    return hashlib.sha256(identity.encode()).hexdigest()


def make_metadata(
    *,
    model: str | None = None,
    adapter_path: str | None = None,
    duration_seconds: float | None = None,
    seed: int | None = None,
    model_id_value: str | None = None,
    data_path: str | None = None,
    eval_data_path: str | None = None,
    eval_data_hash: str | None = None,
    benchmark_suite: str | None = None,
) -> AgentMetadata:
    """Create metadata with current UTC timestamp.

    If ``data_path`` or ``eval_data_path`` are provided, their SHA-256
    content hashes are computed and stored for commensurability checks.
    A pre-computed ``eval_data_hash`` (e.g. from ``derived_eval_hash``)
    takes priority over hashing ``eval_data_path``.
    """
    data_hash_val = file_hash(data_path) if data_path else None
    eval_data_hash_val = (
        eval_data_hash
        if eval_data_hash is not None
        else file_hash(eval_data_path) if eval_data_path else None
    )

    return AgentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model,
        adapter_path=adapter_path,
        duration_seconds=duration_seconds,
        seed=seed,
        model_id=model_id_value,
        data_hash=data_hash_val,
        eval_data_hash=eval_data_hash_val,
        benchmark_suite=benchmark_suite,
    )
