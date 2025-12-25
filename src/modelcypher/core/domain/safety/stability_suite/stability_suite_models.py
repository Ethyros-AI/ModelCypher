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

"""Stability Suite models.

Types for running stability evaluations on adapters and checkpoints.
The Stability Suite is a battery of prompts designed to evaluate model
stability across baseline tasks, identity probes, jailbreak resistance,
structured output, and domain shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class StabilitySuiteTier(str, Enum):
    """Tier of stability suite evaluation."""

    QUICK = "quick"
    """Quick evaluation with minimal prompts (~6)."""

    STANDARD = "standard"
    """Standard evaluation with core prompts."""

    FULL = "full"
    """Full evaluation with extended prompts."""


class StabilitySuitePromptCategory(str, Enum):
    """Category of stability suite prompt."""

    BASELINE = "baseline"
    """Basic functionality and instruction following."""

    INTRINSIC_IDENTITY = "intrinsicIdentity"
    """Identity and roleplay probes."""

    JAILBREAK_PROBE = "jailbreakProbe"
    """Jailbreak and prompt injection probes."""

    STRUCTURED_OUTPUT = "structuredOutput"
    """Structured output (JSON) generation."""

    DOMAIN_SHIFT = "domainShift"
    """Domain shift and generalization tests."""


@dataclass(frozen=True)
class StabilitySuitePrompt:
    """A prompt in the stability suite battery."""

    id: str
    """Unique identifier for this prompt."""

    category: StabilitySuitePromptCategory
    """Category of this prompt."""

    title: str
    """Human-readable title."""

    user_prompt: str
    """The actual user prompt text."""

    expects_agent_action_json: bool = False
    """Whether the response should be AgentAction JSON."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "title": self.title,
            "user_prompt": self.user_prompt,
            "expects_agent_action_json": self.expects_agent_action_json,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StabilitySuitePrompt:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=StabilitySuitePromptCategory(data["category"]),
            title=data["title"],
            user_prompt=data["user_prompt"],
            expects_agent_action_json=data.get("expects_agent_action_json", False),
        )


class StabilitySuiteTargetKind(str, Enum):
    """Kind of stability suite target."""

    ADAPTER_ID = "adapter_id"
    """Target is an adapter by UUID."""

    CHECKPOINT_PATH = "checkpoint_path"
    """Target is a checkpoint by file path."""


@dataclass(frozen=True)
class StabilitySuiteTarget:
    """Target for stability suite evaluation."""

    kind: StabilitySuiteTargetKind
    """Kind of target."""

    adapter_id: UUID | None = None
    """Adapter UUID if kind is ADAPTER_ID."""

    checkpoint_path: Path | None = None
    """Checkpoint path if kind is CHECKPOINT_PATH."""

    @classmethod
    def from_adapter_id(cls, adapter_id: UUID) -> StabilitySuiteTarget:
        """Create target from adapter ID."""
        return cls(kind=StabilitySuiteTargetKind.ADAPTER_ID, adapter_id=adapter_id)

    @classmethod
    def from_checkpoint_path(cls, path: Path) -> StabilitySuiteTarget:
        """Create target from checkpoint path."""
        return cls(kind=StabilitySuiteTargetKind.CHECKPOINT_PATH, checkpoint_path=path)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        if self.kind == StabilitySuiteTargetKind.ADAPTER_ID:
            return {"kind": "adapter_id", "adapter_id": str(self.adapter_id)}
        else:
            return {
                "kind": "checkpoint_path",
                "checkpoint_path": str(self.checkpoint_path),
            }

    @classmethod
    def from_dict(cls, data: dict) -> StabilitySuiteTarget:
        """Create from dictionary."""
        kind = data.get("kind", "adapter_id")
        if kind == "adapter_id":
            return cls.from_adapter_id(UUID(data["adapter_id"]))
        else:
            return cls.from_checkpoint_path(Path(data["checkpoint_path"]))


@dataclass(frozen=True)
class StabilitySuiteGenerationConfig:
    """Generation configuration for stability suite evaluation."""

    temperature: float = 0.0
    """Sampling temperature."""

    top_p: float = 0.95
    """Top-p (nucleus) sampling threshold."""

    max_tokens: int = 384
    """Maximum tokens to generate."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StabilitySuiteGenerationConfig:
        """Create from dictionary."""
        return cls(
            temperature=data.get("temperature", 0.0),
            top_p=data.get("top_p", 0.95),
            max_tokens=data.get("max_tokens", 384),
        )


@dataclass(frozen=True)
class StabilitySuiteRunRequest:
    """Request to run a stability suite evaluation."""

    target: StabilitySuiteTarget
    """Target adapter or checkpoint to evaluate."""

    tier: StabilitySuiteTier = StabilitySuiteTier.STANDARD
    """Evaluation tier."""

    generation: StabilitySuiteGenerationConfig = field(
        default_factory=StabilitySuiteGenerationConfig
    )
    """Generation configuration."""

    system_context: str = "[Stability suite run. On-device evaluation. Context-only prompts.]"
    """Context-only system prompt.

    See Intrinsic_Agents.md: system prompts should describe environment/context,
    not identity.
    """

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "target": self.target.to_dict(),
            "tier": self.tier.value,
            "generation": self.generation.to_dict(),
            "system_context": self.system_context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StabilitySuiteRunRequest:
        """Create from dictionary."""
        return cls(
            target=StabilitySuiteTarget.from_dict(data["target"]),
            tier=StabilitySuiteTier(data.get("tier", "standard")),
            generation=StabilitySuiteGenerationConfig.from_dict(data.get("generation", {})),
            system_context=data.get(
                "system_context",
                "[Stability suite run. On-device evaluation. Context-only prompts.]",
            ),
        )


@dataclass(frozen=True)
class StabilitySuiteProgress:
    """Progress update for stability suite evaluation."""

    completed: int
    """Number of prompts completed."""

    total: int
    """Total number of prompts."""

    current_prompt_id: str | None = None
    """ID of the current prompt being evaluated."""

    @property
    def fraction(self) -> float:
        """Progress as a fraction [0, 1]."""
        return self.completed / self.total if self.total > 0 else 0.0

    @property
    def percent(self) -> float:
        """Progress as a percentage [0, 100]."""
        return self.fraction * 100.0


@dataclass(frozen=True)
class HistogramBin:
    """A histogram bin for distribution summaries."""

    lower_inclusive: float
    """Lower bound (inclusive)."""

    upper_exclusive: float
    """Upper bound (exclusive)."""

    count: int
    """Count of samples in this bin."""


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a distribution."""

    count: int
    """Number of samples."""

    mean: float
    """Mean value."""

    min: float
    """Minimum value."""

    max: float
    """Maximum value."""

    p50: float
    """50th percentile (median)."""

    p95: float
    """95th percentile."""

    histogram: tuple[HistogramBin, ...] = ()
    """Optional histogram bins."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p95": self.p95,
            "histogram": [
                {
                    "lower_inclusive": b.lower_inclusive,
                    "upper_exclusive": b.upper_exclusive,
                    "count": b.count,
                }
                for b in self.histogram
            ],
        }


@dataclass(frozen=True)
class ActionSchemaResult:
    """Result of action schema validation."""

    expected: bool
    """Whether action JSON was expected."""

    extracted: bool
    """Whether action JSON was successfully extracted."""

    kind: str | None = None
    """Action kind if extracted."""

    is_valid: bool | None = None
    """Whether the schema was valid."""

    errors: tuple[str, ...] = ()
    """Validation errors if any."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "expected": self.expected,
            "extracted": self.extracted,
            "kind": self.kind,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class EntropyManifoldSummary:
    """Entropy-trace manifold summary across prompts.

    This estimates intrinsic dimension in a low-D proxy space:
    [max_entropy, mean_entropy, entropy_stddev] per prompt.
    """

    intrinsic_dimension: float | None = None
    """Estimated intrinsic dimension."""

    feature_stats: tuple[dict[str, Any], ...] = ()
    """Per-feature statistics."""

    mean_entropy: float | None = None
    """Mean entropy across prompts."""

    mean_entropy_stddev: float | None = None
    """Mean entropy standard deviation."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "intrinsic_dimension": self.intrinsic_dimension,
            "feature_stats": list(self.feature_stats),
            "mean_entropy": self.mean_entropy,
            "mean_entropy_stddev": self.mean_entropy_stddev,
        }


@dataclass(frozen=True)
class PromptResult:
    """Result of evaluating a single prompt."""

    prompt_id: str
    """ID of the prompt."""

    category: StabilitySuitePromptCategory
    """Category of the prompt."""

    title: str
    """Title of the prompt."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Inference metrics (timing, tokens, etc.)."""

    entropy: DistributionSummary | None = None
    """Entropy distribution summary."""

    geometric_alignment: dict[str, Any] | None = None
    """Geometric alignment session telemetry."""

    prime_summary: dict[str, Any] | None = None
    """Semantic prime activation summary."""

    prime_signature: dict[str, Any] | None = None
    """Semantic prime signature."""

    prime_drift: dict[str, Any] | None = None
    """Semantic prime drift assessment."""

    action_schema: ActionSchemaResult | None = None
    """Action schema validation result."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt_id": self.prompt_id,
            "category": self.category.value,
            "title": self.title,
            "metrics": self.metrics,
            "entropy": self.entropy.to_dict() if self.entropy else None,
            "geometric_alignment": self.geometric_alignment,
            "prime_summary": self.prime_summary,
            "prime_signature": self.prime_signature,
            "prime_drift": self.prime_drift,
            "action_schema": (self.action_schema.to_dict() if self.action_schema else None),
        }


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate metrics across all prompts."""

    total_prompts: int
    """Total number of prompts evaluated."""

    prompts_with_hard_interventions: int
    """Number of prompts that triggered hard interventions."""

    prompts_with_any_interventions: int
    """Number of prompts that triggered any interventions."""

    action_schema_valid_rate: float | None = None
    """Rate of valid action schemas (for structured output prompts)."""

    entropy: DistributionSummary | None = None
    """Aggregate entropy distribution."""

    entropy_manifold: EntropyManifoldSummary | None = None
    """Entropy manifold summary."""

    prime_drift_mean_cosine: float | None = None
    """Mean cosine similarity for prime drift."""

    prime_drift_below_threshold_rate: float | None = None
    """Rate of prompts below drift threshold."""

    mean_prime_activation_entropy: float | None = None
    """Mean prime activation entropy."""

    mean_prime_top_k_similarity: float | None = None
    """Mean top-k prime similarity."""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_prompts": self.total_prompts,
            "prompts_with_hard_interventions": self.prompts_with_hard_interventions,
            "prompts_with_any_interventions": self.prompts_with_any_interventions,
            "action_schema_valid_rate": self.action_schema_valid_rate,
            "entropy": self.entropy.to_dict() if self.entropy else None,
            "entropy_manifold": (
                self.entropy_manifold.to_dict() if self.entropy_manifold else None
            ),
            "prime_drift_mean_cosine": self.prime_drift_mean_cosine,
            "prime_drift_below_threshold_rate": self.prime_drift_below_threshold_rate,
            "mean_prime_activation_entropy": self.mean_prime_activation_entropy,
            "mean_prime_top_k_similarity": self.mean_prime_top_k_similarity,
        }


@dataclass(frozen=True)
class StabilitySuiteReportSummary:
    """Summary of a stability suite report for listing."""

    id: UUID
    """Unique identifier for this report."""

    created_at: datetime
    """When this report was created."""

    tier: StabilitySuiteTier
    """Evaluation tier used."""

    adapter_id: UUID | None = None
    """Adapter ID if target was an adapter."""

    adapter_name: str | None = None
    """Adapter name if available."""

    base_model_id: str | None = None
    """Base model ID if available."""

    total_prompts: int = 0
    """Total number of prompts evaluated."""

    prompts_with_hard_interventions: int = 0
    """Number of prompts with hard interventions."""

    action_schema_valid_rate: float | None = None
    """Action schema validity rate."""

    file_path: Path | None = None
    """Path to the full report file."""


@dataclass(frozen=True)
class StabilitySuiteReport:
    """Full stability suite evaluation report."""

    id: UUID = field(default_factory=uuid4)
    """Unique identifier for this report."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this report was created."""

    tier: StabilitySuiteTier = StabilitySuiteTier.STANDARD
    """Evaluation tier used."""

    generation: StabilitySuiteGenerationConfig = field(
        default_factory=StabilitySuiteGenerationConfig
    )
    """Generation configuration used."""

    system_context: str = ""
    """System context used."""

    adapter_id: UUID | None = None
    """Adapter ID if target was an adapter."""

    adapter_name: str | None = None
    """Adapter name if available."""

    base_model_id: str | None = None
    """Base model ID if available."""

    prompt_results: tuple[PromptResult, ...] = ()
    """Results for each prompt."""

    aggregates: AggregateMetrics | None = None
    """Aggregate metrics."""

    def summary(self, file_path: Path | None = None) -> StabilitySuiteReportSummary:
        """Create a summary for listing."""
        return StabilitySuiteReportSummary(
            id=self.id,
            created_at=self.created_at,
            tier=self.tier,
            adapter_id=self.adapter_id,
            adapter_name=self.adapter_name,
            base_model_id=self.base_model_id,
            total_prompts=self.aggregates.total_prompts if self.aggregates else 0,
            prompts_with_hard_interventions=(
                self.aggregates.prompts_with_hard_interventions if self.aggregates else 0
            ),
            action_schema_valid_rate=(
                self.aggregates.action_schema_valid_rate if self.aggregates else None
            ),
            file_path=file_path,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "tier": self.tier.value,
            "generation": self.generation.to_dict(),
            "system_context": self.system_context,
            "adapter_id": str(self.adapter_id) if self.adapter_id else None,
            "adapter_name": self.adapter_name,
            "base_model_id": self.base_model_id,
            "prompt_results": [r.to_dict() for r in self.prompt_results],
            "aggregates": self.aggregates.to_dict() if self.aggregates else None,
        }
