# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Evaluation types - ONE definition, no duplicates.

These types are shared across all backends. Framework-specific evaluation
engines use these types with Backend protocol for tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class EvaluationMetric(str, Enum):
    """Metrics that can be computed during evaluation."""

    LOSS = "loss"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    BITS_PER_CHARACTER = "bpc"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs.

    Framework-agnostic - works with any backend.
    """

    metrics: list[EvaluationMetric] = field(
        default_factory=lambda: [EvaluationMetric.LOSS, EvaluationMetric.PERPLEXITY]
    )
    batch_size: int = 4
    sequence_length: int = 512
    max_samples: int | None = None

    @classmethod
    def default(cls) -> "EvaluationConfig":
        return cls()


@dataclass
class EvaluationProgress:
    """Progress update during evaluation."""

    samples_processed: int
    total_samples: int
    current_metric: float | None = None

    @property
    def percentage(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return self.samples_processed / self.total_samples


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""

    metrics: dict[EvaluationMetric, float]
    samples_evaluated: int
    tokens_evaluated: int
    duration_seconds: float

    @property
    def loss(self) -> float | None:
        return self.metrics.get(EvaluationMetric.LOSS)

    @property
    def perplexity(self) -> float | None:
        return self.metrics.get(EvaluationMetric.PERPLEXITY)

    @property
    def accuracy(self) -> float | None:
        return self.metrics.get(EvaluationMetric.ACCURACY)

    @property
    def bpc(self) -> float | None:
        return self.metrics.get(EvaluationMetric.BITS_PER_CHARACTER)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": {k.value: v for k, v in self.metrics.items()},
            "samples_evaluated": self.samples_evaluated,
            "tokens_evaluated": self.tokens_evaluated,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        metrics = {
            EvaluationMetric(k): v for k, v in data.get("metrics", {}).items()
        }
        return cls(
            metrics=metrics,
            samples_evaluated=data.get("samples_evaluated", 0),
            tokens_evaluated=data.get("tokens_evaluated", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class EvaluationBatch:
    """A single evaluation batch.

    Uses generic Array type from Backend protocol - the actual tensor
    type (mx.array, torch.Tensor, jnp.ndarray) is determined at runtime.
    """

    inputs: Any  # Backend Array: [batch, seq_len] int32/int64
    targets: Any  # Backend Array: [batch, seq_len] int32/int64
    mask: Any  # Backend Array: [batch, seq_len] float32
    valid_token_counts: list[int]


class EvaluationError(Exception):
    """Evaluation failed."""


__all__ = [
    "EvaluationMetric",
    "EvaluationConfig",
    "EvaluationProgress",
    "EvaluationResult",
    "EvaluationBatch",
    "EvaluationError",
]
