"""Recall evaluator interface for Baranov replication tracks.

EXPERIMENTAL: Not validated for production use.

Defines the ``RecallEvaluator`` protocol and supporting data types for
measuring per-fact recall in both ``raw_completion`` and ``chat_template``
modes.  A standalone ``compute_recall_aggregate`` function computes
aggregate statistics with Clopper-Pearson confidence intervals.

No concrete evaluator implementation is provided in this patchset -- that
is deferred to patchset 2.  Integration tests mock this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from modelcypher.core.domain.statistics import clopper_pearson_interval

if TYPE_CHECKING:
    from modelcypher.experimental.baranov.models import FactTriple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecallMode(str, Enum):
    """Evaluation mode for recall measurement."""

    raw_completion = "raw_completion"
    chat_template = "chat_template"


# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecallOutcome:
    """Per-fact recall evaluation result."""

    fact_id: str
    recalled: bool
    raw_output: str
    confidence: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "recalled": self.recalled,
            "raw_output": self.raw_output,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RecallAggregate:
    """Aggregate recall statistics with optional confidence interval."""

    total: int
    recalled_count: int
    recall_rate: float
    confidence_interval: tuple[float, float] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "recalled_count": self.recalled_count,
            "recall_rate": self.recall_rate,
            "confidence_interval": (
                list(self.confidence_interval)
                if self.confidence_interval is not None
                else None
            ),
        }


@dataclass(frozen=True)
class RecallResult:
    """Combined per-fact and aggregate recall evaluation output."""

    per_fact_outcomes: tuple[RecallOutcome, ...]
    aggregate: RecallAggregate

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_fact_outcomes": [o.as_dict() for o in self.per_fact_outcomes],
            "aggregate": self.aggregate.as_dict(),
        }


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def compute_recall_aggregate(
    outcomes: list[RecallOutcome] | tuple[RecallOutcome, ...],
    *,
    confidence_level: float | None = None,
) -> RecallAggregate:
    """Compute aggregate recall statistics with Clopper-Pearson CI.

    Parameters
    ----------
    outcomes:
        Per-fact evaluation outcomes.
    confidence_level:
        If ``None``, derives ``alpha = 1 / n_total`` (sample-size-derived,
        no arbitrary 0.95 constant).  Otherwise, ``alpha = 1 - confidence_level``.

    Returns
    -------
    RecallAggregate with recall_rate and optional exact confidence interval.
    """
    total = len(outcomes)
    if total == 0:
        return RecallAggregate(
            total=0,
            recalled_count=0,
            recall_rate=0.0,
            confidence_interval=None,
        )

    recalled_count = sum(1 for o in outcomes if o.recalled)
    recall_rate = recalled_count / total

    if confidence_level is not None:
        alpha = 1.0 - confidence_level
    else:
        alpha = 1.0 / total

    ci: tuple[float, float] | None = None
    if 0.0 < alpha < 1.0:
        ci = clopper_pearson_interval(
            n_correct=recalled_count,
            n_total=total,
            alpha=alpha,
        )

    return RecallAggregate(
        total=total,
        recalled_count=recalled_count,
        recall_rate=recall_rate,
        confidence_interval=ci,
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

# generate_fn signature follows the BenchmarkService callback pattern:
#   generate_fn(model, tokenizer, prompt, max_tokens, verbose) -> str
GenerateFn = Callable[..., str]


@runtime_checkable
class RecallEvaluator(Protocol):
    """Protocol for fact-recall evaluation.

    Concrete implementations are deferred to patchset 2.  This protocol
    defines the contract that all evaluators must satisfy.
    """

    def evaluate_recall(
        self,
        facts: list[FactTriple],
        generate_fn: GenerateFn,
        model: Any,
        tokenizer: Any,
        mode: RecallMode = RecallMode.raw_completion,
        chat_template: str | None = None,
    ) -> RecallResult:
        """Evaluate recall of *facts* using the given generation function.

        Parameters
        ----------
        facts:
            Facts to probe.
        generate_fn:
            Model generation callback.
        model:
            The model object (passed through to generate_fn).
        tokenizer:
            The tokenizer (passed through to generate_fn).
        mode:
            ``raw_completion`` or ``chat_template``.
        chat_template:
            Template name when mode is ``chat_template``.

        Returns
        -------
        RecallResult with per-fact outcomes and aggregate statistics.
        """
        ...


__all__ = [
    "GenerateFn",
    "RecallAggregate",
    "RecallEvaluator",
    "RecallMode",
    "RecallOutcome",
    "RecallResult",
    "compute_recall_aggregate",
]
