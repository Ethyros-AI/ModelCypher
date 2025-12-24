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

"""Orchestrates multi-LoRA adapter composition for ensemble inference.

Enables combining multiple specialized LoRA adapters at inference time,
allowing models to leverage multiple skill domains simultaneously.

Two composition strategies are supported:
- **Weight Blending**: High-compatibility adapters merged via linear interpolation
- **Attention Routing**: Different specializations route tokens to specific experts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from uuid import UUID, uuid4


class CompositionStrategy(str, Enum):
    """Strategy for composing multiple adapters."""

    WEIGHT_BLENDING = "weight_blending"
    """Blend adapter weights via linear interpolation: W' = Σ αᵢ * W_loraᵢ

    Best for adapters with similar specializations or high geometric compatibility.
    """

    ATTENTION_ROUTING = "attention_routing"
    """Route tokens to specialized adapters based on content.

    Best for adapters with different specializations (e.g., code vs prose).
    """


@dataclass(frozen=True)
class AdapterInfo:
    """Information about an adapter in an ensemble."""

    id: UUID
    name: str
    compatibility_score: float | None = None


@dataclass(frozen=True)
class Ensemble:
    """Represents an active ensemble of adapters."""

    id: UUID
    """Unique identifier for this ensemble."""

    adapters: list[AdapterInfo]
    """Adapters participating in this ensemble."""

    weights: dict[UUID, float]
    """Weight for each adapter (by adapter ID)."""

    strategy: CompositionStrategy
    """Composition strategy in use."""

    created_at: datetime = field(default_factory=datetime.now)
    """When the ensemble was created."""

    @property
    def dominant_adapter(self) -> AdapterInfo | None:
        """Returns the dominant adapter (highest weight)."""
        if not self.weights:
            return None

        max_weight = max(self.weights.values())
        dominant_id = next(
            (id_ for id_, weight in self.weights.items() if weight == max_weight),
            None,
        )

        if dominant_id is None:
            return None

        return next(
            (adapter for adapter in self.adapters if adapter.id == dominant_id),
            None,
        )


@dataclass(frozen=True)
class EnsembleResult:
    """Result of an ensemble creation operation."""

    ensemble: Ensemble
    strategy_suggested: CompositionStrategy
    compatibility_scores: dict[UUID, float]
    warnings: list[str]


@dataclass
class OrchestratorConfiguration:
    """Configuration for the orchestrator."""

    max_adapters: int = 4
    """Maximum number of adapters in an ensemble."""

    min_fit_score: float = 0.4
    """Minimum fit score required for an adapter to join an ensemble."""

    weight_blending_threshold: float = 0.7
    """Threshold above which weight blending is preferred."""

    auto_select_strategy: bool = True
    """Whether to auto-select composition strategy based on compatibility."""

    @classmethod
    def default(cls) -> OrchestratorConfiguration:
        """Default configuration."""
        return cls()


class EnsembleOrchestratorError(Exception):
    """Errors from ensemble orchestration."""

    pass


class NoAdaptersError(EnsembleOrchestratorError):
    """No adapters provided for ensemble creation."""

    def __init__(self) -> None:
        super().__init__("No adapters provided for ensemble creation")


class NoCompatibleAdaptersError(EnsembleOrchestratorError):
    """No adapters meet the minimum compatibility threshold."""

    def __init__(self) -> None:
        super().__init__("No adapters meet the minimum compatibility threshold")


class TooManyAdaptersError(EnsembleOrchestratorError):
    """Too many adapters for ensemble."""

    def __init__(self, count: int, max_count: int) -> None:
        super().__init__(f"Too many adapters ({count}) for ensemble. Maximum is {max_count}")
        self.count = count
        self.max_count = max_count


class NoActiveEnsembleError(EnsembleOrchestratorError):
    """No active ensemble to modify."""

    def __init__(self) -> None:
        super().__init__("No active ensemble to modify")


class InvalidAdapterIDError(EnsembleOrchestratorError):
    """Invalid adapter ID."""

    def __init__(self, adapter_id: UUID) -> None:
        super().__init__(f"Invalid adapter ID: {adapter_id}")
        self.adapter_id = adapter_id


class EnsembleOrchestrator:
    """Orchestrates multi-LoRA adapter composition for ensemble inference."""

    def __init__(
        self,
        configuration: OrchestratorConfiguration | None = None,
    ):
        self.configuration = configuration or OrchestratorConfiguration.default()
        self._active_ensemble: Ensemble | None = None
        self._stabilizer_adapter: AdapterInfo | None = None

    def create_ensemble(
        self,
        adapters: list[AdapterInfo],
        compatibility_scores: dict[UUID, float] | None = None,
        strategy: CompositionStrategy | None = None,
    ) -> EnsembleResult:
        """Create an ensemble from a set of adapters.

        Evaluates adapter compatibility, assigns weights based on scores,
        and selects composition strategy.

        Args:
            adapters: Adapters to include in the ensemble.
            compatibility_scores: Pre-computed compatibility scores by adapter ID.
            strategy: Composition strategy (None for auto-selection).

        Returns:
            The created ensemble with compatibility information.

        Raises:
            NoAdaptersError: If no adapters provided.
            TooManyAdaptersError: If too many adapters provided.
            NoCompatibleAdaptersError: If no adapters meet threshold.
        """
        if not adapters:
            raise NoAdaptersError()

        if len(adapters) > self.configuration.max_adapters:
            raise TooManyAdaptersError(len(adapters), self.configuration.max_adapters)

        # Use provided scores or compute defaults
        scores = compatibility_scores or {}
        warnings: list[str] = []
        valid_adapters: list[AdapterInfo] = []

        for adapter in adapters:
            score = scores.get(adapter.id, adapter.compatibility_score or 0.5)
            scores[adapter.id] = score

            if score >= self.configuration.min_fit_score:
                valid_adapters.append(adapter)
            else:
                warnings.append(
                    f"Adapter '{adapter.name}' excluded: fit score {score:.2f} below threshold"
                )

        if not valid_adapters:
            raise NoCompatibleAdaptersError()

        # Compute weights from compatibility scores
        weights = self._compute_weights(valid_adapters, scores)

        # Select strategy
        avg_compatibility = sum(scores.values()) / len(scores) if scores else 0.0
        suggested_strategy = (
            CompositionStrategy.WEIGHT_BLENDING
            if avg_compatibility >= self.configuration.weight_blending_threshold
            else CompositionStrategy.ATTENTION_ROUTING
        )

        if strategy is not None:
            selected_strategy = strategy
            if strategy != suggested_strategy:
                warnings.append(
                    f"Using {strategy.value} but {suggested_strategy.value} is recommended based on compatibility"
                )
        elif self.configuration.auto_select_strategy:
            selected_strategy = suggested_strategy
        else:
            selected_strategy = CompositionStrategy.WEIGHT_BLENDING

        ensemble = Ensemble(
            id=uuid4(),
            adapters=list(valid_adapters),
            weights=weights,
            strategy=selected_strategy,
        )

        self._active_ensemble = ensemble

        return EnsembleResult(
            ensemble=ensemble,
            strategy_suggested=suggested_strategy,
            compatibility_scores=scores,
            warnings=warnings,
        )

    def rebalance(self, weights: dict[UUID, float]) -> None:
        """Rebalance weights in the active ensemble.

        Args:
            weights: New weights for each adapter (by adapter ID).

        Raises:
            NoActiveEnsembleError: If no active ensemble.
            InvalidAdapterIDError: If an adapter ID is not in the ensemble.
        """
        if self._active_ensemble is None:
            raise NoActiveEnsembleError()

        # Validate all adapter IDs are in the ensemble
        adapter_ids = {adapter.id for adapter in self._active_ensemble.adapters}
        for adapter_id in weights.keys():
            if adapter_id not in adapter_ids:
                raise InvalidAdapterIDError(adapter_id)

        # Normalize weights to sum to 1.0
        normalized_weights = self._normalize_weights(weights)

        # Create new ensemble with updated weights
        self._active_ensemble = Ensemble(
            id=self._active_ensemble.id,
            adapters=self._active_ensemble.adapters,
            weights=normalized_weights,
            strategy=self._active_ensemble.strategy,
            created_at=self._active_ensemble.created_at,
        )

    def stabilizer_takeover(self) -> None:
        """Emergency takeover: disband ensemble and activate stabilizer adapter.

        Used when ensemble behavior becomes unstable or produces poor outputs.
        The stabilizer is a known-safe adapter that provides baseline functionality.
        """
        if self._stabilizer_adapter is None:
            self._active_ensemble = None
            return

        stabilizer = self._stabilizer_adapter

        # Create single-adapter "ensemble" with the stabilizer
        self._active_ensemble = Ensemble(
            id=uuid4(),
            adapters=[stabilizer],
            weights={stabilizer.id: 1.0},
            strategy=CompositionStrategy.WEIGHT_BLENDING,
        )

    def set_stabilizer(self, adapter: AdapterInfo | None) -> None:
        """Configure the stabilizer adapter for emergency takeover.

        Args:
            adapter: The adapter to use as a stabilizer.
        """
        self._stabilizer_adapter = adapter

    def current_ensemble(self) -> Ensemble | None:
        """Get the currently active ensemble."""
        return self._active_ensemble

    def disband_ensemble(self) -> None:
        """Disband the current ensemble."""
        self._active_ensemble = None

    def _compute_weights(
        self,
        adapters: list[AdapterInfo],
        scores: dict[UUID, float],
    ) -> dict[UUID, float]:
        """Compute weights proportional to compatibility score."""
        total_score = sum(scores.get(adapter.id, 0) for adapter in adapters)

        if total_score <= 0:
            # Equal weights if no scores
            equal_weight = 1.0 / len(adapters)
            return {adapter.id: equal_weight for adapter in adapters}

        return {
            adapter.id: scores.get(adapter.id, 0) / total_score
            for adapter in adapters
        }

    def _normalize_weights(self, weights: dict[UUID, float]) -> dict[UUID, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {id_: weight / total for id_, weight in weights.items()}
