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

"""Entropy Delta Sample: Raw geometric measurements of adapter-base divergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon, inf_value

# =============================================================================
# Baseline Distribution for Geometry-Derived Comparisons
# =============================================================================


@dataclass(frozen=True)
class BaselineDistribution:
    """Baseline distribution learned from calibration data.

    Attributes
    ----------
    mean : float
        Mean value from calibration samples.
    std : float
        Standard deviation from calibration samples.
    """

    mean: float
    std: float

    def z_score(self, value: float) -> float:
        """Compute z-score: how many standard deviations from mean."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        val_arr = backend.array([value])
        mean_arr = backend.array([self.mean])
        std_arr = backend.array([self.std])
        diff = val_arr - mean_arr
        abs_diff = backend.abs(diff)
        eps_arr = backend.array([eps])
        std_small = std_arr < eps_arr
        within = abs_diff < eps_arr
        zero_arr = backend.zeros_like(val_arr)
        inf_arr = backend.full(val_arr.shape, inf_value(backend))
        z_arr = diff / std_arr
        result = backend.where(std_small, backend.where(within, zero_arr, inf_arr), z_arr)
        backend.eval(result)
        return float(backend.to_scalar(result))

    @classmethod
    def from_samples(cls, values: list[float]) -> "BaselineDistribution":
        """Compute baseline from calibration samples."""
        if not values:
            raise ValueError("Cannot compute baseline from empty samples")
        backend = get_default_backend()
        values_arr = backend.array(values)
        mean_arr = backend.mean(values_arr)
        variance_arr = backend.var(values_arr)
        std_arr = backend.sqrt(variance_arr)
        backend.eval(mean_arr, std_arr)
        return cls(
            mean=float(backend.to_scalar(mean_arr)),
            std=float(backend.to_scalar(std_arr)),
        )


# =============================================================================
# Entropy Delta Sample
# =============================================================================


@dataclass(frozen=True)
class EntropyDeltaSample:
    """Raw geometric measurements of adapter-base divergence.

    Notes
    -----
    Entropy distributions are model-specific. Use calibrated baselines:
    - anomaly_z_score(baseline) for normalization against calibration stats
    """

    id: UUID
    token_index: int
    generated_token: int
    base_entropy: float
    base_logit_variance: float
    base_top_token: int
    adapter_entropy: float
    adapter_logit_variance: float
    adapter_top_token: int
    base_logit_margin: float | None = None
    base_token_logit: float | None = None
    base_rank_fraction: float | None = None
    base_frontier_hit: bool | None = None
    kl_divergence_adapter_to_base: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float = 0.0
    correlation_id: UUID | None = None
    source: str | None = None

    @staticmethod
    def create(
        token_index: int,
        generated_token: int,
        base_entropy: float,
        base_logit_variance: float,
        base_top_token: int,
        adapter_entropy: float,
        adapter_logit_variance: float,
        adapter_top_token: int,
        base_logit_margin: float | None = None,
        base_token_logit: float | None = None,
        base_rank_fraction: float | None = None,
        base_frontier_hit: bool | None = None,
        kl_divergence_adapter_to_base: float | None = None,
        latency_ms: float = 0.0,
        correlation_id: UUID | None = None,
        source: str | None = None,
    ) -> "EntropyDeltaSample":
        return EntropyDeltaSample(
            id=uuid4(),
            token_index=token_index,
            generated_token=generated_token,
            base_entropy=base_entropy,
            base_logit_variance=base_logit_variance,
            base_top_token=base_top_token,
            adapter_entropy=adapter_entropy,
            adapter_logit_variance=adapter_logit_variance,
            adapter_top_token=adapter_top_token,
            base_logit_margin=base_logit_margin,
            base_token_logit=base_token_logit,
            base_rank_fraction=base_rank_fraction,
            base_frontier_hit=base_frontier_hit,
            kl_divergence_adapter_to_base=kl_divergence_adapter_to_base,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
            source=source,
        )

    @property
    def delta(self) -> float:
        """Entropy delta: base - adapter."""
        return self.base_entropy - self.adapter_entropy

    @property
    def top_token_disagreement(self) -> bool:
        """Whether base and adapter disagree on top token."""
        return self.base_top_token != self.adapter_top_token

    @property
    def variance_delta(self) -> float:
        """Variance delta: base - adapter."""
        backend = get_default_backend()
        base_arr = backend.array([self.base_logit_variance])
        adapter_arr = backend.array([self.adapter_logit_variance])
        delta_arr = base_arr - adapter_arr
        backend.eval(delta_arr)
        return float(backend.to_scalar(delta_arr))

    @property
    def anomaly_score(self) -> float:
        """Entropy ratio measuring base uncertainty relative to adapter confidence."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        delta_arr = backend.array([self.delta])
        positive_delta = backend.maximum(delta_arr, backend.array([0.0]))
        denom = backend.maximum(backend.array([self.base_entropy]), backend.array([eps]))
        score_arr = positive_delta / denom
        backend.eval(score_arr)
        return float(backend.to_scalar(score_arr))

    def anomaly_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute z-score relative to calibration baseline."""
        return baseline.z_score(self.anomaly_score)


# =============================================================================
# Entropy Delta Session Result
# =============================================================================


@dataclass(frozen=True)
class EntropyDeltaSessionResult:
    """Aggregated entropy delta metrics over a generation session.

    Use raw measurements directly or with BaselineDistribution.z_score()
    for normalization against calibration stats.
    """

    session_id: UUID
    correlation_id: UUID | None
    session_start: datetime
    session_end: datetime
    total_tokens: int
    anomaly_count: int
    max_anomaly_score: float
    avg_delta: float
    disagreement_rate: float
    avg_base_logit_margin: float | None = None
    max_base_logit_margin: float | None = None
    conflict_analysis: ConflictAnalysis | None = None
    samples: list[EntropyDeltaSample] = field(default_factory=list)

    def security_z_score(self, baseline: BaselineDistribution) -> float:
        """Compute security z-score relative to calibration baseline."""
        return baseline.z_score(self.max_anomaly_score)

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return (self.session_end - self.session_start).total_seconds()

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per token in milliseconds."""
        if not self.samples:
            return 0.0
        backend = get_default_backend()
        lat_arr = backend.array([sample.latency_ms for sample in self.samples])
        mean_arr = backend.mean(lat_arr)
        backend.eval(mean_arr)
        return float(backend.to_scalar(mean_arr))
