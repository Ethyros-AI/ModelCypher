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

"""Data types for prime geometry analysis."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

__all__ = [
    "EmbeddingType",
    "BaselineType",
    "PrimeSequence",
    "EigenvalueDistribution",
    "SpectralComparison",
    "PrimeGeometryResult",
    "ConfidenceInterval",
    "EffectSize",
    "HypothesisTest",
    "ComprehensiveResult",
    "ScaleSweepResult",
    "PerturbationResult",
]


class EmbeddingType(Enum):
    """Types of embeddings for prime sequence analysis."""

    TIME_DELAY = "time_delay"
    RESIDUE = "residue"
    DIGIT = "digit"
    POSITION = "position"


class BaselineType(Enum):
    """Types of random baselines for comparison."""

    EXPONENTIAL = "exponential"  # Gaps between Poisson events
    UNIFORM = "uniform"  # Uniform distribution
    POISSON = "poisson"  # Poisson-distributed gaps
    CRAMER = "cramer"  # Cramér probabilistic model
    SHUFFLED = "shuffled"  # Shuffled prime gaps


@dataclass(frozen=True)
class PrimeSequence:
    """A sequence of prime numbers with derived properties."""

    primes: "Array"  # The prime numbers [n_primes]
    gaps: "Array"  # Prime gaps: p[i+1] - p[i] [n_primes - 1]
    count: int
    max_prime: int

    @property
    def gap_count(self) -> int:
        return self.count - 1


@dataclass(frozen=True)
class EigenvalueDistribution:
    """Eigenvalue distribution of a Gram matrix."""

    eigenvalues: "Array"  # Sorted eigenvalues (descending)
    participation_ratio: float  # Effective rank: (sum(λ))^2 / sum(λ^2)
    spectral_entropy: float  # -sum(p * log(p)) where p = λ/sum(λ)
    condition_number: float  # λ_max / λ_min (for positive eigenvalues)
    top_k_ratio: float  # sum(top 10 eigenvalues) / sum(all eigenvalues)


@dataclass(frozen=True)
class SpectralComparison:
    """Comparison of two eigenvalue distributions."""

    source_label: str
    target_label: str
    participation_ratio_diff: float
    spectral_entropy_diff: float
    wasserstein_distance: float  # W1 distance between normalized spectra
    ks_statistic: float  # Kolmogorov-Smirnov statistic


@dataclass(frozen=True)
class PrimeGeometryResult:
    """Complete analysis of prime number geometry."""

    # Source data
    prime_count: int
    embedding_dim: int

    # Eigenvalue analysis
    prime_eigenvalues: EigenvalueDistribution
    random_eigenvalues: EigenvalueDistribution
    comparison: SpectralComparison

    # Intrinsic dimension
    prime_intrinsic_dim: float
    random_intrinsic_dim: float

    # CKA between different representations
    gap_to_position_cka: float  # CKA between gap and position embeddings

    # Raw data for further analysis
    prime_gram: "Array"
    random_gram: "Array"


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap interval bounds from resampling."""

    lower: float
    upper: float
    mean: float
    std: float
    n_bootstrap: int


@dataclass(frozen=True)
class EffectSize:
    """Cohen's d effect size."""

    d: float  # Cohen's d: (mean1 - mean2) / pooled_std

    @staticmethod
    def from_cohens_d(d: float) -> "EffectSize":
        """Create EffectSize from Cohen's d value."""
        return EffectSize(d=d)


@dataclass(frozen=True)
class HypothesisTest:
    """Result of a single hypothesis test."""

    hypothesis_id: str  # H1-H8
    description: str
    passed: bool | None  # Always None (no pass/fail thresholds)
    p_value: float | None  # None when samples unavailable for statistical test
    effect_size: EffectSize
    prime_value: float
    baseline_value: float
    confidence_interval: ConfidenceInterval | None = None


@dataclass
class ComprehensiveResult:
    """Complete results from comprehensive prime geometry analysis."""

    # Metadata
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Scale info
    n_primes: int = 0
    max_prime: int = 0
    embedding_dim: int = 20

    # Results by embedding type
    embedding_results: dict[str, EigenvalueDistribution] = field(default_factory=dict)

    # Results by baseline type
    baseline_results: dict[str, EigenvalueDistribution] = field(default_factory=dict)

    # Pairwise comparisons
    comparisons: dict[str, SpectralComparison] = field(default_factory=dict)

    # Hypothesis tests
    hypothesis_tests: dict[str, HypothesisTest] = field(default_factory=dict)

    # Summary statistics
    summary: dict[str, float] = field(default_factory=dict)


@dataclass
class ScaleSweepResult:
    """Results from testing across multiple scales."""

    scales: list[int] = field(default_factory=list)  # n_primes values tested
    results: list[ComprehensiveResult] = field(default_factory=list)

    # Trend analysis
    effect_size_trend: list[float] = field(default_factory=list)
    p_value_trend: list[float] = field(default_factory=list)
    scale_invariance_fraction: float | None = None


@dataclass
class PerturbationResult:
    """Results from perturbation robustness testing."""

    noise_level: float
    original_participation_ratio: float
    perturbed_participation_ratio: float
    stability_score: float  # 1 - relative change
