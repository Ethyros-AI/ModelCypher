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

"""Signature dataclasses for behavioral analysis.

This module provides immutable dataclasses that capture the geometric signatures
used to detect behavioral drift during model merges. These signatures integrate
with the circuit breaker system for safety monitoring.

Key Metrics:
- Refusal geodesic distance: How far responses are from known refusal patterns
- Persona CKA: Alignment to baseline identity representations
- Factual sensitivity: Knowledge preservation (effect size 0.94)
- Entropy z-score: Stability relative to calibrated baseline
- Trajectory complexity: Intra-layer dynamics (looping vs feedforward)

These signatures capture RAW metrics only. No composite scores or heuristic
thresholds are applied - caller interprets the metrics for their use case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BehavioralSignature:
    """Raw geometric signature for behavioral drift detection.

    All metrics are raw values without thresholds or classifications.
    Callers interpret relative to their baselines and use cases.

    Attributes:
        refusal_geodesic_distance: Geodesic distance to nearest refusal anchor
            in embedding space. Lower = closer to refusal boundary.
            NaN if no refusal probes available.
        refusal_trajectory_slope: Linear trend of refusal distances over probes.
            Negative = approaching refusal boundary. NaN if insufficient probes.
        factual_sensitivity: Mean counterfactual sensitivity across fact pairs.
            Higher = model distinguishes facts from violations (effect size 0.94).
            NaN if no fact pairs available.
        persona_cka_to_baseline: Linear CKA between current and baseline
            identity embeddings. 1.0 = identical personas, 0.0 = orthogonal.
            NaN if no baseline available.
        identity_layer_consistency: CKA consistency across layers for identity
            probes. Higher = more stable identity representation. NaN if
            insufficient layers.
        entropy_z_score: Z-score of response entropy relative to calibrated
            baseline. High positive = unusually high entropy. NaN if no
            baseline calibration available.
        probe_count: Number of probes used in this analysis.
        layer_indices_analyzed: Tuple of layer indices that were analyzed.

    Note:
        This class returns raw metrics only. No composite scores.
        Values may be NaN for degenerate inputs or missing data.
        Caller interprets metrics based on their use case.
    """

    # Refusal behavior
    refusal_geodesic_distance: float  # Distance to refusal anchor - may be NaN
    refusal_trajectory_slope: float  # Trend direction - may be NaN

    # Capability preservation
    factual_sensitivity: float  # Mean counterfactual sensitivity - may be NaN

    # Persona stability
    persona_cka_to_baseline: float  # CKA to baseline - may be NaN
    identity_layer_consistency: float  # Cross-layer CKA - may be NaN

    # Entropy characteristics
    entropy_z_score: float  # Z-score from baseline - may be NaN

    # Metadata
    probe_count: int
    layer_indices_analyzed: tuple[int, ...]

    # Optional entropy data for circuit breaker signal conversion
    mean_entropy: float = float("nan")  # Raw mean entropy for normalization
    vocab_size: int = 0  # Model vocab size for normalization (0 = unknown)

    # Trajectory complexity metrics (optional - computed on demand)
    trajectory_path_ratio: float = float("nan")  # Path length / direct distance
    trajectory_mean_curvature: float = float("nan")  # Mean curvature (radians)
    trajectory_return_cka: float = float("nan")  # Mean non-adjacent layer CKA
    trajectory_effective_rank: float = float("nan")  # Shannon effective rank

    # Layer-wise entropy trajectory (Entropy-Lens approach)
    entropy_trajectory_slope: float = float("nan")  # Linear trend across layers
    entropy_peak_layer_fraction: float = float("nan")  # Peak location [0,1]
    entropy_monotonicity: float = float("nan")  # Spearman correlation [-1,1]
    entropy_early_late_ratio: float = float("nan")  # Early/late mean ratio

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary with all signature fields. NaN values are preserved.
        """
        return {
            "refusal_geodesic_distance": self.refusal_geodesic_distance,
            "refusal_trajectory_slope": self.refusal_trajectory_slope,
            "factual_sensitivity": self.factual_sensitivity,
            "persona_cka_to_baseline": self.persona_cka_to_baseline,
            "identity_layer_consistency": self.identity_layer_consistency,
            "entropy_z_score": self.entropy_z_score,
            "mean_entropy": self.mean_entropy,
            "vocab_size": self.vocab_size,
            "probe_count": self.probe_count,
            "layer_indices_analyzed": list(self.layer_indices_analyzed),
            "trajectory_path_ratio": self.trajectory_path_ratio,
            "trajectory_mean_curvature": self.trajectory_mean_curvature,
            "trajectory_return_cka": self.trajectory_return_cka,
            "trajectory_effective_rank": self.trajectory_effective_rank,
            "entropy_trajectory_slope": self.entropy_trajectory_slope,
            "entropy_peak_layer_fraction": self.entropy_peak_layer_fraction,
            "entropy_monotonicity": self.entropy_monotonicity,
            "entropy_early_late_ratio": self.entropy_early_late_ratio,
        }

    @property
    def has_refusal_data(self) -> bool:
        """Whether refusal metrics are available (not NaN)."""
        return not math.isnan(self.refusal_geodesic_distance)

    @property
    def has_capability_data(self) -> bool:
        """Whether capability metrics are available (not NaN)."""
        return not math.isnan(self.factual_sensitivity)

    @property
    def has_persona_data(self) -> bool:
        """Whether persona metrics are available (not NaN)."""
        return not math.isnan(self.persona_cka_to_baseline)

    @property
    def has_entropy_data(self) -> bool:
        """Whether entropy metrics are available (not NaN)."""
        return not math.isnan(self.entropy_z_score)

    @property
    def has_trajectory_data(self) -> bool:
        """Whether trajectory complexity metrics are available (not NaN)."""
        return not math.isnan(self.trajectory_path_ratio)

    @property
    def has_entropy_trajectory_data(self) -> bool:
        """Whether entropy trajectory metrics are available (not NaN)."""
        return not math.isnan(self.entropy_trajectory_slope)

    @property
    def signal_availability(self) -> float:
        """Fraction of signals that are available (not NaN).

        Returns:
            Float in [0, 1] indicating what fraction of the 4 signal
            categories have valid data.
        """
        available = sum(
            [
                self.has_refusal_data,
                self.has_capability_data,
                self.has_persona_data,
                self.has_entropy_data,
            ]
        )
        return available / 4.0


@dataclass(frozen=True)
class RefusalBoundaryResult:
    """Result from refusal boundary analysis.

    Attributes:
        min_distance: Minimum geodesic distance to any refusal anchor.
        mean_distance: Mean geodesic distance across all refusal anchors.
        distances: Raw distances to each refusal anchor.
        anchor_count: Number of refusal anchors tested.
    """

    min_distance: float
    mean_distance: float
    distances: tuple[float, ...]
    anchor_count: int


@dataclass(frozen=True)
class CapabilityPreservationResult:
    """Result from capability preservation analysis.

    Attributes:
        mean_sensitivity: Mean counterfactual sensitivity across pairs.
        sensitivities: Raw sensitivity for each fact pair.
        pair_count: Number of fact/counterfactual pairs tested.
    """

    mean_sensitivity: float
    sensitivities: tuple[float, ...]
    pair_count: int


@dataclass(frozen=True)
class PersonaStabilityResult:
    """Result from persona stability analysis.

    Attributes:
        cka_to_baseline: CKA between current and baseline persona.
        layer_consistency: CKA consistency across layers.
        layer_cka_values: CKA values for each layer pair.
        layers_analyzed: Layers included in analysis.
    """

    cka_to_baseline: float
    layer_consistency: float
    layer_cka_values: tuple[float, ...]
    layers_analyzed: tuple[int, ...]


@dataclass(frozen=True)
class EntropyAnalysisResult:
    """Result from entropy analysis.

    Attributes:
        mean_entropy: Mean raw entropy across probe texts.
        z_score: Z-score relative to calibrated baseline (if available).
        entropies: Raw entropy values for each probe text.
        probe_count: Number of probes analyzed.
        vocab_size: Model vocabulary size used for normalization.
    """

    mean_entropy: float
    z_score: float
    entropies: tuple[float, ...]
    probe_count: int
    vocab_size: int


@dataclass(frozen=True)
class EntropyTrajectoryResult:
    """Result from layer-wise entropy trajectory analysis.

    Captures how entropy evolves through transformer layers using the
    Entropy-Lens approach (Ali et al., 2025). Each layer's hidden state
    is projected through the unembedding matrix to compute per-layer entropy.

    Trajectory Features:
    - slope: Linear trend of entropy across layers (positive = increasing)
    - peak_layer_fraction: Location of max entropy as fraction of depth [0,1]
    - monotonicity: Spearman correlation with layer order [-1,1]
    - early_late_ratio: Early-layer mean / late-layer mean entropy ratio

    Attributes:
        layer_entropies: Raw entropy values per layer, in layer order.
        layer_indices: Layer indices corresponding to entropy values.
        slope: Linear regression slope of entropy vs layer index.
            Positive = entropy increases through layers.
            Negative = entropy decreases (model becomes more confident).
        peak_layer_fraction: Layer index of max entropy as fraction of depth.
            0.0 = peak at input, 1.0 = peak at output.
        monotonicity: Spearman rank correlation between layer index and entropy.
            1.0 = perfectly monotonic increasing, -1.0 = monotonic decreasing.
        early_late_ratio: Ratio of early-layer mean entropy to late-layer mean.
            >1.0 = higher early entropy, <1.0 = higher late entropy.
        vocab_size: Model vocabulary size (for max entropy calculation).
        max_possible_entropy: ln(vocab_size) - theoretical maximum.
        probe_count: Number of probe prompts used for measurement.
    """

    layer_entropies: tuple[float, ...]
    layer_indices: tuple[int, ...]
    slope: float
    peak_layer_fraction: float
    monotonicity: float
    early_late_ratio: float
    vocab_size: int
    max_possible_entropy: float
    probe_count: int

    @property
    def normalized_trajectory(self) -> tuple[float, ...]:
        """Return entropy trajectory normalized to [0, 1] by max possible."""
        if self.max_possible_entropy <= 0:
            return tuple(0.0 for _ in self.layer_entropies)
        return tuple(e / self.max_possible_entropy for e in self.layer_entropies)

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "layer_entropies": list(self.layer_entropies),
            "layer_indices": list(self.layer_indices),
            "slope": self.slope,
            "peak_layer_fraction": self.peak_layer_fraction,
            "monotonicity": self.monotonicity,
            "early_late_ratio": self.early_late_ratio,
            "vocab_size": self.vocab_size,
            "max_possible_entropy": self.max_possible_entropy,
            "probe_count": self.probe_count,
        }
