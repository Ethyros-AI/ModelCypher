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
            "probe_count": self.probe_count,
            "layer_indices_analyzed": list(self.layer_indices_analyzed),
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
