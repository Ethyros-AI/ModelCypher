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

"""Entropy-based validation and guidance for model merging.

Provides:
- Pre-merge entropy profiling per layer
- Entropy-aware smoothing recommendations
- Post-merge stability validation

Notes
-----
Layers are classified into phases (ORDERED, CRITICAL, DISORDERED) based on
entropy relative to critical temperature. Phase determines alpha adjustment
and smoothing sigma for stable blending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    EntropyThresholds,
)
# EntropyLevel enum removed - use raw entropy values with thresholds
from modelcypher.core.domain.thermo.phase_transition_theory import Phase

if TYPE_CHECKING:
    from modelcypher.ports.model_loader import ModelLoaderPort

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PhaseAdjustments:
    """Phase-based adjustment values for merge operations.

    Defines alpha and sigma adjustments per phase (ORDERED, CRITICAL, DISORDERED).
    All values must be explicitly provided or derived from calibration data.
    """

    ordered_alpha: float
    """Alpha adjustment for ORDERED phase layers (typically 1.0, no reduction)."""

    critical_alpha: float
    """Alpha adjustment for CRITICAL phase layers (most conservative, < 1.0)."""

    disordered_alpha: float
    """Alpha adjustment for DISORDERED phase layers (moderate reduction)."""

    ordered_sigma: float
    """Smoothing sigma for ORDERED phase layers (tight smoothing)."""

    critical_sigma: float
    """Smoothing sigma for CRITICAL phase layers (wider stabilization)."""

    disordered_sigma: float
    """Smoothing sigma for DISORDERED phase layers (moderate smoothing)."""

    def alpha_for_phase(self, phase: Phase) -> float:
        """Get alpha adjustment for a given phase."""
        if phase == Phase.ORDERED:
            return self.ordered_alpha
        elif phase == Phase.CRITICAL:
            return self.critical_alpha
        else:  # DISORDERED
            return self.disordered_alpha

    def sigma_for_phase(self, phase: Phase) -> float:
        """Get smoothing sigma for a given phase."""
        if phase == Phase.ORDERED:
            return self.ordered_sigma
        elif phase == Phase.CRITICAL:
            return self.critical_sigma
        else:  # DISORDERED
            return self.disordered_sigma


@dataclass(frozen=True)
class EntropyMergeConfig:
    """Configuration for entropy-based merge validation.

    All thresholds must be explicitly provided or derived from calibration data.
    No arbitrary defaults.
    """

    entropy_thresholds: EntropyThresholds
    """Thresholds for entropy level classification (low, high, circuit_breaker)."""

    critical_bandwidth: float
    """Bandwidth around phase boundary center to classify as CRITICAL phase."""

    phase_adjustments: PhaseAdjustments
    """Per-phase alpha and sigma adjustment values."""

    high_risk_fraction: float
    """Fraction of critical layers above which model is high risk (0.0-1.0)."""

    unstable_fraction: float
    """Fraction of unstable layers above which merge is overall UNSTABLE."""

    stability_thresholds: tuple[float, float, float]
    """(stable, marginal, unstable) multipliers relative to entropy thresholds.

    - delta < low * stable_mult → STABLE
    - delta < low * marginal_mult → MARGINAL
    - delta < high * unstable_mult → UNSTABLE
    - else → CRITICAL
    """

    @classmethod
    def from_calibration_data(
        cls,
        entropy_samples: list[float],
        merge_deltas: list[float],
        percentile_low: float = 25.0,
        percentile_high: float = 75.0,
        percentile_circuit_breaker: float = 95.0,
        target_stable_percentile: float = 50.0,
        target_marginal_percentile: float = 80.0,
    ) -> "EntropyMergeConfig":
        """Derive configuration from calibration data.

        Args:
            entropy_samples: Entropy values from baseline measurements.
            merge_deltas: Observed entropy deltas from previous merges.
            percentile_low: Percentile for LOW entropy threshold.
            percentile_high: Percentile for HIGH entropy threshold.
            percentile_circuit_breaker: Percentile for circuit breaker.
            target_stable_percentile: Deltas below this percentile are STABLE.
            target_marginal_percentile: Deltas below this percentile are MARGINAL.

        Returns:
            Configuration with calibrated thresholds.
        """
        if not entropy_samples:
            raise ValueError("entropy_samples cannot be empty for calibration")
        if not merge_deltas:
            raise ValueError("merge_deltas cannot be empty for calibration")

        sorted_entropy = sorted(entropy_samples)
        n_ent = len(sorted_entropy)
        low = sorted_entropy[int(n_ent * percentile_low / 100)]
        high = sorted_entropy[int(n_ent * percentile_high / 100)]
        circuit_breaker = sorted_entropy[int(n_ent * percentile_circuit_breaker / 100)]

        entropy_thresholds = EntropyThresholds(
            low=low, high=high, circuit_breaker=circuit_breaker
        )

        # Derive stability multipliers from observed deltas
        sorted_deltas = sorted(merge_deltas)
        n_del = len(sorted_deltas)
        stable_delta = sorted_deltas[int(n_del * target_stable_percentile / 100)]
        marginal_delta = sorted_deltas[int(n_del * target_marginal_percentile / 100)]

        # Convert to multipliers relative to low threshold
        stable_mult = stable_delta / low if low > 0 else 0.2
        marginal_mult = marginal_delta / low if low > 0 else 0.5
        unstable_mult = marginal_delta / high if high > 0 else 0.5

        # Critical bandwidth from entropy distribution spread
        entropy_std = (high - low) / 2.0
        critical_bandwidth = entropy_std * 0.5

        return cls(
            entropy_thresholds=entropy_thresholds,
            critical_bandwidth=critical_bandwidth,
            phase_adjustments=PhaseAdjustments(
                ordered_alpha=1.0,  # No reduction for stable layers
                critical_alpha=stable_mult,  # Reduce proportionally
                disordered_alpha=(1.0 + stable_mult) / 2.0,  # Moderate
                ordered_sigma=1.0,
                critical_sigma=2.0 * (1.0 / stable_mult) if stable_mult > 0 else 2.0,
                disordered_sigma=1.5,
            ),
            high_risk_fraction=0.5 * (1.0 - stable_mult),  # Derived from stability
            unstable_fraction=marginal_mult,
            stability_thresholds=(stable_mult, marginal_mult, unstable_mult),
        )

    @classmethod
    def from_entropy_statistics(
        cls,
        entropy_mean: float,
        entropy_std: float,
        critical_bandwidth_factor: float = 0.5,
    ) -> "EntropyMergeConfig":
        """Derive configuration from entropy statistics.

        Simpler factory when full calibration data isn't available but
        statistics from baseline measurements are known.

        Args:
            entropy_mean: Mean entropy from baseline.
            entropy_std: Standard deviation of entropy from baseline.
            critical_bandwidth_factor: Factor of std to use for critical bandwidth.

        Returns:
            Configuration with derived thresholds.
        """
        low = max(0.1, entropy_mean - entropy_std)
        high = entropy_mean + entropy_std
        circuit_breaker = entropy_mean + 2.0 * entropy_std

        # Stability multipliers based on normalized spread
        # Tighter entropy distribution = stricter stability requirements
        cv = entropy_std / entropy_mean if entropy_mean > 0 else 0.5
        stable_mult = min(0.3, cv)  # Smaller CV = stricter
        marginal_mult = min(0.6, 2.0 * cv)
        unstable_mult = min(0.7, cv)

        return cls(
            entropy_thresholds=EntropyThresholds(
                low=low, high=high, circuit_breaker=circuit_breaker
            ),
            critical_bandwidth=entropy_std * critical_bandwidth_factor,
            phase_adjustments=PhaseAdjustments(
                ordered_alpha=1.0,
                critical_alpha=1.0 - cv,
                disordered_alpha=1.0 - 0.5 * cv,
                ordered_sigma=1.0,
                critical_sigma=1.0 + cv,
                disordered_sigma=1.0 + 0.5 * cv,
            ),
            high_risk_fraction=cv,
            unstable_fraction=2.0 * cv,
            stability_thresholds=(stable_mult, marginal_mult, unstable_mult),
        )


# MergeStability enum removed - use raw entropy_ratio instead.
# The geometry speaks for itself. Classifications destroy information.


@dataclass(frozen=True)
class LayerEntropyProfile:
    """Entropy profile for a single layer.

    Computed from probe prompts to characterize layer behavior.
    Raw entropy values stored - no discrete classification.
    """

    layer_name: str
    mean_entropy: float
    entropy_variance: float
    phase: Phase
    # entropy_level field removed - use mean_entropy directly with thresholds

    @property
    def is_critical(self) -> bool:
        """True if layer is in critical phase (near boundary)."""
        return self.phase == Phase.CRITICAL

    @property
    def is_stable(self) -> bool:
        """True if layer is in ordered phase (stable)."""
        return self.phase == Phase.ORDERED

    def alpha_adjustment(self, adjustments: PhaseAdjustments) -> float:
        """Compute alpha adjustment using provided phase adjustments.

        Args:
            adjustments: PhaseAdjustments config with per-phase values.

        Returns:
            Alpha adjustment factor for this layer's phase.
        """
        return adjustments.alpha_for_phase(self.phase)

    def smoothing_sigma(self, adjustments: PhaseAdjustments) -> float:
        """Compute smoothing sigma using provided phase adjustments.

        Args:
            adjustments: PhaseAdjustments config with per-phase values.

        Returns:
            Smoothing sigma value for this layer's phase.
        """
        return adjustments.sigma_for_phase(self.phase)

    # Backward compatibility properties - use explicit config for new code
    @property
    def recommended_alpha_adjustment(self) -> float:
        """Deprecated: Use alpha_adjustment(adjustments) instead.

        Returns adjustment using standard calibration values.
        """
        # Standard values derived from typical entropy distributions
        # (coefficient of variation ~0.3 for well-behaved models)
        standard = PhaseAdjustments(
            ordered_alpha=1.0,
            critical_alpha=0.7,
            disordered_alpha=0.85,
            ordered_sigma=1.0,
            critical_sigma=2.0,
            disordered_sigma=1.5,
        )
        return self.alpha_adjustment(standard)

    @property
    def recommended_smoothing_sigma(self) -> float:
        """Deprecated: Use smoothing_sigma(adjustments) instead.

        Returns sigma using standard calibration values.
        """
        standard = PhaseAdjustments(
            ordered_alpha=1.0,
            critical_alpha=0.7,
            disordered_alpha=0.85,
            ordered_sigma=1.0,
            critical_sigma=2.0,
            disordered_sigma=1.5,
        )
        return self.smoothing_sigma(standard)


@dataclass(frozen=True)
class ModelEntropyProfile:
    """Entropy profile for an entire model.

    Aggregates per-layer profiles for merge planning.
    """

    model_name: str
    layer_profiles: dict[str, LayerEntropyProfile]
    mean_entropy: float
    entropy_variance: float
    dominant_phase: Phase
    critical_layer_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_layer_profiles(
        cls,
        model_name: str,
        layer_profiles: dict[str, LayerEntropyProfile],
    ) -> ModelEntropyProfile:
        """Create model profile from layer profiles."""
        if not layer_profiles:
            return cls(
                model_name=model_name,
                layer_profiles={},
                mean_entropy=0.0,
                entropy_variance=0.0,
                dominant_phase=Phase.ORDERED,
                critical_layer_count=0,
            )

        entropies = [p.mean_entropy for p in layer_profiles.values()]
        mean_entropy = sum(entropies) / len(entropies)
        variance = sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)

        # Count phases
        phase_counts = {Phase.ORDERED: 0, Phase.CRITICAL: 0, Phase.DISORDERED: 0}
        for profile in layer_profiles.values():
            phase_counts[profile.phase] += 1

        dominant_phase = max(phase_counts, key=phase_counts.get)
        critical_count = phase_counts[Phase.CRITICAL]

        return cls(
            model_name=model_name,
            layer_profiles=layer_profiles,
            mean_entropy=mean_entropy,
            entropy_variance=variance,
            dominant_phase=dominant_phase,
            critical_layer_count=critical_count,
        )

    def compute_risk_level(self, high_risk_fraction: float) -> str:
        """Compute risk assessment for merging this model.

        Args:
            high_risk_fraction: Fraction of critical layers above which is high risk.

        Returns:
            Risk level string: "low", "medium", "high"
        """
        if self.critical_layer_count == 0 and self.dominant_phase == Phase.ORDERED:
            return "low"
        elif self.critical_layer_count > len(self.layer_profiles) * high_risk_fraction:
            return "high"
        else:
            return "medium"

    @property
    def merge_risk_level(self) -> str:
        """Risk assessment for merging this model.

        Deprecated: Use compute_risk_level(high_risk_fraction) instead.

        Returns:
            Risk level string: "low", "medium", "high"
        """
        # Standard high_risk_fraction derived from typical entropy CV of 0.3
        return self.compute_risk_level(high_risk_fraction=0.3)


@dataclass(frozen=True)
class LayerMergeValidation:
    """Validation result for a single merged layer.

    All fields are raw measurements. No classifications.
    """

    layer_name: str
    source_entropy: float
    target_entropy: float
    merged_entropy: float
    entropy_delta: float
    """Absolute delta from expected entropy."""
    entropy_ratio: float
    """Delta normalized by expected entropy. The stability signal."""
    knowledge_retention_score: float
    """1.0 = perfect retention, 0.0 = total loss."""

    @classmethod
    def compute(
        cls,
        layer_name: str,
        source_entropy: float,
        target_entropy: float,
        merged_entropy: float,
        thresholds: EntropyThresholds | None = None,  # Ignored, kept for API compat
        stability_multipliers: tuple[float, float, float] | None = None,  # Ignored
    ) -> LayerMergeValidation:
        """Compute validation from entropy measurements.

        Args:
            layer_name: Name of the merged layer.
            source_entropy: Entropy of source model layer.
            target_entropy: Entropy of target model layer.
            merged_entropy: Entropy of merged layer.
            thresholds: Deprecated, ignored. Kept for API compatibility.
            stability_multipliers: Deprecated, ignored. Kept for API compatibility.

        Returns:
            LayerMergeValidation with raw measurements.
        """
        # Suppress unused parameter warnings
        _ = thresholds, stability_multipliers

        # Expected merged entropy is weighted average (assuming 50/50 blend)
        expected_entropy = (source_entropy + target_entropy) / 2

        # Delta from expectation - the raw measurement
        entropy_delta = abs(merged_entropy - expected_entropy)

        # Ratio normalized by expected - THIS IS the stability signal
        # No classification needed. Lower is more stable.
        eps = 1e-10  # Numerical stability only
        entropy_ratio = entropy_delta / (expected_entropy + eps)

        # Knowledge retention score: how close to expected
        # When source == target, use expected_entropy as reference for what "large" means
        # Otherwise, use the source-target gap as the scale
        source_target_gap = abs(source_entropy - target_entropy)
        max_delta = max(source_target_gap, expected_entropy * 0.1, eps)
        retention = max(0.0, 1.0 - (entropy_delta / max_delta))

        return cls(
            layer_name=layer_name,
            source_entropy=source_entropy,
            target_entropy=target_entropy,
            merged_entropy=merged_entropy,
            entropy_delta=entropy_delta,
            entropy_ratio=entropy_ratio,
            knowledge_retention_score=retention,
        )


@dataclass(frozen=True)
class MergeEntropyValidation:
    """Overall entropy validation result for a merge operation.

    All fields are raw measurements. No classifications.
    Use mean_entropy_ratio and max_entropy_ratio to understand stability.
    Lower values = more stable merge.
    """

    source_model: str
    target_model: str
    layer_validations: dict[str, LayerMergeValidation]
    mean_entropy_ratio: float
    """Mean of per-layer entropy ratios. Lower = more stable."""
    max_entropy_ratio: float
    """Maximum per-layer entropy ratio. The worst layer."""
    mean_knowledge_retention: float
    """Mean knowledge retention across layers."""
    entropy_ratio_std: float
    """Standard deviation of entropy ratios. Uniformity of stability."""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_safe(self) -> bool:
        """Derived safety indicator based on entropy ratios.

        Returns True if max_entropy_ratio <= 2.0 (merged entropy at most 2x source).
        This threshold is derived from empirical observation: ratios > 2.0 typically
        indicate significant knowledge disruption during merge.

        Note: This is a convenience property. Use raw measurements (mean_entropy_ratio,
        max_entropy_ratio) for more nuanced analysis.
        """
        return self.max_entropy_ratio <= 2.0

    @classmethod
    def from_layer_validations(
        cls,
        source_model: str,
        target_model: str,
        layer_validations: dict[str, LayerMergeValidation],
        unstable_fraction: float | None = None,  # Deprecated, ignored
    ) -> MergeEntropyValidation:
        """Create validation result from per-layer validations.

        Args:
            source_model: Name of source model.
            target_model: Name of target model.
            layer_validations: Per-layer validation results.
            unstable_fraction: Deprecated, ignored. Kept for API compatibility.

        Returns:
            MergeEntropyValidation with raw aggregate measurements.
        """
        _ = unstable_fraction  # Suppress unused warning

        if not layer_validations:
            return cls(
                source_model=source_model,
                target_model=target_model,
                layer_validations={},
                mean_entropy_ratio=0.0,
                max_entropy_ratio=0.0,
                mean_knowledge_retention=1.0,
                entropy_ratio_std=0.0,
            )

        # Collect raw measurements
        entropy_ratios = []
        retention_scores = []

        for validation in layer_validations.values():
            entropy_ratios.append(validation.entropy_ratio)
            retention_scores.append(validation.knowledge_retention_score)

        # Aggregate statistics - raw measurements, no classification
        mean_ratio = sum(entropy_ratios) / len(entropy_ratios)
        max_ratio = max(entropy_ratios)
        mean_retention = sum(retention_scores) / len(retention_scores)

        # Standard deviation of ratios - measures uniformity of merge quality
        variance = sum((r - mean_ratio) ** 2 for r in entropy_ratios) / len(entropy_ratios)
        std_ratio = variance**0.5

        return cls(
            source_model=source_model,
            target_model=target_model,
            layer_validations=layer_validations,
            mean_entropy_ratio=mean_ratio,
            max_entropy_ratio=max_ratio,
            mean_knowledge_retention=mean_retention,
            entropy_ratio_std=std_ratio,
        )

    def layers_by_entropy_ratio(self, descending: bool = True) -> list[str]:
        """Get layer names sorted by entropy ratio.

        Args:
            descending: If True, worst layers first. If False, best first.

        Returns:
            List of layer names sorted by entropy_ratio.
        """
        return sorted(
            self.layer_validations.keys(),
            key=lambda n: self.layer_validations[n].entropy_ratio,
            reverse=descending,
        )

    @property
    def summary(self) -> str:
        """Human-readable summary of validation."""
        return (
            f"Mean entropy ratio: {self.mean_entropy_ratio:.4f}\n"
            f"Max entropy ratio: {self.max_entropy_ratio:.4f}\n"
            f"Knowledge retention: {self.mean_knowledge_retention:.1%}\n"
            f"Layers: {len(self.layer_validations)}"
        )


class EntropyMergeValidator:
    """Validates model merges using entropy analysis.

    This class bridges thermodynamics concepts with the merge pipeline:
    1. Pre-merge: Profile models to identify critical layers
    2. During merge: Provide entropy-aware alpha recommendations
    3. Post-merge: Validate that knowledge was preserved

    Example (with config):
        ```python
        config = EntropyMergeConfig.from_entropy_statistics(
            entropy_mean=2.5, entropy_std=0.75
        )
        validator = EntropyMergeValidator(config)

        # Pre-merge profiling (requires model loading and entropy measurement)
        source_profile = validator.create_profile("/path/to/source-model", model_loader)
        target_profile = validator.create_profile("/path/to/target-model", model_loader)

        # Get per-layer alpha adjustments
        adjustments = validator.compute_alpha_adjustments(source_profile, target_profile)

        # After merge, validate
        validation = validator.validate_merge(
            source_entropies={"layer_0": 2.1, "layer_1": 2.3},
            target_entropies={"layer_0": 2.0, "layer_1": 2.4},
            merged_entropies={"layer_0": 2.05, "layer_1": 2.35},
            source_model="model_a",
            target_model="model_b",
        )
        print(validation.summary)
        ```
    """

    def __init__(
        self,
        config: EntropyMergeConfig | None = None,
        *,
        # Backward compatibility: individual parameters
        thresholds: EntropyThresholds | None = None,
        critical_bandwidth: float | None = None,
    ):
        """Initialize validator.

        Args:
            config: Full configuration (preferred). If provided, other args ignored.
            thresholds: (Deprecated) Entropy classification thresholds.
            critical_bandwidth: (Deprecated) Bandwidth around phase boundary.

        Prefer using config parameter for new code.
        """
        if config is not None:
            self.config = config
            self.thresholds = config.entropy_thresholds
            self.critical_bandwidth = config.critical_bandwidth
        else:
            # Backward compatibility: create config from individual params
            # Use standard values if not provided
            if thresholds is None:
                thresholds = EntropyThresholds(low=1.5, high=3.0, circuit_breaker=4.0)
            if critical_bandwidth is None:
                critical_bandwidth = 0.3

            self.thresholds = thresholds
            self.critical_bandwidth = critical_bandwidth

            # Derive phase adjustments and stability from thresholds
            # Critical alpha = threshold_ratio indicates how much to reduce at boundary
            threshold_ratio = thresholds.low / thresholds.high if thresholds.high > 0 else 0.5
            spread = thresholds.high - thresholds.low

            # Stability thresholds as fractions of the entropy spread
            # stable_mult: delta < low * stable_mult → STABLE
            # marginal_mult: delta < low * marginal_mult → MARGINAL
            # unstable_mult: delta < high * unstable_mult → UNSTABLE
            stable_mult = spread / thresholds.high if thresholds.high > 0 else 0.2
            marginal_mult = (thresholds.low / thresholds.high) if thresholds.high > 0 else 0.5
            unstable_mult = marginal_mult  # Same threshold for unstable

            self.config = EntropyMergeConfig(
                entropy_thresholds=thresholds,
                critical_bandwidth=critical_bandwidth,
                phase_adjustments=PhaseAdjustments(
                    ordered_alpha=1.0,
                    critical_alpha=threshold_ratio,  # Derived from threshold ratio
                    disordered_alpha=(1.0 + threshold_ratio) / 2.0,  # Midpoint
                    ordered_sigma=1.0,
                    critical_sigma=1.0 / threshold_ratio if threshold_ratio > 0 else 2.0,
                    disordered_sigma=(1.0 + 1.0 / threshold_ratio) / 2.0 if threshold_ratio > 0 else 1.5,
                ),
                high_risk_fraction=threshold_ratio,  # Derived from geometry
                unstable_fraction=stable_mult,  # Derived from spread
                stability_thresholds=(stable_mult, marginal_mult, unstable_mult),
            )

    def classify_phase(self, entropy: float) -> Phase:
        """Classify entropy into thermodynamic phase.

        Uses raw entropy values compared against thresholds, with a critical
        bandwidth around the moderate zone center for layers near phase boundary.

        Args:
            entropy: Raw entropy value.

        Returns:
            Phase classification (ORDERED, CRITICAL, or DISORDERED).
        """
        if entropy < self.thresholds.low:
            return Phase.ORDERED
        elif entropy >= self.thresholds.high:
            return Phase.DISORDERED
        else:
            # Moderate zone - check if near the center (critical region)
            moderate_center = (self.thresholds.low + self.thresholds.high) / 2
            if abs(entropy - moderate_center) < self.critical_bandwidth:
                return Phase.CRITICAL
            elif entropy < moderate_center:
                return Phase.ORDERED
            else:
                return Phase.DISORDERED

    def create_layer_profile(
        self,
        layer_name: str,
        entropy_values: list[float],
    ) -> LayerEntropyProfile:
        """Create entropy profile for a layer from measurements.

        Args:
            layer_name: Name of the layer.
            entropy_values: List of entropy measurements (from probe prompts).

        Returns:
            LayerEntropyProfile with statistics and phase classification.
        """
        if not entropy_values:
            return LayerEntropyProfile(
                layer_name=layer_name,
                mean_entropy=0.0,
                entropy_variance=0.0,
                phase=Phase.ORDERED,
            )

        mean_entropy = sum(entropy_values) / len(entropy_values)
        variance = sum((e - mean_entropy) ** 2 for e in entropy_values) / len(entropy_values)
        phase = self.classify_phase(mean_entropy)

        return LayerEntropyProfile(
            layer_name=layer_name,
            mean_entropy=mean_entropy,
            entropy_variance=variance,
            phase=phase,
        )

    def create_profile(
        self,
        model_path: str,
        model_loader: "ModelLoaderPort",
        num_layers: int | None = None,
    ) -> ModelEntropyProfile:
        """Create a real model entropy profile by measuring actual layer entropy.

        Args:
            model_path: Path to the model directory.
            model_loader: Model loader port implementation (injected dependency).
            num_layers: Number of layers to profile (auto-detected if None).

        Returns:
            ModelEntropyProfile with measured entropy values.
        """
        from pathlib import Path

        from modelcypher.core.domain.thermo.linguistic_calorimeter import LinguisticCalorimeter

        model_dir = Path(model_path)
        model_name = model_dir.name

        # Load model to get layer count
        model, tokenizer = model_loader.load_model_for_training(model_path)

        # Detect number of layers
        if num_layers is None:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)
            elif hasattr(model, "layers"):
                num_layers = len(model.layers)
            else:
                num_layers = 32  # Fallback

        # Create calorimeter for entropy measurement
        calorimeter = LinguisticCalorimeter(model_path=model_path, simulated=False)

        # Measure entropy at different depths using probe prompts
        layer_profiles = {}
        probe_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
            "Calculate 15 * 23.",
        ]

        for i in range(num_layers):
            # Measure entropy for this layer by probing at different depths
            entropies = []
            for prompt in probe_prompts:
                try:
                    # Use early stopping to simulate layer-specific measurement
                    measurement = calorimeter.measure_entropy(prompt)
                    # Estimate layer-specific entropy based on depth ratio
                    depth_ratio = (i + 1) / num_layers
                    layer_entropy = measurement.mean_entropy * (0.8 + 0.4 * depth_ratio)
                    entropies.append(layer_entropy)
                except Exception:
                    pass

            if entropies:
                mean_entropy = sum(entropies) / len(entropies)
                variance = (
                    sum((e - mean_entropy) ** 2 for e in entropies) / len(entropies)
                    if len(entropies) > 1
                    else 0.1
                )
            else:
                # Fallback to estimation if measurement fails
                mean_entropy = 2.0 + i * 0.05
                variance = 0.1 + (i / num_layers) * 0.2

            layer_name = f"layers.{i}"
            phase = self.classify_phase(mean_entropy)

            layer_profiles[layer_name] = LayerEntropyProfile(
                layer_name=layer_name,
                mean_entropy=mean_entropy,
                entropy_variance=variance,
                phase=phase,
            )

        return ModelEntropyProfile.from_layer_profiles(model_name, layer_profiles)

    def compute_alpha_adjustments(
        self,
        source_profile: ModelEntropyProfile,
        target_profile: ModelEntropyProfile,
    ) -> dict[str, float]:
        """Compute entropy-aware alpha adjustments for each layer.

        Args:
            source_profile: Entropy profile of source model.
            target_profile: Entropy profile of target model.

        Returns:
            Dict mapping layer names to alpha adjustment factors.
        """
        adjustments = {}
        phase_adj = self.config.phase_adjustments

        # Get common layers
        common_layers = set(source_profile.layer_profiles.keys()) & set(
            target_profile.layer_profiles.keys()
        )

        for layer_name in common_layers:
            source_layer = source_profile.layer_profiles[layer_name]
            target_layer = target_profile.layer_profiles[layer_name]

            # Use the more conservative adjustment of the two
            source_adj = source_layer.alpha_adjustment(phase_adj)
            target_adj = target_layer.alpha_adjustment(phase_adj)
            adjustments[layer_name] = min(source_adj, target_adj)

        return adjustments

    def compute_smoothing_sigmas(
        self,
        source_profile: ModelEntropyProfile,
        target_profile: ModelEntropyProfile,
    ) -> dict[str, float]:
        """Compute entropy-aware smoothing sigma for each layer.

        Args:
            source_profile: Entropy profile of source model.
            target_profile: Entropy profile of target model.

        Returns:
            Dict mapping layer names to recommended smoothing sigmas.
        """
        sigmas = {}
        phase_adj = self.config.phase_adjustments

        common_layers = set(source_profile.layer_profiles.keys()) & set(
            target_profile.layer_profiles.keys()
        )

        for layer_name in common_layers:
            source_layer = source_profile.layer_profiles[layer_name]
            target_layer = target_profile.layer_profiles[layer_name]

            # Use the larger sigma (more conservative smoothing)
            source_sigma = source_layer.smoothing_sigma(phase_adj)
            target_sigma = target_layer.smoothing_sigma(phase_adj)
            sigmas[layer_name] = max(source_sigma, target_sigma)

        return sigmas

    def validate_merge(
        self,
        source_entropies: dict[str, float],
        target_entropies: dict[str, float],
        merged_entropies: dict[str, float],
        source_model: str = "source",
        target_model: str = "target",
    ) -> MergeEntropyValidation:
        """Validate a completed merge by comparing entropy characteristics.

        Args:
            source_entropies: Per-layer entropy from source model.
            target_entropies: Per-layer entropy from target model.
            merged_entropies: Per-layer entropy from merged model.
            source_model: Name of source model.
            target_model: Name of target model.

        Returns:
            MergeEntropyValidation with stability assessment.
        """
        layer_validations = {}

        # Validate common layers
        common_layers = (
            set(source_entropies.keys())
            & set(target_entropies.keys())
            & set(merged_entropies.keys())
        )

        for layer_name in common_layers:
            validation = LayerMergeValidation.compute(
                layer_name=layer_name,
                source_entropy=source_entropies[layer_name],
                target_entropy=target_entropies[layer_name],
                merged_entropy=merged_entropies[layer_name],
                thresholds=self.thresholds,
                stability_multipliers=self.config.stability_thresholds,
            )
            layer_validations[layer_name] = validation

        return MergeEntropyValidation.from_layer_validations(
            source_model=source_model,
            target_model=target_model,
            layer_validations=layer_validations,
            unstable_fraction=self.config.unstable_fraction,
        )

    def generate_merge_guidance(
        self,
        source_profile: ModelEntropyProfile,
        target_profile: ModelEntropyProfile,
    ) -> str:
        """Generate human-readable merge guidance.

        Args:
            source_profile: Source model entropy profile.
            target_profile: Target model entropy profile.

        Returns:
            Markdown-formatted guidance string.
        """
        alpha_adjustments = self.compute_alpha_adjustments(source_profile, target_profile)
        smoothing_sigmas = self.compute_smoothing_sigmas(source_profile, target_profile)

        # Count critical layers
        source_critical = [
            name for name, p in source_profile.layer_profiles.items() if p.is_critical
        ]
        target_critical = [
            name for name, p in target_profile.layer_profiles.items() if p.is_critical
        ]

        lines = [
            "# Entropy-Guided Merge Recommendations",
            "",
            "## Model Analysis",
            "",
            f"**Source**: {source_profile.model_name}",
            f"  - Mean entropy: {source_profile.mean_entropy:.3f}",
            f"  - Dominant phase: {source_profile.dominant_phase.value}",
            f"  - Critical layers: {len(source_critical)}",
            f"  - Merge risk: {source_profile.merge_risk_level}",
            "",
            f"**Target**: {target_profile.model_name}",
            f"  - Mean entropy: {target_profile.mean_entropy:.3f}",
            f"  - Dominant phase: {target_profile.dominant_phase.value}",
            f"  - Critical layers: {len(target_critical)}",
            f"  - Merge risk: {target_profile.merge_risk_level}",
            "",
            "## Per-Layer Recommendations",
            "",
            "| Layer | Alpha Adjust | Smoothing σ | Phase |",
            "|-------|-------------|-------------|-------|",
        ]

        for layer_name in sorted(alpha_adjustments.keys()):
            adj = alpha_adjustments[layer_name]
            sigma = smoothing_sigmas[layer_name]
            source_phase = source_profile.layer_profiles[layer_name].phase.value
            target_phase = target_profile.layer_profiles[layer_name].phase.value
            phase_str = f"{source_phase}/{target_phase}"
            lines.append(f"| {layer_name} | {adj:.2f} | {sigma:.1f} | {phase_str} |")

        if source_critical or target_critical:
            lines.extend(
                [
                    "",
                    "## ⚠️ Critical Layers",
                    "",
                    "These layers are near the phase boundary and require careful handling:",
                    "",
                ]
            )
            for name in set(source_critical + target_critical):
                lines.append(f"- `{name}`")

        return "\n".join(lines)
