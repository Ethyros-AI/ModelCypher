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

"""
Regime State Detector (formerly Phase Transition Theory).

Statistical mechanics framework for understanding training regime transitions.
Derives critical temperature T_c from the softmax-Boltzmann equivalence.

Ported 1:1 from the reference Swift implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Minimum temperature to avoid division by zero
MINIMUM_TEMPERATURE: float = 1e-6


class RegimeState(str, Enum):
    """Classification of thermodynamic regime state."""

    ORDERED = "ordered"  # T < T_c: Low entropy, sharp distribution
    CRITICAL = "critical"  # T ≈ T_c: Maximum variance, sign flips possible
    DISORDERED = "disordered"  # T > T_c: High entropy, flat distribution

    @property
    def display_name(self) -> str:
        if self == RegimeState.ORDERED:
            return "Ordered (T < T_c)"
        elif self == RegimeState.CRITICAL:
            return "Critical (T ≈ T_c)"
        else:
            return "Disordered (T > T_c)"

    @property
    def expected_modifier_effect(self) -> str:
        if self == RegimeState.ORDERED:
            return "Entropy reduction (cooling) - modifiers sharpen distribution"
        elif self == RegimeState.CRITICAL:
            return "Unpredictable - near phase boundary, effects can flip"
        else:
            return "Entropy increase (heating) - modifiers flatten distribution"


@dataclass(frozen=True)
class BasinTopology:
    """Energy levels for behavioral attractor basins.

    Basin depths are derived from the entropy landscape of the logit distribution.
    In Boltzmann statistics, depth = -log(probability), so lower entropy states
    correspond to deeper basins (more stable attractors).
    """

    refusal_depth: float
    """Depth of refusal basin (lowest entropy = deepest)."""

    caution_depth: float
    """Depth of caution basin (moderate entropy)."""

    transition_ridge: float
    """Height of transition barrier (maximum entropy)."""

    solution_depth: float
    """Depth of solution basin (moderate entropy)."""

    @classmethod
    def from_logit_geometry(
        cls,
        entropy: float,
        max_entropy: float,
        temperature: float,
        critical_temperature: float,
    ) -> "BasinTopology":
        """Derive basin topology from the current logit geometry.

        The entropy landscape determines basin depths:
        - Refusal: deepest basin at low entropy (concentrated probability)
        - Solution: moderate depth at moderate entropy
        - Caution: shallow basin at higher entropy
        - Transition ridge: at maximum entropy

        The T/T_c ratio modulates the relative depths.

        Args:
            entropy: Current Shannon entropy of the distribution.
            max_entropy: Maximum possible entropy (log of effective vocab size).
            temperature: Current temperature parameter.
            critical_temperature: Estimated critical temperature from logits.

        Returns:
            Basin topology derived from the entropy landscape.
        """
        if max_entropy <= 0:
            max_entropy = 1.0  # Fallback to prevent division by zero

        # Normalized entropy position [0, 1]
        normalized_entropy = min(1.0, entropy / max_entropy)

        # T/T_c ratio determines which basin we're in
        if critical_temperature > 0:
            t_ratio = temperature / critical_temperature
        else:
            t_ratio = 1.0

        # Basin depths derived from entropy structure:
        # - Refusal is deepest (most stable) at low entropy
        # - Solution is moderate depth
        # - Caution is shallow (easily perturbed)
        # Depths scale with how far we are from critical point

        # Distance from critical point determines basin stability
        distance_from_critical = abs(1.0 - t_ratio)

        # Refusal depth: deepest when far from critical, shallow when critical
        refusal_depth = distance_from_critical * (1.0 - normalized_entropy)

        # Caution depth: shallow, scales with entropy
        caution_depth = normalized_entropy * 0.5

        # Solution depth: intermediate, depends on entropy regime
        solution_depth = (1.0 - normalized_entropy) * 0.5 + normalized_entropy * 0.3

        # Transition ridge: at maximum entropy, height from entropy ratio
        transition_ridge = normalized_entropy + distance_from_critical * 0.2

        return cls(
            refusal_depth=refusal_depth,
            caution_depth=caution_depth,
            transition_ridge=min(1.0, transition_ridge),
            solution_depth=solution_depth,
        )

    def basin_weights(self, temperature: float) -> tuple[float, float, float]:
        """Boltzmann weights for each basin at given temperature.

        Returns: (refusal, caution, solution) weights as probabilities.
        """
        if temperature <= 0:
            # At T=0, all weight in deepest basin
            depths = [self.refusal_depth, self.caution_depth, self.solution_depth]
            min_idx = depths.index(min(depths))
            weights = [0.0, 0.0, 0.0]
            weights[min_idx] = 1.0
            return (weights[0], weights[1], weights[2])

        # Boltzmann factors: exp(-E/T)
        z_refusal = math.exp(-self.refusal_depth / temperature)
        z_caution = math.exp(-self.caution_depth / temperature)
        z_solution = math.exp(-self.solution_depth / temperature)
        partition = z_refusal + z_caution + z_solution

        if partition <= 0:
            # Uniform if partition function fails
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        return (
            z_refusal / partition,
            z_caution / partition,
            z_solution / partition,
        )


@dataclass(frozen=True)
class TemperatureSweepResult:
    """Result of analyzing entropy across a temperature sweep."""

    temperatures: list[float]
    entropies: list[float]
    derivatives: list[float]
    estimated_tc: float
    observed_peak_t: float | None


@dataclass(frozen=True)
class RegimeAnalysis:
    """Complete result of regime state analysis."""

    temperature: float
    estimated_tc: float
    state: RegimeState
    logit_variance: float
    effective_vocab_size: int
    predicted_modifier_effect: float
    confidence: float
    basin_weights: tuple[float, float, float] | None


class RegimeStateDetector:
    """
    Detects the current thermodynamic regime (Ordered, Critical, Disordered)
    of the model based on logit statistics and temperature.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        """Initialize with compute backend."""
        self._backend = backend or get_default_backend()

    def analyze(
        self,
        logits: "Array",
        temperature: float,
        intensity_score: float = 0.0,
    ) -> RegimeAnalysis:
        """Perform full regime analysis from logits.

        All thresholds, tolerances, and basin topology are derived directly
        from the logit geometry - no external calibration or presets needed.

        Args:
            logits: Raw logit tensor from model.
            temperature: Temperature parameter for softmax scaling.
            intensity_score: Intensity of any modifier being applied (0-1).

        Returns:
            Complete regime analysis with geometry-derived parameters.
        """
        # Compute all statistics from the logits
        variance = self.compute_logit_variance(logits, temperature)
        _, raw_variance, std_dev = self.compute_logit_statistics(logits)
        v_eff = self.effective_vocabulary_size(logits, 1.0)
        base_entropy = self.compute_entropy(logits, temperature)

        # Maximum possible entropy: log(V_eff)
        max_entropy = math.log(max(1, v_eff))

        # Estimate T_c from logit statistics
        tc = self.estimate_critical_temperature(std_dev, v_eff)

        # Classify state using geometry-derived tolerance
        state = self.classify_state(temperature, tc, variance)

        # Predict modifier effect using all computed statistics
        predicted, confidence = self.predict_modifier_effect(
            state,
            intensity_score,
            base_entropy,
            temperature,
            tc,
            variance,
        )

        # Derive basin topology from the entropy landscape
        topology = BasinTopology.from_logit_geometry(
            entropy=base_entropy,
            max_entropy=max_entropy,
            temperature=temperature,
            critical_temperature=tc,
        )

        # Compute basin weights from topology
        weights = topology.basin_weights(temperature)

        return RegimeAnalysis(
            temperature=temperature,
            estimated_tc=tc,
            state=state,
            logit_variance=variance,
            effective_vocab_size=v_eff,
            predicted_modifier_effect=predicted,
            confidence=confidence,
            basin_weights=weights,
        )

    @staticmethod
    def classify_state(
        temperature: float,
        critical_temperature: float,
        logit_variance: float,
    ) -> RegimeState:
        """Classify current state based on temperature relative to T_c.

        The tolerance (critical region width) is derived from the logit variance:
        higher variance = wider critical region because the system is more
        susceptible to fluctuations.

        Args:
            temperature: Current temperature parameter.
            critical_temperature: Estimated T_c from logit statistics.
            logit_variance: Variance of the logit distribution.

        Returns:
            RegimeState classification.
        """
        if critical_temperature <= 0:
            return RegimeState.ORDERED

        ratio = temperature / critical_temperature

        # Tolerance derived from variance: coefficient of variation of logits
        # Higher variance relative to T_c = wider critical region
        # Clamped to [0.05, 0.5] to maintain physical meaning
        if critical_temperature > 0:
            tolerance = min(0.5, max(0.05, math.sqrt(logit_variance) / critical_temperature))
        else:
            tolerance = 0.1

        if ratio < 1.0 - tolerance:
            return RegimeState.ORDERED
        elif ratio > 1.0 + tolerance:
            return RegimeState.DISORDERED
        else:
            return RegimeState.CRITICAL

    @staticmethod
    def predict_modifier_effect(
        state: RegimeState,
        intensity_score: float,
        base_entropy: float,
        temperature: float,
        critical_temperature: float,
        logit_variance: float,
    ) -> tuple[float, float]:
        """Predict the sign and magnitude of entropy change from modifiers.

        The effect magnitude and confidence are derived from:
        - Distance from critical temperature (how stable the regime is)
        - Logit variance (how susceptible to perturbation)
        - Base entropy (room for change)

        Args:
            state: Current regime state.
            intensity_score: Modifier intensity (0-1).
            base_entropy: Current Shannon entropy.
            temperature: Current temperature parameter.
            critical_temperature: Estimated T_c.
            logit_variance: Variance of logit distribution.

        Returns:
            Tuple of (predicted_delta_h, confidence).
        """
        if critical_temperature <= 0:
            return (0.0, 0.0)

        # Distance from critical point determines effect strength
        t_ratio = temperature / critical_temperature
        distance_from_critical = abs(1.0 - t_ratio)

        # Coefficient of variation of logits: uncertainty measure
        cv = math.sqrt(logit_variance) / critical_temperature if critical_temperature > 0 else 1.0

        if state == RegimeState.ORDERED:
            # Far below T_c: modifiers cause cooling (entropy reduction)
            # Effect magnitude scales with distance from critical point
            # and inversely with current variance (stable systems change less)
            effect_multiplier = distance_from_critical / (1.0 + cv)
            delta_h = -intensity_score * effect_multiplier * base_entropy

            # Confidence: high when far from critical, low variance
            confidence = distance_from_critical / (1.0 + cv)

        elif state == RegimeState.CRITICAL:
            # Near T_c: effects are unpredictable, near zero expected value
            # but high variance in outcomes
            delta_h = 0.0

            # Confidence very low: we can't predict which way it will go
            confidence = cv / (1.0 + distance_from_critical + cv)

        else:  # DISORDERED
            # Above T_c: modifiers cause heating (entropy increase)
            # Effect is weaker than in ordered phase
            effect_multiplier = distance_from_critical / (2.0 + cv)
            delta_h = intensity_score * effect_multiplier * base_entropy

            # Confidence moderate: more predictable than critical but
            # high variance systems are still uncertain
            confidence = distance_from_critical / (1.0 + 2.0 * cv)

        # Clamp confidence to [0, 1]
        confidence = min(1.0, max(0.0, confidence))

        return (delta_h, confidence)

    @staticmethod
    def estimate_critical_temperature(
        logit_std_dev: float,
        effective_vocab_size: int,
    ) -> float:
        """T_c = σ_z / √(2 × ln(V_eff))."""
        if effective_vocab_size <= 1:
            return 1.0
        log_veff = math.log(effective_vocab_size)
        if log_veff <= 0:
            return 1.0
        return logit_std_dev / math.sqrt(2.0 * log_veff)

    def compute_logit_variance(self, logits: "Array", temperature: float) -> float:
        """Compute logit variance under temperature-scaled distribution."""
        if temperature <= 0:
            return 0.0
        safe_temp = max(temperature, MINIMUM_TEMPERATURE)
        b = self._backend

        logits_f32 = b.astype(logits, "float32")

        # Temperature-scaled softmax
        scaled = logits_f32 / safe_temp
        probs = b.softmax(scaled, axis=-1)

        # E[z] = Σ p_i z_i
        mean_z = b.sum(probs * logits_f32, axis=-1)

        # E[z²] = Σ p_i z_i²
        mean_z_squared = b.sum(probs * logits_f32 * logits_f32, axis=-1)

        # Var(z) = E[z²] - E[z]²
        variance = mean_z_squared - mean_z * mean_z
        b.eval(variance)

        return max(0.0, self._scalar_mean(variance))

    def compute_logit_statistics(self, logits: "Array") -> tuple[float, float, float]:
        """Tuple of (mean, variance, std_dev) of raw logits."""
        b = self._backend
        logits_f32 = b.astype(logits, "float32")
        mean_arr = b.mean(logits_f32)
        var_arr = b.var(logits_f32)
        b.eval(mean_arr, var_arr)

        mean_np = b.to_numpy(mean_arr)
        var_np = b.to_numpy(var_arr)
        mean_val = float(mean_np.item())
        var_val = float(var_np.item())
        std_val = math.sqrt(var_val)

        return (mean_val, var_val, std_val)

    def effective_vocabulary_size(
        self,
        logits: "Array",
        temperature: float,
        threshold: float = 1e-4,
    ) -> int:
        """Compute effective vocabulary size (tokens with p > threshold)."""
        if temperature <= 0:
            return 1
        b = self._backend

        logits_f32 = b.astype(logits, "float32")
        scaled = logits_f32 / max(temperature, MINIMUM_TEMPERATURE)
        probs = b.softmax(scaled, axis=-1)

        mask = probs > threshold
        count = b.sum(b.astype(mask, "int32"), axis=-1)
        b.eval(count)

        return max(1, self._scalar_mean_int(count))

    def compute_entropy(self, logits: "Array", temperature: float) -> float:
        """Compute Shannon entropy."""
        if temperature <= 0:
            return 0.0
        b = self._backend

        logits_f32 = b.astype(logits, "float32")
        scaled = logits_f32 / max(temperature, MINIMUM_TEMPERATURE)
        probs = b.softmax(scaled, axis=-1)

        log_probs = b.log(probs + 1e-10)
        entropy = -b.sum(probs * log_probs, axis=-1)
        b.eval(entropy)

        return self._scalar_mean(entropy)

    def _scalar_mean(self, array: "Array") -> float:
        """Compute scalar mean from array."""
        b = self._backend
        reduced = array if array.ndim == 0 else b.mean(array)
        b.eval(reduced)
        reduced_np = b.to_numpy(reduced)
        return float(reduced_np.item())

    def _scalar_mean_int(self, array: "Array") -> int:
        """Compute scalar mean as int from array."""
        b = self._backend
        if array.ndim == 0:
            b.eval(array)
            arr_np = b.to_numpy(array)
            return int(arr_np.item())
        mean_val = self._scalar_mean(b.astype(array, "float32"))
        return int(round(mean_val))
