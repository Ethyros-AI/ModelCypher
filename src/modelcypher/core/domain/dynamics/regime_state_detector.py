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
from typing import TYPE_CHECKING, Optional, List

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Minimum temperature to avoid division by zero
MINIMUM_TEMPERATURE: float = 1e-6


class RegimeState(str, Enum):
    """Classification of thermodynamic regime state."""
    ORDERED = "ordered"      # T < T_c: Low entropy, sharp distribution
    CRITICAL = "critical"    # T ≈ T_c: Maximum variance, sign flips possible
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
    """
    Energy levels for behavioral attractor basins.
    """
    refusal_depth: float = 0.0   # Deepest
    caution_depth: float = 0.2   # Shallow
    transition_ridge: float = 0.8  # Barrier
    solution_depth: float = 0.4  # Moderate

    @classmethod
    def default(cls) -> "BasinTopology":
        return cls()

    def basin_weights(self, temperature: float) -> tuple[float, float, float]:
        """
        Boltzmann weights for each basin at given temperature.
        Returns: (refusal, caution, solution) weights
        """
        if temperature <= 0:
            return (1.0, 0.0, 0.0)

        z_refusal = math.exp(-self.refusal_depth / temperature)
        z_caution = math.exp(-self.caution_depth / temperature)
        z_solution = math.exp(-self.solution_depth / temperature)
        partition = z_refusal + z_caution + z_solution

        if partition <= 0:
            return (0.33, 0.33, 0.33)

        return (
            z_refusal / partition,
            z_caution / partition,
            z_solution / partition,
        )


@dataclass(frozen=True)
class TemperatureSweepResult:
    """Result of analyzing entropy across a temperature sweep."""
    temperatures: List[float]
    entropies: List[float]
    derivatives: List[float]
    estimated_tc: float
    observed_peak_t: Optional[float]


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
    basin_weights: Optional[tuple[float, float, float]]


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
        topology: Optional[BasinTopology] = None,
    ) -> RegimeAnalysis:
        """
        Perform full regime analysis from logits.
        """
        if topology is None:
            topology = BasinTopology.default()

        # Compute statistics
        variance = self.compute_logit_variance(logits, temperature)
        _, _, std_dev = self.compute_logit_statistics(logits)
        v_eff = self.effective_vocabulary_size(logits, 1.0)

        # Estimate T_c
        tc = self.estimate_critical_temperature(std_dev, v_eff)

        # Classify state
        state = self.classify_state(temperature, tc)

        # Predict modifier effect
        base_entropy = self.compute_entropy(logits, temperature)
        predicted, confidence = self.predict_modifier_effect(
            state,
            intensity_score,
            base_entropy,
        )

        # Compute basin weights
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
        tolerance: float = 0.15,
    ) -> RegimeState:
        """Classify current state based on temperature relative to T_c."""
        if critical_temperature <= 0:
            return RegimeState.ORDERED

        ratio = temperature / critical_temperature

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
    ) -> tuple[float, float]:
        """Predict the sign and magnitude of entropy change."""
        if state == RegimeState.ORDERED:
            # Strong cooling
            delta_h = -intensity_score * 0.4 * base_entropy
            return (delta_h, 0.85)
        elif state == RegimeState.CRITICAL:
            # Unpredictable
            return (0.0, 0.3)
        else:  # DISORDERED
            # Mild heating
            delta_h = intensity_score * 0.15 * base_entropy
            return (delta_h, 0.6)

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
