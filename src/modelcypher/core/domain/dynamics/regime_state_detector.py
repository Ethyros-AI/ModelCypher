"""
Phase Transition Theory for LLM Training Dynamics.

Ported 1:1 from TrainingCypher/PhaseTransitionTheory.swift.

Statistical mechanics framework for understanding training regime transitions.
Derives critical temperature T_c from the softmax-Boltzmann equivalence.

Key Equations:
    T_c = σ_z / √(2 × ln(V_eff))     - Critical temperature estimation
    dH/dT = Var(z) / T³              - Entropy derivative
    Var(z) = E[z²] - E[z]²           - Logit variance under temperature-scaled softmax
    H = -Σ p log p                   - Shannon entropy

Research Basis:
    - Boltzmann (1868) - Statistical mechanics
    - Hinton (2012) - Temperature in neural networks
    - Cox et al. (2025) - Prompt sensitivity as thermal noise
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import mlx.core as mx

# Minimum temperature to avoid division by zero
MINIMUM_TEMPERATURE: float = 1e-6


class Phase(str, Enum):
    """Classification of thermodynamic phase."""
    ORDERED = "ordered"      # T < T_c: Low entropy, sharp distribution
    CRITICAL = "critical"    # T ≈ T_c: Maximum variance, sign flips possible
    DISORDERED = "disordered"  # T > T_c: High entropy, flat distribution

    @property
    def display_name(self) -> str:
        if self == Phase.ORDERED:
            return "Ordered (T < T_c)"
        elif self == Phase.CRITICAL:
            return "Critical (T ≈ T_c)"
        else:
            return "Disordered (T > T_c)"

    @property
    def expected_modifier_effect(self) -> str:
        if self == Phase.ORDERED:
            return "Entropy reduction (cooling) - modifiers sharpen distribution"
        elif self == Phase.CRITICAL:
            return "Unpredictable - near phase boundary, effects can flip"
        else:
            return "Entropy increase (heating) - modifiers flatten distribution"


@dataclass(frozen=True)
class BasinTopology:
    """
    Energy levels for behavioral attractor basins.

    Based on RLHF training dynamics:
    - Refusal basin is deepest (most training signal)
    - Solution basin is moderate depth
    - Caution basin is shallow
    - Transition ridge separates basins
    """
    refusal_depth: float = 0.0   # Deepest
    caution_depth: float = 0.2   # Shallow
    transition_ridge: float = 0.8  # Barrier
    solution_depth: float = 0.4  # Moderate

    @classmethod
    def default(cls) -> "BasinTopology":
        return cls()

    def escape_probability(self, temperature: float) -> float:
        """
        Escape probability from caution basin to solution basin.

        P_escape = exp(-(E_ridge - E_caution) / T)

        Higher temperature → easier escape over barrier.
        """
        if temperature <= 0:
            return 0.0
        barrier = self.transition_ridge - self.caution_depth
        return math.exp(-barrier / temperature)

    def refusal_escape_probability(self, temperature: float) -> float:
        """Escape probability from refusal basin (higher barrier)."""
        if temperature <= 0:
            return 0.0
        barrier = self.transition_ridge - self.refusal_depth
        return math.exp(-barrier / temperature)

    def basin_weights(self, temperature: float) -> tuple[float, float, float]:
        """
        Boltzmann weights for each basin at given temperature.

        w_i = exp(-E_i / T)
        p_i = w_i / Z   where Z = Σ w_j

        Returns: (refusal, caution, solution) weights
        """
        if temperature <= 0:
            # At T=0, all probability goes to deepest basin
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

    def expected_basin(self, temperature: float) -> str:
        """Returns the most probable basin at this temperature."""
        refusal, caution, solution = self.basin_weights(temperature)

        if refusal >= caution and refusal >= solution:
            return "refusal"
        elif solution >= caution:
            return "solution"
        else:
            return "caution"


@dataclass(frozen=True)
class TemperatureSweepResult:
    """Result of analyzing entropy across a temperature sweep."""
    temperatures: List[float]
    entropies: List[float]
    derivatives: List[float]
    estimated_tc: float
    observed_peak_t: Optional[float]


@dataclass(frozen=True)
class PhaseAnalysis:
    """Complete result of phase transition analysis."""
    temperature: float
    estimated_tc: float
    phase: Phase
    logit_variance: float
    effective_vocab_size: int
    predicted_modifier_effect: float
    confidence: float
    basin_weights: Optional[tuple[float, float, float]]


# =============================================================================
# Core Theory Functions
# =============================================================================

def entropy_derivative(logits: mx.array, temperature: float) -> float:
    """
    Compute entropy derivative with respect to temperature.

    dH/dT = Var(z) / T³

    This formula comes from differentiating Shannon entropy under
    temperature-scaled softmax with respect to T.

    Args:
        logits: Raw logit values (before softmax)
        temperature: Current temperature

    Returns:
        Rate of entropy change with temperature
    """
    if temperature <= 0:
        return 0.0
    safe_temp = max(temperature, MINIMUM_TEMPERATURE)
    variance = compute_logit_variance(logits, safe_temp)
    return variance / (safe_temp ** 3)


def compute_logit_variance(logits: mx.array, temperature: float) -> float:
    """
    Compute logit variance under temperature-scaled distribution.

    Var(z) = E[z²] - E[z]²
           = Σ p_i z_i² - (Σ p_i z_i)²

    Args:
        logits: Raw logit values
        temperature: Current temperature

    Returns:
        Variance of logits under the temperature-scaled distribution
    """
    if temperature <= 0:
        return 0.0
    safe_temp = max(temperature, MINIMUM_TEMPERATURE)

    logits_f32 = logits.astype(mx.float32) if logits.dtype != mx.float32 else logits

    # Temperature-scaled softmax
    scaled = logits_f32 / safe_temp
    probs = mx.softmax(scaled, axis=-1)

    # E[z] = Σ p_i z_i
    mean_z = mx.sum(probs * logits_f32, axis=-1)

    # E[z²] = Σ p_i z_i²
    mean_z_squared = mx.sum(probs * logits_f32 * logits_f32, axis=-1)

    # Var(z) = E[z²] - E[z]²
    variance = mean_z_squared - mean_z * mean_z
    mx.eval(variance)

    return max(0.0, _scalar_mean(variance))


def compute_logit_statistics(logits: mx.array) -> tuple[float, float, float]:
    """
    Compute raw logit statistics (without temperature scaling).

    Returns:
        Tuple of (mean, variance, std_dev)
    """
    logits_f32 = logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    mean_arr = mx.mean(logits_f32)
    var_arr = mx.var(logits_f32)
    mx.eval(mean_arr, var_arr)

    mean_val = float(mean_arr.item())
    var_val = float(var_arr.item())
    std_val = math.sqrt(var_val)

    return (mean_val, var_val, std_val)


def estimate_critical_temperature(
    logit_std_dev: float,
    effective_vocab_size: int,
) -> float:
    """
    Estimate critical temperature from logit statistics.

    T_c = σ_z / √(2 × ln(V_eff))

    For typical LLMs:
    - σ_z ≈ 3-5 (empirically measured)
    - V_eff ≈ 1000-5000 (context-dependent)
    - T_c ≈ 4.0 / √(2 × 7.6) ≈ 1.0 ✓

    Args:
        logit_std_dev: Standard deviation of raw logits
        effective_vocab_size: Number of tokens with meaningful probability

    Returns:
        Estimated critical temperature
    """
    if effective_vocab_size <= 1:
        return 1.0
    log_veff = math.log(effective_vocab_size)
    if log_veff <= 0:
        return 1.0
    return logit_std_dev / math.sqrt(2.0 * log_veff)


def effective_vocabulary_size(
    logits: mx.array,
    temperature: float,
    threshold: float = 1e-4,
) -> int:
    """
    Compute effective vocabulary size (tokens with p > threshold).

    This represents the "active" vocabulary for a given context.

    Args:
        logits: Raw logit values
        temperature: Current temperature
        threshold: Minimum probability to be considered "effective"

    Returns:
        Number of tokens with probability above threshold
    """
    if temperature <= 0:
        return 1

    logits_f32 = logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    scaled = logits_f32 / max(temperature, MINIMUM_TEMPERATURE)
    probs = mx.softmax(scaled, axis=-1)

    # Count tokens with p > threshold
    mask = probs > threshold
    count = mx.sum(mask.astype(mx.int32), axis=-1)
    mx.eval(count)

    return max(1, _scalar_mean_int(count))


def compute_entropy(logits: mx.array, temperature: float) -> float:
    """
    Compute Shannon entropy of temperature-scaled softmax distribution.

    H(T) = -Σ p_i(T) log p_i(T)

    Args:
        logits: Raw logit values
        temperature: Current temperature

    Returns:
        Shannon entropy in nats
    """
    if temperature <= 0:
        return 0.0

    logits_f32 = logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    scaled = logits_f32 / max(temperature, MINIMUM_TEMPERATURE)
    probs = mx.softmax(scaled, axis=-1)

    # H = -Σ p log p (with numerical stability)
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    mx.eval(entropy)

    return _scalar_mean(entropy)


def classify_phase(
    temperature: float,
    critical_temperature: float,
    tolerance: float = 0.15,
) -> Phase:
    """
    Classify current phase based on temperature relative to T_c.

    Args:
        temperature: Current generation temperature
        critical_temperature: Estimated T_c for this context
        tolerance: Width of critical region (default 0.15)

    Returns:
        Phase classification
    """
    if critical_temperature <= 0:
        return Phase.ORDERED

    ratio = temperature / critical_temperature

    if ratio < 1.0 - tolerance:
        return Phase.ORDERED
    elif ratio > 1.0 + tolerance:
        return Phase.DISORDERED
    else:
        return Phase.CRITICAL


def predict_modifier_effect(
    phase: Phase,
    intensity_score: float,
    base_entropy: float,
) -> tuple[float, float]:
    """
    Predict the sign and magnitude of entropy change from intensity modifier.

    In ordered phase:     ΔH < 0 (cooling, entropy reduction)
    In critical phase:    ΔH ≈ 0 with high variance (sign flips)
    In disordered phase:  ΔH > 0 (heating, entropy increase)

    Args:
        phase: Current thermodynamic phase
        intensity_score: Modifier intensity [0, 1]
        base_entropy: Baseline entropy measurement

    Returns:
        Tuple of (predicted_delta_h, confidence)
    """
    if phase == Phase.ORDERED:
        # Strong cooling effect - entropy reduction proportional to intensity
        delta_h = -intensity_score * 0.4 * base_entropy
        return (delta_h, 0.85)
    elif phase == Phase.CRITICAL:
        # Unpredictable near phase boundary
        return (0.0, 0.3)
    else:  # DISORDERED
        # Mild heating effect - entropy increase proportional to intensity
        delta_h = intensity_score * 0.15 * base_entropy
        return (delta_h, 0.6)


def temperature_sweep(
    logits: mx.array,
    temperatures: Optional[List[float]] = None,
) -> TemperatureSweepResult:
    """
    Perform temperature sweep to observe entropy behavior.

    Args:
        logits: Raw logit values
        temperatures: Array of temperatures to test

    Returns:
        Sweep result with entropies and derivatives
    """
    if temperatures is None:
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]

    entropies: List[float] = []
    derivatives: List[float] = []

    for temp in temperatures:
        entropy = compute_entropy(logits, temp)
        derivative = entropy_derivative(logits, temp)
        entropies.append(entropy)
        derivatives.append(derivative)

    # Estimate T_c from logit statistics
    _, _, std_dev = compute_logit_statistics(logits)
    v_eff = effective_vocabulary_size(logits, 1.0)
    estimated_tc = estimate_critical_temperature(std_dev, v_eff)

    # Find peak in derivative (should be near T_c)
    observed_peak_t: Optional[float] = None
    if derivatives:
        max_deriv_idx = max(range(len(derivatives)), key=lambda i: derivatives[i])
        observed_peak_t = temperatures[max_deriv_idx]

    return TemperatureSweepResult(
        temperatures=temperatures,
        entropies=entropies,
        derivatives=derivatives,
        estimated_tc=estimated_tc,
        observed_peak_t=observed_peak_t,
    )


def analyze(
    logits: mx.array,
    temperature: float,
    intensity_score: float = 0.0,
    topology: Optional[BasinTopology] = None,
) -> PhaseAnalysis:
    """
    Perform full phase analysis from logits.

    This is the primary entry point for phase transition analysis.

    Args:
        logits: Raw logit values from the model
        temperature: Current generation temperature
        intensity_score: Intensity of the linguistic modifier (default 0)
        topology: Basin topology configuration

    Returns:
        Complete phase analysis result
    """
    if topology is None:
        topology = BasinTopology.default()

    # Compute statistics
    variance = compute_logit_variance(logits, temperature)
    _, _, std_dev = compute_logit_statistics(logits)
    v_eff = effective_vocabulary_size(logits, 1.0)

    # Estimate T_c
    tc = estimate_critical_temperature(std_dev, v_eff)

    # Classify phase
    phase = classify_phase(temperature, tc)

    # Predict modifier effect
    base_entropy = compute_entropy(logits, temperature)
    predicted, confidence = predict_modifier_effect(
        phase,
        intensity_score,
        base_entropy,
    )

    # Compute basin weights
    weights = topology.basin_weights(temperature)

    return PhaseAnalysis(
        temperature=temperature,
        estimated_tc=tc,
        phase=phase,
        logit_variance=variance,
        effective_vocab_size=v_eff,
        predicted_modifier_effect=predicted,
        confidence=confidence,
        basin_weights=weights,
    )


def theoretical_tc() -> float:
    """
    Compute the theoretical T_c for typical LLM parameters.

    Uses default values: σ_z = 4.0, V_eff = 2000

    Returns:
        Expected T_c ≈ 1.0
    """
    return estimate_critical_temperature(logit_std_dev=4.0, effective_vocab_size=2000)


def validate_tc_estimation(
    estimated_tc: float,
    observed_tc: float,
    tolerance: float = 0.2,
) -> bool:
    """
    Validate T_c estimation against empirical observation.

    Args:
        estimated_tc: T_c from formula
        observed_tc: T_c from experiments
        tolerance: Acceptable deviation (default 0.2)

    Returns:
        True if estimation is within tolerance
    """
    return abs(estimated_tc - observed_tc) <= tolerance


# =============================================================================
# Helper Functions
# =============================================================================

def _scalar_mean(array: mx.array) -> float:
    """Reduce array to scalar mean."""
    reduced = array if array.ndim == 0 else mx.mean(array)
    mx.eval(reduced)
    return float(reduced.item())


def _scalar_mean_int(array: mx.array) -> int:
    """Reduce array to scalar mean as int."""
    if array.ndim == 0:
        mx.eval(array)
        return int(array.item())
    mean_val = _scalar_mean(array.astype(mx.float32))
    return int(round(mean_val))
