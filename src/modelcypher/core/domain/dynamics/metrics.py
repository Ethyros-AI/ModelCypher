"""
Optimization Metrics Calculator for Training Dynamics.

Integrates with phase_transition.py for physics-based calculations.
"""
from dataclasses import dataclass
from typing import Optional
import math
import mlx.core as mx

from .phase_transition import (
    Phase,
    compute_logit_statistics,
    estimate_critical_temperature,
    classify_phase,
    compute_entropy,
    effective_vocabulary_size,
)


@dataclass
class OptimizationState:
    """State vector for optimization dynamics."""
    perplexity: float
    total_energy: float  # Hamiltonian: H = T + V
    gradient_norm: float
    loss: float
    is_stable: bool
    phase: Optional[Phase] = None
    estimated_tc: Optional[float] = None


class OptimizationMetricCalculator:
    """
    Calculates high-dimensional optimization metrics.

    Integrates physics-based phase transition theory with standard ML metrics:
    - Perplexity: exp(entropy)
    - Gradient Norm: L2 norm of update vector
    - Loss: Function value (Potential)
    - Phase: Thermodynamic phase classification
    """

    # Thresholds for divergence detection
    CRITICAL_PERPLEXITY_THRESHOLD = 100.0
    STABLE_PERPLEXITY_RANGE = (0.5, 5.0)

    def calculate_metrics(
        self,
        loss: float,
        gradient_norm: float,
        entropy: float,
        logits: Optional[mx.array] = None,
        temperature: float = 1.0,
    ) -> OptimizationState:
        """
        Computes optimization state vector.

        Args:
            loss: Current loss value
            gradient_norm: L2 norm of gradients
            entropy: Current entropy value
            logits: Optional raw logits for phase analysis
            temperature: Current temperature (for phase classification)

        Returns:
            OptimizationState with all computed metrics
        """
        # Perplexity from entropy
        perplexity = math.exp(entropy) if entropy < 100 else float('inf')

        # Total "Energy" in optimization landscape (Hamiltonian view)
        # H = T + V (Kinetic + Potential)
        kinetic = gradient_norm ** 2
        potential = loss
        total_energy = kinetic + potential

        is_stable = (
            self.STABLE_PERPLEXITY_RANGE[0] <= perplexity <= self.STABLE_PERPLEXITY_RANGE[1]
        )

        # Phase analysis if logits provided
        phase: Optional[Phase] = None
        estimated_tc: Optional[float] = None

        if logits is not None:
            _, _, std_dev = compute_logit_statistics(logits)
            v_eff = effective_vocabulary_size(logits, 1.0)
            estimated_tc = estimate_critical_temperature(std_dev, v_eff)
            phase = classify_phase(temperature, estimated_tc)

        return OptimizationState(
            perplexity=perplexity,
            total_energy=total_energy,
            gradient_norm=gradient_norm,
            loss=loss,
            is_stable=is_stable,
            phase=phase,
            estimated_tc=estimated_tc,
        )

    def calculate_metrics_from_logits(
        self,
        logits: mx.array,
        loss: float,
        gradient_norm: float,
        temperature: float = 1.0,
    ) -> OptimizationState:
        """
        Calculates metrics directly from logits (full physics integration).

        Args:
            logits: Raw logit values from the model
            loss: Current loss value
            gradient_norm: L2 norm of gradients
            temperature: Current temperature

        Returns:
            OptimizationState with phase transition analysis
        """
        entropy = compute_entropy(logits, temperature)
        return self.calculate_metrics(
            loss=loss,
            gradient_norm=gradient_norm,
            entropy=entropy,
            logits=logits,
            temperature=temperature,
        )

    def analyze_stability(self, state: OptimizationState) -> str:
        """
        Analyze training stability based on optimization state.

        Returns one of: "DIVERGENT", "OVERFITTING", "STABLE", "VOLATILE"
        """
        if state.perplexity > self.CRITICAL_PERPLEXITY_THRESHOLD:
            return "DIVERGENT"
        elif state.perplexity < self.STABLE_PERPLEXITY_RANGE[0]:
            return "OVERFITTING"
        elif state.is_stable:
            return "STABLE"
        else:
            return "VOLATILE"

    def get_phase_recommendation(self, state: OptimizationState) -> str:
        """
        Get recommendation based on current phase.
        """
        if state.phase is None:
            return "No phase data available"

        if state.phase == Phase.ORDERED:
            return "In ordered phase: modifiers will reduce entropy. Consider increasing temperature."
        elif state.phase == Phase.CRITICAL:
            return "Near critical temperature: behavior is unpredictable. Monitor closely."
        else:
            return "In disordered phase: modifiers will increase entropy. Consider decreasing temperature."
