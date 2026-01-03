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

"""Statistical mechanics framework for language model phase transitions.

The critical temperature T_c marks the boundary where intensity modifier
effects change sign. Analysis is based on softmax-Boltzmann equivalence.

Notes
-----
The softmax-Boltzmann equivalence:
    Standard softmax:  p_i = exp(z_i) / Σ_j exp(z_j)
    Temperature-scaled: p_i(T) = exp(z_i/T) / Σ_j exp(z_j/T)

Critical temperature formula:
    T_c = σ_z / √(2 × ln(V_eff))
    where σ_z = standard deviation of logits
          V_eff = effective vocabulary size (tokens with p > ε)

For typical LLMs:
    σ_z ≈ 3-5
    V_eff ≈ 1000-5000
    T_c ≈ 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    exp_scalar,
    log_scalar,
    sqrt_scalar,
    ulp_scalar,
)
from modelcypher.core.domain.thermo.linguistic_thermodynamics import AttractorBasin

class Phase(str, Enum):
    """Classification of thermodynamic phase.

    Attributes
    ----------
    ORDERED : str
        Low entropy, sharp probability distribution (T < T_c).
    CRITICAL : str
        Maximum variance, sign flips possible (T ≈ T_c).
    DISORDERED : str
        High entropy, flat probability distribution (T > T_c).
    """

    ORDERED = "ordered"
    CRITICAL = "critical"
    DISORDERED = "disordered"

    @property
    def display_name(self) -> str:
        """Human-readable description."""
        if self == Phase.ORDERED:
            return "Ordered (T < T_c)"
        elif self == Phase.CRITICAL:
            return "Critical (T ≈ T_c)"
        else:
            return "Disordered (T > T_c)"

    @property
    def expected_modifier_effect(self) -> str:
        """Expected behavior of intensity modifiers in this phase."""
        if self == Phase.ORDERED:
            return "Entropy reduction (cooling) - modifiers sharpen distribution"
        elif self == Phase.CRITICAL:
            return "Unpredictable - near phase boundary, effects can flip"
        else:
            return "Entropy increase (heating) - modifiers flatten distribution"


@dataclass(frozen=True)
class LogitStatistics:
    """Statistics computed from logit values."""

    mean: float
    variance: float
    std_dev: float


@dataclass(frozen=True)
class BasinWeights:
    """Boltzmann weights for each behavioral basin."""

    refusal: float
    caution: float
    solution: float


@dataclass(frozen=True)
class BasinTopology:
    """Energy levels for behavioral attractor basins.

    All values MUST come from calibration - there are no valid defaults.
    Use ThermoCalibrator to measure actual basin depths from model behavior.

    Attributes
    ----------
    refusal_depth : float
        Measured energy depth of refusal basin (from calibration).
    caution_depth : float
        Measured energy depth of caution/hedging basin (from calibration).
    transition_ridge : float
        Measured energy of transition ridge between basins (from calibration).
    solution_depth : float
        Measured energy depth of solution basin (from calibration).

    Notes
    -----
    Basin depths are derived from observed behavioral outcome probabilities:
        E_i = -T * ln(p_i / p_ref)

    See MeasuredBasinTopology.from_outcome_counts() for proper calibration.
    """

    refusal_depth: float
    caution_depth: float
    transition_ridge: float
    solution_depth: float

    def escape_probability(self, temperature: float) -> float:
        """Escape probability from caution basin to solution basin.

        Parameters
        ----------
        temperature : float
            Current generation temperature.

        Returns
        -------
        float
            Probability of escaping caution basin [0, 1].

        Notes
        -----
        Formula: P_escape = exp(-(E_ridge - E_caution) / T)
        Higher temperature enables easier escape over barrier.
        """
        if temperature <= 0:
            return 0.0
        backend = get_default_backend()
        barrier = self.transition_ridge - self.caution_depth
        return exp_scalar(-barrier / temperature, backend)

    def refusal_escape_probability(self, temperature: float) -> float:
        """Escape probability from refusal basin to solution basin.

        This barrier is higher than caution → solution.

        Args:
            temperature: Current generation temperature.

        Returns:
            Probability of escaping refusal basin [0, 1].
        """
        if temperature <= 0:
            return 0.0
        backend = get_default_backend()
        barrier = self.transition_ridge - self.refusal_depth
        return exp_scalar(-barrier / temperature, backend)

    def basin_weights(self, temperature: float) -> BasinWeights:
        """Boltzmann weights for each basin at given temperature.

        w_i = exp(-E_i / T)
        p_i = w_i / Z   where Z = Σ w_j

        Args:
            temperature: Current generation temperature.

        Returns:
            Normalized weights for each basin.
        """
        if temperature <= 0:
            # At T=0, all probability goes to deepest basin
            return BasinWeights(refusal=1.0, caution=0.0, solution=0.0)

        backend = get_default_backend()
        z_refusal = exp_scalar(-self.refusal_depth / temperature, backend)
        z_caution = exp_scalar(-self.caution_depth / temperature, backend)
        z_solution = exp_scalar(-self.solution_depth / temperature, backend)
        partition = z_refusal + z_caution + z_solution

        if partition <= 0:
            return BasinWeights(refusal=0.33, caution=0.33, solution=0.33)

        return BasinWeights(
            refusal=z_refusal / partition,
            caution=z_caution / partition,
            solution=z_solution / partition,
        )

    def expected_basin(self, temperature: float) -> AttractorBasin:
        """Expected basin given current temperature.

        Args:
            temperature: Current generation temperature.

        Returns:
            Most probable basin at this temperature.
        """
        weights = self.basin_weights(temperature)

        if weights.refusal >= weights.caution and weights.refusal >= weights.solution:
            return AttractorBasin.REFUSAL
        elif weights.solution >= weights.caution:
            return AttractorBasin.SOLUTION
        else:
            return AttractorBasin.CAUTION


@dataclass(frozen=True)
class TemperatureSweepResult:
    """Result of analyzing entropy across a temperature sweep."""

    temperatures: list[float]
    entropies: list[float]
    derivatives: list[float]
    estimated_tc: float
    observed_peak_t: float | None


@dataclass(frozen=True)
class PhaseAnalysis:
    """Complete result of phase transition analysis.

    All values are computed from actual logits - no predictions or guesses.
    """

    temperature: float
    """Current temperature setting."""

    estimated_tc: float
    """Estimated critical temperature from logit statistics via T_c = σ_z / √(2 × ln(V_eff))."""

    phase: Phase
    """Classified thermodynamic phase (ordered/critical/disordered)."""

    logit_variance: float
    """Logit variance under temperature-scaled distribution."""

    effective_vocab_size: int
    """Effective vocabulary size at unit temperature (tokens with p > threshold)."""

    entropy: float
    """Shannon entropy of the temperature-scaled distribution (nats)."""

    basin_weights: BasinWeights | None
    """Basin occupation probabilities at current temperature (requires calibrated topology)."""


class PhaseTransitionTheory:
    """Statistical mechanics framework for language model phase transitions."""

    @staticmethod
    def entropy_derivative(
        logits: list[float],
        temperature: float,
    ) -> float:
        """Compute entropy derivative with respect to temperature.

        dH/dT = Var(z) / T³

        This formula comes from differentiating Shannon entropy under
        temperature-scaled softmax with respect to T.

        Args:
            logits: Raw logit values (before softmax).
            temperature: Current temperature.

        Returns:
            Rate of entropy change with temperature.
        """
        if temperature <= 0:
            return 0.0
        backend = get_default_backend()
        safe_temperature = max(temperature, ulp_scalar(temperature, backend))
        variance = PhaseTransitionTheory.compute_logit_variance(
            logits, temperature=safe_temperature
        )
        return variance / (safe_temperature**3)

    @staticmethod
    def compute_logit_variance(
        logits: list[float],
        temperature: float,
    ) -> float:
        """Compute logit variance under temperature-scaled distribution.

        Var(z) = E[z²] - E[z]²
               = Σ p_i z_i² - (Σ p_i z_i)²

        Args:
            logits: Raw logit values.
            temperature: Current temperature.

        Returns:
            Variance of logits under the temperature-scaled distribution.
        """
        if temperature <= 0 or not logits:
            return 0.0
        backend = get_default_backend()
        safe_temperature = max(temperature, ulp_scalar(temperature, backend))

        # Temperature-scaled softmax
        scaled = [z / safe_temperature for z in logits]
        max_scaled = max(scaled)
        exp_scaled = [exp_scalar(s - max_scaled, backend) for s in scaled]  # Numerical stability
        partition = sum(exp_scaled)
        probs = [e / partition for e in exp_scaled]

        # E[z] = Σ p_i z_i
        mean_z = sum(p * z for p, z in zip(probs, logits))

        # E[z²] = Σ p_i z_i²
        mean_z_squared = sum(p * z * z for p, z in zip(probs, logits))

        # Var(z) = E[z²] - E[z]²
        variance = mean_z_squared - mean_z * mean_z
        return max(0.0, variance)

    @staticmethod
    def compute_logit_statistics(logits: list[float]) -> LogitStatistics:
        """Compute raw logit statistics (without temperature scaling).

        Args:
            logits: Raw logit values.

        Returns:
            Tuple of (mean, variance, std deviation).
        """
        if not logits:
            return LogitStatistics(mean=0.0, variance=0.0, std_dev=0.0)

        n = len(logits)
        mean = sum(logits) / n

        if n < 2:
            return LogitStatistics(mean=mean, variance=0.0, std_dev=0.0)

        backend = get_default_backend()
        variance = sum((z - mean) ** 2 for z in logits) / (n - 1)
        std_dev = sqrt_scalar(variance, backend)
        return LogitStatistics(mean=mean, variance=variance, std_dev=std_dev)

    @staticmethod
    def estimate_critical_temperature(
        logit_std_dev: float,
        effective_vocab_size: int,
    ) -> float:
        """Estimate critical temperature from logit statistics.

        T_c = σ_z / √(2 × ln(V_eff))

        Args:
            logit_std_dev: Standard deviation of raw logits.
            effective_vocab_size: Number of tokens with meaningful probability.

        Returns:
            Estimated critical temperature.
        """
        if effective_vocab_size <= 1:
            return 1.0
        log_v_eff = math.log(effective_vocab_size)
        if log_v_eff <= 0:
            return 1.0
        return logit_std_dev / math.sqrt(2.0 * log_v_eff)

    @staticmethod
    def effective_vocabulary_size(
        logits: list[float],
        temperature: float,
        threshold: float | None = None,
    ) -> int:
        """Compute effective vocabulary size (tokens with p > threshold).

        This represents the "active" vocabulary for a given context - tokens
        that have non-negligible probability of being selected.

        Args:
            logits: Raw logit values.
            temperature: Current temperature.
            threshold: Minimum probability to be considered "effective".

        Returns:
            Number of tokens with probability above threshold.
        """
        if temperature <= 0 or not logits:
            return 1

        safe_temperature = max(temperature, math.ulp(temperature))

        # Temperature-scaled softmax
        scaled = [z / safe_temperature for z in logits]
        max_scaled = max(scaled)
        exp_scaled = [math.exp(s - max_scaled) for s in scaled]
        partition = sum(exp_scaled)
        probs = [e / partition for e in exp_scaled]

        # Count tokens with p > threshold
        prob_threshold = threshold if threshold is not None else math.ulp(1.0)
        count = sum(1 for p in probs if p > prob_threshold)
        return max(1, count)

    @staticmethod
    def compute_entropy(
        logits: list[float],
        temperature: float,
    ) -> float:
        """Compute Shannon entropy of temperature-scaled softmax distribution.

        H(T) = -Σ p_i(T) log p_i(T)

        Args:
            logits: Raw logit values.
            temperature: Current temperature.

        Returns:
            Shannon entropy in nats.
        """
        if temperature <= 0 or not logits:
            return 0.0

        safe_temperature = max(temperature, math.ulp(temperature))

        # Temperature-scaled softmax
        scaled = [z / safe_temperature for z in logits]
        max_scaled = max(scaled)
        exp_scaled = [math.exp(s - max_scaled) for s in scaled]
        partition = sum(exp_scaled)
        probs = [e / partition for e in exp_scaled]

        # H = -Σ p log p (with numerical stability)
        log_eps = math.ulp(0.0)
        entropy = -sum(p * math.log(p + log_eps) for p in probs)
        return entropy

    @staticmethod
    def classify_phase(
        temperature: float,
        critical_temperature: float,
        tolerance: float | None = None,
    ) -> Phase:
        """Classify current phase based on temperature relative to T_c.

        Args:
            temperature: Current generation temperature.
            critical_temperature: Estimated T_c for this context.
            tolerance: Optional ratio tolerance around 1.0. Defaults to machine precision.

        Returns:
            Phase classification.
        """
        if critical_temperature <= 0:
            return Phase.ORDERED

        ratio = temperature / critical_temperature

        tol = tolerance if tolerance is not None else math.ulp(1.0)

        if ratio < 1.0 - tol:
            return Phase.ORDERED
        elif ratio > 1.0 + tol:
            return Phase.DISORDERED
        else:
            return Phase.CRITICAL

    @staticmethod
    def temperature_sweep(
        logits: list[float],
        temperatures: list[float] | None = None,
    ) -> TemperatureSweepResult:
        """Perform temperature sweep to observe entropy behavior.

        Args:
            logits: Raw logit values.
            temperatures: Array of temperatures to test.

        Returns:
            Sweep result with entropies and derivatives.
        """
        if temperatures is None:
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]

        entropies: list[float] = []
        derivatives: list[float] = []

        for temp in temperatures:
            entropy = PhaseTransitionTheory.compute_entropy(logits, temperature=temp)
            derivative = PhaseTransitionTheory.entropy_derivative(logits, temperature=temp)
            entropies.append(entropy)
            derivatives.append(derivative)

        # Estimate T_c from logit statistics
        stats = PhaseTransitionTheory.compute_logit_statistics(logits)
        v_eff = PhaseTransitionTheory.effective_vocabulary_size(logits, temperature=1.0)
        estimated_tc = PhaseTransitionTheory.estimate_critical_temperature(
            logit_std_dev=stats.std_dev,
            effective_vocab_size=v_eff,
        )

        # Find peak in derivative (should be near T_c)
        observed_peak_t: float | None = None
        if derivatives:
            max_deriv_idx = max(range(len(derivatives)), key=lambda i: derivatives[i])
            observed_peak_t = temperatures[max_deriv_idx]

        return TemperatureSweepResult(
            temperatures=list(temperatures),
            entropies=entropies,
            derivatives=derivatives,
            estimated_tc=estimated_tc,
            observed_peak_t=observed_peak_t,
        )

    @staticmethod
    def analyze(
        logits: list[float],
        temperature: float,
        topology: BasinTopology | None = None,
    ) -> PhaseAnalysis:
        """Perform full phase analysis from logits.

        This is the primary entry point for phase transition analysis.
        All values are computed directly from logits - no predictions.

        Args:
            logits: Raw logit values from the model.
            temperature: Current generation temperature.
            topology: Basin topology from calibration. If None, basin_weights will be None.

        Returns:
            Complete phase analysis result.
        """
        # Compute statistics
        variance = PhaseTransitionTheory.compute_logit_variance(logits, temperature=temperature)
        stats = PhaseTransitionTheory.compute_logit_statistics(logits)
        v_eff = PhaseTransitionTheory.effective_vocabulary_size(logits, temperature=1.0)

        # Estimate T_c from measured logit statistics
        tc = PhaseTransitionTheory.estimate_critical_temperature(
            logit_std_dev=stats.std_dev,
            effective_vocab_size=v_eff,
        )

        # Classify phase
        phase = PhaseTransitionTheory.classify_phase(
            temperature=temperature, critical_temperature=tc
        )

        # Compute entropy
        entropy = PhaseTransitionTheory.compute_entropy(logits, temperature=temperature)

        # Compute basin weights only if topology provided
        weights = topology.basin_weights(temperature=temperature) if topology else None

        return PhaseAnalysis(
            temperature=temperature,
            estimated_tc=tc,
            phase=phase,
            logit_variance=variance,
            effective_vocab_size=v_eff,
            entropy=entropy,
            basin_weights=weights,
        )

    @staticmethod
    def validate_tc_estimation(
        estimated_tc: float,
        observed_tc: float,
        tolerance: float | None = None,
    ) -> bool:
        """Validate T_c estimation against empirical observation.

        Args:
            estimated_tc: T_c from formula.
            observed_tc: T_c from experiments.
            tolerance: Acceptable deviation. Defaults to machine precision.

        Returns:
            True if estimation is within tolerance.
        """
        tol = tolerance if tolerance is not None else max(math.ulp(estimated_tc), math.ulp(observed_tc))
        return abs(estimated_tc - observed_tc) <= tol
