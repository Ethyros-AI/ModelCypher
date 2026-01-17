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
Model cognitive state representations using raw entropy/variance values.

Notes
-----
Absolute entropy thresholds are model-dependent. Different models operate
at vastly different entropy scales:
- Qwen 0.5B: mean ~5.0, std ~1.08
- Qwen 3B: mean ~7.0, std ~1.05
- Llama 3B 4-bit: mean ~11.2, std ~0.22

Use z-scores relative to model baseline, not absolute thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
)


def _model_eps() -> float:
    b = get_default_backend()
    return machine_epsilon(b, b.array([1.0], dtype=precision_dtype(b)))


# Z-score to confidence level mapping (one-sided, derived from normal CDF)
# These are the standard statistical thresholds, not arbitrary values.
# Formula: confidence = 1 - (1 - erf(z / sqrt(2))) / 2  (one-sided)
#
# | z-score | One-sided confidence | Two-sided confidence |
# |---------|---------------------|---------------------|
# | 1.0     | 84.13%              | 68.27%              |
# | 1.5     | 93.32%              | 86.64%              |
# | 1.645   | 95.00%              | 90.00%              |
# | 1.96    | 97.50%              | 95.00%              |
# | 2.0     | 97.72%              | 95.45%              |
# | 2.576   | 99.50%              | 99.00%              |
#
# Defaults used in this module:
# - is_low: z=-1.5 (93.3% confidence that entropy is below baseline)
# - is_high: z=2.0 (97.7% confidence that entropy is above baseline)
# - is_escalation/recovery: z=1.0 (84.1% confidence for transitions)
Z_CONFIDENCE_84 = 1.0  # 84.13% one-sided confidence
Z_CONFIDENCE_93 = 1.5  # 93.32% one-sided confidence
Z_CONFIDENCE_95 = 1.645  # 95.00% one-sided confidence
Z_CONFIDENCE_97 = 1.96  # 97.50% one-sided confidence
Z_CONFIDENCE_98 = 2.0  # 97.72% one-sided confidence
Z_CONFIDENCE_99 = 2.576  # 99.50% one-sided confidence


@dataclass(frozen=True)
class EntropyBaseline:
    """Calibrated entropy baseline for a specific model.

    Must be computed empirically by running the model on representative prompts.
    Use `mc entropy calibrate --model <path>` to generate.
    """

    mean: float
    std: float
    max_theoretical: float
    model_name: str = ""

    def z_score(self, entropy: float) -> float:
        """Compute z-score of entropy relative to baseline."""
        eps = _model_eps()
        if self.std < eps:
            return 0.0
        return (entropy - self.mean) / self.std

    def is_low(self, entropy: float, z_threshold: float = -Z_CONFIDENCE_93) -> bool:
        """Check if entropy is significantly below baseline (confident).

        Default threshold -1.5 corresponds to 93.3% one-sided confidence.
        """
        return self.z_score(entropy) < z_threshold

    def is_high(self, entropy: float, z_threshold: float = Z_CONFIDENCE_98) -> bool:
        """Check if entropy is significantly above baseline (uncertain).

        Default threshold 2.0 corresponds to 97.7% one-sided confidence.
        """
        return self.z_score(entropy) > z_threshold

    def normalized(self, entropy: float) -> float:
        """Normalize entropy to [0, 1] using theoretical max."""
        eps = _model_eps()
        if self.max_theoretical < eps:
            return 0.0
        return entropy / self.max_theoretical


@dataclass(frozen=True)
class EntropyTransition:
    """Records an entropy transition during generation.

    Use z_score_delta with a baseline for model-appropriate significance testing.

    Attributes
    ----------
    from_entropy : float
        Entropy before transition.
    from_variance : float
        Variance before transition.
    to_entropy : float
        Entropy after transition.
    to_variance : float
        Variance after transition.
    token_index : int
        Token index where transition occurred.
    timestamp : datetime
        When the transition was recorded.
    reason : str or None
        Optional explanation for the transition.
    """

    from_entropy: float
    from_variance: float
    to_entropy: float
    to_variance: float
    token_index: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str | None = None

    @property
    def entropy_delta(self) -> float:
        """Change in entropy. Positive = increasing uncertainty."""
        return self.to_entropy - self.from_entropy

    @property
    def variance_delta(self) -> float:
        """Change in variance."""
        return self.to_variance - self.from_variance

    def z_score_delta(self, baseline: EntropyBaseline) -> float:
        """Change in z-score terms (model-appropriate significance)."""
        eps = _model_eps()
        if baseline.std < eps:
            return 0.0
        return self.entropy_delta / baseline.std

    def is_escalation(self, baseline: EntropyBaseline, z_threshold: float = Z_CONFIDENCE_84) -> bool:
        """Entropy increased significantly (getting more uncertain).

        Args:
            baseline: Uses z-score based significance.
            z_threshold: Z-score threshold for significance (default: 1.0 = 84.1% confidence).
        """
        return self.z_score_delta(baseline) > z_threshold

    def is_recovery(self, baseline: EntropyBaseline, z_threshold: float = Z_CONFIDENCE_84) -> bool:
        """Entropy decreased significantly (getting more confident).

        Args:
            baseline: Uses z-score based significance.
            z_threshold: Z-score threshold for significance (default: 1.0 = 84.1% confidence).
        """
        return self.z_score_delta(baseline) < -z_threshold

    @property
    def description(self) -> str:
        """Human-readable description of the transition."""
        delta = self.entropy_delta
        if delta > 0:
            direction = "increased"
        elif delta < 0:
            direction = "decreased"
        else:
            direction = "unchanged"
        return (
            f"Entropy {direction} from {self.from_entropy:.2f} to "
            f"{self.to_entropy:.2f} (delta={delta:+.2f}) at token {self.token_index}"
        )




def is_confident(entropy: float, baseline: EntropyBaseline) -> bool:
    """Check if entropy indicates confident state.

    Args:
        entropy: Current entropy value.
        baseline: Model entropy baseline.
    """
    return baseline.is_low(entropy)

def is_uncertain(entropy: float, baseline: EntropyBaseline) -> bool:
    """Check if entropy indicates uncertain state.

    Args:
        entropy: Current entropy value.
        baseline: Model entropy baseline.
    """
    return baseline.is_high(entropy)

def is_distressed(
    entropy: float,
    variance: float,
    baseline: EntropyBaseline,
    *,
    z_threshold: float = Z_CONFIDENCE_98,
) -> bool:
    """Check if entropy indicates distress (high entropy + low variance).

    High entropy with low variance suggests the model is "stuck" - uncertain
    but not exploring different options.

    Args:
        entropy: Current entropy value.
        variance: Current variance.
        baseline: Model entropy baseline.
        z_threshold: Z-score threshold for "high entropy" (default: 2.0 = 97.7% confidence).
    """
    # High entropy relative to baseline
    is_high_entropy = baseline.z_score(entropy) > z_threshold
    # Low variance relative to baseline standard deviation
    is_low_variance = variance < baseline.std
    return is_high_entropy and is_low_variance


def requires_caution(entropy: float, variance: float, baseline: EntropyBaseline) -> bool:
    """Check if current state warrants caution.

    Returns True if entropy is uncertain OR distressed (high entropy + low variance).
    """
    return is_uncertain(entropy, baseline) or is_distressed(entropy, variance, baseline)
