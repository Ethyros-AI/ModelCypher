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
        if self.std < 1e-10:
            return 0.0
        return (entropy - self.mean) / self.std

    def is_low(self, entropy: float, z_threshold: float = -1.5) -> bool:
        """Check if entropy is significantly below baseline (confident)."""
        return self.z_score(entropy) < z_threshold

    def is_high(self, entropy: float, z_threshold: float = 2.0) -> bool:
        """Check if entropy is significantly above baseline (uncertain)."""
        return self.z_score(entropy) > z_threshold

    def normalized(self, entropy: float) -> float:
        """Normalize entropy to [0, 1] using theoretical max."""
        if self.max_theoretical < 1e-10:
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
        if baseline.std < 1e-10:
            return 0.0
        return self.entropy_delta / baseline.std

    def is_escalation(self, baseline: EntropyBaseline | None = None, z_threshold: float = 1.0) -> bool:
        """Entropy increased significantly (getting more uncertain).

        Args:
            baseline: If provided, uses z-score based significance (recommended).
                     If None, uses raw delta > 0.5 (legacy, model-dependent).
            z_threshold: Z-score threshold for significance (default: 1.0 std dev).
        """
        if baseline is not None:
            return self.z_score_delta(baseline) > z_threshold
        # Legacy fallback - avoid if possible
        return self.entropy_delta > 0.5

    def is_recovery(self, baseline: EntropyBaseline | None = None, z_threshold: float = 1.0) -> bool:
        """Entropy decreased significantly (getting more confident).

        Args:
            baseline: If provided, uses z-score based significance (recommended).
                     If None, uses raw delta < -0.5 (legacy, model-dependent).
            z_threshold: Z-score threshold for significance (default: 1.0 std dev).
        """
        if baseline is not None:
            return self.z_score_delta(baseline) < -z_threshold
        # Legacy fallback - avoid if possible
        return self.entropy_delta < -0.5

    @property
    def description(self) -> str:
        """Human-readable description of the transition."""
        delta = self.entropy_delta
        if delta > 0.5:
            direction = "escalated"
        elif delta < -0.5:
            direction = "recovered"
        else:
            direction = "changed"
        return (
            f"Entropy {direction} from {self.from_entropy:.2f} to "
            f"{self.to_entropy:.2f} at token {self.token_index}"
        )


# Backward compatibility alias
StateTransition = EntropyTransition


def is_confident(entropy: float, variance: float, baseline: EntropyBaseline | None = None) -> bool:
    """Check if entropy indicates confident state.

    Args:
        entropy: Current entropy value.
        variance: Current variance (unused, kept for API compatibility).
        baseline: Model entropy baseline. If None, returns False (can't determine).
    """
    if baseline is None:
        # Without baseline, we can't determine confidence - different models
        # have vastly different entropy ranges
        return False
    return baseline.is_low(entropy)


def is_uncertain(entropy: float, variance: float, baseline: EntropyBaseline | None = None) -> bool:
    """Check if entropy indicates uncertain state.

    Args:
        entropy: Current entropy value.
        variance: Current variance (unused, kept for API compatibility).
        baseline: Model entropy baseline. If None, returns False (can't determine).
    """
    if baseline is None:
        return False
    return baseline.is_high(entropy)


def is_distressed(entropy: float, variance: float, baseline: EntropyBaseline | None = None) -> bool:
    """Check if entropy indicates distress (high entropy + low variance).

    High entropy with low variance suggests the model is "stuck" - uncertain
    but not exploring different options.

    Args:
        entropy: Current entropy value.
        variance: Current variance.
        baseline: Model entropy baseline. If None, returns False (can't determine).
    """
    if baseline is None:
        return False
    # High entropy (> 2 std above mean) + low variance relative to entropy
    # Variance should scale with entropy; low relative variance is suspicious
    is_high_entropy = baseline.z_score(entropy) > 2.0
    expected_variance = baseline.std * 0.5  # Rough heuristic
    is_low_variance = variance < expected_variance
    return is_high_entropy and is_low_variance


def requires_caution(entropy: float, variance: float, baseline: EntropyBaseline | None = None) -> bool:
    """Check if current state warrants caution."""
    return is_uncertain(entropy, variance, baseline) or is_distressed(entropy, variance, baseline)
