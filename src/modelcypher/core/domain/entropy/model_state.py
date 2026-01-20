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



