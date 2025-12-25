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

The entropy and variance values ARE the cognitive state - no classification needed.

Information-theoretic thresholds (not arbitrary):
- Confident: entropy < ln(e²) ≈ 2.0 (probability mass concentrated)
- Uncertain: entropy > 3.0 (high uncertainty)
- Distress signature: high entropy + low variance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EntropyTransition:
    """Records an entropy transition during generation.

    Raw entropy/variance values before and after. The delta IS the severity change.
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

    @property
    def is_escalation(self) -> bool:
        """Entropy increased significantly (getting more uncertain)."""
        return self.entropy_delta > 0.5

    @property
    def is_recovery(self) -> bool:
        """Entropy decreased significantly (getting more confident)."""
        return self.entropy_delta < -0.5

    @property
    def description(self) -> str:
        """Human-readable description of the transition."""
        if self.is_escalation:
            direction = "escalated"
        elif self.is_recovery:
            direction = "recovered"
        else:
            direction = "changed"
        return (
            f"Entropy {direction} from {self.from_entropy:.2f} to "
            f"{self.to_entropy:.2f} at token {self.token_index}"
        )


# Backward compatibility alias
StateTransition = EntropyTransition


def is_confident(entropy: float, variance: float) -> bool:
    """Check if entropy indicates confident state (low entropy)."""
    return entropy < 2.0


def is_uncertain(entropy: float, variance: float) -> bool:
    """Check if entropy indicates uncertain state (high entropy)."""
    return entropy > 3.0


def is_distressed(entropy: float, variance: float) -> bool:
    """Check if entropy indicates distress (high entropy + low variance)."""
    return entropy > 3.5 and variance < 0.2


def requires_caution(entropy: float, variance: float) -> bool:
    """Check if current state warrants caution."""
    return is_uncertain(entropy, variance) or is_distressed(entropy, variance)
