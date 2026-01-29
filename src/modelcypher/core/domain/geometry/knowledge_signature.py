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

"""Signature dataclasses for knowledge discovery.

This module provides immutable dataclasses that capture the geometric signatures
used to distinguish factual knowledge from opinions in neural network representations.

Key Findings (from research):
- Counterfactual sensitivity effect size: 0.94 (strong)
- Facts: high sensitivity (~0.2+), representation changes when fact is violated
- Opinions: low sensitivity (~0.06), representation similar regardless of content

These signatures capture RAW metrics only. No composite scores or heuristic
thresholds are applied - caller interprets the metrics for their use case.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeSignature:
    """Raw geometric signature for factual vs opinion classification.

    The primary metric is counterfactual_sensitivity - how much the representation
    changes when the fact is violated. This has effect size 0.94 for distinguishing
    facts from opinions in research experiments.

    Attributes:
        statement: The original statement being analyzed.
        counterfactual: The counterfactual version (fact-violating statement).
        counterfactual_sensitivity: PRIMARY METRIC - cosine distance between
            original and counterfactual representations. NaN if degenerate.
        effective_rank: Shannon effective rank of the representation.
        spectral_entropy: Entropy of normalized singular values.

    Note:
        This class returns raw metrics only. No composite scores.
        Values may be NaN for degenerate inputs.
        Caller interprets metrics based on their use case.
    """

    statement: str
    counterfactual: str
    counterfactual_sensitivity: float  # PRIMARY METRIC - may be NaN
    effective_rank: float
    spectral_entropy: float

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "statement": self.statement,
            "counterfactual": self.counterfactual,
            "counterfactual_sensitivity": self.counterfactual_sensitivity,
            "effective_rank": self.effective_rank,
            "spectral_entropy": self.spectral_entropy,
        }


@dataclass(frozen=True)
class InvariantSignature:
    """Raw multi-metric geometric signature for knowledge detection.

    This signature captures multiple geometric metrics that characterize
    representation stability. No composite scores - caller interprets.

    Attributes:
        prompt: The prompt being analyzed.
        kurtosis: Fourth standardized moment - distribution peakedness. NaN if degenerate.
        spectral_entropy: Entropy of normalized singular values.
        effective_rank: exp(spectral_entropy) - continuous dimensionality.
        layer_consistency: CKA across consecutive layers. NaN if insufficient layers.
        repetition_consistency: Cosine similarity across runs. NaN if insufficient reps.

    Note:
        Returns raw metrics only. No composite scores.
        Values may be NaN for degenerate inputs.
    """

    prompt: str
    kurtosis: float  # may be NaN
    spectral_entropy: float
    effective_rank: float
    layer_consistency: float  # may be NaN
    repetition_consistency: float  # may be NaN

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "kurtosis": self.kurtosis,
            "spectral_entropy": self.spectral_entropy,
            "effective_rank": self.effective_rank,
            "layer_consistency": self.layer_consistency,
            "repetition_consistency": self.repetition_consistency,
        }
