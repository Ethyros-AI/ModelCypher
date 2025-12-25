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

"""Feature vector extracted from adapter weights for lightweight risk scoring.

P0 stores summary statistics; future phases can append additional dimensions
without breaking persistence by using versioned keys.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FEATURE_VERSION = "P0-lite"


@dataclass(frozen=True)
class DeltaFeatureSet:
    """Feature vector extracted from adapter weights for lightweight risk scoring.

    Contains statistical features computed from LoRA adapter weights that can
    be used to detect unusual or potentially malicious weight patterns.
    """

    l2_norms: tuple[float, ...] = ()
    """L2 norms of adapter delta tensors (per target module)."""

    sparsity: tuple[float, ...] = ()
    """Sparsity ratios (fraction of near-zero elements) per module."""

    cosine_to_aligned: tuple[float, ...] = ()
    """Cosine similarity to aligned-direction vectors per module (if available)."""

    suspect_layer_indices: tuple[int, ...] = ()
    """Ranked suspect layers (indices into target modules)."""

    feature_version: str = DEFAULT_FEATURE_VERSION
    """Version tag for the feature extractor."""

    @property
    def layer_count(self) -> int:
        """Number of layers analyzed."""
        return len(self.l2_norms)

    @property
    def has_suspect_layers(self) -> bool:
        """Whether any layers were flagged as suspect."""
        return len(self.suspect_layer_indices) > 0

    @property
    def suspect_layer_fraction(self) -> float:
        """Fraction of layers flagged as suspect."""
        if self.layer_count == 0:
            return 0.0
        return len(self.suspect_layer_indices) / self.layer_count

    @property
    def mean_l2_norm(self) -> float:
        """Mean L2 norm across all layers."""
        if not self.l2_norms:
            return 0.0
        return sum(self.l2_norms) / len(self.l2_norms)

    @property
    def max_l2_norm(self) -> float:
        """Maximum L2 norm across all layers."""
        if not self.l2_norms:
            return 0.0
        return max(self.l2_norms)

    @property
    def mean_sparsity(self) -> float:
        """Mean sparsity across all layers."""
        if not self.sparsity:
            return 0.0
        return sum(self.sparsity) / len(self.sparsity)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "l2_norms": list(self.l2_norms),
            "sparsity": list(self.sparsity),
            "cosine_to_aligned": list(self.cosine_to_aligned),
            "suspect_layer_indices": list(self.suspect_layer_indices),
            "feature_version": self.feature_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DeltaFeatureSet:
        """Create from dictionary."""
        return cls(
            l2_norms=tuple(data.get("l2_norms", [])),
            sparsity=tuple(data.get("sparsity", [])),
            cosine_to_aligned=tuple(data.get("cosine_to_aligned", [])),
            suspect_layer_indices=tuple(data.get("suspect_layer_indices", [])),
            feature_version=data.get("feature_version", DEFAULT_FEATURE_VERSION),
        )
