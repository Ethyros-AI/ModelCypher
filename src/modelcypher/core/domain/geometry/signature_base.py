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

"""Base class for vector signature types.

This module provides a mixin class for signature vectors that need
L2 normalization and cosine similarity operations. All signature
classes in the agents domain should inherit from this mixin.

Signature classes include:
- SemanticPrimeSignature
- EmotionConceptSignature
- ComputationalGateSignature
- etc.

By using this mixin, these classes share a single, tested implementation
of l2_normalized() and cosine_similarity() rather than duplicating
the same logic in each class.
"""

from __future__ import annotations

from typing import TypeVar

from modelcypher.core.domain.geometry.vector_math import VectorMath

# Type variable for self-referential return types
T = TypeVar("T", bound="SignatureMixin")


class SignatureMixin:
    """Mixin class providing L2 normalization and cosine similarity for signatures.

    Classes using this mixin must have a `values: list[float]` attribute.

    This is the canonical implementation - do not duplicate these methods
    in subclasses.

    Example usage:
        @dataclass
        class MySignature(SignatureMixin):
            dimension_labels: list[str]
            values: list[float]

            # The mixin provides:
            # - cosine_similarity(self, other)
            # - l2_normalized(self) -> MySignature
            # - l2_norm(self) -> float | None
    """

    values: list[float]  # Required attribute

    def l2_norm(self) -> float | None:
        """Compute L2 norm of this signature's values.

        Returns:
            L2 norm, or None if values are empty or all zeros.
        """
        return VectorMath.l2_norm(self.values)

    def l2_normalized(self: T) -> T:
        """Return a copy of this signature with L2-normalized values.

        Returns:
            New signature instance with normalized values.
        """
        normalized = VectorMath.l2_normalized(self.values)
        # Create new instance with same attributes but normalized values
        return self._with_values(normalized)

    def cosine_similarity(self, other: "SignatureMixin") -> float:
        """Compute cosine similarity with another signature.

        For same-dimension signatures, uses direct cosine similarity (fast path).
        For different dimensions, truncates to shared dimension and computes
        cosine - different dimensions are just different compression levels
        of the same geometry.

        Args:
            other: Another signature to compare with.

        Returns:
            Similarity in [-1, 1]. Never returns None - geometry always fits.
            -1 = opposite, 0 = orthogonal, 1 = identical direction.
        """
        # Same dimension: fast cosine similarity
        if self._has_same_dimensions(other):
            cos_sim = VectorMath.cosine_similarity(self.values, other.values)
            return cos_sim if cos_sim is not None else 0.0

        # Different dimensions: truncate to shared dimension
        # This preserves the geometry in the shared subspace
        min_dim = min(len(self.values), len(other.values))
        truncated_self = self.values[:min_dim]
        truncated_other = other.values[:min_dim]
        cos_sim = VectorMath.cosine_similarity(truncated_self, truncated_other)
        return cos_sim if cos_sim is not None else 0.0

    def _with_values(self: T, new_values: list[float]) -> T:
        """Create a copy of this signature with new values.

        Subclasses can override this for custom construction logic.
        Default implementation uses __class__ constructor with values replaced.

        Args:
            new_values: New values list.

        Returns:
            New signature instance.
        """
        # Try to use dataclass replace if available
        try:
            from dataclasses import replace

            return replace(self, values=new_values)
        except (TypeError, ImportError):
            # Fallback: try direct attribute copy
            import copy

            new_sig = copy.copy(self)
            new_sig.values = new_values
            return new_sig

    def _has_same_dimensions(self, other: "SignatureMixin") -> bool:
        """Check if another signature has the same dimension count.

        This is an observation about the compression level, not a compatibility
        judgment. Different dimensions are just different compressions of the
        same underlying geometry - both are valid and comparable.

        Args:
            other: Another signature.

        Returns:
            True if signatures have matching dimensions (enables fast cosine path).
        """
        if not hasattr(other, "values"):
            return False
        return len(self.values) == len(other.values)


class LabeledSignatureMixin(SignatureMixin):
    """Extended mixin for signatures with dimension labels.

    Classes using this mixin must have:
    - values: list[float]
    - A labels attribute (e.g., prime_ids, emotion_ids, gate_ids)

    The mixin will check label compatibility during comparison.
    """

    def _get_labels(self) -> list[str] | None:
        """Get the dimension labels for this signature.

        Subclasses should override to return their specific label attribute.
        Default looks for common attribute names.
        """
        for attr in ["prime_ids", "emotion_ids", "gate_ids", "labels", "dimension_ids"]:
            if hasattr(self, attr):
                return getattr(self, attr)
        return None

    def _has_same_dimensions(self, other: "LabeledSignatureMixin") -> bool:
        """Check if signatures have matching dimensions AND labels.

        Different labels still represent the same conceptual space - just
        different naming conventions. Both are valid for comparison.

        Args:
            other: Another labeled signature.

        Returns:
            True if dimensions and labels match (enables fast comparison path).
        """
        if not super()._has_same_dimensions(other):
            return False

        # Check labels if available - matching labels enable direct comparison
        self_labels = self._get_labels()
        other_labels = other._get_labels() if hasattr(other, "_get_labels") else None

        if self_labels is not None and other_labels is not None:
            return self_labels == other_labels

        return True
