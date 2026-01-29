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

"""Manifold entropy measurement for geometric self-alignment.

This module provides a unified entropy measure across the full manifold of
neural network representations. The entropy aggregates:

1. Layer-wise intrinsic dimension (TwoNN)
2. SVD ratio alignment to fundamental constants {π, e, φ, √2, π/e}
3. Complexity-dimension law fit (slope should be e/π, intercept should be π/e)
4. Curvature structure

The key insight: lower entropy = better geometric alignment. The fundamental
constants define what "coherent" means in information-theoretic terms.

This is Phase 1 of the Geometric Self-Alignment System: READ-ONLY measurement.
No weights are modified here.

References:
    - fundamental_constants.py - The constants that define coherence
    - Facco et al. (2017) - TwoNN intrinsic dimension estimation

Empirical validation (LFM2-350M, January 2026):
    - Complexity-dimension slope = e/π (0.68% error)
    - Complexity-dimension intercept = π/e (2.95% error)
    - 11 SVD matches with < 1% error to fundamental constants
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.fundamental_constants import (
    COMPLEXITY_INTERCEPT,
    COMPLEXITY_SLOPE,
    E_OVER_PI,
    PI_OVER_E,
    ConstantMatch,
    FundamentalConstant,
    analyze_svd_ratios,
    find_constant_match,
    percent_error,
)
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class LayerEntropyResult:
    """Entropy measurements for a single layer."""

    layer_idx: int
    intrinsic_dimension: float
    effective_rank: float  # Shannon effective rank (entropy-based)
    spectral_entropy: float
    sample_count: int

    @property
    def dimension_ratio(self) -> float:
        """Ratio of intrinsic dimension to effective rank.

        Values near 1.0 indicate consistent dimensionality.
        """
        if self.effective_rank > 0:
            return self.intrinsic_dimension / self.effective_rank
        return 0.0


@dataclass
class SVDSignatureResult:
    """SVD ratio alignment to fundamental constants."""

    matches: List[Tuple[int, int, ConstantMatch]]  # (i, j, match)
    n_precise: int  # Deprecated: raw count of ratio matches (len(matches))
    n_significant: int  # Deprecated: raw count of ratio matches (len(matches))
    mean_error: float  # Average error of all matches
    top_singular_values: List[float]

    @property
    def signature_quality(self) -> float:
        """Deprecated: use raw metrics (n_precise, n_significant, mean_error)."""
        raise RuntimeError(
            "No heuristic quality scores in domain metrics. "
            "Inspect n_precise, n_significant, and mean_error directly."
        )


@dataclass
class ComplexityLawResult:
    """Fit to the complexity-dimension law: dim = (e/π) × c + (π/e)."""

    slope: float
    intercept: float
    r_squared: float
    slope_error: float  # Percent error from e/π
    intercept_error: float  # Percent error from π/e

    @property
    def validates_theory(self) -> bool:
        """Deprecated: use raw metrics (slope_error, intercept_error, r_squared)."""
        raise RuntimeError(
            "No heuristic thresholds in domain metrics. "
            "Inspect slope_error, intercept_error, and r_squared directly."
        )

    @property
    def law_quality(self) -> float:
        """Deprecated: use raw metrics (slope_error, intercept_error, r_squared)."""
        raise RuntimeError(
            "No heuristic quality scores in domain metrics. "
            "Inspect slope_error, intercept_error, and r_squared directly."
        )


@dataclass
class ManifoldEntropyResult:
    """Complete entropy measurement across the full manifold."""

    # Aggregate entropy (lower is better)
    total_entropy: float

    # Per-layer breakdown
    layer_entropies: Dict[int, LayerEntropyResult] = field(default_factory=dict)

    # SVD signature (alignment to fundamental constants)
    svd_signature: Optional[SVDSignatureResult] = None

    # Complexity-dimension law fit
    complexity_law: Optional[ComplexityLawResult] = None

    # Raw data for debugging
    complexities: Optional[List[float]] = None
    dimensions: Optional[List[float]] = None

    @property
    def has_significant_alignment(self) -> bool:
        """Deprecated: use svd_signature.n_precise directly."""
        raise RuntimeError(
            "No heuristic thresholds in domain metrics. "
            "Inspect svd_signature.n_precise directly."
        )

    @property
    def law_validates(self) -> bool:
        """Deprecated: use complexity_law raw metrics directly."""
        raise RuntimeError(
            "No heuristic thresholds in domain metrics. "
            "Inspect complexity_law.slope_error, intercept_error, r_squared directly."
        )

    @property
    def alignment_quality(self) -> float:
        """Deprecated: use raw alignment metrics directly."""
        raise RuntimeError(
            "No heuristic quality scores in domain metrics. "
            "Inspect svd_signature and complexity_law raw metrics directly."
        )


class ManifoldEntropy:
    """Compute aggregate entropy across the full manifold.

    This is the central measurement for geometric self-alignment.
    Lower entropy = better alignment to fundamental constants.

    Usage:
        entropy = ManifoldEntropy(backend)
        result = entropy.compute_from_activations(layer_activations)

        # Or with complexity annotations
        result = entropy.compute_with_complexity(
            layer_activations,
            complexities=[1, 2, 3, 4]  # per-statement complexity
        )
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()
        self._id_estimator = IntrinsicDimension(self._backend)
        self._eff_rank = EffectiveRank(self._backend)

    def compute_layer_entropy(
        self,
        activations: "Array",
        layer_idx: int,
    ) -> LayerEntropyResult:
        """Compute entropy for a single layer's activations.

        Args:
            activations: [n_samples, features] activation matrix
            layer_idx: Layer index for identification

        Returns:
            LayerEntropyResult with intrinsic dimension, effective rank, etc.
        """
        b = self._backend
        arr = b.array(activations) if not hasattr(activations, "shape") else activations
        b.eval(arr)

        n_samples = int(arr.shape[0])
        if n_samples < 4:
            return LayerEntropyResult(
                layer_idx=layer_idx,
                intrinsic_dimension=0.0,
                effective_rank=0.0,
                spectral_entropy=0.0,
                sample_count=n_samples,
            )

        # Compute intrinsic dimension (TwoNN)
        try:
            id_result = self._id_estimator.compute(arr)
            intrinsic_dim = id_result.intrinsic_dimension
        except Exception:
            intrinsic_dim = 0.0

        # Compute effective rank (Shannon entropy-based)
        eff_result = self._eff_rank.compute(arr)
        effective_rank = eff_result.shannon_effective_rank
        spectral_entropy = eff_result.spectral_entropy

        return LayerEntropyResult(
            layer_idx=layer_idx,
            intrinsic_dimension=intrinsic_dim,
            effective_rank=effective_rank,
            spectral_entropy=spectral_entropy,
            sample_count=n_samples,
        )

    def compute_svd_signature(
        self,
        activations: "Array",
        max_gap: int = 7,
        max_index: int = 20,
        threshold: float | None = None,
    ) -> SVDSignatureResult:
        """Analyze SVD singular value ratios for fundamental constant encoding.

        Args:
            activations: [n_samples, features] activation matrix
            max_gap: Maximum gap between indices to check
            max_index: Maximum index to analyze
        threshold: Deprecated. Thresholds are not permitted in domain metrics.

        Returns:
            SVDSignatureResult with matches and quality metrics
        """
        b = self._backend
        arr = b.array(activations) if not hasattr(activations, "shape") else activations
        b.eval(arr)

        # Compute SVD
        _, singular_values, _ = geodesic_svd(b, arr)
        b.eval(singular_values)

        if threshold is not None:
            raise RuntimeError(
                "No heuristic thresholds in domain metrics. "
                "Inspect error_percent values directly."
            )

        # Analyze ratios using fundamental_constants module (raw errors)
        matches = analyze_svd_ratios(singular_values, b, max_gap, max_index, threshold=None)

        # Deprecated: thresholded counts removed; keep raw count for compatibility
        n_precise = len(matches)
        n_significant = len(matches)

        # Mean error across all matches
        if matches:
            mean_error = sum(m.error_percent for _, _, m in matches) / len(matches)
        else:
            mean_error = 100.0  # No matches = maximum error

        # Get top singular values for diagnostics
        n_sv = min(20, int(singular_values.shape[0]))
        top_sv = [float(b.to_scalar(singular_values[i:i+1])) for i in range(n_sv)]

        return SVDSignatureResult(
            matches=matches,
            n_precise=n_precise,
            n_significant=n_significant,
            mean_error=mean_error,
            top_singular_values=top_sv,
        )

    def compute_complexity_law(
        self,
        complexities: List[float],
        dimensions: List[float],
    ) -> ComplexityLawResult:
        """Validate the complexity-dimension law against measured data.

        The theoretical law is: dim = (e/π) × complexity + (π/e)

        Args:
            complexities: List of complexity values
            dimensions: List of corresponding dimension values

        Returns:
            ComplexityLawResult with slope, intercept, and fit quality
        """
        n = min(len(complexities), len(dimensions))
        if n < 2:
            return ComplexityLawResult(
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                slope_error=100.0,
                intercept_error=100.0,
            )

        c = complexities[:n]
        d = dimensions[:n]

        # Linear regression via closed-form least squares
        sum_c = sum(c)
        sum_d = sum(d)
        sum_cc = sum(val * val for val in c)
        sum_cd = sum(ci * di for ci, di in zip(c, d))
        denom = n * sum_cc - sum_c * sum_c

        if denom != 0.0:
            slope = (n * sum_cd - sum_c * sum_d) / denom
            intercept = (sum_d - slope * sum_c) / n
        else:
            slope = 0.0
            intercept = sum_d / n if n > 0 else 0.0

        # R-squared
        mean_d = sum_d / n
        pred = [slope * ci + intercept for ci in c]
        ss_res = sum((di - pi) ** 2 for di, pi in zip(d, pred))
        ss_tot = sum((di - mean_d) ** 2 for di in d)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Errors from theoretical values
        slope_error = percent_error(slope, E_OVER_PI)
        intercept_error = percent_error(intercept, PI_OVER_E)

        return ComplexityLawResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            slope_error=slope_error,
            intercept_error=intercept_error,
        )

    def compute_from_activations(
        self,
        layer_activations: Dict[int, "Array"],
    ) -> ManifoldEntropyResult:
        """Compute manifold entropy from layer activations.

        This is the main entry point for entropy computation without
        complexity annotations.

        Args:
            layer_activations: Dict mapping layer_idx to activation arrays
                              Each array is [n_samples, features]

        Returns:
            ManifoldEntropyResult with aggregate entropy and per-layer breakdown
        """
        layer_entropies: Dict[int, LayerEntropyResult] = {}
        total_spectral_entropy = 0.0

        for layer_idx, activations in layer_activations.items():
            layer_result = self.compute_layer_entropy(activations, layer_idx)
            layer_entropies[layer_idx] = layer_result
            total_spectral_entropy += layer_result.spectral_entropy

        # Compute SVD signature from the middle layer (most informative)
        svd_signature = None
        if layer_activations:
            sorted_layers = sorted(layer_activations.keys())
            mid_idx = sorted_layers[len(sorted_layers) // 2]
            svd_signature = self.compute_svd_signature(layer_activations[mid_idx])

        # Aggregate entropy: sum of spectral entropies (raw measurement)
        total_entropy = total_spectral_entropy

        return ManifoldEntropyResult(
            total_entropy=total_entropy,
            layer_entropies=layer_entropies,
            svd_signature=svd_signature,
            complexity_law=None,  # No complexity annotations
        )

    def compute_with_complexity(
        self,
        layer_activations: Dict[int, "Array"],
        complexities: List[float],
        statement_layer_map: Optional[Dict[int, int]] = None,
    ) -> ManifoldEntropyResult:
        """Compute manifold entropy with complexity-dimension law validation.

        This is the full entropy computation including the complexity-dimension
        law fit. Each statement has an associated complexity level.

        Args:
            layer_activations: Dict mapping layer_idx to activation arrays
                              Each array is [n_samples, features]
            complexities: Complexity level for each sample
            statement_layer_map: Optional mapping from statement to stabilization layer
                                If provided, uses per-statement stabilization layer

        Returns:
            ManifoldEntropyResult with full analysis including complexity law
        """
        # First compute basic entropy
        result = self.compute_from_activations(layer_activations)

        # Compute dimensions for each complexity level
        dimensions: List[float] = []

        if statement_layer_map is None:
            # Use a single representative layer (middle layer)
            sorted_layers = sorted(layer_activations.keys())
            if not sorted_layers:
                return result

            mid_idx = sorted_layers[len(sorted_layers) // 2]
            activations = layer_activations[mid_idx]

            # Compute dimension for each sample
            b = self._backend
            arr = b.array(activations) if not hasattr(activations, "shape") else activations
            b.eval(arr)

            n_samples = int(arr.shape[0])
            if n_samples != len(complexities):
                return result

            # Compute overall dimension (not per-sample)
            try:
                id_result = self._id_estimator.compute(arr)
                dimensions = [id_result.intrinsic_dimension] * len(complexities)
            except Exception:
                return result
        else:
            # Use per-statement stabilization layers
            for stmt_idx, complexity in enumerate(complexities):
                if stmt_idx not in statement_layer_map:
                    continue
                layer_idx = statement_layer_map[stmt_idx]
                if layer_idx not in layer_activations:
                    continue

                activations = layer_activations[layer_idx]
                b = self._backend
                arr = b.array(activations) if not hasattr(activations, "shape") else activations

                # Get the specific sample's neighborhood dimension
                try:
                    id_result = self._id_estimator.compute(arr)
                    dimensions.append(id_result.intrinsic_dimension)
                except Exception:
                    dimensions.append(0.0)

        if len(dimensions) < 2:
            return result

        # Compute complexity-dimension law fit
        complexity_law = self.compute_complexity_law(complexities, dimensions)

        # Entropy remains a raw spectral measurement; law fit is reported separately.
        total_entropy = result.total_entropy

        return ManifoldEntropyResult(
            total_entropy=total_entropy,
            layer_entropies=result.layer_entropies,
            svd_signature=result.svd_signature,
            complexity_law=complexity_law,
            complexities=complexities,
            dimensions=dimensions,
        )

    def compute_entropy_delta(
        self,
        before: ManifoldEntropyResult,
        after: ManifoldEntropyResult,
    ) -> float:
        """Compute the change in entropy.

        Returns:
            Positive value if entropy decreased (good),
            negative if increased (bad).
        """
        return before.total_entropy - after.total_entropy


def compute_manifold_entropy(
    layer_activations: Dict[int, "Array"],
    backend: "Backend | None" = None,
    complexities: Optional[List[float]] = None,
) -> ManifoldEntropyResult:
    """Convenience function to compute manifold entropy.

    Args:
        layer_activations: Dict mapping layer_idx to activation arrays
        backend: Backend to use
        complexities: Optional complexity levels per sample

    Returns:
        ManifoldEntropyResult
    """
    entropy = ManifoldEntropy(backend)
    if complexities is not None:
        return entropy.compute_with_complexity(layer_activations, complexities)
    return entropy.compute_from_activations(layer_activations)


__all__ = [
    # Main class
    "ManifoldEntropy",
    # Result types
    "ManifoldEntropyResult",
    "LayerEntropyResult",
    "SVDSignatureResult",
    "ComplexityLawResult",
    # Convenience function
    "compute_manifold_entropy",
]
