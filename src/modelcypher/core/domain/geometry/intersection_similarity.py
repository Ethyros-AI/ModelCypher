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

"""Intersection similarity modes and computation for manifold stitching.

This module provides various similarity metrics for computing dimension
correlations between source and target models during manifold stitching.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.vector_math import geodesic_cosine_sparse

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ActivationFingerprint,
        DimensionCorrelation,
        IntersectionMap,
        LayerConfidence,
    )

__all__ = [
    "IntersectionSimilarityMode",
    "compute_jaccard_similarity",
    "compute_weighted_jaccard_similarity",
    "compute_cosine_similarity",
    "build_layer_correlations",
    "build_intersection_map",
    "intersection_map_from_dict",
]


class IntersectionSimilarityMode(str, Enum):
    """
    Similarity mode for building intersection maps.

    Controls how dimension correlations are computed between source and target.
    """

    JACCARD = "jaccard"  # Binary set overlap (sparse activation patterns)
    WEIGHTED_JACCARD = "weighted_jaccard"  # Magnitude-weighted Jaccard
    CKA = "cka"  # Centered Kernel Alignment
    GROMOV_WASSERSTEIN = "gromov_wasserstein"  # Optimal transport-based


def compute_jaccard_similarity(
    source_dims: set[int],
    target_dims: set[int],
) -> float:
    """
    Compute Jaccard similarity between two sets of activated dimensions.

    Jaccard = |intersection| / |union|
    """
    if not source_dims and not target_dims:
        return 0.0
    intersection = source_dims & target_dims
    union = source_dims | target_dims
    return len(intersection) / len(union) if union else 0.0


def compute_weighted_jaccard_similarity(
    source_activations: dict[int, float],
    target_activations: dict[int, float],
) -> float:
    """
    Compute magnitude-weighted Jaccard similarity.

    Weighted Jaccard = sum(min(a, b)) / sum(max(a, b))
    """
    all_dims = set(source_activations.keys()) | set(target_activations.keys())
    if not all_dims:
        return 0.0

    min_sum = 0.0
    max_sum = 0.0

    for dim in all_dims:
        a = abs(source_activations.get(dim, 0.0))
        b = abs(target_activations.get(dim, 0.0))
        min_sum += min(a, b)
        max_sum += max(a, b)

    return min_sum / max_sum if max_sum > 0 else 0.0


def compute_cosine_similarity(
    source_activations: dict[int, float],
    target_activations: dict[int, float],
) -> float:
    """
    Compute cosine similarity between sparse activation vectors.
    """
    backend = get_default_backend()
    try:
        return geodesic_cosine_sparse(source_activations, target_activations, backend)
    except ValueError:
        return 0.0


def build_layer_correlations(
    source_fingerprints: list["ActivationFingerprint"],
    target_fingerprints: list["ActivationFingerprint"],
    layer: int,
    mode: IntersectionSimilarityMode = IntersectionSimilarityMode.JACCARD,
) -> list["DimensionCorrelation"]:
    """
    Build dimension correlations for a layer using the specified similarity mode.

    Args:
        source_fingerprints: Fingerprints from source model
        target_fingerprints: Fingerprints from target model
        layer: Layer index to analyze
        mode: Similarity mode to use
    Returns:
        List of dimension correlations
    """
    from modelcypher.core.domain.geometry.manifold_stitcher import DimensionCorrelation

    # Collect all activated dimensions across fingerprints
    # Structure: fp.activated_dimensions is dict[int, list[ActivatedDimension]]
    #            where key is layer index, value is list of ActivatedDimension
    #            ActivatedDimension has .index (dimension within layer) and .activation
    source_dim_activations: dict[int, dict[str, float]] = {}  # dim_index -> {prime_id: activation}
    target_dim_activations: dict[int, dict[str, float]] = {}

    for fp in source_fingerprints:
        if layer not in fp.activated_dimensions:
            continue
        for dim in fp.activated_dimensions[layer]:
            if dim.index not in source_dim_activations:
                source_dim_activations[dim.index] = {}
            source_dim_activations[dim.index][fp.prime_id] = dim.activation

    for fp in target_fingerprints:
        if layer not in fp.activated_dimensions:
            continue
        for dim in fp.activated_dimensions[layer]:
            if dim.index not in target_dim_activations:
                target_dim_activations[dim.index] = {}
            target_dim_activations[dim.index][fp.prime_id] = dim.activation

    correlations = []

    # Compute correlations between all pairs of dimensions
    for s_dim, s_primes in source_dim_activations.items():
        best_correlation = 0.0
        best_target_dim = -1

        for t_dim, t_primes in target_dim_activations.items():
            # Compute similarity based on mode
            if mode == IntersectionSimilarityMode.JACCARD:
                similarity = compute_jaccard_similarity(set(s_primes.keys()), set(t_primes.keys()))
            elif mode == IntersectionSimilarityMode.WEIGHTED_JACCARD:
                # Build activation vectors using common primes
                common_primes = set(s_primes.keys()) & set(t_primes.keys())
                if not common_primes:
                    similarity = 0.0
                else:
                    s_vec = {i: s_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    t_vec = {i: t_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    similarity = compute_weighted_jaccard_similarity(s_vec, t_vec)
            elif mode == IntersectionSimilarityMode.CKA:
                common_primes = set(s_primes.keys()) & set(t_primes.keys())
                if not common_primes:
                    similarity = 0.0
                else:
                    s_vec = {i: s_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    t_vec = {i: t_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    cosine = compute_cosine_similarity(s_vec, t_vec)
                    similarity = cosine * cosine  # CKA ≈ cos^2 for centered vectors
            elif mode == IntersectionSimilarityMode.GROMOV_WASSERSTEIN:
                # GW requires full pairwise distance matrices from raw activations
                raise NotImplementedError(
                    "GROMOV_WASSERSTEIN mode requires raw activations, not semantic prime "
                    "signatures. Use gromov_wasserstein.py directly with activation matrices."
                )
            else:
                similarity = 0.0

            if similarity > best_correlation:
                best_correlation = similarity
                best_target_dim = t_dim

        if best_target_dim >= 0:
            correlations.append(
                DimensionCorrelation(
                    source_dim=s_dim,
                    target_dim=best_target_dim,
                    correlation=best_correlation,
                )
            )

    return correlations


def build_intersection_map(
    source_fingerprints: list["ActivationFingerprint"],
    target_fingerprints: list["ActivationFingerprint"],
    source_model: str,
    target_model: str,
    mode: IntersectionSimilarityMode = IntersectionSimilarityMode.JACCARD,
) -> "IntersectionMap":
    """
    Build an intersection map between source and target fingerprints.

    Routes to appropriate similarity computation based on mode.

    Args:
        source_fingerprints: Fingerprints from source model
        target_fingerprints: Fingerprints from target model
        source_model: Source model identifier
        target_model: Target model identifier
        mode: Similarity mode to use
    Returns:
        IntersectionMap with dimension correlations and layer confidences
    """
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        IntersectionMap,
        LayerConfidence,
    )

    # Collect all layers
    # Structure: fp.activated_dimensions is dict[int, list[ActivatedDimension]]
    #            where key is layer index
    all_layers: set[int] = set()
    for fp in source_fingerprints:
        all_layers.update(fp.activated_dimensions.keys())
    for fp in target_fingerprints:
        all_layers.update(fp.activated_dimensions.keys())

    dimension_correlations: dict[int, list[DimensionCorrelation]] = {}
    layer_confidences = []

    total_aligned = 0
    total_source_dims = 0
    total_target_dims = 0

    for layer in sorted(all_layers):
        correlations = build_layer_correlations(
            source_fingerprints=source_fingerprints,
            target_fingerprints=target_fingerprints,
            layer=layer,
            mode=mode,
        )

        dimension_correlations[layer] = correlations

        values = [c.correlation for c in correlations]
        mean_corr = sum(values) / len(values) if values else 0.0

        layer_confidences.append(
            LayerConfidence(
                layer=layer,
                confidence=mean_corr,
                correlation_count=len(values),
            )
        )

        total_aligned += len(correlations)

    # Estimate total dimensions (rough)
    # Structure: fp.activated_dimensions is dict[int, list[ActivatedDimension]]
    #            ActivatedDimension has .index (dimension within layer)
    source_dims_per_layer: set[tuple[int, int]] = set()
    target_dims_per_layer: set[tuple[int, int]] = set()
    for fp in source_fingerprints:
        for layer_idx, dims in fp.activated_dimensions.items():
            for dim in dims:
                source_dims_per_layer.add((layer_idx, dim.index))
    for fp in target_fingerprints:
        for layer_idx, dims in fp.activated_dimensions.items():
            for dim in dims:
                target_dims_per_layer.add((layer_idx, dim.index))

    total_source_dims = len(source_dims_per_layer)
    total_target_dims = len(target_dims_per_layer)

    # Mean layer CKA from sparse fingerprint correlations
    # NOTE: This is NOT a geometric invariant - just sparse matching quality
    mean_layer_cka = (
        sum(lc.confidence for lc in layer_confidences) / len(layer_confidences)
        if layer_confidences
        else 0.0
    )

    return IntersectionMap(
        source_model=source_model,
        target_model=target_model,
        dimension_correlations=dimension_correlations,
        mean_layer_cka=mean_layer_cka,
        aligned_dimension_count=total_aligned,
        total_source_dims=total_source_dims,
        total_target_dims=total_target_dims,
        layer_confidences=layer_confidences,
    )


def intersection_map_from_dict(payload: dict[str, Any]) -> "IntersectionMap":
    """Parse an IntersectionMap from a dictionary payload."""
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        DimensionCorrelation,
        IntersectionMap,
        LayerConfidence,
    )

    def _get(key: str, fallback: str | None = None) -> Any:
        if key in payload:
            return payload[key]
        if fallback and fallback in payload:
            return payload[fallback]
        return None

    raw_correlations = _get("dimensionCorrelations", "dimension_correlations") or {}
    dimension_correlations: dict[int, list[DimensionCorrelation]] = {}
    for layer_key, entries in raw_correlations.items():
        try:
            layer = int(layer_key)
        except (TypeError, ValueError):
            continue
        parsed: list[DimensionCorrelation] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            source_dim = entry.get("sourceDim", entry.get("source_dim"))
            target_dim = entry.get("targetDim", entry.get("target_dim"))
            correlation = entry.get("correlation")
            if source_dim is None or target_dim is None or correlation is None:
                continue
            parsed.append(
                DimensionCorrelation(
                    source_dim=int(source_dim),
                    target_dim=int(target_dim),
                    correlation=float(correlation),
                )
            )
        if parsed:
            dimension_correlations[layer] = parsed

    raw_layer_confidences = _get("layerConfidences", "layer_confidences") or []
    layer_confidences: list[LayerConfidence] = []
    for entry in raw_layer_confidences:
        if not isinstance(entry, dict):
            continue
        layer = entry.get("layer")
        confidence = entry.get("confidence")
        count = entry.get("correlationCount", entry.get("correlation_count"))
        if layer is None or confidence is None or count is None:
            continue
        layer_confidences.append(
            LayerConfidence(
                layer=int(layer),
                confidence=float(confidence),
                correlation_count=int(count),
            )
        )

    # Support both new and legacy field names for backwards compatibility
    mean_cka = _get("meanLayerCka", "mean_layer_cka")
    if mean_cka is None:
        # Fallback to legacy field names
        mean_cka = _get("rawFingerprintSimilarity", "raw_fingerprint_similarity")
    if mean_cka is None:
        mean_cka = _get("overallCorrelation", "overall_correlation")
    mean_layer_cka = float(mean_cka or 0.0)

    return IntersectionMap(
        source_model=str(_get("sourceModel", "source_model") or ""),
        target_model=str(_get("targetModel", "target_model") or ""),
        dimension_correlations=dimension_correlations,
        mean_layer_cka=mean_layer_cka,
        aligned_dimension_count=int(_get("alignedDimensionCount", "aligned_dimension_count") or 0),
        total_source_dims=int(_get("totalSourceDims", "total_source_dims") or 0),
        total_target_dims=int(_get("totalTargetDims", "total_target_dims") or 0),
        layer_confidences=layer_confidences,
    )
