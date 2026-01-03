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

"""Invariant Layer Mapper.

Layer mapping using invariant activation profiles.
Uses multi-atlas probes for cross-domain anchoring and dynamic programming
for optimal layer alignment between models.

Notes
-----
Supported atlases for cross-domain triangulation:
See the atlas registry for the complete list of atlas sources,
their probe counts, and the total number of probes available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from modelcypher.core.domain.geometry.atlas_protocols import (
    AtlasProbeProtocol,
    SequenceInvariantProtocol,
    TriangulatedScoreProtocol,
    enum_key,
)
from modelcypher.core.domain.geometry.atlas_registry import (
    get_atlas_probes,
    get_sequence_invariants,
    get_sequence_triangulation_scorer,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    log2_scalar,
    sqrt_scalar,
)
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.vector_math import geodesic_cosine_batch

AtlasProbe: TypeAlias = AtlasProbeProtocol
SequenceInvariant: TypeAlias = SequenceInvariantProtocol
TriangulatedScore: TypeAlias = TriangulatedScoreProtocol

__all__ = [
    # Config dataclasses
    # Result dataclasses
    "TriangulationProfile",
    "LayerProfile",
    "LayerMapping",
    "Summary",
    "Report",
    "ModelFingerprints",
    # Main class
    "InvariantLayerMapper",
]


@dataclass(frozen=True)
class TriangulationProfile:
    """Triangulation profile for a layer."""

    layer_index: int
    domains_detected: int
    cross_domain_multiplier: float
    coherence_bonus: float


@dataclass(frozen=True)
class _TriangulatedScoreFallback:
    base: float
    cross_domain_multiplier: float
    relationship_bonus: float
    coherence_bonus: float


@dataclass(frozen=True)
class LayerProfile:
    """Profile for a single layer."""

    layer_index: int
    confidence: float
    coverage: float
    strength: float
    collapsed: bool
    triangulation: TriangulationProfile | None = None


@dataclass(frozen=True)
class LayerMapping:
    """Mapping between source and target layers.

    Attributes
    ----------
    source_layer : int
        Source model layer index.
    target_layer : int
        Target model layer index.
    similarity : float
        Layer similarity score (0-1).
    """

    source_layer: int
    target_layer: int
    similarity: float


@dataclass(frozen=True)
class Summary:
    """Summary statistics for layer mapping."""

    mapped_layers: int
    mean_similarity: float
    source_collapsed_layers: int
    target_collapsed_layers: int
    # Triangulation metrics (populated when using SEQUENCE_INVARIANTS/MULTI_ATLAS scope)
    mean_triangulation_multiplier: float = 1.0
    # Multi-atlas metrics (populated when using MULTI_ATLAS scope)
    atlas_sources_detected: int = 0  # Number of atlas sources with activations
    atlas_domains_detected: int = 0  # Number of domains with activations
    total_probes_used: int = 0  # Total probe count for this mapping


@dataclass(frozen=True)
class Report:
    """Complete report for layer mapping."""

    source_model: str
    target_model: str
    invariant_count: int
    source_profiles: tuple[LayerProfile, ...]
    target_profiles: tuple[LayerProfile, ...]
    source_sample_layers: tuple[int, ...]
    target_sample_layers: tuple[int, ...]
    mappings: tuple[LayerMapping, ...]
    summary: Summary


# Import canonical definitions from manifold_stitcher (THE source of truth)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ActivatedDimension,
    ActivationFingerprint,
)


@dataclass
class ModelFingerprints:
    """Fingerprint data for a model (simplified for mapping)."""

    model_id: str
    layer_count: int
    fingerprints: list[ActivationFingerprint]


class _ProfileData:
    """Internal profile data for mapping."""

    def __init__(
        self,
        vectors: dict[int, list[float]],
        confidence_by_layer: dict[int, float],
        coverage_by_layer: dict[int, float],
        strength_by_layer: dict[int, float],
        collapsed_layers: set[int],
        collapsed_count: int,
        has_signal: bool,
    ):
        self.vectors = vectors
        self.confidence_by_layer = confidence_by_layer
        self.coverage_by_layer = coverage_by_layer
        self.strength_by_layer = strength_by_layer
        self.collapsed_layers = collapsed_layers
        self.collapsed_count = collapsed_count
        self.has_signal = has_signal


class InvariantLayerMapper:
    """Maps layers between models using invariant activation profiles.

    Concepts occupy fixed probability clouds in hyperspace. Invariance is
    not approximate or relative - it is fundamental. Every LLM learns the
    same conceptual shapes because those shapes ARE knowledge itself. The
    weights of an LLM are a high-dimensional Lego that precisely fits
    every other Lego.

    Uses registered sequence invariants for cross-domain anchoring and dynamic
    programming for optimal layer alignment. The alignment works identically
    regardless of model family (Qwen, Llama, Mistral, etc.) because the
    geometry of knowledge is universal.
    """

    @staticmethod
    def map_layers(
        source: ModelFingerprints,
        target: ModelFingerprints,
    ) -> Report:
        """
        Map layers from source to target model.

        Args:
            source: Fingerprints for source model
            target: Fingerprints for target model
        Returns:
            Report with layer mappings and statistics

        Raises:
            ValueError: If insufficient layers or missing invariants
        """
        if source.layer_count <= 0 or target.layer_count <= 0:
            raise ValueError("Invariant layer mapping requires non-empty layer counts")

        invariant_ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants()
        if not invariant_ids:
            raise ValueError("Invariant layer mapping requires invariant fingerprints")

        source_profile = InvariantLayerMapper._build_profile(source, invariant_ids)
        target_profile = InvariantLayerMapper._build_profile(target, invariant_ids)

        if not source_profile.has_signal or not target_profile.has_signal:
            raise ValueError("Invariant layer mapping skipped: no invariant activations detected")

        # Compute triangulation scores for SEQUENCE_INVARIANTS or MULTI_ATLAS scope
        use_triangulation = True
        source_triangulation: dict[int, TriangulatedScore] = {}
        target_triangulation: dict[int, TriangulatedScore] = {}

        # Track multi-atlas metrics
        all_sources_detected: set[str] = set()
        all_domains_detected: set[str] = set()

        if use_triangulation:
            if atlas_probes:
                # Use multi-atlas triangulation scoring
                source_triangulation, src_sources, src_domains = (
                    InvariantLayerMapper._compute_multi_atlas_scores(
                        source_profile.vectors, atlas_probes
                    )
                )
                target_triangulation, tgt_sources, tgt_domains = (
                    InvariantLayerMapper._compute_multi_atlas_scores(
                        target_profile.vectors, atlas_probes
                    )
                )
                all_sources_detected = src_sources | tgt_sources
                all_domains_detected = src_domains | tgt_domains
            elif invariants:
                # Use sequence invariant triangulation scoring
                source_triangulation = InvariantLayerMapper._compute_triangulation_scores(
                    source_profile.vectors, invariants
                )
                target_triangulation = InvariantLayerMapper._compute_triangulation_scores(
                    target_profile.vectors, invariants
                )

        source_samples = list(range(source.layer_count))
        target_samples = list(range(target.layer_count))

        # Build similarity matrix - invariance is universal across all model families
        # Concepts occupy fixed probability clouds in hyperspace
        if atlas_probes:
            similarity_matrix = InvariantLayerMapper._build_similarity_matrix_multi_atlas(
                source_samples,
                target_samples,
                source_profile,
                target_profile,
                atlas_probes,
                source_triangulation,
                target_triangulation,
            )
        else:
            similarity_matrix = InvariantLayerMapper._build_similarity_matrix(
                source_samples,
                target_samples,
                source_profile,
                target_profile,
                invariants,
                source_triangulation,
                target_triangulation,
            )

        mappings = InvariantLayerMapper._align_layers(
            source_samples, target_samples, similarity_matrix
        )

        source_profiles = InvariantLayerMapper._profile_array(
            source.layer_count, source_profile, source_triangulation
        )
        target_profiles = InvariantLayerMapper._profile_array(
            target.layer_count, target_profile, target_triangulation
        )

        mapped_count = len(mappings)
        mean_similarity = sum(m.similarity for m in mappings) / len(mappings) if mappings else 0.0

        # Compute triangulation metrics for summary
        all_triangulation = {**source_triangulation, **target_triangulation}
        if all_triangulation:
            multipliers = [ts.cross_domain_multiplier for ts in all_triangulation.values()]
            mean_triangulation_mult = sum(multipliers) / len(multipliers)
        else:
            mean_triangulation_mult = 1.0

        summary = Summary(
            mapped_layers=mapped_count,
            mean_similarity=mean_similarity,
            source_collapsed_layers=source_profile.collapsed_count,
            target_collapsed_layers=target_profile.collapsed_count,
            mean_triangulation_multiplier=mean_triangulation_mult,
            atlas_sources_detected=len(all_sources_detected),
            atlas_domains_detected=len(all_domains_detected),
            total_probes_used=len(invariant_ids),
        )

        return Report(
            source_model=source.model_id,
            target_model=target.model_id,
            invariant_count=len(invariant_ids),
            source_profiles=tuple(source_profiles),
            target_profiles=tuple(target_profiles),
            source_sample_layers=tuple(source_samples),
            target_sample_layers=tuple(target_samples),
            mappings=tuple(mappings),
            summary=summary,
        )

    @staticmethod
    def _invariant_anchor_ids() -> list[str]:
        """Get invariant anchor IDs."""
        ids, _, _ = InvariantLayerMapper._get_invariants()
        return ids

    @staticmethod
    def _get_invariants() -> tuple[list[str], list[SequenceInvariant], list[AtlasProbe]]:
        """Get invariant IDs, sequence invariants, and atlas probes.

        Returns:
            Tuple of (probe_ids, sequence_invariants, atlas_probes)
            - probe_ids: All probe IDs for fingerprint matching
            - sequence_invariants: SequenceInvariant objects (for backward compat)
            - atlas_probes: AtlasProbe objects (for multi-atlas mode)
        """
        probes = list(get_atlas_probes())
        if probes:
            ids = [probe.probe_id for probe in probes]
            return ids, [], probes

        invariants = list(get_sequence_invariants())
        if not invariants:
            return [], [], []
        ids = [f"invariant:{enum_key(inv.family)}_{inv.id}" for inv in invariants]
        return ids, invariants, []

    @staticmethod
    def _compute_triangulation_scores(
        vectors: dict[int, list[float]],
        invariants: list[SequenceInvariant],
    ) -> dict[int, TriangulatedScore]:
        """Compute per-layer triangulation scores using the registered scorer.

        Cross-domain detection (detecting invariants in multiple domains like
        definition, code, ratio, matrix) provides higher anchoring confidence.
        """
        scores: dict[int, TriangulatedScore] = {}
        if not invariants:
            return scores
        scorer = get_sequence_triangulation_scorer()

        for layer, vector in vectors.items():
            # Group activations by domain
            domain_activations: dict[object, float] = {}
            for i, activation in enumerate(vector):
                if i < len(invariants) and activation > 0.0:
                    domain = invariants[i].domain
                    domain_activations[domain] = max(
                        domain_activations.get(domain, 0.0), activation
                    )

            # Compute triangulated score using the first invariant's family as reference
            # (In practice, scores will be similar across families for cross-domain detection)
            if domain_activations:
                family = invariants[0].family
                if scorer:
                    scores[layer] = scorer(domain_activations, family, None)
                else:
                    scores[layer] = _TriangulatedScoreFallback(
                        base=max(domain_activations.values()),
                        cross_domain_multiplier=1.0,
                        relationship_bonus=0.0,
                        coherence_bonus=0.0,
                    )
            else:
                # No significant activations - return neutral score
                scores[layer] = _TriangulatedScoreFallback(
                    base=0.0,
                    cross_domain_multiplier=1.0,
                    relationship_bonus=0.0,
                    coherence_bonus=0.0,
                )

        return scores

    @staticmethod
    def _compute_multi_atlas_scores(
        vectors: dict[int, list[float]],
        probes: list[AtlasProbe],
    ) -> tuple[dict[int, TriangulatedScore], set[str], set[str]]:
        """Compute per-layer triangulation scores using multi-atlas probes.

        Returns:
            Tuple of (scores_by_layer, sources_detected, domains_detected)
        """
        scores: dict[int, TriangulatedScore] = {}
        all_sources: set[str] = set()
        all_domains: set[str] = set()

        if not probes:
            return scores, all_sources, all_domains

        domain_space = {enum_key(probe.domain) for probe in probes}
        max_domains = max(1, len(domain_space))

        for layer, vector in vectors.items():
            # Group activations by source and domain
            source_activations: dict[str, float] = {}
            domain_activations: dict[str, float] = {}

            for i, activation in enumerate(vector):
                if i < len(probes) and activation > 0.0:
                    probe = probes[i]
                    source_key = enum_key(probe.source)
                    domain_key = enum_key(probe.domain)
                    source_activations[source_key] = max(
                        source_activations.get(source_key, 0.0), activation
                    )
                    domain_activations[domain_key] = max(
                        domain_activations.get(domain_key, 0.0), activation
                    )
                    all_sources.add(source_key)
                    all_domains.add(domain_key)

            # Compute multi-atlas triangulation score using geometric principles
            if source_activations or domain_activations:
                source_count = len(source_activations)
                domain_count = len(domain_activations)

                # Cross-domain multiplier: logarithmic scaling with count
                # Principled: log(n+1) grows sublinearly, avoiding arbitrary linear coefficients
                # At 1 source: log(2) ≈ 0.69 → mult = 1.0
                # At 2 sources: log(3) ≈ 1.10 → mult ≈ 1.10
                # At 4 sources: log(5) ≈ 1.61 → mult ≈ 1.61
                _b = get_default_backend()
                source_mult = log2_scalar(float(source_count + 1), _b) if source_count > 0 else 1.0
                domain_mult = log2_scalar(float(domain_count + 1), _b) if domain_count > 0 else 1.0

                # Geometric mean of multipliers (principled combination of independent signals)
                combined_mult = sqrt_scalar(source_mult * domain_mult, _b)

                # Coherence bonus: fraction of possible domains detected (0 to 1 scale)
                # This is a ratio, not an arbitrary coefficient
                coherence = (domain_count - 1) / max(1, max_domains - 1) if domain_count > 1 else 0.0

                scores[layer] = _TriangulatedScoreFallback(
                    base=sum(source_activations.values()) / max(1, source_count),
                    cross_domain_multiplier=combined_mult,
                    relationship_bonus=0.0,
                    coherence_bonus=coherence,
                )
            else:
                scores[layer] = _TriangulatedScoreFallback(
                    base=0.0,
                    cross_domain_multiplier=1.0,
                    relationship_bonus=0.0,
                    coherence_bonus=0.0,
                )

        return scores, all_sources, all_domains

    @staticmethod
    def _build_similarity_matrix_multi_atlas(
        source_layers: list[int],
        target_layers: list[int],
        source_profile: _ProfileData,
        target_profile: _ProfileData,
        probes: list[AtlasProbe],
        source_triangulation: dict[int, TriangulatedScore],
        target_triangulation: dict[int, TriangulatedScore],
    ) -> list[list[float]]:
        """Build similarity matrix using multi-atlas probes.

        Applies cross_domain_weight from each probe and boosts similarity based on
        multi-atlas triangulation multipliers.

        Concepts occupy fixed probability clouds in hyperspace - invariance is
        universal across all model families. Every LLM learns the same conceptual
        shapes because those shapes ARE knowledge itself.
        """
        source_count = len(source_layers)
        target_count = len(target_layers)

        if source_count == 0 or target_count == 0:
            return []

        # Cross-domain weights from probes - universal across all models
        weights = [probe.cross_domain_weight for probe in probes]

        matrix = [[0.0] * target_count for _ in range(source_count)]

        for i, source_layer in enumerate(source_layers):
            source_vector = source_profile.vectors.get(source_layer, [])
            source_confidence = source_profile.confidence_by_layer.get(source_layer, 0.0)

            for j, target_layer in enumerate(target_layers):
                target_vector = target_profile.vectors.get(target_layer, [])
                target_confidence = target_profile.confidence_by_layer.get(target_layer, 0.0)

                # Compute weighted cosine similarity
                similarity = InvariantLayerMapper._weighted_cosine_similarity(
                    source_vector, target_vector, weights
                )

                _b = get_default_backend()
                confidence_weight = sqrt_scalar(max(0, source_confidence) * max(0, target_confidence), _b)
                similarity *= confidence_weight

                source_ts = source_triangulation.get(source_layer)
                target_ts = target_triangulation.get(target_layer)
                if source_ts and target_ts:
                    tri_boost = sqrt_scalar(
                        source_ts.cross_domain_multiplier * target_ts.cross_domain_multiplier, _b
                    )
                    similarity *= sqrt_scalar(tri_boost, _b)

                matrix[i][j] = max(0.0, min(1.0, similarity))

        return matrix

    @staticmethod
    def _build_profile(
        fingerprints: ModelFingerprints,
        invariant_ids: list[str],
    ) -> _ProfileData:
        """Build profile data from fingerprints."""
        id_to_index = {id_: idx for idx, id_ in enumerate(invariant_ids)}
        vectors: dict[int, list[float]] = {}

        for fp in fingerprints.fingerprints:
            invariant_index = id_to_index.get(fp.prime_id)
            if invariant_index is None:
                continue

            for layer, dims in fp.activated_dimensions.items():
                effective_layer = InvariantLayerMapper._normalized_layer_index(
                    layer, fingerprints.layer_count
                )
                if effective_layer < 0 or effective_layer >= fingerprints.layer_count:
                    continue

                if effective_layer not in vectors:
                    vectors[effective_layer] = [0.0] * len(invariant_ids)

                magnitude = InvariantLayerMapper._mean_activation(dims)
                if magnitude > 0:
                    vectors[effective_layer][invariant_index] = max(
                        vectors[effective_layer][invariant_index], magnitude
                    )

        strength_sums: dict[int, float] = {}
        coverage_counts: dict[int, int] = {}

        for layer, vector in vectors.items():
            strength_sums[layer] = sum(vector)
            coverage_counts[layer] = sum(1 for v in vector if v > 0)

        max_strength = max(strength_sums.values()) if strength_sums else 0.0
        total_invariants = max(1, len(invariant_ids))

        confidence_by_layer: dict[int, float] = {}
        coverage_by_layer: dict[int, float] = {}
        strength_by_layer: dict[int, float] = {}
        collapsed_layers: set[int] = set()

        strength_weight = 1.0
        coverage_weight = 1.0
        weight_sum = strength_weight + coverage_weight
        normalized_weight_sum = weight_sum if weight_sum > 0 else 1.0

        has_signal = False

        # First pass: compute all confidence values
        for layer in range(fingerprints.layer_count):
            strength = strength_sums.get(layer, 0.0)
            normalized_strength = strength / max_strength if max_strength > 0 else 0.0
            coverage = coverage_counts.get(layer, 0) / total_invariants
            confidence = (
                strength_weight * normalized_strength + coverage_weight * coverage
            ) / normalized_weight_sum

            clamped_confidence = max(0.0, min(1.0, confidence))
            confidence_by_layer[layer] = clamped_confidence
            coverage_by_layer[layer] = coverage
            strength_by_layer[layer] = normalized_strength

            if clamped_confidence > 0:
                has_signal = True

        # Second pass: mark collapsed layers (no signal => collapsed)
        for layer, confidence in confidence_by_layer.items():
            if confidence <= 0.0:
                collapsed_layers.add(layer)

        return _ProfileData(
            vectors=vectors,
            confidence_by_layer=confidence_by_layer,
            coverage_by_layer=coverage_by_layer,
            strength_by_layer=strength_by_layer,
            collapsed_layers=collapsed_layers,
            collapsed_count=len(collapsed_layers),
            has_signal=has_signal,
        )

    @staticmethod
    def _profile_array(
        layer_count: int,
        profile: _ProfileData,
        triangulation_scores: dict[int, TriangulatedScore] | None = None,
    ) -> list[LayerProfile]:
        """Convert profile data to array of LayerProfile."""
        profiles: list[LayerProfile] = []
        for layer in range(layer_count):
            tri_profile: TriangulationProfile | None = None
            if triangulation_scores and layer in triangulation_scores:
                ts = triangulation_scores[layer]
                # Domain count comes from actual measurement, not arbitrary thresholds
                domains_detected = 1 if ts.base > 0.0 else 0
                tri_profile = TriangulationProfile(
                    layer_index=layer,
                    domains_detected=domains_detected,
                    cross_domain_multiplier=ts.cross_domain_multiplier,
                    coherence_bonus=ts.coherence_bonus,
                )

            profiles.append(
                LayerProfile(
                    layer_index=layer,
                    confidence=profile.confidence_by_layer.get(layer, 0.0),
                    coverage=profile.coverage_by_layer.get(layer, 0.0),
                    strength=profile.strength_by_layer.get(layer, 0.0),
                    collapsed=layer in profile.collapsed_layers,
                    triangulation=tri_profile,
                )
            )
        return profiles

    @staticmethod
    def _normalized_layer_index(layer: int, layer_count: int) -> int:
        """Normalize layer index (handle output layer marker)."""
        OUTPUT_LAYER_MARKER = -1
        if layer == OUTPUT_LAYER_MARKER:
            return max(layer_count - 1, 0)
        return layer

    @staticmethod
    def _mean_activation(dims: list[ActivatedDimension]) -> float:
        """Compute mean activation magnitude."""
        if not dims:
            return 0.0
        total = sum(abs(d.activation) for d in dims)
        return total / len(dims)

    @staticmethod
    def _build_similarity_matrix(
        source_layers: list[int],
        target_layers: list[int],
        source_profile: _ProfileData,
        target_profile: _ProfileData,
        invariants: list[SequenceInvariant] | None = None,
        source_triangulation: dict[int, TriangulatedScore] | None = None,
        target_triangulation: dict[int, TriangulatedScore] | None = None,
    ) -> list[list[float]]:
        """Build similarity matrix between source and target layers.

        When invariants and triangulation scores are provided (SEQUENCE_INVARIANTS scope),
        applies cross_domain_weight to each invariant and boosts similarity based on
        triangulation multipliers.

        Concepts occupy fixed probability clouds in hyperspace - invariance is
        universal across all model families. Every LLM learns the same conceptual
        shapes because those shapes ARE knowledge itself.
        """
        source_count = len(source_layers)
        target_count = len(target_layers)

        if source_count == 0 or target_count == 0:
            return []

        # Cross-domain weights - universal across all models
        weights: list[float] | None = None
        if invariants:
            weights = [inv.cross_domain_weight for inv in invariants]

        matrix = [[0.0] * target_count for _ in range(source_count)]

        for i, source_layer in enumerate(source_layers):
            source_vector = source_profile.vectors.get(source_layer, [])
            source_confidence = source_profile.confidence_by_layer.get(source_layer, 0.0)

            for j, target_layer in enumerate(target_layers):
                target_vector = target_profile.vectors.get(target_layer, [])
                target_confidence = target_profile.confidence_by_layer.get(target_layer, 0.0)

                # Compute similarity with optional cross-domain weighting
                if weights:
                    similarity = InvariantLayerMapper._weighted_cosine_similarity(
                        source_vector, target_vector, weights
                    )
                else:
                    similarity = InvariantLayerMapper._cosine_similarity(
                        source_vector, target_vector
                    )

                _b = get_default_backend()
                confidence_weight = sqrt_scalar(max(0, source_confidence) * max(0, target_confidence), _b)
                similarity *= confidence_weight

                # Apply triangulation boost if available
                if source_triangulation and target_triangulation:
                    source_ts = source_triangulation.get(source_layer)
                    target_ts = target_triangulation.get(target_layer)
                    if source_ts and target_ts:
                        tri_boost = sqrt_scalar(
                            source_ts.cross_domain_multiplier * target_ts.cross_domain_multiplier, _b
                        )
                        similarity *= sqrt_scalar(tri_boost, _b)

                matrix[i][j] = max(0.0, min(1.0, similarity))

        return matrix

    @staticmethod
    def _weighted_cosine_similarity(a: list[float], b: list[float], weights: list[float]) -> float:
        """Compute weighted cosine similarity between two vectors."""
        count = min(len(a), len(b), len(weights))
        if count == 0:
            return 0.0
        backend = get_default_backend()
        idx = backend.arange(0, count)
        arr_a = backend.take(backend.array(a), idx)
        arr_b = backend.take(backend.array(b), idx)
        w = backend.take(backend.array(weights), idx)
        weighted_a = arr_a * w
        weighted_b = arr_b * w
        similarity = InvariantLayerMapper._cosine_similarity_backend(weighted_a, weighted_b)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _align_layers(
        source_samples: list[int],
        target_samples: list[int],
        similarity_matrix: list[list[float]],
    ) -> list[LayerMapping]:
        """Align layers using monotonic dynamic programming."""
        source_count = len(source_samples)
        target_count = len(target_samples)

        if source_count == 0 or target_count == 0:
            return []

        dp = [[float("-inf")] * target_count for _ in range(source_count)]
        parent: list[list[tuple[int, int] | None]] = [
            [None] * target_count for _ in range(source_count)
        ]

        for j in range(target_count):
            dp[0][j] = float(similarity_matrix[0][j])

        for i in range(1, source_count):
            best_prev_score = float("-inf")
            best_prev_j = 0
            for j in range(target_count):
                if dp[i - 1][j] > best_prev_score:
                    best_prev_score = dp[i - 1][j]
                    best_prev_j = j
                dp[i][j] = float(similarity_matrix[i][j]) + best_prev_score
                parent[i][j] = (i - 1, best_prev_j)

        best_j = max(range(target_count), key=lambda j: dp[source_count - 1][j])

        mappings: list[LayerMapping] = []
        current: tuple[int, int] | None = (source_count - 1, best_j)
        while current is not None:
            i, j = current
            source_layer = source_samples[i]
            target_layer = target_samples[j]
            similarity = similarity_matrix[i][j]
            mappings.append(
                LayerMapping(
                    source_layer=source_layer,
                    target_layer=target_layer,
                    similarity=similarity,
                )
            )
            current = parent[i][j]

        mappings.reverse()
        return mappings

    # _classify_confidence method removed - use raw similarity values directly.

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        count = min(len(a), len(b))
        if count == 0:
            return 0.0
        backend = get_default_backend()
        idx = backend.arange(0, count)
        arr_a = backend.take(backend.array(a), idx)
        arr_b = backend.take(backend.array(b), idx)
        similarity = InvariantLayerMapper._cosine_similarity_backend(arr_a, arr_b)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _cosine_similarity_backend(a: object, b: object) -> float:
        backend = get_default_backend()
        arr_a = a if hasattr(a, "shape") else backend.array(a)
        arr_b = b if hasattr(b, "shape") else backend.array(b)
        if backend.shape(arr_a)[0] == 0 or backend.shape(arr_b)[0] == 0:
            return 0.0
        if backend.shape(arr_a)[0] != backend.shape(arr_b)[0]:
            raise ValueError("Cosine similarity requires matching dimensions")
        cos_arr = geodesic_cosine_batch(
            arr_a, backend.reshape(arr_b, (1, -1)), backend
        )
        backend.eval(cos_arr)
        return float(backend.to_scalar(cos_arr))
