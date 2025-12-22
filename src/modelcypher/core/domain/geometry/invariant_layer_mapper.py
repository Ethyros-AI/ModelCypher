"""
Invariant Layer Mapper.

Layer mapping strategy using invariant activation profiles and collapse-aware confidence.
Uses SequenceInvariantAtlas for cross-domain anchoring and dynamic programming
for optimal layer alignment between models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
    DEFAULT_FAMILIES,
)


class InvariantScope(str, Enum):
    """Scope of invariants to use for mapping."""
    INVARIANTS = "invariants"
    LOGIC_ONLY = "logicOnly"


class ConfidenceLevel(str, Enum):
    """Confidence level for layer mapping."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Config:
    """Configuration for invariant layer mapping."""
    invariant_scope: InvariantScope = InvariantScope.INVARIANTS
    family_allowlist: Optional[frozenset[SequenceFamily]] = None
    sample_layer_count: Optional[int] = 12
    min_similarity: float = 0.2
    max_skip: int = 0
    skip_penalty: float = 0.15
    collapse_threshold: float = 0.35
    collapse_mismatch_penalty: float = 0.3
    strength_weight: float = 0.6
    coverage_weight: float = 0.4
    high_confidence_threshold: float = 0.65
    medium_confidence_threshold: float = 0.45


@dataclass(frozen=True)
class LayerProfile:
    """Profile for a single layer."""
    layer_index: int
    confidence: float
    coverage: float
    strength: float
    collapsed: bool


@dataclass(frozen=True)
class LayerMapping:
    """Mapping between source and target layers."""
    source_layer: int
    target_layer: int
    similarity: float
    confidence: ConfidenceLevel
    is_skipped: bool


@dataclass(frozen=True)
class Summary:
    """Summary statistics for layer mapping."""
    mapped_layers: int
    skipped_layers: int
    mean_similarity: float
    alignment_quality: float
    source_collapsed_layers: int
    target_collapsed_layers: int


@dataclass(frozen=True)
class Report:
    """Complete report for layer mapping."""
    source_model: str
    target_model: str
    config: Config
    invariant_count: int
    source_profiles: tuple[LayerProfile, ...]
    target_profiles: tuple[LayerProfile, ...]
    source_sample_layers: tuple[int, ...]
    target_sample_layers: tuple[int, ...]
    mappings: tuple[LayerMapping, ...]
    summary: Summary


@dataclass
class ModelFingerprints:
    """Fingerprint data for a model (simplified for mapping)."""
    model_id: str
    layer_count: int
    fingerprints: list[ActivationFingerprint]


@dataclass
class ActivationFingerprint:
    """Activation fingerprint for a prime/invariant."""
    prime_id: str
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass
class ActivatedDimension:
    """Single activated dimension."""
    dimension: int
    activation: float


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
    """
    Maps layers between models using invariant activation profiles.

    Uses SequenceInvariantInventory for cross-domain anchoring and
    dynamic programming for optimal layer alignment.
    """

    @staticmethod
    def map_layers(
        source: ModelFingerprints,
        target: ModelFingerprints,
        config: Optional[Config] = None,
    ) -> Report:
        """
        Map layers from source to target model.

        Args:
            source: Fingerprints for source model
            target: Fingerprints for target model
            config: Optional mapping configuration

        Returns:
            Report with layer mappings and statistics

        Raises:
            ValueError: If insufficient layers or missing invariants
        """
        if config is None:
            config = Config()

        if source.layer_count <= 0 or target.layer_count <= 0:
            raise ValueError("Invariant layer mapping requires non-empty layer counts")

        invariant_ids = InvariantLayerMapper._invariant_anchor_ids(config)
        if not invariant_ids:
            raise ValueError("Invariant layer mapping requires invariant fingerprints")

        source_profile = InvariantLayerMapper._build_profile(source, invariant_ids, config)
        target_profile = InvariantLayerMapper._build_profile(target, invariant_ids, config)

        if not source_profile.has_signal or not target_profile.has_signal:
            raise ValueError("Invariant layer mapping skipped: no invariant activations detected")

        source_samples = InvariantLayerMapper._sample_layers(source.layer_count, config.sample_layer_count)
        target_samples = InvariantLayerMapper._sample_layers(target.layer_count, config.sample_layer_count)

        similarity_matrix = InvariantLayerMapper._build_similarity_matrix(
            source_samples, target_samples, source_profile, target_profile, config
        )

        mappings = InvariantLayerMapper._align_layers(source_samples, target_samples, similarity_matrix, config)

        source_profiles = InvariantLayerMapper._profile_array(source.layer_count, source_profile)
        target_profiles = InvariantLayerMapper._profile_array(target.layer_count, target_profile)

        mapped_count = len(mappings)
        skipped_count = sum(1 for m in mappings if m.is_skipped)
        mean_similarity = sum(m.similarity for m in mappings) / len(mappings) if mappings else 0.0
        valid_mappings = [m for m in mappings if not m.is_skipped]
        alignment_quality = sum(m.similarity for m in valid_mappings) / len(valid_mappings) if valid_mappings else 0.0

        summary = Summary(
            mapped_layers=mapped_count,
            skipped_layers=skipped_count,
            mean_similarity=mean_similarity,
            alignment_quality=alignment_quality,
            source_collapsed_layers=source_profile.collapsed_count,
            target_collapsed_layers=target_profile.collapsed_count,
        )

        return Report(
            source_model=source.model_id,
            target_model=target.model_id,
            config=config,
            invariant_count=len(invariant_ids),
            source_profiles=tuple(source_profiles),
            target_profiles=tuple(target_profiles),
            source_sample_layers=tuple(source_samples),
            target_sample_layers=tuple(target_samples),
            mappings=tuple(mappings),
            summary=summary,
        )

    @staticmethod
    def _invariant_anchor_ids(config: Config) -> list[str]:
        """Get invariant anchor IDs based on config."""
        if config.invariant_scope == InvariantScope.LOGIC_ONLY:
            base_families = frozenset([SequenceFamily.LOGIC])
        else:
            base_families = DEFAULT_FAMILIES

        families = config.family_allowlist.intersection(base_families) if config.family_allowlist else base_families
        invariants = SequenceInvariantInventory.probes_for_families(set(families))
        return [f"invariant:{inv.family.value}_{inv.id}" for inv in invariants]

    @staticmethod
    def _build_profile(
        fingerprints: ModelFingerprints,
        invariant_ids: list[str],
        config: Config,
    ) -> _ProfileData:
        """Build profile data from fingerprints."""
        id_to_index = {id_: idx for idx, id_ in enumerate(invariant_ids)}
        vectors: dict[int, list[float]] = {}

        for fp in fingerprints.fingerprints:
            invariant_index = id_to_index.get(fp.prime_id)
            if invariant_index is None:
                continue

            for layer, dims in fp.activated_dimensions.items():
                effective_layer = InvariantLayerMapper._normalized_layer_index(layer, fingerprints.layer_count)
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

        weight_sum = max(0, config.strength_weight) + max(0, config.coverage_weight)
        normalized_weight_sum = weight_sum if weight_sum > 0 else 1.0

        has_signal = False

        for layer in range(fingerprints.layer_count):
            strength = strength_sums.get(layer, 0.0)
            normalized_strength = strength / max_strength if max_strength > 0 else 0.0
            coverage = coverage_counts.get(layer, 0) / total_invariants
            confidence = (
                max(0, config.strength_weight) * normalized_strength
                + max(0, config.coverage_weight) * coverage
            ) / normalized_weight_sum

            clamped_confidence = max(0.0, min(1.0, confidence))
            confidence_by_layer[layer] = clamped_confidence
            coverage_by_layer[layer] = coverage
            strength_by_layer[layer] = normalized_strength

            if clamped_confidence > 0:
                has_signal = True

            if clamped_confidence < max(0, config.collapse_threshold):
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
    def _profile_array(layer_count: int, profile: _ProfileData) -> list[LayerProfile]:
        """Convert profile data to array of LayerProfile."""
        return [
            LayerProfile(
                layer_index=layer,
                confidence=profile.confidence_by_layer.get(layer, 0.0),
                coverage=profile.coverage_by_layer.get(layer, 0.0),
                strength=profile.strength_by_layer.get(layer, 0.0),
                collapsed=layer in profile.collapsed_layers,
            )
            for layer in range(layer_count)
        ]

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
    def _sample_layers(layer_count: int, sample_count: Optional[int]) -> list[int]:
        """Sample layers evenly across the model."""
        if layer_count <= 0:
            return []
        if sample_count is None or sample_count <= 0 or sample_count >= layer_count:
            return list(range(layer_count))

        last_index = layer_count - 1
        indices: list[int] = []

        for i in range(sample_count):
            position = i / max(1, sample_count - 1)
            index = int(round(position * last_index))
            indices.append(index)

        return sorted(set(indices))

    @staticmethod
    def _build_similarity_matrix(
        source_layers: list[int],
        target_layers: list[int],
        source_profile: _ProfileData,
        target_profile: _ProfileData,
        config: Config,
    ) -> list[list[float]]:
        """Build similarity matrix between source and target layers."""
        source_count = len(source_layers)
        target_count = len(target_layers)

        if source_count == 0 or target_count == 0:
            return []

        matrix = [[0.0] * target_count for _ in range(source_count)]

        for i, source_layer in enumerate(source_layers):
            source_vector = source_profile.vectors.get(source_layer, [])
            source_confidence = source_profile.confidence_by_layer.get(source_layer, 0.0)
            source_collapsed = source_layer in source_profile.collapsed_layers

            for j, target_layer in enumerate(target_layers):
                target_vector = target_profile.vectors.get(target_layer, [])
                target_confidence = target_profile.confidence_by_layer.get(target_layer, 0.0)
                target_collapsed = target_layer in target_profile.collapsed_layers

                similarity = InvariantLayerMapper._cosine_similarity(source_vector, target_vector)
                confidence_weight = math.sqrt(max(0, source_confidence) * max(0, target_confidence))
                similarity *= confidence_weight

                if source_collapsed != target_collapsed:
                    penalty = max(0.0, min(1.0, config.collapse_mismatch_penalty))
                    similarity *= (1 - penalty)

                matrix[i][j] = max(0.0, min(1.0, similarity))

        return matrix

    @staticmethod
    def _align_layers(
        source_samples: list[int],
        target_samples: list[int],
        similarity_matrix: list[list[float]],
        config: Config,
    ) -> list[LayerMapping]:
        """Align layers using dynamic programming."""
        source_count = len(source_samples)
        target_count = len(target_samples)

        if source_count == 0 or target_count == 0:
            return []

        NEG_INF = float('-inf')

        # DP table
        dp = [[NEG_INF] * (target_count + 1) for _ in range(source_count + 1)]
        parent: list[list[Optional[tuple[int, int, str]]]] = [
            [None] * (target_count + 1) for _ in range(source_count + 1)
        ]

        dp[0][0] = 0.0

        for i in range(source_count + 1):
            for j in range(target_count + 1):
                current = dp[i][j]
                if current <= NEG_INF / 2:
                    continue

                # Match
                if i < source_count and j < target_count:
                    score = current + similarity_matrix[i][j]
                    if score > dp[i + 1][j + 1]:
                        dp[i + 1][j + 1] = score
                        parent[i + 1][j + 1] = (i, j, "match")

                # Skip source
                if i < source_count:
                    if InvariantLayerMapper._allow_skip(parent, i, j, "skip_source", config.max_skip):
                        score = current - config.skip_penalty
                        if score > dp[i + 1][j]:
                            dp[i + 1][j] = score
                            parent[i + 1][j] = (i, j, "skip_source")

                # Skip target
                if j < target_count:
                    if InvariantLayerMapper._allow_skip(parent, i, j, "skip_target", config.max_skip):
                        score = current - config.skip_penalty
                        if score > dp[i][j + 1]:
                            dp[i][j + 1] = score
                            parent[i][j + 1] = (i, j, "skip_target")

        # Backtrack to get mappings
        mappings: list[LayerMapping] = []
        i, j = source_count, target_count

        while i > 0 or j > 0:
            step = parent[i][j]
            if step is None:
                break

            source_idx, target_idx, move = step

            if move == "match":
                source_layer = source_samples[source_idx]
                target_layer = target_samples[target_idx]
                similarity = similarity_matrix[source_idx][target_idx]
                confidence = InvariantLayerMapper._classify_confidence(similarity, config)
                is_skipped = similarity < config.min_similarity

                mappings.append(LayerMapping(
                    source_layer=source_layer,
                    target_layer=target_layer,
                    similarity=similarity,
                    confidence=confidence,
                    is_skipped=is_skipped,
                ))

            i, j = source_idx, target_idx

        mappings.reverse()
        return mappings

    @staticmethod
    def _allow_skip(
        parent: list[list[Optional[tuple[int, int, str]]]],
        source_index: int,
        target_index: int,
        move: str,
        max_skip: int,
    ) -> bool:
        """Check if a skip is allowed based on max consecutive skips."""
        if max_skip <= 0:
            return True

        skips = 0
        current_source = source_index
        current_target = target_index

        while current_source >= 0 and current_target >= 0:
            step = parent[current_source][current_target]
            if step is None:
                break

            _, _, step_move = step
            if step_move != move:
                break

            skips += 1
            if skips >= max_skip:
                return False

            current_source, current_target, _ = step

        return True

    @staticmethod
    def _classify_confidence(similarity: float, config: Config) -> ConfidenceLevel:
        """Classify confidence level based on similarity."""
        if similarity >= config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        if similarity >= config.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        if similarity >= config.min_similarity:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        count = min(len(a), len(b))
        if count == 0:
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for i in range(count):
            va = a[i]
            vb = b[i]
            dot += va * vb
            norm_a += va * va
            norm_b += vb * vb

        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        return max(0.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))
