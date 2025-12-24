"""
Invariant Layer Mapper.

Layer mapping strategy using invariant activation profiles and collapse-aware confidence.
Uses multi-atlas probes for cross-domain anchoring and dynamic programming
for optimal layer alignment between models.

Supported atlases:
- Sequence Invariants: 68 probes (mathematical/logical)
- Semantic Primes: 65 probes (linguistic/mental)
- Computational Gates: 72 probes (computational/structural)
- Emotion Concepts: 32 probes (affective/relational)

Total: 237 probes for cross-domain triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional
import math

# Type hints only - prevents circular import with agents package
if TYPE_CHECKING:
    from modelcypher.core.domain.agents.sequence_invariant_atlas import (
        SequenceFamily,
        SequenceInvariant,
        TriangulatedScore,
        ExpressionDomain,
    )
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasProbe,
        AtlasSource,
        AtlasDomain,
    )


def _get_sequence_invariants():
    """Lazy import for sequence invariant types."""
    from modelcypher.core.domain.agents.sequence_invariant_atlas import (
        SequenceFamily,
        SequenceInvariant,
        SequenceInvariantInventory,
        TriangulationScorer,
        TriangulatedScore,
        ExpressionDomain,
        DEFAULT_FAMILIES,
    )
    return (
        SequenceFamily, SequenceInvariant, SequenceInvariantInventory,
        TriangulationScorer, TriangulatedScore, ExpressionDomain, DEFAULT_FAMILIES
    )


def _get_unified_atlas():
    """Lazy import for unified atlas types."""
    from modelcypher.core.domain.agents.unified_atlas import (
        AtlasProbe,
        AtlasSource,
        AtlasDomain,
        UnifiedAtlasInventory,
        MultiAtlasTriangulationScorer,
        MultiAtlasTriangulationScore,
        DEFAULT_ATLAS_SOURCES,
    )
    return (
        AtlasProbe, AtlasSource, AtlasDomain, UnifiedAtlasInventory,
        MultiAtlasTriangulationScorer, MultiAtlasTriangulationScore, DEFAULT_ATLAS_SOURCES
    )


class LayerMappingStrategy(str, Enum):
    """Strategy for mapping layers between models.

    CRM: CRM-based CKA alignment using centered kernel alignment
         to match layers by representation similarity.

    INVARIANT_COLLAPSE: Invariant-only mapping with collapse-awareness.
         Uses semantic/sequence invariant probes and penalizes
         collapsed layer mismatches.
    """
    CRM = "crm"
    INVARIANT_COLLAPSE = "invariant_collapse"


class InvariantScope(str, Enum):
    """Scope of invariants to use for mapping."""
    INVARIANTS = "invariants"
    LOGIC_ONLY = "logicOnly"
    SEQUENCE_INVARIANTS = "sequenceInvariants"  # Full 68-probe system with triangulation
    MULTI_ATLAS = "multiAtlas"  # Full 237-probe system across all atlases


class ConfidenceLevel(str, Enum):
    """Confidence level for layer mapping."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class LayerMatchCategory(str, Enum):
    """Categories of layer matching criteria."""
    ACTIVATION_PATTERN = "activation_pattern"  # Raw activation similarity
    INVARIANT_COVERAGE = "invariant_coverage"  # Semantic prime coverage
    COLLAPSE_STATE = "collapse_state"  # Layer collapse detection
    TRIANGULATION = "triangulation"  # Cross-domain triangulation quality
    CKA_ALIGNMENT = "cka_alignment"  # Centered kernel alignment score


@dataclass(frozen=True)
class LayerMatchCategoryWeights:
    """Weights for different layer matching criteria.

    Used by INVARIANT_COLLAPSE strategy to combine multiple
    similarity signals into a final layer match score.
    """
    activation_pattern: float = 0.3
    invariant_coverage: float = 0.25
    collapse_state: float = 0.15
    triangulation: float = 0.2
    cka_alignment: float = 0.1

    def normalized(self) -> "LayerMatchCategoryWeights":
        """Return weights normalized to sum to 1.0."""
        total = (
            self.activation_pattern + self.invariant_coverage +
            self.collapse_state + self.triangulation + self.cka_alignment
        )
        if total <= 0:
            return LayerMatchCategoryWeights()
        return LayerMatchCategoryWeights(
            activation_pattern=self.activation_pattern / total,
            invariant_coverage=self.invariant_coverage / total,
            collapse_state=self.collapse_state / total,
            triangulation=self.triangulation / total,
            cka_alignment=self.cka_alignment / total,
        )

    def as_dict(self) -> dict[LayerMatchCategory, float]:
        """Return weights as dictionary keyed by category."""
        return {
            LayerMatchCategory.ACTIVATION_PATTERN: self.activation_pattern,
            LayerMatchCategory.INVARIANT_COVERAGE: self.invariant_coverage,
            LayerMatchCategory.COLLAPSE_STATE: self.collapse_state,
            LayerMatchCategory.TRIANGULATION: self.triangulation,
            LayerMatchCategory.CKA_ALIGNMENT: self.cka_alignment,
        }


@dataclass(frozen=True)
class CRMMappingConfig:
    """Configuration for CRM-based layer mapping.

    CRM (Centered Representational Model) uses CKA to find optimal
    layer alignments between models with potentially different depths.
    """
    cka_kernel: str = "linear"  # "linear", "rbf", "polynomial"
    rbf_sigma: float = 1.0  # Sigma for RBF kernel
    normalize_activations: bool = True
    min_cka_score: float = 0.3  # Minimum CKA to consider alignment
    use_debiased_cka: bool = True  # Use debiased CKA estimator


@dataclass(frozen=True)
class InvariantCollapseMappingConfig:
    """Configuration for invariant-collapse layer mapping.

    Uses semantic invariants with collapse detection to map layers
    while penalizing mismatched collapse states.
    """
    collapse_threshold: float = 0.3  # Below this confidence = collapsed
    min_invariant_coverage: float = 0.5  # Min fraction of invariants active
    collapse_mismatch_penalty: float = 0.35  # Penalty for state mismatch
    allow_many_to_one: bool = False  # Allow multiple source to one target
    category_weights: Optional[LayerMatchCategoryWeights] = None
    use_triangulation_boost: bool = True  # Boost scores with triangulation


@dataclass(frozen=True)
class Config:
    """Configuration for invariant layer mapping."""
    # Strategy selection
    strategy: LayerMappingStrategy = LayerMappingStrategy.INVARIANT_COLLAPSE
    crm_config: Optional[CRMMappingConfig] = None
    invariant_collapse_config: Optional[InvariantCollapseMappingConfig] = None

    # Legacy / common options
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

    # Layer match category weights (used with INVARIANT_COLLAPSE strategy)
    layer_match_category_weights: Optional[LayerMatchCategoryWeights] = None

    # Triangulation scoring options (used with SEQUENCE_INVARIANTS scope)
    use_cross_domain_weighting: bool = True
    triangulation_threshold: float = 0.3
    multi_domain_bonus: bool = True

    # Multi-atlas configuration (used with MULTI_ATLAS scope)
    atlas_sources: Optional[frozenset[AtlasSource]] = None  # None = all sources
    atlas_domains: Optional[frozenset[AtlasDomain]] = None  # None = all domains

    # CKA integration (used with CRM strategy or as auxiliary signal)
    use_cka_auxiliary: bool = False  # Use CKA as auxiliary signal in invariant mode
    cka_auxiliary_weight: float = 0.2  # Weight for CKA auxiliary signal


@dataclass(frozen=True)
class TriangulationProfile:
    """Triangulation profile for a layer."""
    layer_index: int
    domains_detected: int
    cross_domain_multiplier: float
    coherence_bonus: float


@dataclass(frozen=True)
class LayerProfile:
    """Profile for a single layer."""
    layer_index: int
    confidence: float
    coverage: float
    strength: float
    collapsed: bool
    triangulation: Optional[TriangulationProfile] = None


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
    # Triangulation metrics (populated when using SEQUENCE_INVARIANTS/MULTI_ATLAS scope)
    mean_triangulation_multiplier: float = 1.0
    triangulation_quality: str = "none"  # "high", "medium", "low", "none"
    # Multi-atlas metrics (populated when using MULTI_ATLAS scope)
    atlas_sources_detected: int = 0  # Number of atlas sources with activations
    atlas_domains_detected: int = 0  # Number of domains with activations
    total_probes_used: int = 0       # Total probe count for this mapping


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

        invariant_ids, invariants, atlas_probes = InvariantLayerMapper._get_invariants(config)
        if not invariant_ids:
            raise ValueError("Invariant layer mapping requires invariant fingerprints")

        source_profile = InvariantLayerMapper._build_profile(source, invariant_ids, config)
        target_profile = InvariantLayerMapper._build_profile(target, invariant_ids, config)

        if not source_profile.has_signal or not target_profile.has_signal:
            raise ValueError("Invariant layer mapping skipped: no invariant activations detected")

        # Compute triangulation scores for SEQUENCE_INVARIANTS or MULTI_ATLAS scope
        use_triangulation = (
            config.invariant_scope in (InvariantScope.SEQUENCE_INVARIANTS, InvariantScope.MULTI_ATLAS)
            and config.multi_domain_bonus
        )
        source_triangulation: dict[int, TriangulatedScore] = {}
        target_triangulation: dict[int, TriangulatedScore] = {}

        # Track multi-atlas metrics
        all_sources_detected: set[AtlasSource] = set()
        all_domains_detected: set[AtlasDomain] = set()

        if use_triangulation:
            if config.invariant_scope == InvariantScope.MULTI_ATLAS and atlas_probes:
                # Use multi-atlas triangulation scoring
                source_triangulation, src_sources, src_domains = InvariantLayerMapper._compute_multi_atlas_scores(
                    source_profile.vectors, atlas_probes, config
                )
                target_triangulation, tgt_sources, tgt_domains = InvariantLayerMapper._compute_multi_atlas_scores(
                    target_profile.vectors, atlas_probes, config
                )
                all_sources_detected = src_sources | tgt_sources
                all_domains_detected = src_domains | tgt_domains
            elif invariants:
                # Use sequence invariant triangulation scoring
                source_triangulation = InvariantLayerMapper._compute_triangulation_scores(
                    source_profile.vectors, invariants, config
                )
                target_triangulation = InvariantLayerMapper._compute_triangulation_scores(
                    target_profile.vectors, invariants, config
                )

        source_samples = InvariantLayerMapper._sample_layers(source.layer_count, config.sample_layer_count)
        target_samples = InvariantLayerMapper._sample_layers(target.layer_count, config.sample_layer_count)

        # Build similarity matrix with appropriate weights
        if config.invariant_scope == InvariantScope.MULTI_ATLAS and atlas_probes:
            similarity_matrix = InvariantLayerMapper._build_similarity_matrix_multi_atlas(
                source_samples, target_samples, source_profile, target_profile, config,
                atlas_probes, source_triangulation, target_triangulation,
            )
        else:
            similarity_matrix = InvariantLayerMapper._build_similarity_matrix(
                source_samples, target_samples, source_profile, target_profile, config,
                invariants, source_triangulation, target_triangulation,
            )

        mappings = InvariantLayerMapper._align_layers(source_samples, target_samples, similarity_matrix, config)

        source_profiles = InvariantLayerMapper._profile_array(
            source.layer_count, source_profile, source_triangulation
        )
        target_profiles = InvariantLayerMapper._profile_array(
            target.layer_count, target_profile, target_triangulation
        )

        mapped_count = len(mappings)
        skipped_count = sum(1 for m in mappings if m.is_skipped)
        mean_similarity = sum(m.similarity for m in mappings) / len(mappings) if mappings else 0.0
        valid_mappings = [m for m in mappings if not m.is_skipped]
        alignment_quality = sum(m.similarity for m in valid_mappings) / len(valid_mappings) if valid_mappings else 0.0

        # Compute triangulation metrics for summary
        all_triangulation = {**source_triangulation, **target_triangulation}
        if all_triangulation:
            multipliers = [ts.cross_domain_multiplier for ts in all_triangulation.values()]
            mean_triangulation_mult = sum(multipliers) / len(multipliers)
            if mean_triangulation_mult >= 1.5:
                tri_quality = "high"
            elif mean_triangulation_mult >= 1.2:
                tri_quality = "medium"
            elif mean_triangulation_mult > 1.0:
                tri_quality = "low"
            else:
                tri_quality = "none"
        else:
            mean_triangulation_mult = 1.0
            tri_quality = "none"

        summary = Summary(
            mapped_layers=mapped_count,
            skipped_layers=skipped_count,
            mean_similarity=mean_similarity,
            alignment_quality=alignment_quality,
            source_collapsed_layers=source_profile.collapsed_count,
            target_collapsed_layers=target_profile.collapsed_count,
            mean_triangulation_multiplier=mean_triangulation_mult,
            triangulation_quality=tri_quality,
            atlas_sources_detected=len(all_sources_detected),
            atlas_domains_detected=len(all_domains_detected),
            total_probes_used=len(invariant_ids),
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
        ids, _, _ = InvariantLayerMapper._get_invariants(config)
        return ids

    @staticmethod
    def _get_invariants(
        config: Config,
    ) -> tuple[list[str], list[SequenceInvariant], list[AtlasProbe]]:
        """Get invariant IDs, sequence invariants, and atlas probes for config.

        Returns:
            Tuple of (probe_ids, sequence_invariants, atlas_probes)
            - probe_ids: All probe IDs for fingerprint matching
            - sequence_invariants: SequenceInvariant objects (for backward compat)
            - atlas_probes: AtlasProbe objects (for multi-atlas mode)
        """
        # Lazy imports to avoid circular dependency with agents package
        (
            SequenceFamily, SequenceInvariant, SequenceInvariantInventory,
            TriangulationScorer, TriangulatedScore, ExpressionDomain, DEFAULT_FAMILIES
        ) = _get_sequence_invariants()
        (
            AtlasProbe, AtlasSource, AtlasDomain, UnifiedAtlasInventory,
            MultiAtlasTriangulationScorer, MultiAtlasTriangulationScore, DEFAULT_ATLAS_SOURCES
        ) = _get_unified_atlas()
        
        # Handle MULTI_ATLAS scope - return all atlas probes
        if config.invariant_scope == InvariantScope.MULTI_ATLAS:
            sources = config.atlas_sources or DEFAULT_ATLAS_SOURCES
            if config.atlas_domains:
                probes = [
                    p for p in UnifiedAtlasInventory.probes_by_source(sources)
                    if p.domain in config.atlas_domains
                ]
            else:
                probes = UnifiedAtlasInventory.probes_by_source(sources)

            ids = [probe.probe_id for probe in probes]
            # Return empty sequence invariants list for multi-atlas mode
            return ids, [], probes

        # Handle sequence-only scopes (backward compatible)
        all_families = frozenset(SequenceFamily)

        if config.invariant_scope == InvariantScope.SEQUENCE_INVARIANTS:
            # Full 68-probe system with all 10 families (including tribonacci)
            base_families = all_families
        elif config.invariant_scope == InvariantScope.LOGIC_ONLY:
            base_families = frozenset([SequenceFamily.LOGIC])
        else:
            base_families = DEFAULT_FAMILIES

        families = config.family_allowlist.intersection(base_families) if config.family_allowlist else base_families
        invariants = SequenceInvariantInventory.probes_for_families(set(families))
        ids = [f"invariant:{inv.family.value}_{inv.id}" for inv in invariants]
        return ids, invariants, []

    @staticmethod
    def _compute_triangulation_scores(
        vectors: dict[int, list[float]],
        invariants: list[SequenceInvariant],
        config: Config,
    ) -> dict[int, TriangulatedScore]:
        """Compute per-layer triangulation scores using TriangulationScorer.

        Cross-domain detection (detecting invariants in multiple domains like
        definition, code, ratio, matrix) provides stronger anchoring.
        """
        # Lazy imports for runtime access
        (
            SequenceFamily, SequenceInvariant, SequenceInvariantInventory,
            TriangulationScorer, TriangulatedScore, ExpressionDomain, DEFAULT_FAMILIES
        ) = _get_sequence_invariants()
        
        scores: dict[int, TriangulatedScore] = {}
        if not invariants:
            return scores

        for layer, vector in vectors.items():
            # Group activations by domain
            domain_activations: dict[ExpressionDomain, float] = {}
            for i, activation in enumerate(vector):
                if i < len(invariants) and activation > config.triangulation_threshold:
                    domain = invariants[i].domain
                    domain_activations[domain] = max(
                        domain_activations.get(domain, 0.0), activation
                    )

            # Compute triangulated score using the first invariant's family as reference
            # (In practice, scores will be similar across families for cross-domain detection)
            if domain_activations:
                family = invariants[0].family
                scores[layer] = TriangulationScorer.compute_score(
                    domain_activations, family, None
                )
            else:
                # No significant activations - return neutral score
                scores[layer] = TriangulatedScore(
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
        config: Config,
    ) -> tuple[dict[int, TriangulatedScore], set[AtlasSource], set[AtlasDomain]]:
        """Compute per-layer triangulation scores using multi-atlas probes.

        Returns:
            Tuple of (scores_by_layer, sources_detected, domains_detected)
        """
        # Lazy imports for runtime access
        (
            SequenceFamily, SequenceInvariant, SequenceInvariantInventory,
            TriangulationScorer, TriangulatedScore, ExpressionDomain, DEFAULT_FAMILIES
        ) = _get_sequence_invariants()
        (
            AtlasProbe, AtlasSource, AtlasDomain, UnifiedAtlasInventory,
            MultiAtlasTriangulationScorer, MultiAtlasTriangulationScore, DEFAULT_ATLAS_SOURCES
        ) = _get_unified_atlas()
        
        scores: dict[int, TriangulatedScore] = {}
        all_sources: set[AtlasSource] = set()
        all_domains: set[AtlasDomain] = set()

        if not probes:
            return scores, all_sources, all_domains

        for layer, vector in vectors.items():
            # Group activations by source and domain
            source_activations: dict[AtlasSource, float] = {}
            domain_activations: dict[AtlasDomain, float] = {}

            for i, activation in enumerate(vector):
                if i < len(probes) and activation > config.triangulation_threshold:
                    probe = probes[i]
                    source_activations[probe.source] = max(
                        source_activations.get(probe.source, 0.0), activation
                    )
                    domain_activations[probe.domain] = max(
                        domain_activations.get(probe.domain, 0.0), activation
                    )
                    all_sources.add(probe.source)
                    all_domains.add(probe.domain)

            # Compute multi-atlas triangulation score
            if source_activations or domain_activations:
                source_count = len(source_activations)
                domain_count = len(domain_activations)

                # Source multiplier: boost when detected across multiple atlas sources
                source_mult = 1.0 + (source_count - 1) * 0.1 if source_count > 0 else 1.0

                # Domain multiplier: boost when detected across multiple domains
                domain_mult = 1.0 + (domain_count - 1) * 0.15 if domain_count > 0 else 1.0

                # Combined multiplier
                combined_mult = (source_mult * domain_mult) ** 0.5

                scores[layer] = TriangulatedScore(
                    base=sum(source_activations.values()) / max(1, source_count),
                    cross_domain_multiplier=combined_mult,
                    relationship_bonus=0.0,
                    coherence_bonus=(domain_count - 1) * 0.05 if domain_count > 1 else 0.0,
                )
            else:
                scores[layer] = TriangulatedScore(
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
        config: Config,
        probes: list[AtlasProbe],
        source_triangulation: dict[int, TriangulatedScore],
        target_triangulation: dict[int, TriangulatedScore],
    ) -> list[list[float]]:
        """Build similarity matrix using multi-atlas probes.

        Applies cross_domain_weight from each probe and boosts similarity based on
        multi-atlas triangulation multipliers.
        """
        source_count = len(source_layers)
        target_count = len(target_layers)

        if source_count == 0 or target_count == 0:
            return []

        # Pre-compute cross-domain weights from probes
        weights = [probe.cross_domain_weight for probe in probes]

        matrix = [[0.0] * target_count for _ in range(source_count)]

        for i, source_layer in enumerate(source_layers):
            source_vector = source_profile.vectors.get(source_layer, [])
            source_confidence = source_profile.confidence_by_layer.get(source_layer, 0.0)
            source_collapsed = source_layer in source_profile.collapsed_layers

            for j, target_layer in enumerate(target_layers):
                target_vector = target_profile.vectors.get(target_layer, [])
                target_confidence = target_profile.confidence_by_layer.get(target_layer, 0.0)
                target_collapsed = target_layer in target_profile.collapsed_layers

                # Compute weighted cosine similarity
                similarity = InvariantLayerMapper._weighted_cosine_similarity(
                    source_vector, target_vector, weights
                )

                confidence_weight = math.sqrt(max(0, source_confidence) * max(0, target_confidence))
                similarity *= confidence_weight

                # Apply multi-atlas triangulation boost
                if config.multi_domain_bonus:
                    source_ts = source_triangulation.get(source_layer)
                    target_ts = target_triangulation.get(target_layer)
                    if source_ts and target_ts:
                        tri_boost = math.sqrt(
                            source_ts.cross_domain_multiplier * target_ts.cross_domain_multiplier
                        )
                        similarity *= math.sqrt(tri_boost)

                if source_collapsed != target_collapsed:
                    penalty = max(0.0, min(1.0, config.collapse_mismatch_penalty))
                    similarity *= (1 - penalty)

                matrix[i][j] = max(0.0, min(1.0, similarity))

        return matrix

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
    def _profile_array(
        layer_count: int,
        profile: _ProfileData,
        triangulation_scores: Optional[dict[int, TriangulatedScore]] = None,
    ) -> list[LayerProfile]:
        """Convert profile data to array of LayerProfile."""
        profiles: list[LayerProfile] = []
        for layer in range(layer_count):
            tri_profile: Optional[TriangulationProfile] = None
            if triangulation_scores and layer in triangulation_scores:
                ts = triangulation_scores[layer]
                # Count domains by checking which had activation above threshold
                domains_detected = 1 if ts.cross_domain_multiplier > 1.0 else 0
                if ts.cross_domain_multiplier >= 1.2:
                    domains_detected = 2
                if ts.cross_domain_multiplier >= 1.5:
                    domains_detected = 3
                if ts.cross_domain_multiplier >= 1.8:
                    domains_detected = 4
                tri_profile = TriangulationProfile(
                    layer_index=layer,
                    domains_detected=domains_detected,
                    cross_domain_multiplier=ts.cross_domain_multiplier,
                    coherence_bonus=ts.coherence_bonus,
                )

            profiles.append(LayerProfile(
                layer_index=layer,
                confidence=profile.confidence_by_layer.get(layer, 0.0),
                coverage=profile.coverage_by_layer.get(layer, 0.0),
                strength=profile.strength_by_layer.get(layer, 0.0),
                collapsed=layer in profile.collapsed_layers,
                triangulation=tri_profile,
            ))
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
        invariants: Optional[list[SequenceInvariant]] = None,
        source_triangulation: Optional[dict[int, TriangulatedScore]] = None,
        target_triangulation: Optional[dict[int, TriangulatedScore]] = None,
    ) -> list[list[float]]:
        """Build similarity matrix between source and target layers.

        When invariants and triangulation scores are provided (SEQUENCE_INVARIANTS scope),
        applies cross_domain_weight to each invariant and boosts similarity based on
        triangulation multipliers.
        """
        source_count = len(source_layers)
        target_count = len(target_layers)

        if source_count == 0 or target_count == 0:
            return []

        # Pre-compute cross-domain weights if using weighting
        weights: Optional[list[float]] = None
        if config.use_cross_domain_weighting and invariants:
            weights = [inv.cross_domain_weight for inv in invariants]

        matrix = [[0.0] * target_count for _ in range(source_count)]

        for i, source_layer in enumerate(source_layers):
            source_vector = source_profile.vectors.get(source_layer, [])
            source_confidence = source_profile.confidence_by_layer.get(source_layer, 0.0)
            source_collapsed = source_layer in source_profile.collapsed_layers

            for j, target_layer in enumerate(target_layers):
                target_vector = target_profile.vectors.get(target_layer, [])
                target_confidence = target_profile.confidence_by_layer.get(target_layer, 0.0)
                target_collapsed = target_layer in target_profile.collapsed_layers

                # Compute similarity with optional cross-domain weighting
                if weights:
                    similarity = InvariantLayerMapper._weighted_cosine_similarity(
                        source_vector, target_vector, weights
                    )
                else:
                    similarity = InvariantLayerMapper._cosine_similarity(source_vector, target_vector)

                confidence_weight = math.sqrt(max(0, source_confidence) * max(0, target_confidence))
                similarity *= confidence_weight

                # Apply triangulation boost if available
                if config.multi_domain_bonus and source_triangulation and target_triangulation:
                    source_ts = source_triangulation.get(source_layer)
                    target_ts = target_triangulation.get(target_layer)
                    if source_ts and target_ts:
                        # Geometric mean of multipliers
                        tri_boost = math.sqrt(
                            source_ts.cross_domain_multiplier * target_ts.cross_domain_multiplier
                        )
                        # Apply as a mild boost (sqrt to dampen)
                        similarity *= math.sqrt(tri_boost)

                if source_collapsed != target_collapsed:
                    penalty = max(0.0, min(1.0, config.collapse_mismatch_penalty))
                    similarity *= (1 - penalty)

                matrix[i][j] = max(0.0, min(1.0, similarity))

        return matrix

    @staticmethod
    def _weighted_cosine_similarity(a: list[float], b: list[float], weights: list[float]) -> float:
        """Compute weighted cosine similarity between two vectors."""
        count = min(len(a), len(b), len(weights))
        if count == 0:
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0

        for i in range(count):
            w = weights[i]
            va = a[i] * w
            vb = b[i] * w
            dot += va * vb
            norm_a += va * va
            norm_b += vb * vb

        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        return max(0.0, min(1.0, dot / (math.sqrt(norm_a) * math.sqrt(norm_b))))

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


# =============================================================================
# Strategy-Based Layer Mapping
# =============================================================================


@dataclass(frozen=True)
class LayerCategoryScores:
    """Per-layer scores broken down by matching category."""
    layer_index: int
    activation_pattern: float
    invariant_coverage: float
    collapse_state: float  # 1.0 if not collapsed, 0.0 if collapsed
    triangulation: float
    cka_alignment: float
    combined: float  # Weighted combination


@dataclass(frozen=True)
class StrategyMappingResult:
    """Result of strategy-based layer mapping."""
    source_model: str
    target_model: str
    strategy: LayerMappingStrategy
    mappings: tuple[LayerMapping, ...]
    source_category_scores: tuple[LayerCategoryScores, ...]
    target_category_scores: tuple[LayerCategoryScores, ...]
    summary: Summary
    # CRM-specific metrics
    mean_cka_alignment: Optional[float] = None
    cka_matrix: Optional[tuple[tuple[float, ...], ...]] = None


class StrategyLayerMapper:
    """
    Strategy-based layer mapper supporting CRM and INVARIANT_COLLAPSE strategies.

    CRM Strategy:
        Uses CKA (Centered Kernel Alignment) to find optimal layer mappings
        based on representation similarity. Good for models with different
        architectures but similar representational structure.

    INVARIANT_COLLAPSE Strategy:
        Uses semantic invariant probes with collapse detection. Maps layers
        by invariant activation patterns while penalizing collapsed layer
        mismatches. Better for merging adapters or fine-tuned models.
    """

    @staticmethod
    def map_layers_with_strategy(
        source: ModelFingerprints,
        target: ModelFingerprints,
        config: Config,
        source_activations: Optional[dict[int, list[list[float]]]] = None,
        target_activations: Optional[dict[int, list[list[float]]]] = None,
    ) -> StrategyMappingResult:
        """
        Map layers using the configured strategy.

        Args:
            source: Fingerprints for source model
            target: Fingerprints for target model
            config: Mapping configuration with strategy selection
            source_activations: Raw activations by layer for CRM (optional)
            target_activations: Raw activations by layer for CRM (optional)

        Returns:
            StrategyMappingResult with mappings and per-layer scores
        """
        if config.strategy == LayerMappingStrategy.CRM:
            return StrategyLayerMapper._map_with_crm(
                source, target, config, source_activations, target_activations
            )
        else:
            return StrategyLayerMapper._map_with_invariant_collapse(
                source, target, config, source_activations, target_activations
            )

    @staticmethod
    def _map_with_crm(
        source: ModelFingerprints,
        target: ModelFingerprints,
        config: Config,
        source_activations: Optional[dict[int, list[list[float]]]] = None,
        target_activations: Optional[dict[int, list[list[float]]]] = None,
    ) -> StrategyMappingResult:
        """Map layers using CRM-based CKA alignment."""
        crm_cfg = config.crm_config or CRMMappingConfig()

        # Build CKA matrix between source and target layers
        source_samples = InvariantLayerMapper._sample_layers(
            source.layer_count, config.sample_layer_count
        )
        target_samples = InvariantLayerMapper._sample_layers(
            target.layer_count, config.sample_layer_count
        )

        cka_matrix: list[list[float]] = []

        if source_activations and target_activations:
            # Use provided activations for CKA computation
            cka_matrix = StrategyLayerMapper._compute_cka_matrix(
                source_samples, target_samples,
                source_activations, target_activations, crm_cfg
            )
        else:
            # Fall back to fingerprint-based similarity
            invariant_ids, _, _ = InvariantLayerMapper._get_invariants(config)
            source_profile = InvariantLayerMapper._build_profile(source, invariant_ids, config)
            target_profile = InvariantLayerMapper._build_profile(target, invariant_ids, config)

            cka_matrix = [[0.0] * len(target_samples) for _ in range(len(source_samples))]
            for i, src_layer in enumerate(source_samples):
                src_vec = source_profile.vectors.get(src_layer, [])
                for j, tgt_layer in enumerate(target_samples):
                    tgt_vec = target_profile.vectors.get(tgt_layer, [])
                    cka_matrix[i][j] = InvariantLayerMapper._cosine_similarity(src_vec, tgt_vec)

        # Find optimal alignment using Hungarian algorithm or greedy
        mappings = StrategyLayerMapper._align_with_cka(
            source_samples, target_samples, cka_matrix, config
        )

        # Build category scores
        source_scores = StrategyLayerMapper._build_category_scores_crm(
            source_samples, cka_matrix, True
        )
        target_scores = StrategyLayerMapper._build_category_scores_crm(
            target_samples, cka_matrix, False
        )

        # Compute mean CKA
        all_cka = [cka_matrix[i][j] for i in range(len(source_samples)) for j in range(len(target_samples))]
        mean_cka = sum(all_cka) / len(all_cka) if all_cka else 0.0

        mapped_count = len(mappings)
        skipped_count = sum(1 for m in mappings if m.is_skipped)
        mean_sim = sum(m.similarity for m in mappings) / len(mappings) if mappings else 0.0
        valid_mappings = [m for m in mappings if not m.is_skipped]
        alignment_quality = sum(m.similarity for m in valid_mappings) / len(valid_mappings) if valid_mappings else 0.0

        summary = Summary(
            mapped_layers=mapped_count,
            skipped_layers=skipped_count,
            mean_similarity=mean_sim,
            alignment_quality=alignment_quality,
            source_collapsed_layers=0,
            target_collapsed_layers=0,
        )

        return StrategyMappingResult(
            source_model=source.model_id,
            target_model=target.model_id,
            strategy=LayerMappingStrategy.CRM,
            mappings=tuple(mappings),
            source_category_scores=tuple(source_scores),
            target_category_scores=tuple(target_scores),
            summary=summary,
            mean_cka_alignment=mean_cka,
            cka_matrix=tuple(tuple(row) for row in cka_matrix),
        )

    @staticmethod
    def _map_with_invariant_collapse(
        source: ModelFingerprints,
        target: ModelFingerprints,
        config: Config,
        source_activations: Optional[dict[int, list[list[float]]]] = None,
        target_activations: Optional[dict[int, list[list[float]]]] = None,
    ) -> StrategyMappingResult:
        """Map layers using invariant-collapse strategy."""
        ic_cfg = config.invariant_collapse_config or InvariantCollapseMappingConfig()
        weights = (
            config.layer_match_category_weights or
            ic_cfg.category_weights or
            LayerMatchCategoryWeights()
        ).normalized()

        # Use the existing InvariantLayerMapper for base mapping
        base_report = InvariantLayerMapper.map_layers(source, target, config)

        # Compute per-layer category scores
        source_scores = StrategyLayerMapper._build_category_scores_invariant(
            base_report.source_profiles, weights, source_activations, config
        )
        target_scores = StrategyLayerMapper._build_category_scores_invariant(
            base_report.target_profiles, weights, target_activations, config
        )

        return StrategyMappingResult(
            source_model=source.model_id,
            target_model=target.model_id,
            strategy=LayerMappingStrategy.INVARIANT_COLLAPSE,
            mappings=base_report.mappings,
            source_category_scores=tuple(source_scores),
            target_category_scores=tuple(target_scores),
            summary=base_report.summary,
            mean_cka_alignment=None,
            cka_matrix=None,
        )

    @staticmethod
    def _compute_cka_matrix(
        source_layers: list[int],
        target_layers: list[int],
        source_activations: dict[int, list[list[float]]],
        target_activations: dict[int, list[list[float]]],
        crm_cfg: CRMMappingConfig,
    ) -> list[list[float]]:
        """Compute CKA similarity matrix between layers."""
        matrix = [[0.0] * len(target_layers) for _ in range(len(source_layers))]

        for i, src_layer in enumerate(source_layers):
            src_acts = source_activations.get(src_layer)
            if not src_acts:
                continue

            for j, tgt_layer in enumerate(target_layers):
                tgt_acts = target_activations.get(tgt_layer)
                if not tgt_acts:
                    continue

                # Compute CKA between activations
                cka = StrategyLayerMapper._compute_linear_cka(src_acts, tgt_acts, crm_cfg)
                matrix[i][j] = cka

        return matrix

    @staticmethod
    def _compute_linear_cka(
        x: list[list[float]],
        y: list[list[float]],
        crm_cfg: CRMMappingConfig,
    ) -> float:
        """Compute linear CKA between two activation matrices.

        CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

        For linear kernel: K = XX^T, L = YY^T
        """
        n = min(len(x), len(y))
        if n == 0:
            return 0.0

        # Truncate to same number of samples
        x = x[:n]
        y = y[:n]

        # Center the data if configured
        if crm_cfg.normalize_activations:
            x = StrategyLayerMapper._center_matrix(x)
            y = StrategyLayerMapper._center_matrix(y)

        # Compute gram matrices for linear kernel
        # K = XX^T, L = YY^T
        gram_x = StrategyLayerMapper._gram_matrix(x)
        gram_y = StrategyLayerMapper._gram_matrix(y)

        # Center gram matrices
        gram_x = StrategyLayerMapper._center_gram(gram_x)
        gram_y = StrategyLayerMapper._center_gram(gram_y)

        # HSIC = trace(KHLH) / (n-1)^2 where H is centering matrix
        # For centered gram matrices: HSIC = trace(KL) / (n-1)^2
        hsic_xy = StrategyLayerMapper._frobenius_inner(gram_x, gram_y)
        hsic_xx = StrategyLayerMapper._frobenius_inner(gram_x, gram_x)
        hsic_yy = StrategyLayerMapper._frobenius_inner(gram_y, gram_y)

        if hsic_xx <= 0 or hsic_yy <= 0:
            return 0.0

        cka = hsic_xy / math.sqrt(hsic_xx * hsic_yy)
        return max(0.0, min(1.0, cka))

    @staticmethod
    def _center_matrix(x: list[list[float]]) -> list[list[float]]:
        """Center columns to zero mean."""
        if not x or not x[0]:
            return x

        n = len(x)
        d = len(x[0])

        # Compute column means
        means = [sum(x[i][j] for i in range(n)) / n for j in range(d)]

        # Subtract means
        return [[x[i][j] - means[j] for j in range(d)] for i in range(n)]

    @staticmethod
    def _gram_matrix(x: list[list[float]]) -> list[list[float]]:
        """Compute gram matrix K = XX^T."""
        n = len(x)
        gram = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                dot = sum(x[i][k] * x[j][k] for k in range(len(x[i])))
                gram[i][j] = dot
                gram[j][i] = dot

        return gram

    @staticmethod
    def _center_gram(gram: list[list[float]]) -> list[list[float]]:
        """Center a gram matrix: H K H where H = I - 1/n * 1*1^T."""
        n = len(gram)
        if n == 0:
            return gram

        # Row means
        row_means = [sum(gram[i]) / n for i in range(n)]
        # Grand mean
        grand_mean = sum(row_means) / n

        # Center: K_ij - mean_i - mean_j + grand_mean
        centered = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                centered[i][j] = gram[i][j] - row_means[i] - row_means[j] + grand_mean

        return centered

    @staticmethod
    def _frobenius_inner(a: list[list[float]], b: list[list[float]]) -> float:
        """Compute Frobenius inner product trace(A^T B)."""
        n = min(len(a), len(b))
        total = 0.0
        for i in range(n):
            for j in range(min(len(a[i]), len(b[i]))):
                total += a[i][j] * b[i][j]
        return total

    @staticmethod
    def _align_with_cka(
        source_layers: list[int],
        target_layers: list[int],
        cka_matrix: list[list[float]],
        config: Config,
    ) -> list[LayerMapping]:
        """Align layers using CKA matrix (greedy optimal assignment)."""
        crm_cfg = config.crm_config or CRMMappingConfig()
        mappings: list[LayerMapping] = []

        # Greedy assignment: for each source, find best available target
        used_targets: set[int] = set()

        for i, src_layer in enumerate(source_layers):
            best_j = -1
            best_cka = -1.0

            for j, tgt_layer in enumerate(target_layers):
                if j in used_targets:
                    continue
                if cka_matrix[i][j] > best_cka:
                    best_cka = cka_matrix[i][j]
                    best_j = j

            if best_j >= 0:
                used_targets.add(best_j)
                tgt_layer = target_layers[best_j]

                is_skipped = best_cka < crm_cfg.min_cka_score
                confidence = InvariantLayerMapper._classify_confidence(best_cka, config)

                mappings.append(LayerMapping(
                    source_layer=src_layer,
                    target_layer=tgt_layer,
                    similarity=best_cka,
                    confidence=confidence,
                    is_skipped=is_skipped,
                ))

        return mappings

    @staticmethod
    def _build_category_scores_crm(
        layers: list[int],
        cka_matrix: list[list[float]],
        is_source: bool,
    ) -> list[LayerCategoryScores]:
        """Build category scores for CRM strategy."""
        scores: list[LayerCategoryScores] = []

        for idx, layer in enumerate(layers):
            # For CRM, CKA alignment is the primary signal
            if is_source:
                row = cka_matrix[idx] if idx < len(cka_matrix) else []
                max_cka = max(row) if row else 0.0
                mean_cka = sum(row) / len(row) if row else 0.0
            else:
                col = [cka_matrix[i][idx] if idx < len(cka_matrix[i]) else 0.0
                       for i in range(len(cka_matrix))]
                max_cka = max(col) if col else 0.0
                mean_cka = sum(col) / len(col) if col else 0.0

            scores.append(LayerCategoryScores(
                layer_index=layer,
                activation_pattern=mean_cka,
                invariant_coverage=0.0,  # Not used in CRM
                collapse_state=1.0,  # Assumed not collapsed in CRM
                triangulation=0.0,  # Not used in CRM
                cka_alignment=max_cka,
                combined=max_cka,
            ))

        return scores

    @staticmethod
    def _build_category_scores_invariant(
        profiles: tuple[LayerProfile, ...],
        weights: LayerMatchCategoryWeights,
        activations: Optional[dict[int, list[list[float]]]],
        config: Config,
    ) -> list[LayerCategoryScores]:
        """Build category scores for invariant-collapse strategy."""
        scores: list[LayerCategoryScores] = []
        w = weights.as_dict()

        for profile in profiles:
            activation_score = profile.strength
            coverage_score = profile.coverage
            collapse_score = 0.0 if profile.collapsed else 1.0

            tri_score = 0.0
            if profile.triangulation:
                tri_score = min(1.0, profile.triangulation.cross_domain_multiplier / 2.0)

            # CKA score placeholder (would need activations)
            cka_score = 0.0

            combined = (
                w[LayerMatchCategory.ACTIVATION_PATTERN] * activation_score +
                w[LayerMatchCategory.INVARIANT_COVERAGE] * coverage_score +
                w[LayerMatchCategory.COLLAPSE_STATE] * collapse_score +
                w[LayerMatchCategory.TRIANGULATION] * tri_score +
                w[LayerMatchCategory.CKA_ALIGNMENT] * cka_score
            )

            scores.append(LayerCategoryScores(
                layer_index=profile.layer_index,
                activation_pattern=activation_score,
                invariant_coverage=coverage_score,
                collapse_state=collapse_score,
                triangulation=tri_score,
                cka_alignment=cka_score,
                combined=combined,
            ))

        return scores


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_layer_alignment_confidence(
    mappings: tuple[LayerMapping, ...],
    config: Config,
) -> float:
    """
    Compute overall confidence in layer alignment.

    Returns a value between 0 and 1 indicating how confident
    we are in the layer mappings.
    """
    if not mappings:
        return 0.0

    valid_mappings = [m for m in mappings if not m.is_skipped]
    if not valid_mappings:
        return 0.0

    # Weight by confidence level
    confidence_values = {
        ConfidenceLevel.HIGH: 1.0,
        ConfidenceLevel.MEDIUM: 0.7,
        ConfidenceLevel.LOW: 0.4,
        ConfidenceLevel.UNCERTAIN: 0.1,
    }

    total_confidence = sum(confidence_values[m.confidence] for m in valid_mappings)
    mean_confidence = total_confidence / len(valid_mappings)

    # Factor in coverage (what fraction of mappings are valid)
    coverage = len(valid_mappings) / len(mappings)

    return mean_confidence * coverage


def select_optimal_strategy(
    source: ModelFingerprints,
    target: ModelFingerprints,
    has_activations: bool = False,
) -> LayerMappingStrategy:
    """
    Select optimal layer mapping strategy based on model characteristics.

    Heuristics:
    - If raw activations available and models have similar depth: CRM
    - If models have very different depths: INVARIANT_COLLAPSE
    - If one model has many collapsed layers: INVARIANT_COLLAPSE
    """
    depth_ratio = min(source.layer_count, target.layer_count) / max(source.layer_count, target.layer_count, 1)

    # CRM works well when depths are similar and we have activations
    if has_activations and depth_ratio >= 0.7:
        return LayerMappingStrategy.CRM

    # Default to invariant-collapse for robustness
    return LayerMappingStrategy.INVARIANT_COLLAPSE
