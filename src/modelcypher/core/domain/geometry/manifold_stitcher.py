from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from typing import ClassVar


class Thresholds:
    strong_correlation: ClassVar[float] = 0.7
    moderate_correlation: ClassVar[float] = 0.4
    strong_weight: ClassVar[float] = 1.0
    moderate_weight: ClassVar[float] = 0.6
    weak_weight: ClassVar[float] = 0.2


@dataclass(frozen=True)
class DimensionCorrelation:
    source_dim: int
    target_dim: int
    correlation: float

    @property
    def is_strong_correlation(self) -> bool:
        return self.correlation > Thresholds.strong_correlation

    @property
    def is_moderate_correlation(self) -> bool:
        return Thresholds.moderate_correlation < self.correlation <= Thresholds.strong_correlation

    @property
    def is_weak_correlation(self) -> bool:
        return self.correlation <= Thresholds.moderate_correlation


@dataclass(frozen=True)
class LayerConfidence:
    layer: int
    strong_correlations: int
    moderate_correlations: int
    weak_correlations: int
    confidence: float = field(init=False)

    def __post_init__(self) -> None:
        total = self.strong_correlations + self.moderate_correlations + self.weak_correlations
        if total > 0:
            weighted = (
                float(self.strong_correlations) * Thresholds.strong_weight
                + float(self.moderate_correlations) * Thresholds.moderate_weight
                + float(self.weak_correlations) * Thresholds.weak_weight
            )
            value = weighted / float(total)
        else:
            value = 0.0
        object.__setattr__(self, "confidence", value)

    @property
    def total_correlations(self) -> int:
        return self.strong_correlations + self.moderate_correlations + self.weak_correlations


@dataclass(frozen=True)
class IntersectionMap:
    source_model: str
    target_model: str
    dimension_correlations: dict[int, list[DimensionCorrelation]]
    overall_correlation: float
    aligned_dimension_count: int
    total_source_dims: int
    total_target_dims: int
    layer_confidences: list[LayerConfidence]


# =============================================================================
# Intersection Similarity Modes (Phase 4 Parity)
# =============================================================================


class IntersectionSimilarityMode(str, Enum):
    """
    Similarity mode for building intersection maps.

    Controls how dimension correlations are computed between source and target.
    """

    JACCARD = "jaccard"  # Binary set overlap (sparse activation patterns)
    WEIGHTED_JACCARD = "weighted_jaccard"  # Magnitude-weighted Jaccard
    CKA = "cka"  # Centered Kernel Alignment
    ENSEMBLE = "ensemble"  # Combined weighted Jaccard + CKA + cosine
    GROMOV_WASSERSTEIN = "gromov_wasserstein"  # Optimal transport-based


@dataclass
class EnsembleWeights:
    """Weights for ensemble similarity mode."""

    weighted_jaccard: float = 0.4
    cka: float = 0.4
    cosine: float = 0.2

    def normalized(self) -> "EnsembleWeights":
        """Return normalized weights summing to 1."""
        total = self.weighted_jaccard + self.cka + self.cosine
        if total <= 0:
            return EnsembleWeights(1 / 3, 1 / 3, 1 / 3)
        return EnsembleWeights(
            weighted_jaccard=self.weighted_jaccard / total,
            cka=self.cka / total,
            cosine=self.cosine / total,
        )


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
    all_dims = set(source_activations.keys()) | set(target_activations.keys())
    if not all_dims:
        return 0.0

    dot_product = 0.0
    source_norm_sq = 0.0
    target_norm_sq = 0.0

    for dim in all_dims:
        a = source_activations.get(dim, 0.0)
        b = target_activations.get(dim, 0.0)
        dot_product += a * b
        source_norm_sq += a * a
        target_norm_sq += b * b

    norm_product = (source_norm_sq ** 0.5) * (target_norm_sq ** 0.5)
    return dot_product / norm_product if norm_product > 1e-8 else 0.0


def compute_ensemble_similarity(
    source_activations: dict[int, float],
    target_activations: dict[int, float],
    weights: Optional[EnsembleWeights] = None,
) -> float:
    """
    Compute ensemble similarity combining multiple metrics.

    Ensemble = w_j * weighted_jaccard + w_c * cka + w_cos * cosine
    """
    if weights is None:
        weights = EnsembleWeights()
    weights = weights.normalized()

    weighted_jaccard = compute_weighted_jaccard_similarity(
        source_activations, target_activations
    )
    cosine = compute_cosine_similarity(source_activations, target_activations)

    # CKA for sparse vectors is approximately cosine^2 for centered data
    # For sparse activations, use squared cosine as CKA approximation
    cka = cosine * cosine

    return (
        weights.weighted_jaccard * weighted_jaccard
        + weights.cka * cka
        + weights.cosine * max(0.0, cosine)  # Cosine can be negative
    )


def build_layer_correlations(
    source_fingerprints: list["ActivationFingerprint"],
    target_fingerprints: list["ActivationFingerprint"],
    layer: int,
    mode: IntersectionSimilarityMode = IntersectionSimilarityMode.JACCARD,
    ensemble_weights: Optional[EnsembleWeights] = None,
    correlation_threshold: float = 0.3,
) -> list[DimensionCorrelation]:
    """
    Build dimension correlations for a layer using the specified similarity mode.

    Args:
        source_fingerprints: Fingerprints from source model
        target_fingerprints: Fingerprints from target model
        layer: Layer index to analyze
        mode: Similarity mode to use
        ensemble_weights: Weights for ensemble mode
        correlation_threshold: Minimum correlation to include

    Returns:
        List of dimension correlations above threshold
    """
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
                similarity = compute_jaccard_similarity(
                    set(s_primes.keys()), set(t_primes.keys())
                )
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
            elif mode == IntersectionSimilarityMode.ENSEMBLE:
                common_primes = set(s_primes.keys()) & set(t_primes.keys())
                if not common_primes:
                    similarity = 0.0
                else:
                    s_vec = {i: s_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    t_vec = {i: t_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    similarity = compute_ensemble_similarity(s_vec, t_vec, ensemble_weights)
            elif mode == IntersectionSimilarityMode.GROMOV_WASSERSTEIN:
                # GW requires full pairwise distance matrices - defer to external module
                # For now, fall back to CKA as approximation
                common_primes = set(s_primes.keys()) & set(t_primes.keys())
                if not common_primes:
                    similarity = 0.0
                else:
                    s_vec = {i: s_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    t_vec = {i: t_primes.get(p, 0.0) for i, p in enumerate(common_primes)}
                    cosine = compute_cosine_similarity(s_vec, t_vec)
                    similarity = cosine * cosine
            else:
                similarity = 0.0

            if similarity > best_correlation:
                best_correlation = similarity
                best_target_dim = t_dim

        if best_correlation >= correlation_threshold and best_target_dim >= 0:
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
    ensemble_weights: Optional[EnsembleWeights] = None,
    correlation_threshold: float = 0.3,
) -> IntersectionMap:
    """
    Build an intersection map between source and target fingerprints.

    Routes to appropriate similarity computation based on mode.

    Args:
        source_fingerprints: Fingerprints from source model
        target_fingerprints: Fingerprints from target model
        source_model: Source model identifier
        target_model: Target model identifier
        mode: Similarity mode to use
        ensemble_weights: Weights for ensemble mode
        correlation_threshold: Minimum correlation to include

    Returns:
        IntersectionMap with dimension correlations and layer confidences
    """
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
            ensemble_weights=ensemble_weights,
            correlation_threshold=correlation_threshold,
        )

        dimension_correlations[layer] = correlations

        # Count correlations by strength
        strong = sum(1 for c in correlations if c.is_strong_correlation)
        moderate = sum(1 for c in correlations if c.is_moderate_correlation)
        weak = sum(1 for c in correlations if c.is_weak_correlation)

        layer_confidences.append(
            LayerConfidence(
                layer=layer,
                strong_correlations=strong,
                moderate_correlations=moderate,
                weak_correlations=weak,
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

    # Overall correlation as mean of layer confidences
    overall_correlation = (
        sum(lc.confidence for lc in layer_confidences) / len(layer_confidences)
        if layer_confidences
        else 0.0
    )

    return IntersectionMap(
        source_model=source_model,
        target_model=target_model,
        dimension_correlations=dimension_correlations,
        overall_correlation=overall_correlation,
        aligned_dimension_count=total_aligned,
        total_source_dims=total_source_dims,
        total_target_dims=total_target_dims,
        layer_confidences=layer_confidences,
    )


class ProbeSpace(str, Enum):
    prelogits_hidden = "prelogits-hidden"
    output_logits = "output-logits"

output_layer_marker = -1


import logging
import math
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from modelcypher.core.domain._backend import get_default_backend

# Import unified_atlas lazily to avoid circular imports
# (geometry -> anchor_invariance_analyzer -> manifold_stitcher -> unified_atlas -> semantic_prime_atlas -> geometry)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _ensure_proper_rotation(
    u: "Array",
    vt: "Array",
    omega: "Array",
    backend: "Backend",
) -> "Array":
    """Ensure rotation matrix has det(R) = +1 (proper rotation, not reflection).

    For SVD-based Procrustes: R = U @ V^T
    If det(R) < 0, we have a reflection. Fix by flipping the last column of U.

    This follows the standard approach from Schönemann (1966) and is used in
    generalized_procrustes.py for consistency.

    Args:
        u: Left singular vectors from SVD
        vt: Right singular vectors (transposed) from SVD
        omega: Current rotation matrix (U @ Vt)
        backend: Backend for array operations

    Returns:
        Proper rotation matrix with det(R) = +1
    """
    try:
        # Convert to numpy for reliable determinant calculation
        omega_np = backend.to_numpy(omega)
        det_val = np.linalg.det(omega_np)

        if det_val < 0:
            # Flip last column of U to change sign of determinant
            u_np = backend.to_numpy(u)
            u_np[:, -1] *= -1
            u_fixed = backend.array(u_np)
            omega = backend.matmul(u_fixed, vt)
            logger.debug("Fixed reflection (det=%.3f) to proper rotation", det_val)

        return omega

    except np.linalg.LinAlgError as e:
        # SVD or det computation failed - return original omega with warning
        logger.warning("Could not compute determinant for sign correction: %s", e)
        return omega


@dataclass
class ContinuousFingerprint:
    """
    Continuous activation fingerprint preserving magnitude information.
    """
    prime_id: str
    prime_text: str
    
    # Layer -> Full activation vector
    activation_vectors: Dict[int, List[float]]
    
    # Layer -> L2 Magnitude
    magnitudes: Dict[int, float]
    
    # Layer -> Entropy (0-1)
    entropies: Dict[int, float]
    
    # Layer -> Sparsity (0-1)
    sparsities: Dict[int, float]

    @staticmethod
    def from_activations(
        prime_id: str,
        prime_text: str,
        layer_activations: Dict[int, List[float]],
        backend: "Backend | None" = None,
    ) -> "ContinuousFingerprint":
        b = backend or get_default_backend()
        magnitudes = {}
        entropies = {}
        sparsities = {}

        for layer, activations in layer_activations.items():
            arr = b.array(activations)
            magnitudes[layer] = float(b.norm(arr))

            logits = arr
            max_val = b.max(logits)
            exp_acts = b.exp(logits - max_val)
            probs = exp_acts / (b.sum(exp_acts) + 1e-10)

            log_probs = b.log(probs + 1e-10)
            entropy = -float(b.to_numpy(b.sum(probs * log_probs)).item())
            max_entropy = math.log(max(len(activations), 1))
            entropies[layer] = min(max(entropy / max_entropy, 0.0), 1.0) if max_entropy > 0 else 0.0

            abs_acts = b.abs(arr)
            threshold = 0.01 * float(b.to_numpy(b.max(abs_acts)).item())
            near_zero = float(b.to_numpy(b.sum(abs_acts < threshold)).item())
            sparsities[layer] = near_zero / max(len(activations), 1)
            
        return ContinuousFingerprint(prime_id, prime_text, layer_activations, magnitudes, entropies, sparsities)

@dataclass(frozen=True)
class StitchingConstants:
    epsilon: float = 1e-8
    similarity_weight: float = 0.6
    cosine_weight: float = 0.2
    magnitude_weight: float = 0.1
    entropy_weight: float = 0.1
    relationship_bonus: float = 0.1
    cross_domain_multiplier: float = 1.2

@dataclass
class ContinuousCorrelationResult:
    cka: float
    cosine_similarity: float
    magnitude_ratio: float
    entropy_delta: float
    
    @property
    def compatibility_score(self) -> float:
        # Note: For 1D vectors (single prime activations), Linear CKA is equivalent to squared cosine similarity.
        # CKA(x, y) = <x, y>^2 / (||x||^2 ||y||^2) = cosine_sim(x, y)^2
        
        cka_score = self.cka if self.cosine_similarity >= 0 else 0.0
        
        # Weighted combination of geometric invariants
        return (StitchingConstants.similarity_weight * cka_score + 
                StitchingConstants.cosine_weight * max(0.0, self.cosine_similarity) + 
                StitchingConstants.magnitude_weight * (1.0 - min(abs(self.magnitude_ratio - 1.0), 1.0)) + 
                StitchingConstants.entropy_weight * (1.0 - min(abs(self.entropy_delta), 1.0)))

@dataclass
class ContinuousModelFingerprints:
    """
    Collection of continuous fingerprints for a model.
    """
    model_id: str
    hidden_dim: int
    layer_count: int
    fingerprints: List[ContinuousFingerprint]
    
    @property
    def mean_entropy(self) -> float:
        vals = [e for fp in self.fingerprints for e in fp.entropies.values()]
        return sum(vals) / len(vals) if vals else 0.0
    
    @property
    def mean_sparsity(self) -> float:
        vals = [s for fp in self.fingerprints for s in fp.sparsities.values()]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def from_model_fingerprints(source: "ModelFingerprints") -> Optional["ContinuousModelFingerprints"]:
        if not hasattr(source, "activation_vectors") or not source.activation_vectors:
            return None
            
        fingerprints_by_prime: Dict[str, Dict[int, List[float]]] = {}
        for key, vec in source.activation_vectors.items():
            if "_layer" not in key: continue
            idx = key.rfind("_layer")
            try:
                layer = int(key[idx+6:])
                prime_id = key[:idx]
                if prime_id not in fingerprints_by_prime: fingerprints_by_prime[prime_id] = {}
                fingerprints_by_prime[prime_id][layer] = vec
            except ValueError: continue
            
        prime_texts = {fp.prime_id: fp.prime_text for fp in source.fingerprints}
        continuous_fps = [
            ContinuousFingerprint.from_activations(pid, prime_texts.get(pid, pid), layers)
            for pid, layers in fingerprints_by_prime.items()
        ]
        return ContinuousModelFingerprints(source.model_id, source.hidden_dim, source.layer_count, continuous_fps)

    def get_layer_profile(self, layer: int) -> Optional["LayerContinuousProfile"]:
        """Get aggregated profile for a specific layer."""
        layer_entropies = []
        layer_sparsities = []
        layer_magnitudes = []

        for fp in self.fingerprints:
            if layer in fp.entropies:
                layer_entropies.append(fp.entropies[layer])
            if layer in fp.sparsities:
                layer_sparsities.append(fp.sparsities[layer])
            if layer in fp.magnitudes:
                layer_magnitudes.append(fp.magnitudes[layer])

        if not layer_entropies:
            return None

        return LayerContinuousProfile(
            layer_index=layer,
            mean_entropy=sum(layer_entropies) / len(layer_entropies),
            mean_sparsity=sum(layer_sparsities) / len(layer_sparsities) if layer_sparsities else 0.0,
            mean_magnitude=sum(layer_magnitudes) / len(layer_magnitudes) if layer_magnitudes else 0.0,
            probe_count=len(layer_entropies),
            entropy_std=_compute_std(layer_entropies),
            sparsity_std=_compute_std(layer_sparsities) if layer_sparsities else 0.0,
        )

    def get_all_layer_profiles(self) -> Dict[int, "LayerContinuousProfile"]:
        """Get profiles for all layers."""
        profiles = {}
        for layer in range(self.layer_count):
            profile = self.get_layer_profile(layer)
            if profile:
                profiles[layer] = profile
        return profiles


@dataclass(frozen=True)
class LayerContinuousProfile:
    """Aggregated continuous profile for a single layer."""
    layer_index: int
    mean_entropy: float
    mean_sparsity: float
    mean_magnitude: float
    probe_count: int
    entropy_std: float = 0.0
    sparsity_std: float = 0.0

    @property
    def is_collapsed(self) -> bool:
        """Layer is considered collapsed if very low entropy or very high sparsity."""
        return self.mean_entropy < 0.1 or self.mean_sparsity > 0.95

    @property
    def confidence(self) -> float:
        """Confidence based on entropy and probe count."""
        # More probes and moderate entropy = higher confidence
        probe_factor = min(self.probe_count / 20.0, 1.0)
        entropy_factor = 1.0 - abs(self.mean_entropy - 0.5) * 2  # Peak at 0.5
        return probe_factor * max(0.0, entropy_factor)


@dataclass(frozen=True)
class TriangulatedProbeResult:
    """Result of triangulated probing across multiple domains."""
    probe_id: str
    primary_domain: str
    activation_score: float
    cross_domain_score: float
    triangulation_multiplier: float
    domains_detected: List[str]
    layer_index: int

    @property
    def triangulated_score(self) -> float:
        """Combined score with triangulation boost."""
        return self.activation_score * self.triangulation_multiplier


@dataclass
class TriangulatedProbingConfig:
    """Configuration for triangulated probing."""
    include_sequence_invariants: bool = True
    include_metaphor_invariants: bool = True
    include_conceptual_genealogy: bool = True
    triangulation_threshold: float = 0.3
    cross_domain_bonus: float = 0.15
    max_domains_for_full_bonus: int = 3


class TriangulatedProbeBuilder:
    """
    Builds triangulated probe sets for enhanced fingerprinting.

    Now uses UnifiedAtlasInventory (321 probes across 7 atlas sources)
    instead of hardcoded probe lists. This provides:
    - SEQUENCE_INVARIANT: 68 probes
    - SEMANTIC_PRIME: 65 probes
    - COMPUTATIONAL_GATE: 76 probes
    - EMOTION_CONCEPT: 32 probes
    - TEMPORAL_CONCEPT: 25 probes
    - SOCIAL_CONCEPT: 25 probes
    - MORAL_CONCEPT: 30 probes
    """

    @staticmethod
    def build_triangulated_probes(
        config: Optional[TriangulatedProbingConfig] = None,
    ) -> List[Any]:
        """
        Build probe set from UnifiedAtlasInventory (321 probes).

        Returns list of AtlasProbe objects with:
        - probe_id: Unique identifier
        - support_texts: Probe texts for embedding
        - source: Atlas source (SEMANTIC_PRIME, SEQUENCE_INVARIANT, etc.)
        - domain: Triangulation domain
        - cross_domain_weight: Weight for cross-domain detection
        """
        # Lazy import to avoid circular dependency
        from modelcypher.core.domain.agents.unified_atlas import (
            UnifiedAtlasInventory,
            AtlasSource,
        )

        if config is None:
            config = TriangulatedProbingConfig()

        sources: set[AtlasSource] = set()

        # Semantic primes always included
        sources.add(AtlasSource.SEMANTIC_PRIME)

        # Add sequence invariants if enabled
        if config.include_sequence_invariants:
            sources.add(AtlasSource.SEQUENCE_INVARIANT)

        # Add other sources based on config
        if config.include_metaphor_invariants:
            # Metaphor invariants map to emotion/temporal/moral concepts
            sources.add(AtlasSource.EMOTION_CONCEPT)
            sources.add(AtlasSource.TEMPORAL_CONCEPT)

        if config.include_conceptual_genealogy:
            # Conceptual genealogy maps to moral/social concepts
            sources.add(AtlasSource.MORAL_CONCEPT)
            sources.add(AtlasSource.SOCIAL_CONCEPT)

        # Always include computational gates for cross-domain triangulation
        sources.add(AtlasSource.COMPUTATIONAL_GATE)

        return UnifiedAtlasInventory.probes_by_source(sources)

    @staticmethod
    def build_all_probes() -> List[Any]:
        """Get all 321 probes for full triangulation."""
        # Lazy import to avoid circular dependency
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
        return UnifiedAtlasInventory.all_probes()

    @staticmethod
    def build_probes_for_sources(sources: set) -> List[Any]:
        """Get probes from specific atlas sources."""
        # Lazy import to avoid circular dependency
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
        return UnifiedAtlasInventory.probes_by_source(sources)

    @staticmethod
    def to_legacy_format(probes: List[Any]) -> List[Dict[str, str]]:
        """Convert AtlasProbe objects to legacy dict format for compatibility."""
        return [
            {
                "probe_id": probe.probe_id,
                "probe_text": probe.support_texts[0] if probe.support_texts else "",
                "primary_domain": probe.source.value,
                "cross_domain_weight": str(probe.cross_domain_weight),
            }
            for probe in probes
        ]

    @staticmethod
    def compute_triangulation_score(
        activations_by_domain: Dict[str, float],
        config: Optional[TriangulatedProbingConfig] = None,
    ) -> Tuple[float, float]:
        """
        Compute triangulation score from multi-domain activations.

        Args:
            activations_by_domain: Activation scores keyed by domain
            config: Configuration

        Returns:
            Tuple of (base_score, triangulation_multiplier)
        """
        if config is None:
            config = TriangulatedProbingConfig()

        if not activations_by_domain:
            return 0.0, 1.0

        # Base score is mean of significant activations
        significant = [v for v in activations_by_domain.values() if v >= config.triangulation_threshold]
        if not significant:
            return 0.0, 1.0

        base_score = sum(significant) / len(significant)

        # Triangulation bonus based on domain count
        domain_count = len(significant)
        bonus_domains = min(domain_count - 1, config.max_domains_for_full_bonus - 1)
        multiplier = 1.0 + bonus_domains * config.cross_domain_bonus

        return base_score, multiplier


def _compute_std(values: List[float]) -> float:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


class ManifoldStitcher:
    """
    Manifold stitching for cross-architecture model merging.
    Implementation of Continuous CKA-based stitching.
    """
    OUTPUT_LAYER_MARKER = 999999
    
    @staticmethod
    def compute_continuous_correlation(
        source: ContinuousFingerprint,
        target: ContinuousFingerprint,
        layer: int,
        backend: "Backend | None" = None,
    ) -> Optional[ContinuousCorrelationResult]:
        b = backend or get_default_backend()
        if layer not in source.activation_vectors or layer not in target.activation_vectors:
            return None
        s_vec = b.array(source.activation_vectors[layer])
        t_vec = b.array(target.activation_vectors[layer])
        if s_vec.size == 0 or t_vec.size == 0:
            return None

        min_len = min(s_vec.size, t_vec.size)
        s_trunc, t_trunc = s_vec[:min_len], t_vec[:min_len]

        dot_prod = float(b.to_numpy(b.sum(s_trunc * t_trunc)).item())
        s_norm = float(b.to_numpy(b.norm(s_vec)).item())
        t_norm = float(b.to_numpy(b.norm(t_vec)).item())
        cosine = dot_prod / (s_norm * t_norm) if (s_norm > 1e-8 and t_norm > 1e-8) else 0.0

        mag_ratio = source.magnitudes.get(layer, 1.0) / target.magnitudes.get(layer, 1.0) if target.magnitudes.get(layer, 1.0) > 1e-8 else 1.0
        entropy_delta = source.entropies.get(layer, 0.0) - target.entropies.get(layer, 0.0)

        return ContinuousCorrelationResult(cosine * cosine, cosine, mag_ratio, entropy_delta)

    @staticmethod
    def compute_cka_matrix(
        source: ContinuousModelFingerprints,
        target: ContinuousModelFingerprints,
        layer: int,
        backend: "Backend | None" = None,
    ) -> Tuple[Any, List[str], List[str]]:
        """
        Compute pairwise CKA matrix between all primes at a given layer.
        Returns: (matrix, source_prime_ids, target_prime_ids)
        """
        b = backend or get_default_backend()
        s_fps = [fp for fp in source.fingerprints if layer in fp.activation_vectors]
        t_fps = [fp for fp in target.fingerprints if layer in fp.activation_vectors]
        if not s_fps or not t_fps:
            return (b.array([]), [], [])

        matrix = []
        for s_fp in s_fps:
            row = []
            for t_fp in t_fps:
                res = ManifoldStitcher.compute_continuous_correlation(s_fp, t_fp, layer, backend=b)
                row.append(res.cka if res else 0.0)
            matrix.append(row)
        return (b.array(matrix), [fp.prime_id for fp in s_fps], [fp.prime_id for fp in t_fps])

    @staticmethod
    def compute_intersection_rotation(
        intersection: IntersectionMap,
        layer: int,
        source_basis: Any,
        target_basis: Any,
        backend: "Backend | None" = None,
    ) -> Tuple[Any, float]:
        """
        Computes a targeted rotation matrix using the intersection map.
        Strong correlations -> tight rotation (trust mapping).
        """
        b = backend or get_default_backend()
        correlations = intersection.dimension_correlations.get(layer, [])
        if not correlations:
            return (b.eye(source_basis.shape[1]), 0.0)

        dim_s = source_basis.shape[0]
        dim_t = target_basis.shape[0]

        # Filter valid correlations
        filtered = [
            c for c in correlations
            if c.source_dim < dim_s and c.target_dim < dim_t
        ]

        if len(filtered) < 2:
            k = min(source_basis.shape[1], target_basis.shape[1])
            return (b.eye(k), 0.0)

        # Build index arrays
        source_indices = [c.source_dim for c in filtered]
        target_indices = [c.target_dim for c in filtered]

        # Take rows from basis matrices using indexing
        z_source = source_basis[source_indices]
        z_target = target_basis[target_indices]

        # Weighting
        weights = [max(0.0, c.correlation) for c in filtered]
        sqrt_weights = b.reshape(b.sqrt(b.array(weights)), (-1, 1))

        z_source = z_source * sqrt_weights
        z_target = z_target * sqrt_weights

        # Procrustes: R = U @ V^T minimizes ||source @ R - target||_F
        m = b.matmul(b.transpose(z_source), z_target)
        u, _, vt = b.svd(m)
        omega = b.matmul(u, vt)

        # Ensure proper rotation (det = +1), not reflection
        omega = _ensure_proper_rotation(u, vt, omega, b)

        confidence = sum(weights) / max(len(weights), 1)
        return (omega, confidence)

    @staticmethod
    def cluster_activations(
        source_activations: Dict[str, List[float]],  # PrimeID -> Activation (Layer 0)
        target_activations: Dict[str, List[float]],
        cluster_count: int = 8,
        backend: "Backend | None" = None,
    ) -> List["AlignmentCluster"]:  # Forward ref string since defined later
        """
        Clusters activations to identify alignment regions.
        """
        b = backend or get_default_backend()
        keys = sorted(list(set(source_activations.keys()) & set(target_activations.keys())))
        if not keys:
            return []

        source_vecs = [source_activations[k] for k in keys]
        target_vecs = [target_activations[k] for k in keys]

        # K-Means on source
        assignments, _ = ManifoldStitcher.k_means(source_vecs, cluster_count, backend=b)

        clusters = []
        shared_dim = min(len(source_vecs[0]), len(target_vecs[0]))

        for cluster_id in range(cluster_count):
            indices = [i for i, a in enumerate(assignments) if a == cluster_id]
            if not indices:
                continue

            s_members = b.array([source_vecs[i][:shared_dim] for i in indices])
            t_members = b.array([target_vecs[i][:shared_dim] for i in indices])

            s_mean = b.mean(s_members, axis=0)
            t_mean = b.mean(t_members, axis=0)

            # Local rotation
            s_centered = s_members - s_mean
            t_centered = t_members - t_mean

            m = b.matmul(b.transpose(s_centered), t_centered)
            u, _, vt = b.svd(m)
            omega = b.matmul(u, vt)

            # Ensure proper rotation (det = +1), not reflection
            omega = _ensure_proper_rotation(u, vt, omega, b)

            # Error
            projected = b.matmul(s_centered, omega)
            error = projected - t_centered
            error_norm = float(b.to_numpy(b.sqrt(b.sum(error * error))).item())
            target_norm = float(b.to_numpy(b.sqrt(b.sum(t_centered * t_centered))).item())
            procrustes_error = error_norm / target_norm if target_norm > 1e-6 else 0.0

            clusters.append(AlignmentCluster(
                id=cluster_id,
                centroid_source=b.to_numpy(s_mean).tolist(),
                centroid_target=b.to_numpy(t_mean).tolist(),
                local_rotation=omega,
                procrustes_error=procrustes_error,
                member_count=len(indices)
            ))

        return clusters

    @staticmethod
    def k_means(
        points: List[List[float]],
        k: int,
        max_iterations: int = 50,
        backend: "Backend | None" = None,
    ) -> Tuple[List[int], List[List[float]]]:
        b = backend or get_default_backend()
        n = len(points)
        if n == 0 or k <= 0:
            return ([], [])

        pts = b.array(points)
        d_dim = pts.shape[1]

        # K-Means++ Initialization
        # 1. Choose first centroid uniformly
        centroids = b.zeros((k, d_dim))
        first_idx = int(b.to_numpy(b.random_randint(0, n, shape=(1,))).item())

        # Update first centroid (manual copy)
        centroids_np = b.to_numpy(centroids)
        centroids_np[0] = b.to_numpy(pts[first_idx])
        centroids = b.array(centroids_np)

        # 2. Choose remaining k-1 centroids
        for i in range(1, k):
            # Compute dists to nearest existing centroid
            current_centroids = centroids[:i]
            # Expand dims for broadcasting
            pts_exp = b.reshape(pts, (n, 1, d_dim))
            cent_exp = b.reshape(current_centroids, (1, i, d_dim))
            diff = pts_exp - cent_exp
            # Norm over last axis
            d = b.sqrt(b.sum(diff * diff, axis=2))  # (N, i)
            min_dists = b.min(d, axis=1)  # (N,)
            probs = min_dists ** 2
            probs = probs / b.sum(probs)

            # Sample next centroid using cumulative sum
            cumsum = b.cumsum(probs)
            r = b.random_uniform(shape=(1,))
            # Find index where cumsum > r
            next_idx = int(b.to_numpy(b.argmax(cumsum > float(b.to_numpy(r).item()))).item())

            # Update centroid
            centroids_np = b.to_numpy(centroids)
            centroids_np[i] = b.to_numpy(pts[next_idx])
            centroids = b.array(centroids_np)

        assignments = b.zeros((n,), dtype="int32")

        for _ in range(max_iterations):
            # Compute distances
            pts_exp = b.reshape(pts, (n, 1, d_dim))
            cent_exp = b.reshape(centroids, (1, k, d_dim))
            diff = pts_exp - cent_exp
            dists = b.sqrt(b.sum(diff * diff, axis=2))  # (N, K)
            new_assignments = b.argmin(dists, axis=1)

            # Check convergence via numpy comparison
            if (b.to_numpy(assignments) == b.to_numpy(new_assignments)).all():
                break
            assignments = new_assignments

            # Update centroids
            centroids_np = b.to_numpy(centroids)
            assignments_np = b.to_numpy(assignments)
            pts_np = b.to_numpy(pts)
            for c in range(k):
                mask = assignments_np == c
                if mask.sum() > 0:
                    centroids_np[c] = pts_np[mask].mean(axis=0)
            centroids = b.array(centroids_np)

        return (b.to_numpy(assignments).tolist(), b.to_numpy(centroids).tolist())

    @staticmethod
    def soft_rotation(
        weight: Any,
        clusters: List["AlignmentCluster"],
        temperature: float = 0.3,
        backend: "Backend | None" = None,
    ) -> Any:
        b = backend or get_default_backend()
        if not clusters:
            return weight
        if weight.ndim != 2:
            return weight

        in_dim = weight.shape[1]
        cluster_dim = clusters[0].local_rotation.shape[0]
        if in_dim != cluster_dim:
            return weight

        # Weighted average
        weights = []
        for c in clusters:
            w = math.exp(-c.procrustes_error / temperature) * c.member_count
            weights.append(w)

        total_weight = sum(weights)
        if total_weight <= 0:
            return weight

        weighted_omega = b.zeros((cluster_dim, cluster_dim))
        for i, c in enumerate(clusters):
            norm_w = weights[i] / total_weight
            weighted_omega = weighted_omega + (c.local_rotation * norm_w)

        # Re-orthogonalize
        u, _, vt = b.svd(weighted_omega)
        omega = b.matmul(u, vt)

        return b.matmul(weight, b.transpose(omega))

    @staticmethod
    async def validate_merged_model(
        merged_model_ctx: Any, # ModelContext
        merged_model_id: str,
        target_fingerprints: ModelFingerprints,
        top_k: int = 32
    ) -> ValidationResult:
        """
        Validates a merged model by comparing its fingerprints to the original target.
        """
        # Determine layers to probe
        target_layers = set()
        for fp in target_fingerprints.fingerprints:
            target_layers.update(fp.activated_dimensions.keys())
            
        intermediate_layers = {l for l in target_layers if l > 0 and l != ManifoldStitcher.OUTPUT_LAYER_MARKER}
        probe_layers = list(intermediate_layers) if intermediate_layers else None
        
        # Probe merged model
        merged_fingerprints = await ManifoldStitcher.probe_with_primes(
            model_ctx=merged_model_ctx,
            model_id=merged_model_id,
            probe_space=target_fingerprints.probe_space,
            top_k=top_k,
            layer_indices=probe_layers
        )
        
        # Compare
        layer_deltas = []
        all_layers = target_layers.union(
            {l for fp in merged_fingerprints.fingerprints for l in fp.activated_dimensions.keys()}
        )
        
        merged_map = {fp.prime_id: fp for fp in merged_fingerprints.fingerprints}
        target_map = {fp.prime_id: fp for fp in target_fingerprints.fingerprints}
        
        prime_ids = set(merged_map.keys()) & set(target_map.keys())
        
        for layer in sorted(list(all_layers)):
            merged_dims = set()
            target_dims = set()
            
            for pid in prime_ids:
                if layer in merged_map[pid].activated_dimensions:
                    merged_dims.update([d.index for d in merged_map[pid].activated_dimensions[layer]])
                if layer in target_map[pid].activated_dimensions:
                    target_dims.update([d.index for d in target_map[pid].activated_dimensions[layer]])
                    
            overlap = merged_dims & target_dims
            union = merged_dims | target_dims
            jaccard = len(overlap) / len(union) if union else 0.0
            
            layer_deltas.append(LayerDelta(
                layer=layer,
                overlap_count=len(overlap),
                merged_count=len(merged_dims),
                target_count=len(target_dims),
                jaccard_similarity=jaccard
            ))
            
        mean_jaccard = sum(d.jaccard_similarity for d in layer_deltas) / len(layer_deltas) if layer_deltas else 0.0
        
        if mean_jaccard > 0.7: status = ValidationStatus.EXCELLENT
        elif mean_jaccard > 0.5: status = ValidationStatus.GOOD
        elif mean_jaccard > 0.3: status = ValidationStatus.FAIR
        else: status = ValidationStatus.POOR
        
        return ValidationResult(
            merged_model=merged_model_id,
            target_model=target_fingerprints.model_id,
            layer_deltas=layer_deltas,
            overall_similarity=mean_jaccard,
            status=status
        )

    @staticmethod
    async def probe_with_primes(
        model_ctx: Any,
        model_id: str,
        probe_space: ProbeSpace,
        top_k: int,
        layer_indices: Optional[List[int]] = None
    ) -> ModelFingerprints:
        # Placeholder for actual probing logic
        # In a real implementation, this would use UnifiedAtlas probes and run inference
        # capturing activations. For now, return empty fingerprint set.
        
        # We will assume for now this is handled by external service or just return empty for parity structure.
        return ModelFingerprints(
            model_id=model_id,
            probe_space=probe_space,
            probe_capture_key=None,
            hidden_dim=0,
            layer_count=0,
            fingerprints=[]
        )



@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

    def __lt__(self, other: "ActivatedDimension") -> bool:
        return self.activation > other.activation


@dataclass(frozen=True)
class ActivationFingerprint:
    prime_id: str
    prime_text: str
    activated_dimensions: dict[int, list[ActivatedDimension]]


@dataclass(frozen=True)
class SparseActivationVector:
    indices: list[int]
    values: list[float]
    length: int

    def dot(self, other: "SparseActivationVector") -> float:
        count_a = min(len(self.indices), len(self.values))
        count_b = min(len(other.indices), len(other.values))
        if count_a <= 0 or count_b <= 0:
            return 0.0
        i = 0
        j = 0
        total = 0.0
        while i < count_a and j < count_b:
            idx_a = self.indices[i]
            idx_b = other.indices[j]
            if idx_a == idx_b:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif idx_a < idx_b:
                i += 1
            else:
                j += 1
        return total

    def dot_dense(self, dense: list[float]) -> float:
        count = min(len(self.indices), len(self.values))
        if count <= 0:
            return 0.0
        total = 0.0
        for i in range(count):
            idx = self.indices[i]
            if 0 <= idx < len(dense):
                total += self.values[i] * dense[idx]
        return total


@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    probe_space: ProbeSpace
    probe_capture_key: Optional[str]
    hidden_dim: int
    layer_count: int
    fingerprints: list[ActivationFingerprint]
    activation_vectors: Optional[dict[str, list[float]]] = None
    activation_sparse_vectors: Optional[dict[str, SparseActivationVector]] = None


class ClusterClassification(str, Enum):
    ALIGNED = "aligned"
    TRANSLATABLE = "translatable"
    DIVERGENT = "divergent"

@dataclass
class AlignmentCluster:
    id: int
    centroid_source: List[float]
    centroid_target: List[float]
    local_rotation: Any  # Array type from backend
    procrustes_error: float
    member_count: int
    
    @property
    def classification(self) -> ClusterClassification:
        if self.procrustes_error < 0.3: return ClusterClassification.ALIGNED
        if self.procrustes_error < 0.7: return ClusterClassification.TRANSLATABLE
        return ClusterClassification.DIVERGENT

@dataclass
class LayerDelta:
    layer: int
    overlap_count: int
    merged_count: int
    target_count: int
    jaccard_similarity: float

class ValidationStatus(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ValidationResult:
    merged_model: str
    target_model: str
    layer_deltas: List[LayerDelta]
    overall_similarity: float
    status: ValidationStatus



def intersection_map_from_dict(payload: dict[str, Any]) -> IntersectionMap:
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
        strong = entry.get("strongCorrelations", entry.get("strong_correlations"))
        moderate = entry.get("moderateCorrelations", entry.get("moderate_correlations"))
        weak = entry.get("weakCorrelations", entry.get("weak_correlations"))
        if layer is None or strong is None or moderate is None or weak is None:
            continue
        layer_confidences.append(
            LayerConfidence(
                layer=int(layer),
                strong_correlations=int(strong),
                moderate_correlations=int(moderate),
                weak_correlations=int(weak),
            )
        )

    return IntersectionMap(
        source_model=str(_get("sourceModel", "source_model") or ""),
        target_model=str(_get("targetModel", "target_model") or ""),
        dimension_correlations=dimension_correlations,
        overall_correlation=float(_get("overallCorrelation", "overall_correlation") or 0.0),
        aligned_dimension_count=int(_get("alignedDimensionCount", "aligned_dimension_count") or 0),
        total_source_dims=int(_get("totalSourceDims", "total_source_dims") or 0),
        total_target_dims=int(_get("totalTargetDims", "total_target_dims") or 0),
        layer_confidences=layer_confidences,
    )
