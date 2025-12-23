"""
Unified Manifold Merger: Geometry-Aware Model Merging with Adaptive Blending.

Ported from the reference Swift implementation (core features only).

Key Features:
1. Adaptive Alpha Profile - Per-layer blending weights based on confidence
2. Gaussian Smoothing - Prevents "tearing" from sharp alpha transitions
3. Spectral Penalty - Reduces trust in poorly-conditioned rotations
4. Dimension Blending - Per-dimension weights from intersection maps

Mathematical Foundation:
- confidence(l) = intersection map confidence for layer l
- α(l) = blending factor = f(confidence(l), procrustes_error(l))
- W_merged = α * W_target + (1-α) * project(W_source, Ω_out, Ω_in)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from modelcypher.core.domain._backend import get_default_backend

from .rotational_merger import MergeOptions, RotationalModelMerger

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
    from .entropy_merge_validator import MergeEntropyValidation

logger = logging.getLogger("modelcypher.merging.unified_manifold_merger")


# =============================================================================
# Configuration
# =============================================================================


class BlendMode(str, Enum):
    """Module-level blending behavior."""
    SOFT = "soft"             # Standard alpha blending
    HARD_SWAP = "hard_swap"   # Binary choice (source or target)
    SKIP = "skip"             # Don't merge this module


class LayerMappingStrategy(str, Enum):
    """Strategy used to map layers across architectures."""
    CRM = "crm"                        # CRM-based CKA alignment
    INVARIANT_COLLAPSE = "invariant_collapse"  # Invariant-only, collapse-aware


class MLPInternalIntersectionMode(str, Enum):
    """Intersection map mode for internal MLP projections (gate/up)."""
    FULL = "full"              # Use full intersection map
    INVARIANTS = "invariants"  # Use only invariant anchors
    LOGIC_ONLY = "logic_only"  # Use only logic invariants


class IntersectionSimilarityMode(str, Enum):
    """Similarity mode for building intersection maps."""
    JACCARD = "jaccard"                    # Binary set overlap
    WEIGHTED_JACCARD = "weighted_jaccard"  # Magnitude-weighted Jaccard
    CKA = "cka"                            # Centered Kernel Alignment
    ENSEMBLE = "ensemble"                  # Combined weighted Jaccard + CKA
    GROMOV_WASSERSTEIN = "gromov_wasserstein"  # Optimal transport-based


class ModuleScope(str, Enum):
    """Module scope for merging operations."""
    ALL = "all"                          # All weight matrices
    ATTENTION_ONLY = "attention_only"    # Q/K/V/O projections only
    MLP_ONLY = "mlp_only"                # Gate/Up/Down projections only
    MLP_INTERNAL = "mlp_internal"        # Gate/Up only (preserve residual)
    ATTENTION_AND_MLP_DOWN = "attention_and_mlp_down"  # Attention + down projection


class SequenceFamily(str, Enum):
    """Sequence families for invariant probing."""
    FIBONACCI = "fibonacci"
    LUCAS = "lucas"
    PRIMES = "primes"
    CATALAN = "catalan"


# =============================================================================
# Nested Configuration Classes
# =============================================================================


@dataclass
class InvariantLayerMapperConfig:
    """Configuration for invariant-only layer mapping."""
    collapse_threshold: float = 0.3
    min_invariant_coverage: float = 0.5
    use_collapse_detection: bool = True


@dataclass
class TangentSpaceConfig:
    """Configuration for tangent-space alignment metrics."""
    local_neighborhood_size: int = 10
    normalization_mode: str = "unit"  # "unit", "standard", "none"
    use_weighted_average: bool = True


@dataclass
class SharedSubspaceConfig:
    """Configuration for shared subspace projection (CCA/SVD)."""
    projection_rank: int = 32
    method: str = "cca"  # "cca", "svd", "procrustes"
    regularization: float = 1e-4


@dataclass
class TransportGuidedConfig:
    """Configuration for transport-guided (Gromov-Wasserstein) merger."""
    entropic_regularization: float = 0.01
    num_iterations: int = 100
    convergence_threshold: float = 1e-6
    use_sliced_gw: bool = False
    num_slices: int = 50


@dataclass
class AffineStitchingConfig:
    """Configuration for affine stitching layer."""
    learning_rate: float = 0.01
    num_iterations: int = 100
    regularization: float = 1e-4
    use_bias: bool = True


@dataclass
class VerbNounConfig:
    """
    Configuration for verb/noun dimension classification.

    Reference: Wierzbicka (1996) "Semantics: Primes and Universals"
    - Verb dimensions: Skill/trajectory (trust Source more)
    - Noun dimensions: Knowledge/position (trust Target more)
    """
    verb_alpha_bias: float = -0.15   # Push alpha down for verbs (trust source)
    noun_alpha_bias: float = 0.15    # Push alpha up for nouns (trust target)
    classification_threshold: float = 0.5
    use_semantic_prime_anchors: bool = True


@dataclass
class IntersectionEnsembleWeights:
    """Ensemble weights when using .ensemble similarity mode."""
    weighted_jaccard: float = 0.4
    cka: float = 0.4
    cosine: float = 0.2


@dataclass
class AnchorCategoryWeights:
    """Anchor-category weights for layer correspondence matching."""
    semantic_primes: float = 1.0
    sequence_invariants: float = 0.5
    metaphor_invariants: float = 0.3
    conceptual_genealogy: float = 0.3


@dataclass
class ModuleBlendPolicy:
    """
    Policy that maps module kinds to blending strategies.
    
    Allows different treatment for attention vs MLP modules based on
    empirical observations about which benefit from blending vs swapping.
    """
    soft_blend_kinds: set = field(default_factory=lambda: {
        "q_proj", "k_proj", "gate_proj", "up_proj", "down_proj"
    })
    hard_swap_kinds: set = field(default_factory=lambda: {"v_proj"})
    skip_kinds: set = field(default_factory=lambda: {"o_proj"})
    soft_blend_max_error: float = 1.3
    hard_swap_advantage_threshold: float = 0.05


@dataclass
class UnifiedMergeConfig:
    """
    Configuration for unified manifold merging.

    This configuration controls all aspects of the merge pipeline including:
    - Core alignment parameters
    - Probing system (semantic primes, sequence invariants, metaphors, genealogy)
    - Intersection map similarity modes
    - Layer mapping strategies
    - Domain signal-based alpha adjustment
    - Transition and consistency gating
    - Integration with specialized mergers (transport-guided, affine stitching)
    - Verb/noun dimension decomposition
    """

    # ==========================================================================
    # Core Parameters
    # ==========================================================================
    alignment_rank: int = 32
    base_alpha: float = 0.5  # Fallback when confidence unavailable

    # ==========================================================================
    # Confidence Thresholds
    # ==========================================================================
    permutation_confidence_threshold: float = 0.6
    rotation_confidence_threshold: float = 0.4

    # ==========================================================================
    # Permutation & Relational Structure
    # ==========================================================================
    use_anchor_activation_permutation: bool = False
    use_relational_structure_gate: bool = False

    # ==========================================================================
    # Layer Mapping
    # ==========================================================================
    layer_mapping_strategy: LayerMappingStrategy = LayerMappingStrategy.CRM
    layer_match_category_weights: Optional[AnchorCategoryWeights] = None
    invariant_layer_mapping_config: InvariantLayerMapperConfig = field(
        default_factory=InvariantLayerMapperConfig
    )

    # ==========================================================================
    # Topology
    # ==========================================================================
    topology_blend_strength: float = 0.0

    # ==========================================================================
    # Probing System
    # ==========================================================================
    use_enriched_primes: bool = True
    include_sequence_invariants: bool = False
    include_metaphor_invariants: bool = False
    include_conceptual_genealogy: bool = False
    sequence_families_for_probing: Optional[set] = None  # Set[SequenceFamily]
    conceptual_genealogy_weight: float = 0.25
    probe_layer_sample_count: Optional[int] = None

    # ==========================================================================
    # Anchor & SVD
    # ==========================================================================
    enable_anchor_spanning_analysis: bool = True
    use_anchor_basis_for_svd: bool = False

    # ==========================================================================
    # Module Scope & Policy
    # ==========================================================================
    module_scope: ModuleScope = ModuleScope.ALL
    use_module_blend_policy: bool = True
    module_blend_policy: ModuleBlendPolicy = field(default_factory=ModuleBlendPolicy)

    # ==========================================================================
    # MLP Internal Projections
    # ==========================================================================
    mlp_internal_gate_strength: float = 0.0
    mlp_internal_intersection_mode: MLPInternalIntersectionMode = (
        MLPInternalIntersectionMode.FULL
    )
    mlp_gate_intersection_mode: Optional[MLPInternalIntersectionMode] = None
    mlp_up_intersection_mode: Optional[MLPInternalIntersectionMode] = None

    # ==========================================================================
    # Dimension Blending
    # ==========================================================================
    use_dimension_blending: bool = True
    dimension_blend_threshold: float = 0.3

    # ==========================================================================
    # Adaptive Alpha Smoothing
    # ==========================================================================
    use_adaptive_alpha_smoothing: bool = True
    adaptive_alpha_smoothing_window: int = 2
    min_alpha: float = 0.1
    max_alpha: float = 0.95

    # ==========================================================================
    # Fisher Blending
    # ==========================================================================
    fisher_blend_strength: float = 0.0
    fisher_epsilon: float = 1e-6

    # ==========================================================================
    # Spectral Penalty
    # ==========================================================================
    spectral_penalty_strength: float = 0.5
    spectral_epsilon: float = 1e-6

    # ==========================================================================
    # Domain Signal Blending
    # ==========================================================================
    domain_signal_strength: float = 0.0
    domain_signal_sparsity_weight: float = 0.5
    domain_signal_smoothness_weight: float = 0.5
    domain_signal_epsilon: float = 1e-6
    domain_signal_prompt_target: int = 8
    domain_signal_gradient_sample_target: int = 8
    domain_signal_min_alpha: float = 0.2
    domain_signal_max_alpha: float = 0.95

    # ==========================================================================
    # Transition Gating
    # ==========================================================================
    transition_gate_strength: float = 0.0
    transition_gate_min_ratio: float = 0.7
    transition_gate_max_ratio: float = 1.3

    # ==========================================================================
    # Consistency Gating
    # ==========================================================================
    consistency_gate_strength: float = 0.0
    consistency_gate_layer_sample_count: int = 6

    # ==========================================================================
    # Tangent Space Alignment
    # ==========================================================================
    use_tangent_space_alignment: bool = False
    tangent_space_config: TangentSpaceConfig = field(
        default_factory=TangentSpaceConfig
    )

    # ==========================================================================
    # Shared Subspace Projection
    # ==========================================================================
    use_shared_subspace_projection: bool = False
    shared_subspace_config: SharedSubspaceConfig = field(
        default_factory=SharedSubspaceConfig
    )
    shared_subspace_blend_weight: float = 0.5

    # ==========================================================================
    # Gromov-Wasserstein
    # ==========================================================================
    gromov_wasserstein_blend_strength: float = 0.0
    gromov_wasserstein_min_score: float = 0.3

    # ==========================================================================
    # Transport-Guided Merging
    # ==========================================================================
    use_transport_guided: bool = False
    transport_guided_config: TransportGuidedConfig = field(
        default_factory=TransportGuidedConfig
    )

    # ==========================================================================
    # Affine Stitching
    # ==========================================================================
    use_affine_stitching: bool = False
    affine_stitching_config: AffineStitchingConfig = field(
        default_factory=AffineStitchingConfig
    )

    # ==========================================================================
    # Verb/Noun Dimension Decomposition
    # ==========================================================================
    use_verb_noun_decomposition: bool = False
    verb_noun_config: VerbNounConfig = field(default_factory=VerbNounConfig)
    verb_noun_blend_strength: float = 0.7

    # ==========================================================================
    # Intersection Similarity Mode
    # ==========================================================================
    intersection_similarity_mode: IntersectionSimilarityMode = (
        IntersectionSimilarityMode.JACCARD
    )
    intersection_ensemble_weights: Optional[IntersectionEnsembleWeights] = None
    intersection_correlation_threshold: float = 0.3

    # ==========================================================================
    # Gradient Boundary Smoothing (Gap 3)
    # ==========================================================================
    use_gradient_boundary_smoothing: bool = False
    """Enable gradient-aware smoothing at layer boundaries."""

    gradient_snr_discontinuity_threshold: float = 0.5
    """SNR difference threshold to flag boundary as discontinuous."""

    gradient_base_smoothing_sigma: float = 1.0
    """Base sigma for Gaussian smoothing of alpha profile."""

    gradient_max_smoothing_multiplier: float = 3.0
    """Maximum multiplier for sigma at discontinuity points."""

    gradient_use_hessian_penalty: bool = True
    """Whether to incorporate Hessian curvature in boundary detection."""

    gradient_adjustment_strength: float = 0.2
    """How much to adjust alpha based on gradient SNR (0 = none, 1 = full)."""

    # ==========================================================================
    # Presets
    # ==========================================================================

    @classmethod
    def conservative(cls) -> "UnifiedMergeConfig":
        """Preset for conservative merging (prefer target in uncertain regions)."""
        return cls(
            base_alpha=0.7,
            permutation_confidence_threshold=0.7,
            rotation_confidence_threshold=0.5,
            spectral_penalty_strength=0.5,
        )

    @classmethod
    def aggressive(cls) -> "UnifiedMergeConfig":
        """Preset for aggressive merging (trust source more)."""
        return cls(
            alignment_rank=48,
            base_alpha=0.3,
            permutation_confidence_threshold=0.4,
            rotation_confidence_threshold=0.3,
            spectral_penalty_strength=0.3,
        )

    @classmethod
    def transport_guided(cls) -> "UnifiedMergeConfig":
        """Preset for transport-guided merging using Gromov-Wasserstein."""
        return cls(
            use_transport_guided=True,
            gromov_wasserstein_blend_strength=0.7,
            use_dimension_blending=True,
            use_adaptive_alpha_smoothing=True,
        )

    @classmethod
    def verb_noun_aware(cls) -> "UnifiedMergeConfig":
        """Preset for verb/noun-aware merging based on semantic primes."""
        return cls(
            use_verb_noun_decomposition=True,
            verb_noun_blend_strength=0.7,
            use_enriched_primes=True,
            include_sequence_invariants=True,
        )

    @classmethod
    def full_geometric(cls) -> "UnifiedMergeConfig":
        """
        Preset enabling all geometric features.

        Combines transport-guided, affine stitching, verb/noun decomposition,
        domain signals, gradient boundary smoothing, and consistency gating
        for maximum alignment quality.
        """
        return cls(
            alignment_rank=48,
            use_transport_guided=True,
            use_affine_stitching=True,
            use_verb_noun_decomposition=True,
            use_shared_subspace_projection=True,
            domain_signal_strength=0.5,
            transition_gate_strength=0.3,
            consistency_gate_strength=0.3,
            use_enriched_primes=True,
            include_sequence_invariants=True,
            include_metaphor_invariants=True,
            intersection_similarity_mode=IntersectionSimilarityMode.ENSEMBLE,
            use_gradient_boundary_smoothing=True,
            gradient_adjustment_strength=0.2,
        )

    @classmethod
    def gradient_smoothed(cls) -> "UnifiedMergeConfig":
        """Preset enabling gradient-aware boundary smoothing."""
        return cls(
            use_gradient_boundary_smoothing=True,
            gradient_snr_discontinuity_threshold=0.5,
            gradient_base_smoothing_sigma=1.5,
            gradient_max_smoothing_multiplier=3.0,
            gradient_adjustment_strength=0.3,
            use_adaptive_alpha_smoothing=True,
        )


# =============================================================================
# Layer Alpha Profile
# =============================================================================


@dataclass
class LayerAlphaProfile:
    """
    Per-layer alpha profile with smoothing to prevent "tearing".
    
    Instead of computing alpha independently for each layer, this profile:
    1. Computes raw alpha based on confidence and Procrustes error
    2. Applies Gaussian smoothing across adjacent layers
    3. Clamps values to prevent extreme blending
    """
    alpha_by_layer: Dict[int, float]
    smoothing_window: int
    base_alpha: float
    used_procrustes_error: bool
    
    @property
    def mean_alpha(self) -> float:
        """Average alpha across all layers."""
        values = list(self.alpha_by_layer.values())
        if not values:
            return self.base_alpha
        return sum(values) / len(values)
    
    @property
    def alpha_variance(self) -> float:
        """Alpha variance (smoothing effectiveness indicator)."""
        values = list(self.alpha_by_layer.values())
        if len(values) <= 1:
            return 0.0
        mean = self.mean_alpha
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def alpha(self, layer: int) -> float:
        """Gets alpha for a specific layer, falling back to base alpha."""
        return self.alpha_by_layer.get(layer, self.base_alpha)


def compute_adaptive_alpha_profile(
    layer_confidences: Dict[int, float],
    base_alpha: float = 0.5,
    smoothing_window: int = 2,
    procrustes_error_by_layer: Optional[Dict[int, float]] = None,
    min_alpha: float = 0.1,
    max_alpha: float = 0.95,
) -> LayerAlphaProfile:
    """
    Computes an adaptive alpha profile with Gaussian smoothing.
    
    Alpha Formula (per layer):
        raw_alpha = 1.0 - (confidence * 0.7)  # High confidence → trust source
        
        If Procrustes error available:
            error_adjustment = clamp(procrustes_error * 0.5, 0, 0.3)
            raw_alpha += error_adjustment  # Higher error → trust target more
    
    Smoothing: Gaussian weighted average across adjacent layers prevents
    sharp alpha transitions that cause tearing artifacts.
    
    Args:
        layer_confidences: Per-layer confidence values (from intersection map)
        base_alpha: Fallback alpha for layers without data
        smoothing_window: Number of layers on each side for smoothing
        procrustes_error_by_layer: Optional per-layer Procrustes error
        min_alpha: Minimum allowed alpha
        max_alpha: Maximum allowed alpha
    
    Returns:
        Smoothed per-layer alpha profile
    """
    if not layer_confidences:
        return LayerAlphaProfile(
            alpha_by_layer={},
            smoothing_window=smoothing_window,
            base_alpha=base_alpha,
            used_procrustes_error=procrustes_error_by_layer is not None,
        )
    
    # Step 1: Compute raw alpha for each layer
    raw_alphas: Dict[int, float] = {}
    
    for layer, confidence in layer_confidences.items():
        # Base formula: high confidence → lower alpha → trust source more
        alpha = 1.0 - (confidence * 0.7)
        
        # Incorporate Procrustes error if available
        if procrustes_error_by_layer and layer in procrustes_error_by_layer:
            error = procrustes_error_by_layer[layer]
            # Higher error → increase alpha → trust target more
            error_adjustment = min(0.3, error * 0.5)
            alpha += error_adjustment
        
        # Clamp before smoothing
        raw_alphas[layer] = max(min_alpha, min(max_alpha, alpha))
    
    if smoothing_window <= 0:
        return LayerAlphaProfile(
            alpha_by_layer=raw_alphas,
            smoothing_window=smoothing_window,
            base_alpha=base_alpha,
            used_procrustes_error=procrustes_error_by_layer is not None,
        )
    
    # Step 2: Apply Gaussian smoothing
    sorted_layers = sorted(raw_alphas.keys())
    smoothed_alphas: Dict[int, float] = {}
    
    # Precompute Gaussian weights
    sigma = smoothing_window / 2.0
    gaussian_weights = []
    for offset in range(-smoothing_window, smoothing_window + 1):
        weight = math.exp(-(offset * offset) / (2 * sigma * sigma))
        gaussian_weights.append(weight)
    weight_sum = sum(gaussian_weights)
    gaussian_weights = [w / weight_sum for w in gaussian_weights]
    
    for layer in sorted_layers:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for offset_idx, offset in enumerate(range(-smoothing_window, smoothing_window + 1)):
            neighbor_layer = layer + offset
            if neighbor_layer in raw_alphas:
                weight = gaussian_weights[offset_idx]
                weighted_sum += raw_alphas[neighbor_layer] * weight
                total_weight += weight
        
        # Normalize and clamp
        fallback_alpha = raw_alphas.get(layer, base_alpha)
        smoothed_alpha = weighted_sum / total_weight if total_weight > 0 else fallback_alpha
        smoothed_alphas[layer] = max(min_alpha, min(max_alpha, smoothed_alpha))
    
    logger.debug(
        f"Alpha profile: {len(smoothed_alphas)} layers, "
        f"mean={sum(smoothed_alphas.values())/len(smoothed_alphas):.2f}, "
        f"window={smoothing_window}"
    )
    
    return LayerAlphaProfile(
        alpha_by_layer=smoothed_alphas,
        smoothing_window=smoothing_window,
        base_alpha=base_alpha,
        used_procrustes_error=procrustes_error_by_layer is not None,
    )


# =============================================================================
# Spectral Penalty
# =============================================================================


def compute_spectral_penalty(
    weight: "Array",
    epsilon: float = 1e-6,
    backend: "Backend | None" = None,
) -> float:
    """
    Computes spectral penalty based on condition number.

    High condition number indicates near-singular matrices that are
    unreliable for rotation. Returns value in [0, 1] where:
    - 0 = well-conditioned (low condition number, trustworthy)
    - 1 = ill-conditioned (high condition number, untrustworthy)

    Args:
        weight: Weight matrix to analyze
        epsilon: Numerical stability epsilon
        backend: Backend for array operations

    Returns:
        Penalty value in [0, 1]
    """
    b = backend or get_default_backend()

    if weight.ndim != 2:
        return 0.0

    try:
        # Compute singular values
        _, s, _ = b.svd(b.astype(weight, "float32"))
        b.eval(s)
        s_list = b.to_numpy(s).tolist()

        if not s_list:
            return 0.0

        s_max = max(s_list)
        s_min = min(abs(x) for x in s_list if abs(x) > epsilon)

        if s_min < epsilon:
            return 1.0  # Ill-conditioned

        condition_number = s_max / s_min

        # Map condition number to penalty:
        # - < 10: penalty ≈ 0
        # - 10-100: penalty scales linearly
        # - > 100: penalty ≈ 1
        normalized = (math.log10(max(condition_number, 1.0)) - 1.0) / 1.0
        return max(0.0, min(1.0, normalized))

    except Exception:
        return 0.5  # Default to moderate penalty on error


def apply_spectral_penalty_to_alpha(
    alpha: float,
    source_weight: "Array",
    target_weight: "Array",
    strength: float = 0.5,
    backend: "Backend | None" = None,
) -> float:
    """
    Adjusts alpha based on spectral properties of weights.

    If source has high condition number (ill-conditioned), increase alpha
    to trust target more. If target has high condition number, decrease
    alpha to trust source more.

    Args:
        alpha: Base alpha value
        source_weight: Source model weight
        target_weight: Target model weight
        strength: Strength of penalty effect (0 = disabled, 1 = full)
        backend: Backend for array operations

    Returns:
        Adjusted alpha value
    """
    if strength <= 0:
        return alpha

    source_penalty = compute_spectral_penalty(source_weight, backend=backend)
    target_penalty = compute_spectral_penalty(target_weight, backend=backend)

    # If source is ill-conditioned, push toward target (increase alpha)
    # If target is ill-conditioned, push toward source (decrease alpha)
    penalty_diff = source_penalty - target_penalty
    adjustment = penalty_diff * strength * 0.3

    adjusted_alpha = max(0.1, min(0.95, alpha + adjustment))
    return adjusted_alpha


# =============================================================================
# Transition and Consistency Gating (Phase 3 Parity)
# =============================================================================


@dataclass
class TransitionContext:
    """
    Context for transition-based alpha adjustment.

    Transition alignment measures how well the model's behavior changes
    (transitions) align between source and target, compared to static
    state alignment. If transitions align better, we can trust geometric
    alignment more; if states align better, be more conservative.
    """

    # Mean CKA between source and target transition activations
    mean_transition_cka: float

    # Mean CKA between source and target state activations
    mean_state_cka: float

    # Advantage of transition alignment over state alignment
    transition_advantage: float

    # Whether transitions align better than states
    transition_better_than_state: bool

    # Number of transitions analyzed
    transition_count: int

    # Per-layer delta alignment ratio (transition_cka / state_cka)
    # Values > 1 mean transitions align better for that layer
    delta_alignment_by_layer: Dict[int, float]

    @staticmethod
    def compute(
        transition_ckas: Dict[int, float],
        state_ckas: Dict[int, float],
    ) -> "TransitionContext":
        """
        Compute transition context from per-layer CKA values.

        Args:
            transition_ckas: Per-layer CKA for transition activations
            state_ckas: Per-layer CKA for state activations

        Returns:
            TransitionContext with computed metrics
        """
        if not transition_ckas or not state_ckas:
            return TransitionContext(
                mean_transition_cka=0.0,
                mean_state_cka=0.0,
                transition_advantage=0.0,
                transition_better_than_state=False,
                transition_count=0,
                delta_alignment_by_layer={},
            )

        common_layers = set(transition_ckas.keys()) & set(state_ckas.keys())
        if not common_layers:
            return TransitionContext(
                mean_transition_cka=0.0,
                mean_state_cka=0.0,
                transition_advantage=0.0,
                transition_better_than_state=False,
                transition_count=0,
                delta_alignment_by_layer={},
            )

        mean_transition = sum(transition_ckas[l] for l in common_layers) / len(common_layers)
        mean_state = sum(state_ckas[l] for l in common_layers) / len(common_layers)

        delta_by_layer = {}
        for layer in common_layers:
            state_cka = state_ckas[layer]
            trans_cka = transition_ckas[layer]
            # Ratio: > 1 means transitions align better
            ratio = trans_cka / (state_cka + 1e-8)
            delta_by_layer[layer] = ratio

        return TransitionContext(
            mean_transition_cka=mean_transition,
            mean_state_cka=mean_state,
            transition_advantage=mean_transition - mean_state,
            transition_better_than_state=mean_transition > mean_state,
            transition_count=len(common_layers),
            delta_alignment_by_layer=delta_by_layer,
        )


def transition_adjusted_alpha(
    base_alpha: float,
    ctx: TransitionContext,
    layer: int,
    strength: float = 0.5,
    min_ratio: float = 0.7,
    max_ratio: float = 1.3,
) -> float:
    """
    Adjust alpha based on transition alignment context.

    Algorithm (from Swift lines 819-839):
        ratio = delta_alignment_by_layer[layer]
        if ratio < min_ratio:
            adjustment = (ratio - min_ratio) / (1 - min_ratio)  # Push toward target
        elif ratio > max_ratio:
            adjustment = (ratio - max_ratio) / (2 - max_ratio)  # Push toward source
        else:
            adjustment = 0

        adjusted_alpha = base_alpha + adjustment * strength

    Args:
        base_alpha: Starting alpha value
        ctx: Transition context with per-layer alignment ratios
        layer: Layer index to adjust for
        strength: How much to apply transition adjustment [0, 1]
        min_ratio: Minimum ratio threshold (below = transitions misaligned)
        max_ratio: Maximum ratio threshold (above = transitions well aligned)

    Returns:
        Adjusted alpha value
    """
    if strength <= 0 or layer not in ctx.delta_alignment_by_layer:
        return base_alpha

    ratio = ctx.delta_alignment_by_layer[layer]
    adjustment = 0.0

    if ratio < min_ratio:
        # Transitions misaligned → push toward target (increase alpha)
        # Normalize to [0, 1] range for adjustment magnitude
        adjustment = (min_ratio - ratio) / max(0.01, 1.0 - min_ratio)
        adjustment = min(1.0, adjustment)  # Cap at 1.0
    elif ratio > max_ratio:
        # Transitions well aligned → push toward source (decrease alpha)
        adjustment = -(ratio - max_ratio) / max(0.01, 2.0 - max_ratio)
        adjustment = max(-1.0, adjustment)  # Cap at -1.0

    adjusted_alpha = base_alpha + adjustment * strength
    return max(0.1, min(0.95, adjusted_alpha))


@dataclass
class ConsistencyContext:
    """
    Context for layer-consistency-based alpha adjustment.

    Measures how consistent the anchor responses are within each model,
    and uses this to determine which model to trust more for each layer.
    """

    # Number of anchors used for consistency measurement
    anchor_count: int

    # Number of layers sampled
    sample_layer_count: int

    # Mean distance between anchor pairs in source model
    mean_source_distance: float

    # Mean distance between anchor pairs in target model
    mean_target_distance: float

    # Per-layer weight for target (higher = trust target more)
    # Based on relative consistency: more consistent model is trusted
    target_weight_by_layer: Dict[int, float]

    @staticmethod
    def compute(
        source_distances: Dict[int, float],
        target_distances: Dict[int, float],
    ) -> "ConsistencyContext":
        """
        Compute consistency context from per-layer anchor distances.

        Lower distance = more consistent = more trustworthy.

        Args:
            source_distances: Per-layer mean distance between anchors in source
            target_distances: Per-layer mean distance between anchors in target

        Returns:
            ConsistencyContext with computed metrics
        """
        if not source_distances or not target_distances:
            return ConsistencyContext(
                anchor_count=0,
                sample_layer_count=0,
                mean_source_distance=0.0,
                mean_target_distance=0.0,
                target_weight_by_layer={},
            )

        common_layers = set(source_distances.keys()) & set(target_distances.keys())
        if not common_layers:
            return ConsistencyContext(
                anchor_count=0,
                sample_layer_count=len(source_distances),
                mean_source_distance=0.0,
                mean_target_distance=0.0,
                target_weight_by_layer={},
            )

        mean_source = sum(source_distances[l] for l in common_layers) / len(common_layers)
        mean_target = sum(target_distances[l] for l in common_layers) / len(common_layers)

        target_weight_by_layer = {}
        for layer in common_layers:
            src_dist = source_distances[layer]
            tgt_dist = target_distances[layer]
            # Lower distance = more consistent = more trustworthy
            # Sigmoid-like mapping: if target more consistent, weight > 0.5
            total = src_dist + tgt_dist + 1e-8
            target_weight = src_dist / total  # Higher source distance → trust target more
            target_weight_by_layer[layer] = target_weight

        return ConsistencyContext(
            anchor_count=0,  # Will be filled by caller if available
            sample_layer_count=len(common_layers),
            mean_source_distance=mean_source,
            mean_target_distance=mean_target,
            target_weight_by_layer=target_weight_by_layer,
        )


def consistency_adjusted_alpha(
    base_alpha: float,
    ctx: ConsistencyContext,
    layer: int,
    strength: float = 0.5,
) -> float:
    """
    Adjust alpha based on layer consistency context.

    Formula:
        target_weight = ctx.target_weight_by_layer[layer]  # [0, 1]
        adjustment = (target_weight - 0.5) * 2 * strength
        adjusted_alpha = base_alpha + adjustment

    Higher target_weight means target is more consistent at this layer,
    so we increase alpha to trust target more.

    Args:
        base_alpha: Starting alpha value
        ctx: Consistency context with per-layer target weights
        layer: Layer index to adjust for
        strength: How much to apply consistency adjustment [0, 1]

    Returns:
        Adjusted alpha value
    """
    if strength <= 0 or layer not in ctx.target_weight_by_layer:
        return base_alpha

    target_weight = ctx.target_weight_by_layer[layer]

    # Map target_weight [0, 1] to adjustment [-strength, +strength]
    # 0.5 = neutral, < 0.5 = trust source, > 0.5 = trust target
    adjustment = (target_weight - 0.5) * 2.0 * strength

    adjusted_alpha = base_alpha + adjustment
    return max(0.1, min(0.95, adjusted_alpha))


# =============================================================================
# MLP Internal Intersection Modes (Phase 6 Parity)
# =============================================================================


class ModuleKind(str, Enum):
    """Classification of weight matrix types."""

    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    O_PROJ = "o_proj"
    GATE_PROJ = "gate_proj"
    UP_PROJ = "up_proj"
    DOWN_PROJ = "down_proj"
    UNKNOWN = "unknown"

    @property
    def is_attention(self) -> bool:
        return self in {ModuleKind.Q_PROJ, ModuleKind.K_PROJ, ModuleKind.V_PROJ, ModuleKind.O_PROJ}

    @property
    def is_mlp(self) -> bool:
        return self in {ModuleKind.GATE_PROJ, ModuleKind.UP_PROJ, ModuleKind.DOWN_PROJ}

    @property
    def is_mlp_internal(self) -> bool:
        """Gate and Up projections (internal to MLP, before down projection)."""
        return self in {ModuleKind.GATE_PROJ, ModuleKind.UP_PROJ}

    @property
    def is_residual_output(self) -> bool:
        """Modules that output to the residual stream."""
        return self in {ModuleKind.O_PROJ, ModuleKind.DOWN_PROJ}


def classify_module_kind(key: str) -> ModuleKind:
    """
    Classify a weight key into its module kind.

    Args:
        key: Weight key like "model.layers.5.mlp.gate_proj.weight"

    Returns:
        ModuleKind classification
    """
    key_lower = key.lower()

    if "q_proj" in key_lower:
        return ModuleKind.Q_PROJ
    if "k_proj" in key_lower:
        return ModuleKind.K_PROJ
    if "v_proj" in key_lower:
        return ModuleKind.V_PROJ
    if "o_proj" in key_lower:
        return ModuleKind.O_PROJ
    if "gate_proj" in key_lower or "gate" in key_lower:
        return ModuleKind.GATE_PROJ
    if "up_proj" in key_lower:
        return ModuleKind.UP_PROJ
    if "down_proj" in key_lower:
        return ModuleKind.DOWN_PROJ

    return ModuleKind.UNKNOWN


@dataclass
class MLPInternalGatingResult:
    """Result of MLP internal gating decision."""

    key: str
    module_kind: ModuleKind
    original_alpha: float
    gated_alpha: float
    gating_applied: bool
    intersection_mode_used: MLPInternalIntersectionMode


def apply_mlp_internal_gating(
    key: str,
    base_alpha: float,
    config: "UnifiedMergeConfig",
    intersection_confidences: Optional[Dict[int, float]] = None,
    invariant_confidences: Optional[Dict[int, float]] = None,
    logic_confidences: Optional[Dict[int, float]] = None,
) -> MLPInternalGatingResult:
    """
    Apply MLP internal gating to adjust alpha for gate/up projections.

    MLP internal projections (gate, up) may need different treatment than
    output projections to prevent disrupting learned gate patterns.

    Args:
        key: Weight key
        base_alpha: Starting alpha value
        config: Merge configuration
        intersection_confidences: Full intersection map confidences by layer
        invariant_confidences: Invariant-only confidences by layer
        logic_confidences: Logic-invariant-only confidences by layer

    Returns:
        MLPInternalGatingResult with gating decision
    """
    module_kind = classify_module_kind(key)

    # Only apply gating to MLP internal projections
    if not module_kind.is_mlp_internal:
        return MLPInternalGatingResult(
            key=key,
            module_kind=module_kind,
            original_alpha=base_alpha,
            gated_alpha=base_alpha,
            gating_applied=False,
            intersection_mode_used=MLPInternalIntersectionMode.FULL,
        )

    # Determine which intersection mode to use for this module
    if module_kind == ModuleKind.GATE_PROJ and config.mlp_gate_intersection_mode:
        mode = config.mlp_gate_intersection_mode
    elif module_kind == ModuleKind.UP_PROJ and config.mlp_up_intersection_mode:
        mode = config.mlp_up_intersection_mode
    else:
        mode = config.mlp_internal_intersection_mode

    # Select confidence source based on mode
    layer = _extract_layer_from_key(key)

    if mode == MLPInternalIntersectionMode.FULL:
        confidences = intersection_confidences
    elif mode == MLPInternalIntersectionMode.INVARIANTS:
        confidences = invariant_confidences or intersection_confidences
    elif mode == MLPInternalIntersectionMode.LOGIC_ONLY:
        confidences = logic_confidences or invariant_confidences or intersection_confidences
    else:
        confidences = intersection_confidences

    # Get confidence for this layer
    confidence = confidences.get(layer, 0.5) if confidences else 0.5

    # Apply gating strength
    # Higher confidence → trust alignment → allow lower alpha (trust source)
    # Lower confidence → be conservative → push toward target (higher alpha)
    gate_strength = config.mlp_internal_gate_strength
    if gate_strength <= 0:
        return MLPInternalGatingResult(
            key=key,
            module_kind=module_kind,
            original_alpha=base_alpha,
            gated_alpha=base_alpha,
            gating_applied=False,
            intersection_mode_used=mode,
        )

    # Conservative gating: low confidence → increase alpha toward target
    # confidence [0, 1] → gating adjustment [+gate_strength, 0]
    gating_adjustment = (1.0 - confidence) * gate_strength
    gated_alpha = min(0.95, base_alpha + gating_adjustment)

    return MLPInternalGatingResult(
        key=key,
        module_kind=module_kind,
        original_alpha=base_alpha,
        gated_alpha=gated_alpha,
        gating_applied=True,
        intersection_mode_used=mode,
    )


def _extract_layer_from_key(key: str) -> int:
    """Extract layer index from weight key."""
    import re

    match = re.search(r"layers\.(\d+)", key)
    if match:
        return int(match.group(1))
    return 0


def filter_weights_by_module_scope(
    weights: Dict[str, Any],
    scope: ModuleScope,
) -> Dict[str, Any]:
    """
    Filter weights to only include modules matching the scope.

    Args:
        weights: All weights
        scope: Module scope to filter by

    Returns:
        Filtered weights dictionary
    """
    if scope == ModuleScope.ALL:
        return weights

    filtered = {}
    for key, value in weights.items():
        kind = classify_module_kind(key)

        if scope == ModuleScope.ATTENTION_ONLY and kind.is_attention:
            filtered[key] = value
        elif scope == ModuleScope.MLP_ONLY and kind.is_mlp:
            filtered[key] = value
        elif scope == ModuleScope.MLP_INTERNAL and kind.is_mlp_internal:
            filtered[key] = value
        elif scope == ModuleScope.ATTENTION_AND_MLP_DOWN:
            if kind.is_attention or kind == ModuleKind.DOWN_PROJ:
                filtered[key] = value

    return filtered


# =============================================================================
# Dimension Blending Weights
# =============================================================================


@dataclass
class DimensionBlendingWeights:
    """
    Per-dimension blending weights based on intersection correlations.

    Instead of a single scalar alpha, this uses per-dimension weights
    derived from how well dimensions correlate between source and target.
    """
    weights: Any  # Array type: [hidden_dim] or [out_dim, in_dim]
    threshold: float
    mean_weight: float
    covered_fraction: float  # Fraction of dimensions with high correlation


def compute_dimension_blending_weights(
    source_activations: "Array",
    target_activations: "Array",
    threshold: float = 0.3,
    fallback_weight: float = 0.5,
    backend: "Backend | None" = None,
) -> DimensionBlendingWeights:
    """
    Computes per-dimension blending weights from activation correlations.

    High correlation → trust source more (lower weight)
    Low correlation → trust target more (higher weight)

    Args:
        source_activations: Source model activations [samples, hidden_dim]
        target_activations: Target model activations [samples, hidden_dim]
        threshold: Correlation threshold for "high confidence"
        fallback_weight: Weight for dimensions below threshold
        backend: Backend for array operations

    Returns:
        DimensionBlendingWeights with per-dimension values
    """
    b = backend or get_default_backend()

    if source_activations.shape != target_activations.shape:
        raise ValueError("Activation shapes must match")

    hidden_dim = source_activations.shape[-1]

    # Normalize
    source_norm = source_activations - b.mean(source_activations, axis=0, keepdims=True)
    target_norm = target_activations - b.mean(target_activations, axis=0, keepdims=True)

    source_std = b.sqrt(b.sum(source_norm ** 2, axis=0) + 1e-8)
    target_std = b.sqrt(b.sum(target_norm ** 2, axis=0) + 1e-8)

    # Per-dimension correlation
    correlations = b.sum(source_norm * target_norm, axis=0) / (source_std * target_std)
    b.eval(correlations)

    corr_list = b.to_numpy(correlations).tolist()

    # Convert correlation to weight:
    # High correlation (>threshold) → trust source → lower alpha
    # Low correlation (<threshold) → trust target → higher alpha
    weights = []
    high_conf_count = 0

    for corr in corr_list:
        abs_corr = abs(corr)
        if abs_corr >= threshold:
            # High correlation: weight = 1 - abs_corr (low = trust source)
            weight = max(0.1, 1.0 - abs_corr)
            high_conf_count += 1
        else:
            weight = fallback_weight
        weights.append(weight)

    weights_array = b.astype(b.array(weights), "float32")
    mean_weight = sum(weights) / len(weights)
    covered_fraction = high_conf_count / hidden_dim

    return DimensionBlendingWeights(
        weights=weights_array,
        threshold=threshold,
        mean_weight=mean_weight,
        covered_fraction=covered_fraction,
    )


# =============================================================================
# Unified Manifold Merger
# =============================================================================


@dataclass
class UnifiedMergeResult:
    """Result of unified manifold merging."""
    merged_weights: Dict[str, Any]  # Array type from backend
    alpha_profile: LayerAlphaProfile
    layers_merged: int
    mean_alpha: float
    spectral_penalty_applied: bool
    dimension_blending_applied: bool
    # Integration tracking flags
    verb_noun_applied: bool = False
    transport_guided_applied: bool = False
    affine_stitching_applied: bool = False
    shared_subspace_applied: bool = False
    transition_gating_applied: bool = False
    consistency_gating_applied: bool = False
    domain_signal_applied: bool = False
    gradient_boundary_smoothing_applied: bool = False
    # Entropy validation (optional, from EntropyMergeValidator)
    entropy_validation: Optional["MergeEntropyValidation"] = None


class UnifiedManifoldMerger:
    """
    Unified Manifold Merger combining geometric alignment methods.
    
    Pipeline Stages:
    1. PROBE - Run semantic primes through models → Intersection map
    2. PERMUTE - Use intersection to guide Re-Basin on MLP neurons
    3. ROTATE - Apply Procrustes on strongly-correlated dimensions
    4. BLEND - Use intersection confidence as per-layer adaptive alpha
    5. PROPAGATE - Carry alignment forward geometrically (zipper)
    
    This implementation provides the core blending logic. The full
    probing and intersection mapping should be done externally.
    """
    
    def __init__(
        self,
        config: UnifiedMergeConfig = None,
        backend: "Backend | None" = None,
    ):
        self.config = config or UnifiedMergeConfig()
        self._backend = backend or get_default_backend()
        self._rotational_merger = RotationalModelMerger(
            MergeOptions(
                alignment_rank=self.config.alignment_rank,
                alpha=self.config.base_alpha,
            ),
            backend=self._backend,
        )

    def merge_with_confidence(
        self,
        source_weights: Dict[str, Any],
        target_weights: Dict[str, Any],
        layer_confidences: Dict[int, float],
        procrustes_errors: Optional[Dict[int, float]] = None,
        source_activations: Optional[Dict[int, List[List[float]]]] = None,
        target_activations: Optional[Dict[int, List[List[float]]]] = None,
        transition_context: Optional["TransitionContext"] = None,
        consistency_context: Optional["ConsistencyContext"] = None,
        domain_signal_decisions: Optional[Dict[int, "DomainSignalDecision"]] = None,
    ) -> UnifiedMergeResult:
        """
        Merge weights using confidence-adaptive blending with full integration.

        This method integrates multiple merge strategies:
        - Adaptive alpha profiling with Gaussian smoothing
        - Spectral penalty for rank preservation
        - Verb/noun decomposition for semantic dimension handling
        - Transport-guided Gromov-Wasserstein alignment
        - Affine stitching for cross-architecture merging
        - Shared subspace projection for dimension-reduced blending
        - Transition and consistency gating
        - Domain signal adjustments

        Args:
            source_weights: Source model weights by key
            target_weights: Target model weights by key
            layer_confidences: Per-layer confidence from intersection map
            procrustes_errors: Optional per-layer Procrustes errors
            source_activations: Optional raw activations by layer for advanced methods
            target_activations: Optional raw activations by layer for advanced methods
            transition_context: Optional transition gating context
            consistency_context: Optional consistency gating context
            domain_signal_decisions: Optional per-layer domain signal adjustments

        Returns:
            UnifiedMergeResult with merged weights and diagnostics
        """
        # Compute base adaptive alpha profile
        alpha_profile = compute_adaptive_alpha_profile(
            layer_confidences=layer_confidences,
            base_alpha=self.config.base_alpha,
            smoothing_window=self.config.adaptive_alpha_smoothing_window,
            procrustes_error_by_layer=procrustes_errors,
            min_alpha=self.config.min_alpha,
            max_alpha=self.config.max_alpha,
        )

        merged_weights: Dict[str, Any] = {}
        b = self._backend
        spectral_applied = False
        dimension_blending_applied = False
        verb_noun_applied = False
        transport_guided_applied = False
        affine_stitching_applied = False
        shared_subspace_applied = False
        transition_gating_applied = False
        consistency_gating_applied = False
        domain_signal_applied = False
        gradient_boundary_smoothing_applied = False
        gradient_boundary_profile: Optional["GradientBoundaryProfile"] = None

        # Apply gradient boundary smoothing if enabled
        if self.config.use_gradient_boundary_smoothing:
            try:
                from modelcypher.core.domain.merging.gradient_boundary_smoother import (
                    GradientBoundaryConfig,
                    smooth_merge_boundaries,
                )

                gradient_config = GradientBoundaryConfig(
                    snr_discontinuity_threshold=self.config.gradient_snr_discontinuity_threshold,
                    base_smoothing_sigma=self.config.gradient_base_smoothing_sigma,
                    max_smoothing_multiplier=self.config.gradient_max_smoothing_multiplier,
                    use_hessian_penalty=self.config.gradient_use_hessian_penalty,
                    smoothing_window=self.config.adaptive_alpha_smoothing_window,
                )

                # Apply gradient-aware smoothing to the alpha profile
                smoothed_alpha, gradient_boundary_profile = smooth_merge_boundaries(
                    alpha_by_layer=alpha_profile.alpha_by_layer,
                    per_sample_gradients=None,  # Would require gradient collection during validation
                    config=gradient_config,
                    base_alpha=self.config.base_alpha,
                    min_alpha=self.config.min_alpha,
                    max_alpha=self.config.max_alpha,
                )

                # Update alpha profile with smoothed values
                alpha_profile = LayerAlphaProfile(
                    alpha_by_layer=smoothed_alpha,
                    smoothing_window=alpha_profile.smoothing_window,
                    base_alpha=alpha_profile.base_alpha,
                    used_procrustes_error=alpha_profile.used_procrustes_error,
                )
                gradient_boundary_smoothing_applied = True
                logger.debug(
                    f"Applied gradient boundary smoothing to {len(smoothed_alpha)} layers"
                )

            except ImportError:
                logger.warning(
                    "Gradient boundary smoothing requested but module not available"
                )

        # Pre-compute verb/noun classifications if enabled
        verb_noun_classifications: Optional[Dict[int, "VerbNounClassification"]] = None
        if self.config.use_verb_noun_decomposition and source_activations:
            verb_noun_classifications = self._compute_verb_noun_classifications(
                source_activations, target_activations
            )

        # Pre-compute shared subspace if enabled
        shared_basis: Optional[Any] = None
        if self.config.use_shared_subspace_projection and source_activations and target_activations:
            shared_basis = self._compute_shared_subspace_basis(
                source_activations, target_activations
            )
            if shared_basis is not None:
                shared_subspace_applied = True

        for key in source_weights:
            if key not in target_weights:
                continue

            # Apply module scope filtering if not ALL
            if self.config.module_scope != ModuleScope.ALL:
                module_kind = classify_module_kind(key)
                scope = self.config.module_scope

                # Check if this module should be included based on scope
                should_include = False
                if scope == ModuleScope.ATTENTION_ONLY and module_kind.is_attention:
                    should_include = True
                elif scope == ModuleScope.MLP_ONLY and module_kind.is_mlp:
                    should_include = True
                elif scope == ModuleScope.MLP_INTERNAL and module_kind.is_mlp_internal:
                    should_include = True
                elif scope == ModuleScope.ATTENTION_AND_MLP_DOWN:
                    if module_kind.is_attention or module_kind == ModuleKind.DOWN_PROJ:
                        should_include = True

                if not should_include:
                    # Copy target weight unchanged for non-matching modules
                    merged_weights[key] = target_weights[key]
                    continue

            source_w = source_weights[key]
            target_w = target_weights[key]

            # Extract layer index from key
            layer = self._extract_layer_index(key)
            alpha = alpha_profile.alpha(layer)

            # Apply transition gating if context provided
            if transition_context and self.config.transition_gate_strength > 0:
                alpha = transition_adjusted_alpha(
                    base_alpha=alpha,
                    ctx=transition_context,
                    layer=layer,
                    config=self.config,
                )
                transition_gating_applied = True

            # Apply consistency gating if context provided
            if consistency_context and self.config.consistency_gate_strength > 0:
                alpha = consistency_adjusted_alpha(
                    base_alpha=alpha,
                    ctx=consistency_context,
                    layer=layer,
                    config=self.config,
                )
                consistency_gating_applied = True

            # Apply domain signal adjustments if provided
            if domain_signal_decisions and layer in domain_signal_decisions:
                decision = domain_signal_decisions[layer]
                alpha = decision.adjusted_alpha
                # Clamp after domain signal adjustment to prevent out-of-bounds
                alpha = max(self.config.min_alpha, min(self.config.max_alpha, alpha))
                domain_signal_applied = True

            # Apply MLP internal gating if enabled
            if self.config.mlp_internal_intersection_mode != MLPInternalIntersectionMode.FULL:
                gating_result = apply_mlp_internal_gating(
                    key=key,
                    base_alpha=alpha,
                    config=self.config,
                    intersection_confidences=layer_confidences,
                )
                # Fix: Use correct attribute name (gated_alpha, not adjusted_alpha)
                alpha = gating_result.gated_alpha
                # Clamp after MLP gating to ensure bounded blending
                alpha = max(self.config.min_alpha, min(self.config.max_alpha, alpha))

            # Apply spectral penalty if enabled
            if self.config.spectral_penalty_strength > 0 and source_w.ndim == 2:
                alpha = apply_spectral_penalty_to_alpha(
                    alpha=alpha,
                    source_weight=source_w,
                    target_weight=target_w,
                    strength=self.config.spectral_penalty_strength,
                )
                spectral_applied = True

            # Apply verb/noun modulation if enabled
            if verb_noun_classifications and layer in verb_noun_classifications:
                vn_class = verb_noun_classifications[layer]
                alpha = self._modulate_alpha_with_verb_noun(
                    alpha, vn_class, key
                )
                verb_noun_applied = True

            # Choose merge strategy based on configuration
            if self.config.use_transport_guided and source_w.ndim == 2:
                # Transport-guided Gromov-Wasserstein merge
                merged = self._apply_transport_guided_merge(
                    source_w, target_w, alpha, layer
                )
                transport_guided_applied = True
            elif self.config.use_affine_stitching and source_w.ndim == 2:
                # Affine stitching merge
                merged = self._apply_affine_stitching(
                    source_w, target_w, alpha, layer,
                    source_activations, target_activations
                )
                affine_stitching_applied = True
            elif shared_basis is not None and source_w.ndim == 2:
                # Project through shared subspace
                merged = self._apply_shared_subspace_blend(
                    source_w, target_w, alpha, shared_basis
                )
            elif (
                self.config.use_dimension_blending
                and source_activations
                and target_activations
                and layer in source_activations
                and layer in target_activations
                and source_w.ndim == 2
            ):
                # Per-dimension blending based on activation correlations
                merged = self._apply_dimension_blending(
                    source_w, target_w, alpha, layer,
                    source_activations, target_activations
                )
                if merged is not None:
                    dimension_blending_applied = True
                else:
                    # Fallback to scalar blend
                    merged = alpha * target_w + (1.0 - alpha) * source_w
            else:
                # Standard linear blend
                merged = alpha * target_w + (1.0 - alpha) * source_w

            b.eval(merged)
            merged_weights[key] = merged

        return UnifiedMergeResult(
            merged_weights=merged_weights,
            alpha_profile=alpha_profile,
            layers_merged=len(merged_weights),
            mean_alpha=alpha_profile.mean_alpha,
            spectral_penalty_applied=spectral_applied,
            dimension_blending_applied=dimension_blending_applied,
            verb_noun_applied=verb_noun_applied,
            transport_guided_applied=transport_guided_applied,
            affine_stitching_applied=affine_stitching_applied,
            shared_subspace_applied=shared_subspace_applied,
            transition_gating_applied=transition_gating_applied,
            consistency_gating_applied=consistency_gating_applied,
            domain_signal_applied=domain_signal_applied,
            gradient_boundary_smoothing_applied=gradient_boundary_smoothing_applied,
        )

    def _compute_verb_noun_classifications(
        self,
        source_activations: Dict[int, List[List[float]]],
        target_activations: Optional[Dict[int, List[List[float]]]],
    ) -> Dict[int, "VerbNounClassification"]:
        """Compute verb/noun classifications for each layer."""
        # Lazy import to avoid circular dependency
        try:
            from ..geometry.verb_noun_classifier import (
                VerbNounDimensionClassifier,
                VerbNounConfig,
            )
        except ImportError:
            logger.warning(
                "Verb/noun classifier unavailable, "
                "skipping verb/noun decomposition"
            )
            return {}

        classifications: Dict[int, "VerbNounClassification"] = {}
        config = VerbNounConfig()

        for layer, acts in source_activations.items():
            if not acts:
                continue
            # Convert to appropriate format and classify
            # Note: This is a simplified integration - full version would use
            # actual fingerprint data
            classifications[layer] = VerbNounDimensionClassifier.classify_layer_basic(
                layer_index=layer,
                activations=acts,
                config=config,
            )

        return classifications

    def _modulate_alpha_with_verb_noun(
        self,
        alpha: float,
        classification: "VerbNounClassification",
        key: str,
    ) -> float:
        """Modulate alpha based on verb/noun classification."""
        strength = self.config.verb_noun_blend_strength
        if strength <= 0:
            return alpha

        # Reduce alpha for verb-heavy dimensions (preserve action semantics)
        # Increase alpha for noun-heavy dimensions (transfer entity representations)
        if hasattr(classification, 'verb_score') and hasattr(classification, 'noun_score'):
            verb_bias = classification.verb_score - classification.noun_score
            # Verb-heavy -> reduce blending, noun-heavy -> increase blending
            adjustment = -verb_bias * strength * 0.2
            alpha = max(self.config.min_alpha, min(self.config.max_alpha, alpha + adjustment))

        return alpha

    def _compute_shared_subspace_basis(
        self,
        source_activations: Dict[int, List[List[float]]],
        target_activations: Dict[int, List[List[float]]],
    ) -> Optional[Any]:
        """Compute shared subspace basis for projection."""
        try:
            from ..geometry.shared_subspace_projector import (
                SharedSubspaceProjector,
                Config as SSConfig,
            )
        except ImportError:
            logger.warning(
                "Shared subspace projector unavailable, "
                "skipping subspace projection"
            )
            return None

        # Aggregate activations across layers
        all_source = []
        all_target = []
        for layer in sorted(source_activations.keys()):
            if layer in target_activations:
                all_source.extend(source_activations[layer])
                all_target.extend(target_activations[layer])

        if not all_source or not all_target:
            logger.warning(
                "Insufficient activation data for shared subspace, "
                "skipping subspace projection"
            )
            return None

        # Use shared subspace projector to find common basis
        # This returns None if insufficient data
        ss_config = SSConfig() if self.config.shared_subspace_config is None else self.config.shared_subspace_config
        result = SharedSubspaceProjector.find_shared_basis(
            source_data=all_source,
            target_data=all_target,
            config=ss_config,
        )

        return result.basis if result else None

    def _apply_transport_guided_merge(
        self,
        source_w: Any,
        target_w: Any,
        alpha: float,
        layer: int,
    ) -> Any:
        """Apply transport-guided merge using Gromov-Wasserstein."""
        try:
            from ..geometry.transport_guided_merger import TransportGuidedMerger
        except ImportError:
            logger.warning(
                f"Transport-guided merger unavailable (layer {layer}), "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w

        # Use transport-guided merger for alignment-aware blending
        tg_config = self.config.transport_guided_config
        gw_strength = self.config.gromov_wasserstein_blend_strength

        if gw_strength <= 0:
            logger.debug(
                f"GW blend strength is 0 (layer {layer}), using linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w

        # Compute optimal transport plan and apply
        merged = TransportGuidedMerger.merge_with_transport(
            source_weight=source_w,
            target_weight=target_w,
            alpha=alpha,
            transport_strength=gw_strength,
        )

        if merged is None:
            logger.warning(
                f"Transport-guided merge failed (layer {layer}), "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w
        return merged

    def _apply_affine_stitching(
        self,
        source_w: Any,
        target_w: Any,
        alpha: float,
        layer: int,
        source_activations: Optional[Dict[int, List[List[float]]]],
        target_activations: Optional[Dict[int, List[List[float]]]],
    ) -> Any:
        """Apply affine stitching for cross-architecture merge."""
        try:
            from ..geometry.affine_stitching_layer import (
                AffineStitchingLayer,
                Config as ASConfig,
            )
        except ImportError:
            logger.warning(
                f"Affine stitching layer unavailable (layer {layer}), "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w

        # Need activations to compute stitching transform
        if not source_activations or not target_activations:
            logger.warning(
                f"Affine stitching requires activations (layer {layer}), "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w

        src_acts = source_activations.get(layer)
        tgt_acts = target_activations.get(layer)

        if not src_acts or not tgt_acts:
            logger.warning(
                f"Missing activations for layer {layer}, "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w

        as_config = self.config.affine_stitching_config or ASConfig()

        # Train stitching layer and apply
        result = AffineStitchingLayer.train_and_apply(
            source_activations=src_acts,
            target_activations=tgt_acts,
            source_weight=source_w,
            target_weight=target_w,
            alpha=alpha,
            config=as_config,
        )

        if result is None:
            logger.warning(
                f"Affine stitching training failed (layer {layer}), "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w
        return result.merged_weight

    def _apply_dimension_blending(
        self,
        source_w: Any,
        target_w: Any,
        base_alpha: float,
        layer: int,
        source_activations: Dict[int, List[List[float]]],
        target_activations: Dict[int, List[List[float]]],
    ) -> Optional[Any]:
        """
        Apply per-dimension blending based on activation correlations.

        Uses compute_dimension_blending_weights() to determine per-dimension
        alpha values based on how well each dimension correlates between
        source and target activations.

        Returns:
            Merged weights with per-dimension blending, or None if failed.
        """
        b = self._backend
        src_acts = source_activations.get(layer)
        tgt_acts = target_activations.get(layer)

        if not src_acts or not tgt_acts:
            logger.debug(f"No activations for layer {layer}, skipping dimension blending")
            return None

        try:
            # Convert to array using backend
            src_array = b.astype(b.array(src_acts), "float32")
            tgt_array = b.astype(b.array(tgt_acts), "float32")

            # Ensure shapes match
            if src_array.shape != tgt_array.shape:
                logger.warning(
                    f"Activation shape mismatch for layer {layer}: "
                    f"{src_array.shape} vs {tgt_array.shape}, "
                    "falling back to scalar blend"
                )
                return None

            # Compute per-dimension blending weights
            dim_weights = compute_dimension_blending_weights(
                source_activations=src_array,
                target_activations=tgt_array,
                threshold=self.config.dimension_blend_threshold,
                fallback_weight=base_alpha,
                backend=b,
            )

            # Apply per-dimension weights
            # weights shape: [hidden_dim] - one weight per output dimension
            # source_w shape: [out_dim, in_dim]
            # We apply per-output-dimension blending

            out_dim = source_w.shape[0]
            weight_dim = dim_weights.weights.shape[0]

            if weight_dim != out_dim:
                # Dimensions don't match - try to align or fallback
                logger.debug(
                    f"Dimension mismatch: weights {weight_dim} vs out_dim {out_dim}, "
                    "using scalar blend"
                )
                return None

            # Reshape weights for broadcasting: [out_dim, 1]
            alpha_per_dim = b.reshape(dim_weights.weights, (out_dim, 1))

            # Per-dimension blend: merged[d] = alpha[d] * target[d] + (1-alpha[d]) * source[d]
            merged = alpha_per_dim * target_w + (1.0 - alpha_per_dim) * source_w

            logger.debug(
                f"Dimension blending layer {layer}: "
                f"mean_weight={dim_weights.mean_weight:.3f}, "
                f"coverage={dim_weights.covered_fraction:.2%}"
            )

            return merged

        except Exception as e:
            logger.warning(
                f"Dimension blending failed for layer {layer}: {e}, "
                "falling back to scalar blend"
            )
            return None

    def _apply_shared_subspace_blend(
        self,
        source_w: Any,
        target_w: Any,
        alpha: float,
        shared_basis: Any,
    ) -> Any:
        """Blend weights through shared subspace projection."""
        b = self._backend
        blend_weight = self.config.shared_subspace_blend_weight
        if blend_weight <= 0 or shared_basis is None:
            logger.debug("Shared subspace blend disabled or no basis, using linear blend")
            return alpha * target_w + (1.0 - alpha) * source_w

        # Project source and target onto shared basis, blend, project back
        # source_proj = source_w @ shared_basis
        # target_proj = target_w @ shared_basis
        # blended_proj = alpha * target_proj + (1 - alpha) * source_proj
        # reconstructed = blended_proj @ shared_basis.T

        # Mix projected blend with direct blend
        try:
            source_proj = b.matmul(source_w, shared_basis)
            target_proj = b.matmul(target_w, shared_basis)
            blended_proj = alpha * target_proj + (1.0 - alpha) * source_proj
            subspace_blend = b.matmul(blended_proj, b.transpose(shared_basis))

            direct_blend = alpha * target_w + (1.0 - alpha) * source_w

            return blend_weight * subspace_blend + (1.0 - blend_weight) * direct_blend
        except Exception as e:
            logger.warning(
                f"Shared subspace projection failed: {e}, "
                "falling back to linear blend"
            )
            return alpha * target_w + (1.0 - alpha) * source_w
    
    def _extract_layer_index(self, key: str) -> int:
        """Extract layer index from weight key like 'model.layers.5.mlp.up_proj.weight'."""
        import re
        match = re.search(r'layers\.(\d+)', key)
        if match:
            return int(match.group(1))
        return 0
    
    def compute_blend_mode(self, key: str, procrustes_error: float) -> BlendMode:
        """
        Determines blend mode for a module based on policy.
        
        Args:
            key: Weight key
            procrustes_error: Procrustes alignment error
        
        Returns:
            BlendMode for this module
        """
        if not self.config.use_module_blend_policy:
            return BlendMode.SOFT
        
        policy = self.config.module_blend_policy
        
        # Check module kind
        for kind in policy.skip_kinds:
            if kind in key:
                return BlendMode.SKIP
        
        for kind in policy.hard_swap_kinds:
            if kind in key:
                # Hard swap if error exceeds threshold
                if procrustes_error > policy.soft_blend_max_error:
                    return BlendMode.HARD_SWAP
        
        return BlendMode.SOFT
