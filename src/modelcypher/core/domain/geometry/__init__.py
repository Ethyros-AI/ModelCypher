"""
Geometry domain - geometric analysis and transformation of model representations.

Core modules for analyzing manifold structure, computing alignments, and
understanding the geometric properties of language model weight spaces.
"""
from __future__ import annotations

from typing import Optional

# =============================================================================
# Core Types (from types.py)
# =============================================================================

from .types import (
    # Permutation Aligner
    AlignmentConfig,
    PermutationAlignmentResult,
    RebasinResult,
    # Concept Detection
    ConceptConfiguration,
    DetectedConcept,
    DetectionResult,
    ConceptComparisonResult,
    # Refusal/Safety
    ContrastivePair,
    RefusalConfig,
    RefusalDirection,
    RefusalDistanceMetrics,
    # Transport Merger
    GWConfig,
    MergerConfig,
    MergerResult,
    BatchMergerResult,
    # Manifold Clusterer
    ManifoldPoint,
    ManifoldRegion,
    ClusteringResult,
    ClusteringConfiguration,
    # Intrinsic Dimension
    IntrinsicDimensionResult,
    # Fingerprints
    ActivatedDimension,
    Fingerprint,
    ModelFingerprints,
    ProjectionMethod,
    ProjectionFeature,
    ProjectionPoint,
    ProjectionResult,
    # Compositional Probes
    CompositionCategory,
    CompositionProbe,
    CompositionAnalysis,
    ConsistencyResult,
    # Procrustes
    ProcrustesConfig,
    ProcrustesResult,
)

# =============================================================================
# Vector and Set Math
# =============================================================================


class VectorMath:
    @staticmethod
    def dot(lhs: list[float], rhs: list[float]) -> Optional[float]:
        if len(lhs) != len(rhs) or not lhs:
            return None
        return sum(float(a) * float(b) for a, b in zip(lhs, rhs))

    @staticmethod
    def l2_norm(vector: list[float]) -> Optional[float]:
        if not vector:
            return None
        sum_sq = sum(float(v) * float(v) for v in vector)
        if sum_sq <= 0:
            return None
        return sum_sq ** 0.5

    @staticmethod
    def l2_normalized(vector: list[float]) -> list[float]:
        norm = VectorMath.l2_norm(vector)
        if not norm or norm <= 0:
            return vector
        inv_norm = 1.0 / norm
        return [float(v) * inv_norm for v in vector]

    @staticmethod
    def cosine_similarity(lhs: list[float], rhs: list[float]) -> Optional[float]:
        if len(lhs) != len(rhs) or not lhs:
            return None
        dot = 0.0
        lhs_norm = 0.0
        rhs_norm = 0.0
        for a, b in zip(lhs, rhs):
            dot += float(a) * float(b)
            lhs_norm += float(a) * float(a)
            rhs_norm += float(b) * float(b)
        if lhs_norm <= 0 or rhs_norm <= 0:
            return None
        return dot / ((lhs_norm ** 0.5) * (rhs_norm ** 0.5))


class SetMath:
    @staticmethod
    def intersection_count(lhs: set, rhs: set) -> int:
        if not lhs or not rhs:
            return 0
        smaller, larger = (lhs, rhs) if len(lhs) <= len(rhs) else (rhs, lhs)
        return sum(1 for element in smaller if element in larger)

    @staticmethod
    def jaccard_similarity(lhs: set, rhs: set) -> float:
        if not lhs or not rhs:
            return 0.0
        intersection = SetMath.intersection_count(lhs, rhs)
        union = len(lhs) + len(rhs) - intersection
        if union <= 0:
            return 0.0
        return float(intersection) / float(union)


# =============================================================================
# DoRA Decomposition (from dora_decomposition.py)
# =============================================================================

from .dora_decomposition import (
    DoRADecomposition,
    DoRAConfig,
    ChangeType,
    ChangeInterpretation,
    MagnitudeDirectionMetrics,
    DecompositionResult,
    DoRAMetricKey,
    to_metrics_dict,
)

# Aliases for backward compatibility
DoRAConfiguration = DoRAConfig


# =============================================================================
# Procrustes and Alignment
# =============================================================================

from .generalized_procrustes import (
    GeneralizedProcrustes,
    Config as GPAConfig,
    Result as GPAResult,
    LayerRotationResult,
    RotationContinuityResult,
    RotationContinuityAnalyzer,
)

from .permutation_aligner import PermutationAligner

# =============================================================================
# Concept Detection
# =============================================================================

from .concept_response_matrix import ConceptResponseMatrix
from .concept_detector import ConceptDetector
from .gate_detector import GateDetector
from .verb_noun_classifier import (
    VerbNounDimensionClassifier,
    VerbNounClassification,
    DimensionClass,
    DimensionResult,
    VerbNounConfig,
)

# =============================================================================
# Path Geometry
# =============================================================================

from .path_geometry import (
    PathNode,
    PathSignature,
    PathComparison,
)

from .traversal_coherence import (
    TraversalCoherence,
    Path,
    Result as TraversalResult,
)

# =============================================================================
# Sparse Region Analysis
# =============================================================================

from .sparse_region_locator import SparseRegionLocator
from .sparse_region_prober import SparseRegionProber
from .sparse_region_validator import SparseRegionValidator
from .sparse_region_domains import SparseRegionDomains

# =============================================================================
# Manifold Analysis
# =============================================================================

from .manifold_clusterer import ManifoldClusterer
from .manifold_dimensionality import ManifoldDimensionality
from .manifold_profile import ManifoldProfile

# =============================================================================
# Transport and Merging
# =============================================================================

from .transport_guided_merger import TransportGuidedMerger
from .gromov_wasserstein import GromovWassersteinDistance

# =============================================================================
# Refusal Direction
# =============================================================================

from .refusal_direction_detector import RefusalDirectionDetector
from .refusal_direction_cache import RefusalDirectionCache

# =============================================================================
# Geometry Validation
# =============================================================================

from .geometry_validation_suite import GeometryValidationSuite
