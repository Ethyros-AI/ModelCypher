from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# --- Permutation Aligner Types ---

@dataclass(frozen=True)
class AlignmentConfig:
    min_match_threshold: float = 0.1
    use_anchor_grounding: bool = True
    top_k: int = 5

@dataclass
class PermutationAlignmentResult:
    permutation: Any # MLX Array or List
    signs: Any # MLX Array or List
    match_quality: float
    match_confidences: List[float]
    sign_flip_count: int
    is_sparse_permutation: bool = False
    assignment_indices: Optional[List[int]] = None

@dataclass
class RebasinResult:
    aligned_weights: Dict[str, Any]
    quality: float
    sign_flip_count: int

# --- Concept Detection Types ---

@dataclass(frozen=True)
class ConceptConfiguration:
    detection_threshold: float = 0.3
    window_sizes: List[int] = field(default_factory=lambda: [10, 20, 30])
    stride: int = 5
    collapse_consecutive: bool = True
    max_concepts_per_response: int = 30
    source_modality_hint: Optional[str] = None

@dataclass(frozen=True)
class DetectedConcept:
    concept_id: str
    category: str
    confidence: float
    character_span: slice # slice(start, end)
    trigger_text: str
    cross_modal_confidence: Optional[float] = None

@dataclass(frozen=True)
class DetectionResult:
    model_id: str
    prompt_id: str
    response_text: str
    detected_concepts: List[DetectedConcept]
    mean_confidence: float
    mean_cross_modal_confidence: Optional[float]
    timestamp: float = field(default_factory=time.time)
    
    @property
    def concept_sequence(self) -> List[str]:
        return [c.concept_id for c in self.detected_concepts]

@dataclass
class ConceptComparisonResult:
    model_a: str
    model_b: str
    concept_path_a: List[str]
    concept_path_b: List[str]
    cka: Optional[float]
    cosine_similarity: Optional[float]
    aligned_concepts: List[str]
    unique_to_a: List[str]
    unique_to_b: List[str]
    
    @property
    def alignment_ratio(self) -> float:
        total = len(set(self.concept_path_a + self.concept_path_b))
        if total == 0: return 1.0
        return len(self.aligned_concepts) / float(total)

# --- Refusal / Safety Types ---

@dataclass(frozen=True)
class ContrastivePair:
    harmful: str
    harmless: str

@dataclass(frozen=True)
class RefusalConfig:
    activation_difference_threshold: float = 0.1
    normalize_direction: bool = True
    target_layers: Optional[List[int]] = None

@dataclass(frozen=True)
class RefusalDirection:
    direction: Any # Vector
    layer_index: int
    hidden_size: int
    strength: float
    explained_variance: float
    model_id: str
    computed_at: float = field(default_factory=time.time)

@dataclass(frozen=True)
class RefusalDistanceMetrics:
    distance_to_refusal: float
    projection_magnitude: float
    is_approaching: bool
    layer_index: int
    token_index: int
    assessment: str # "likely", "possible", "unlikely", "neutral"

# --- Transport Guided Merger Types ---

@dataclass(frozen=True)
class GWConfig:
    epsilon: float = 0.01
    max_iter: int = 100
    threshold: float = 1e-4

@dataclass(frozen=True)
class MergerConfig:
    coupling_threshold: float = 0.001
    normalize_rows: bool = True
    blend_alpha: float = 0.5
    use_intersection_confidence: bool = True
    min_samples: int = 5
    gw_config: GWConfig = field(default_factory=GWConfig)

@dataclass
class MergerResult:
    merged_weights: Any # Matrix
    gw_distance: float
    marginal_error: float
    effective_rank: int
    converged: bool
    iterations: int
    dimension_confidences: List[float]

@dataclass
class BatchMergerResult:
    layer_results: Dict[str, MergerResult]
    mean_gw_distance: float
    mean_marginal_error: float
    failed_layers: List[str]
    quality_score: float

# --- Manifold Clusterer Types ---

@dataclass
class ManifoldPoint:
    id: uuid.UUID
    embedding: Any # Vector (list[float] or Array)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ManifoldRegion:
    id: int
    centroid: Any
    points: List[ManifoldPoint]
    density: float
    volume: float

@dataclass
class ClusteringResult:
    regions: List[ManifoldRegion]
    noise_points: List[ManifoldPoint]
    metrics: Dict[str, float]

@dataclass
class ClusteringConfiguration:
    epsilon: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"

# --- Intrinsic Dimension Types ---

@dataclass
class IntrinsicDimensionResult:
    estimated_dimension: float
    method: str
    details: Dict[str, Any]

# --- Fingerprints Types ---

@dataclass(frozen=True)
class ActivatedDimension:
    index: int
    activation: float

@dataclass(frozen=True)
class Fingerprint:
    prime_id: str
    prime_text: str
    activated_dimensions: Dict[int, List[ActivatedDimension]]

@dataclass(frozen=True)
class ModelFingerprints:
    model_id: str
    fingerprints: List[Fingerprint]

class ProjectionMethod(Enum):
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"

@dataclass
class ProjectionFeature:
    layer: int
    dimension: int
    frequency: int
    
    @property
    def key(self) -> str:
        return f"{self.layer}:{self.dimension}"

@dataclass
class ProjectionPoint:
    id: str
    prime_id: str
    prime_text: str
    x: float
    y: float

@dataclass
class ProjectionResult:
    model_id: str
    method: ProjectionMethod
    max_features: int
    included_layers: Optional[List[List[int]]]
    features: List[ProjectionFeature]
    points: List[ProjectionPoint]

# --- Compositional Probes Types ---

class CompositionCategory(Enum):
    MENTAL_PREDICATE = "mentalPredicate"
    ACTION = "action"
    EVALUATIVE = "evaluative"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    QUANTIFIED = "quantified"
    RELATIONAL = "relational"

@dataclass(frozen=True)
class CompositionProbe:
    phrase: str
    components: List[str]
    category: CompositionCategory

@dataclass(frozen=True)
class CompositionAnalysis:
    probe: CompositionProbe
    barycentric_weights: List[float]
    residual_norm: float
    centroid_similarity: float
    component_angles: List[float]
    
    @property
    def is_compositional(self) -> bool:
        return self.residual_norm < 0.5 and self.centroid_similarity > 0.3

@dataclass(frozen=True)
class ConsistencyResult:
    probe_count: int
    analyses_a: List[CompositionAnalysis]
    analyses_b: List[CompositionAnalysis]
    barycentric_correlation: float
    angular_correlation: float
    consistency_score: float
    is_compatible: bool
    interpretation: str

# --- Generalized Procrustes Types ---

@dataclass(frozen=True)
class ProcrustesConfig:
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    allow_reflections: bool = False
    min_models: int = 2
    allow_scaling: bool = False
    
    @staticmethod
    def default() -> "ProcrustesConfig":
        return ProcrustesConfig()

@dataclass(frozen=True)
class ProcrustesResult:
    consensus: List[List[float]]
    rotations: List[List[List[float]]]
    scales: List[float]
    residuals: List[List[List[float]]]
    converged: bool
    iterations: int
    alignment_error: float
    per_model_errors: List[float]
    consensus_variance_ratio: float
    sample_count: int
    dimension: int
    model_count: int
