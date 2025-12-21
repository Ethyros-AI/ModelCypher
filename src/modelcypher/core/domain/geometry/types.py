
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

# --- Permutation Aligner Types ---

@dataclass
class PermutationAlignmentResult:
    weight_map: Dict[str, Any] # Placeholder for aligned weights (numpy/mlx array agnostic in type definition if possible, or Any)
    # Ideally should be backend agnostic. For now using Any or list.
    # But usually backend returns Tensor/Array. 
    # Let's keep it generic "Any" for the port definition.
    metrics: Dict[str, float]

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
