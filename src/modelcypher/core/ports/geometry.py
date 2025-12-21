
from typing import Protocol, List, Any, Optional, Set, Dict, runtime_checkable
from modelcypher.core.domain.geometry.types import (
    ManifoldPoint, ClusteringResult, ClusteringConfiguration,
    IntrinsicDimensionResult,
    ModelFingerprints, ProjectionResult, ProjectionMethod,
    CompositionProbe, CompositionAnalysis, ConsistencyResult,
    ProcrustesConfig, ProcrustesResult,
    AlignmentConfig, PermutationAlignmentResult, RebasinResult,
    RefusalConfig, RefusalDirection, RefusalDistanceMetrics,
    MergerConfig, MergerResult, BatchMergerResult
)

@runtime_checkable
class GeometryPort(Protocol):
    """
    Abstract interface for high-dimensional geometry operations.
    Adapters (MLX, CUDA) must implement this.
    """
    
    # --- Permutation Alignment ---

    async def align_permutations(
        self,
        source_weight: Any,
        target_weight: Any,
        anchors: Optional[Any],
        config: AlignmentConfig
    ) -> PermutationAlignmentResult:
        ...

    async def align_via_anchor_projection(
        self,
        source_weight: Any,
        target_weight: Any,
        anchors: Any,
        config: AlignmentConfig
    ) -> PermutationAlignmentResult:
        ...

    async def rebasin_mlp(
        self,
        source_weights: Dict[str, Any],
        target_weights: Dict[str, Any],
        anchors: Any,
        config: AlignmentConfig
    ) -> RebasinResult:
        ...

    # --- Safety / Refusal Geometry ---

    async def compute_refusal_direction(
        self,
        harmful_activations: Any, # [N, D]
        harmless_activations: Any, # [N, D]
        config: RefusalConfig,
        metadata: Dict[str, Any] # e.g. layer_id, model_id
    ) -> Optional[RefusalDirection]:
        ...
        
    async def measure_refusal_distance(
        self,
        activation: Any, # [D]
        direction: RefusalDirection,
        token_index: int,
        previous_projection: Optional[float] = None
    ) -> RefusalDistanceMetrics:
        ...
        
    # --- Transport Guided Merger ---
    
    async def merge_models_transport(
        self,
        source_weights: Any, # Matrix or Dict of matrices
        target_weights: Any,
        source_activations: Any,
        target_activations: Any,
        config: MergerConfig
    ) -> Union[MergerResult, BatchMergerResult]:
        ...
        
    # --- Manifold Analysis ---
    
    async def cluster_manifold(
        self,
        points: List[ManifoldPoint],
        config: ClusteringConfiguration
    ) -> ClusteringResult:
        ...

    async def estimate_intrinsic_dimension(
        self,
        points: List[Any], # Vectors
        method: str = "mle"
    ) -> IntrinsicDimensionResult:
        ...

    # --- Alignment & Projection ---

    async def project_fingerprints(
        self,
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: Optional[Set[int]] = None,
        seed: int = 42
    ) -> ProjectionResult:
        ...
        
    async def align_procrustes(
        self,
        activations: List[List[List[float]]],
        config: ProcrustesConfig
    ) -> Optional[ProcrustesResult]:
        ...

    # --- Compositional Analysis ---

    async def analyze_composition(
        self,
        composition_embedding: Any,
        component_embeddings: Any, # Array/Tensor [N, D]
        probe: CompositionProbe
    ) -> CompositionAnalysis:
        ...
        
    async def check_consistency(
        self,
        analyses_a: List[CompositionAnalysis],
        analyses_b: List[CompositionAnalysis]
    ) -> ConsistencyResult:
        ...
