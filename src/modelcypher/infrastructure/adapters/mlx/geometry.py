
from typing import List, Any, Optional, Set
import mlx.core as mx
import numpy as np

from modelcypher.core.ports.geometry import GeometryPort
from modelcypher.core.domain.geometry.types import (
    ManifoldPoint, ClusteringResult, ClusteringConfiguration, ManifoldRegion,
    IntrinsicDimensionResult,
    ModelFingerprints, ProjectionResult, ProjectionMethod, ProjectionPoint, ProjectionFeature,
    CompositionProbe, CompositionAnalysis, ConsistencyResult,
    ProcrustesConfig, ProcrustesResult
)

from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer as MLXManifoldClusterer
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimensionEstimator
from modelcypher.core.domain.geometry.fingerprints import ModelFingerprintsProjection
from modelcypher.core.domain.geometry.probes import CompositionalProbes
from modelcypher.core.domain.generalized_procrustes import GeneralizedProcrustes, Config as GPAConfig

class MLXGeometryAdapter(GeometryPort):
    
    async def cluster_manifold(
        self,
        points: List[ManifoldPoint],
        config: ClusteringConfiguration
    ) -> ClusteringResult:
        # Convert config
        mlx_config = MLXManifoldClusterer.Configuration(
            epsilon=config.epsilon,
            min_points=config.min_samples, # Note: config calls it min_samples, impl calls it min_points
            metric=config.metric # Note: impl might not support metric arg in constructor?
        )
        # Check ManifoldClusterer.Configuration definition in file view above:
        # epsilon: float = 0.3
        # min_points: int = 5
        # compute_intrinsic_dimension: bool = True
        # max_clusters: int = 50
        # IT DOES NOT HAVE metric.
        
        # So I should remove metric from constructor call or handle it if my new types have it.
        # My types.py has ClusteringConfiguration(metric="euclidean").
        # The MLX impl ignores metric (uses Euclidean hardcoded).
        
        # Correct logic:
        mlx_config = MLXManifoldClusterer.Configuration(
             epsilon=config.epsilon,
             min_points=config.min_samples
        )
        
        return MLXManifoldClusterer(mlx_config).cluster(points) # Wait, cluster is instance method or static?
        # File view line 101: def cluster(self, points: List[ManifoldPoint]) -> ClusteringResult:
        # It's an INSTANCE method.
        # My previous adapter code did: MLXManifoldClusterer.cluster(points, mlx_config) which is wrong.
        
        # Correct usage:
        # clusterer = MLXManifoldClusterer(mlx_config)
        # return clusterer.cluster(points)

    async def estimate_intrinsic_dimension(
        self,
        points: List[Any], 
        method: str = "mle"
    ) -> IntrinsicDimensionResult:
        pts = points
        if not isinstance(points, mx.array):
             try:
                 pts = mx.array(points)
             except:
                 pass
        
        estimate = IntrinsicDimensionEstimator.estimate_two_nn(pts) 
        
        return IntrinsicDimensionResult(
            estimated_dimension=estimate,
            method=method,
            details={}
        )

    async def project_fingerprints(
        self,
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: Optional[Set[int]] = None,
        seed: int = 42
    ) -> ProjectionResult:
        return ModelFingerprintsProjection.project_2d(
            fingerprints, method, max_features, layers, seed
        )

    async def align_procrustes(
        self,
        activations: List[List[List[float]]],
        config: ProcrustesConfig
    ) -> Optional[ProcrustesResult]:
        
        gpa_config = GPAConfig(
            max_iterations=config.max_iterations,
            convergence_threshold=config.convergence_threshold,
            allow_reflections=config.allow_reflections,
            min_models=config.min_models,
            allow_scaling=config.allow_scaling
        )
        
        res = GeneralizedProcrustes.align(activations, gpa_config)
        if not res: return None
        
        return ProcrustesResult(
            consensus=res.consensus,
            rotations=res.rotations,
            scales=res.scales,
            residuals=res.residuals,
            converged=res.converged,
            iterations=res.iterations,
            alignment_error=res.alignment_error,
            per_model_errors=res.per_model_errors,
            consensus_variance_ratio=res.consensus_variance_ratio,
            sample_count=res.sample_count,
            dimension=res.dimension,
            model_count=res.model_count
        )

    async def analyze_composition(
        self,
        composition_embedding: Any,
        component_embeddings: Any, 
        probe: CompositionProbe
    ) -> CompositionAnalysis:
        comp = composition_embedding if isinstance(composition_embedding, mx.array) else mx.array(composition_embedding)
        comps = component_embeddings if isinstance(component_embeddings, mx.array) else mx.array(component_embeddings)
        
        return CompositionalProbes.analyze_composition(comp, comps, probe)
        
    async def check_consistency(
        self,
        analyses_a: List[CompositionAnalysis],
        analyses_b: List[CompositionAnalysis]
    ) -> ConsistencyResult:
        return CompositionalProbes.check_consistency(analyses_a, analyses_b)
