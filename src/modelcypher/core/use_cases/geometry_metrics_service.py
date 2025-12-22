"""
Geometry Metrics Service.

Exposes standalone geometry metrics as CLI/MCP-consumable operations.
These are the unique value propositions of ModelCypher - geometric
diagnostics that no other tool provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from modelcypher.core.domain.geometry.gromov_wasserstein import (
    Config as GWConfig,
    GromovWassersteinDistance,
    Result as GWResult,
)
from modelcypher.core.domain.geometry.intrinsic_dimension_estimator import (
    IntrinsicDimensionEstimator,
    TwoNNConfiguration,
    BootstrapConfiguration,
)
from modelcypher.core.domain.geometry.topological_fingerprint import (
    TopologicalFingerprint,
)


@dataclass(frozen=True)
class GromovWassersteinResult:
    """Result of Gromov-Wasserstein distance computation."""
    distance: float
    normalized_distance: float
    compatibility_score: float
    converged: bool
    iterations: int
    coupling_shape: tuple[int, int]
    interpretation: str


@dataclass(frozen=True)
class IntrinsicDimensionResult:
    """Result of intrinsic dimension estimation."""
    dimension: float
    confidence_lower: float
    confidence_upper: float
    sample_count: int
    method: str
    interpretation: str


@dataclass(frozen=True)
class TopologicalFingerprintResult:
    """Result of topological fingerprint computation."""
    betti_0: int  # Connected components
    betti_1: int  # Loops/holes
    persistence_entropy: float
    total_persistence: float
    interpretation: str


class GeometryMetricsService:
    """
    Service for standalone geometry metrics.

    These metrics provide the unique geometric diagnostics that differentiate
    ModelCypher from other ML tools.
    """

    def compute_gromov_wasserstein(
        self,
        source_points: list[list[float]],
        target_points: list[list[float]],
        epsilon: float = 0.05,
        max_iterations: int = 50,
    ) -> GromovWassersteinResult:
        """
        Compute Gromov-Wasserstein distance between two point clouds.

        This measures the structural similarity of representation spaces
        without requiring point-to-point correspondence.

        Args:
            source_points: First point cloud (N x D)
            target_points: Second point cloud (M x D)
            epsilon: Entropic regularization parameter
            max_iterations: Maximum outer iterations

        Returns:
            GromovWassersteinResult with distance and interpretation
        """
        config = GWConfig(
            epsilon=epsilon,
            max_outer_iterations=max_iterations,
        )

        source_distances = GromovWassersteinDistance.compute_pairwise_distances(source_points)
        target_distances = GromovWassersteinDistance.compute_pairwise_distances(target_points)

        result = GromovWassersteinDistance.compute(
            source_distances=source_distances,
            target_distances=target_distances,
            config=config,
        )

        # Generate interpretation
        if result.normalized_distance < 0.1:
            interpretation = "Highly similar structure. Representation spaces are nearly isomorphic."
        elif result.normalized_distance < 0.3:
            interpretation = "Moderately similar. Core structure preserved with some divergence."
        elif result.normalized_distance < 0.5:
            interpretation = "Significant structural differences. Careful alignment needed before merging."
        else:
            interpretation = "Very different structures. Merging may cause capability loss."

        if not result.converged:
            interpretation += " Warning: solver did not converge; results may be approximate."

        return GromovWassersteinResult(
            distance=result.distance,
            normalized_distance=result.normalized_distance,
            compatibility_score=result.compatibility_score,
            converged=result.converged,
            iterations=result.iterations,
            coupling_shape=(len(source_points), len(target_points)),
            interpretation=interpretation,
        )

    def estimate_intrinsic_dimension(
        self,
        points: list[list[float]],
        use_regression: bool = True,
        bootstrap_samples: int = 200,
    ) -> IntrinsicDimensionResult:
        """
        Estimate intrinsic dimension of a point cloud using TwoNN.

        This reveals the effective degrees of freedom in a representation
        space, which can indicate model capacity and generalization.

        Args:
            points: Point cloud (N x D)
            use_regression: Use regression method (more accurate)
            bootstrap_samples: Number of bootstrap iterations for confidence

        Returns:
            IntrinsicDimensionResult with dimension and confidence bounds
        """
        config = TwoNNConfiguration(
            use_regression=use_regression,
            bootstrap=BootstrapConfiguration(resamples=bootstrap_samples) if bootstrap_samples > 0 else None,
        )

        estimate = IntrinsicDimensionEstimator.estimate_two_nn(points, config)

        # Extract confidence intervals if available
        if estimate.ci is not None:
            lower = estimate.ci.lower
            upper = estimate.ci.upper
        else:
            lower = estimate.intrinsic_dimension * 0.8
            upper = estimate.intrinsic_dimension * 1.2

        # Generate interpretation
        dimension = estimate.intrinsic_dimension
        ambient_dim = len(points[0]) if points else 0
        ratio = dimension / ambient_dim if ambient_dim > 0 else 0

        if ratio < 0.1:
            interpretation = f"Low intrinsic dimension ({dimension:.1f}). Representations are highly structured/compressed."
        elif ratio < 0.3:
            interpretation = f"Moderate intrinsic dimension ({dimension:.1f}). Balanced capacity utilization."
        elif ratio < 0.6:
            interpretation = f"High intrinsic dimension ({dimension:.1f}). Rich representations with many degrees of freedom."
        else:
            interpretation = f"Very high intrinsic dimension ({dimension:.1f}). May indicate noise or overfitting."

        return IntrinsicDimensionResult(
            dimension=dimension,
            confidence_lower=lower,
            confidence_upper=upper,
            sample_count=estimate.sample_count,
            method="TwoNN" + (" (regression)" if use_regression else " (maximum likelihood)"),
            interpretation=interpretation,
        )

    def compute_topological_fingerprint(
        self,
        points: list[list[float]],
        max_dimension: int = 1,
        max_edge_length: float | None = None,
    ) -> TopologicalFingerprintResult:
        """
        Compute topological fingerprint using persistent homology.

        This reveals the shape of the representation manifold, including
        connected components, loops, and voids.

        Args:
            points: Point cloud (N x D)
            max_dimension: Maximum homology dimension to compute
            max_edge_length: Maximum edge length for Rips complex

        Returns:
            TopologicalFingerprintResult with Betti numbers and persistence
        """
        fingerprint = TopologicalFingerprint()
        summary = fingerprint.compute(
            points=points,
            max_dimension=max_dimension,
            max_edge_length=max_edge_length,
        )

        # Generate interpretation
        if summary.betti_0 == 1 and summary.betti_1 == 0:
            interpretation = "Simple connected topology. Single coherent representation cluster."
        elif summary.betti_0 > 1 and summary.betti_1 == 0:
            interpretation = f"Fragmented topology ({summary.betti_0} components). Multiple distinct representation clusters."
        elif summary.betti_1 > 0:
            interpretation = f"Complex topology with {summary.betti_1} loop(s). May indicate cyclic or periodic structure."
        else:
            interpretation = "Standard topology with moderate complexity."

        if summary.persistence_entropy > 0.8:
            interpretation += " High persistence entropy suggests stable features."
        elif summary.persistence_entropy < 0.3:
            interpretation += " Low persistence entropy indicates transient features."

        return TopologicalFingerprintResult(
            betti_0=summary.betti_0,
            betti_1=summary.betti_1,
            persistence_entropy=summary.persistence_entropy,
            total_persistence=summary.total_persistence,
            interpretation=interpretation,
        )

    def analyze_sparse_regions(
        self,
        points: list[list[float]],
        sparsity_threshold: float = 0.1,
        min_region_size: int = 3,
    ) -> SparseRegionResult:
        """
        Analyze sparse regions in representation space.

        Sparse regions often correspond to underutilized capacity or
        potential areas for improvement.

        Args:
            points: Point cloud (N x D)
            sparsity_threshold: Threshold for considering a region sparse
            min_region_size: Minimum points to form a region

        Returns:
            SparseRegionResult with region analysis
        """
        locator = SparseRegionLocator(
            sparsity_threshold=sparsity_threshold,
            min_region_size=min_region_size,
        )
        validator = SparseRegionValidator()

        regions = locator.locate(points)
        validation = validator.validate(regions, points)

        if not regions:
            return SparseRegionResult(
                region_count=0,
                total_volume_fraction=0.0,
                max_sparsity=0.0,
                mean_sparsity=0.0,
                coverage_quality="complete",
                interpretation="No sparse regions detected. Representation space is well-utilized.",
            )

        sparsities = [r.sparsity for r in regions]
        total_volume = sum(r.volume_fraction for r in regions)

        # Determine coverage quality
        if total_volume < 0.1:
            quality = "excellent"
            interp = "Minimal sparse regions. Representation space is well-utilized."
        elif total_volume < 0.3:
            quality = "good"
            interp = f"Some sparse regions ({len(regions)}). Consider targeted data augmentation."
        elif total_volume < 0.5:
            quality = "moderate"
            interp = f"Significant sparse regions ({len(regions)}). May limit generalization."
        else:
            quality = "poor"
            interp = f"Large sparse regions ({len(regions)}). Representations may be inefficient."

        return SparseRegionResult(
            region_count=len(regions),
            total_volume_fraction=total_volume,
            max_sparsity=max(sparsities),
            mean_sparsity=sum(sparsities) / len(sparsities),
            coverage_quality=quality,
            interpretation=interp,
        )

    def detect_refusal_direction(
        self,
        activations: list[list[float]],
        baseline: list[list[float]] | None = None,
    ) -> RefusalDirectionResult:
        """
        Detect proximity to refusal direction in activation space.

        This measures how close the model's activations are to the
        learned refusal direction, indicating potential safety behavior.

        Args:
            activations: Current activation vectors
            baseline: Optional baseline activations for comparison

        Returns:
            RefusalDirectionResult with distance and risk assessment
        """
        detector = RefusalDirectionDetector()

        result = detector.detect(
            activations=activations,
            baseline=baseline,
        )

        # Determine risk level
        if result.distance > 0.8:
            risk = "low"
            interp = "Far from refusal direction. Normal generation behavior."
        elif result.distance > 0.5:
            risk = "moderate"
            interp = "Moderate proximity to refusal direction. Monitor for boundary behavior."
        elif result.distance > 0.2:
            risk = "elevated"
            interp = "Approaching refusal direction. May trigger safety mechanisms soon."
        else:
            risk = "high"
            interp = "Very close to refusal direction. Safety mechanisms likely active."

        if result.is_approaching:
            interp += " Trajectory is moving toward refusal."

        return RefusalDirectionResult(
            distance=result.distance,
            is_approaching=result.is_approaching,
            direction_magnitude=result.magnitude,
            risk_level=risk,
            interpretation=interp,
        )

    @staticmethod
    def gromov_wasserstein_payload(result: GromovWassersteinResult) -> dict:
        """Convert GW result to CLI/MCP payload."""
        return {
            "distance": result.distance,
            "normalizedDistance": result.normalized_distance,
            "compatibilityScore": result.compatibility_score,
            "converged": result.converged,
            "iterations": result.iterations,
            "couplingShape": list(result.coupling_shape),
            "interpretation": result.interpretation,
        }

    @staticmethod
    def intrinsic_dimension_payload(result: IntrinsicDimensionResult) -> dict:
        """Convert ID result to CLI/MCP payload."""
        return {
            "dimension": result.dimension,
            "confidenceLower": result.confidence_lower,
            "confidenceUpper": result.confidence_upper,
            "sampleCount": result.sample_count,
            "method": result.method,
            "interpretation": result.interpretation,
        }

    @staticmethod
    def topological_fingerprint_payload(result: TopologicalFingerprintResult) -> dict:
        """Convert TF result to CLI/MCP payload."""
        return {
            "betti0": result.betti_0,
            "betti1": result.betti_1,
            "persistenceEntropy": result.persistence_entropy,
            "totalPersistence": result.total_persistence,
            "interpretation": result.interpretation,
        }

    @staticmethod
    def sparse_region_payload(result: SparseRegionResult) -> dict:
        """Convert SR result to CLI/MCP payload."""
        return {
            "regionCount": result.region_count,
            "totalVolumeFraction": result.total_volume_fraction,
            "maxSparsity": result.max_sparsity,
            "meanSparsity": result.mean_sparsity,
            "coverageQuality": result.coverage_quality,
            "interpretation": result.interpretation,
        }

    @staticmethod
    def refusal_direction_payload(result: RefusalDirectionResult) -> dict:
        """Convert RD result to CLI/MCP payload."""
        return {
            "distance": result.distance,
            "isApproaching": result.is_approaching,
            "directionMagnitude": result.direction_magnitude,
            "riskLevel": result.risk_level,
            "interpretation": result.interpretation,
        }
