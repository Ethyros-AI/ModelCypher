# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Geometry Service for model representation analysis.

Provides geometric analysis of model activations including gate detection,
path comparison, and validation suite execution. Use this service to compare
how different models process the same inputs.

Example:
    service = GeometryService(embedder=embedder)
    result = service.compare_paths(model_a, model_b, prompt)
    print(result.comparison.cka_similarity)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from modelcypher.core.domain.geometry.gate_detector import DetectionResult, GateDetector
from modelcypher.core.domain.geometry.geometry_validation_suite import (
    Config as ValidationConfig,
)
from modelcypher.core.domain.geometry.geometry_validation_suite import (
    GeometryValidationSuite,
    Report,
)
from modelcypher.core.domain.geometry.path_geometry import (
    ComprehensiveComparison,
    PathComparison,
    PathGeometry,
    PathSignature,
)
from modelcypher.ports.embedding import EmbeddingProvider


@dataclass(frozen=True)
class PathComparisonResult:
    model_a: str
    model_b: str
    prompt_id: str
    detection_a: DetectionResult
    detection_b: DetectionResult
    path_a: PathSignature
    path_b: PathSignature
    comparison: PathComparison
    comprehensive: ComprehensiveComparison | None = None


class GeometryService:
    def __init__(
        self,
        detector: GateDetector | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        if detector is None and embedder is not None:
            detector = GateDetector(embedder=embedder)
        self.detector = detector

    def validate(self, include_fixtures: bool = False) -> Report:
        base = ValidationConfig.default()
        config = ValidationConfig(
            include_fixtures=include_fixtures,
            thresholds=base.thresholds,
            gromov_wasserstein=base.gromov_wasserstein,
        )
        suite = GeometryValidationSuite()
        return suite.run(config)

    def detect_path(
        self,
        text: str,
        model_id: str,
        prompt_id: str,
        entropy_trace: list[float] | None = None,
    ) -> DetectionResult:
        if self.detector is None:
            raise ValueError("GateDetector not configured. Provide embedder or detector.")
        detector = self.detector
        return detector.detect(
            text=text,
            model_id=model_id,
            prompt_id=prompt_id,
            entropy_trace=entropy_trace,
        )

    def compare_paths(
        self,
        text_a: str,
        text_b: str,
        model_a: str,
        model_b: str,
        prompt_id: str = "compare",
        comprehensive: bool = False,
    ) -> PathComparisonResult:
        if self.detector is None:
            raise ValueError("GateDetector not configured. Provide embedder or detector.")
        detector = self.detector
        result_a = detector.detect(text=text_a, model_id=model_a, prompt_id=prompt_id)
        result_b = detector.detect(text=text_b, model_id=model_b, prompt_id=prompt_id)

        gate_embeddings = detector.get_gate_embeddings()
        path_a = result_a.to_path_signature(gate_embeddings=gate_embeddings)
        path_b = result_b.to_path_signature(gate_embeddings=gate_embeddings)

        comparison = PathGeometry.compare(path_a, path_b, gate_embeddings)
        comprehensive_result = (
            PathGeometry.comprehensive_compare(path_a, path_b, gate_embeddings)
            if comprehensive
            else None
        )

        return PathComparisonResult(
            model_a=model_a,
            model_b=model_b,
            prompt_id=prompt_id,
            detection_a=result_a,
            detection_b=result_b,
            path_a=path_a,
            path_b=path_b,
            comparison=comparison,
            comprehensive=comprehensive_result,
        )

    @staticmethod
    def detection_payload(result: DetectionResult) -> dict:
        return {
            "modelID": result.model_id,
            "promptID": result.prompt_id,
            "responseText": result.response_text,
            "detectedGates": [
                {
                    "gateID": gate.gate_id,
                    "gateName": gate.gate_name,
                    "confidence": gate.confidence,
                    "characterSpan": {
                        "lowerBound": gate.character_span[0],
                        "upperBound": gate.character_span[1],
                    },
                    "triggerText": gate.trigger_text,
                    "localEntropy": gate.local_entropy,
                }
                for gate in result.detected_gates
            ],
            "meanConfidence": result.mean_confidence,
            "timestamp": GeometryService._iso_timestamp(result.timestamp),
        }

    @staticmethod
    def path_comparison_payload(result: PathComparisonResult) -> dict:
        payload = {
            "modelA": result.model_a,
            "modelB": result.model_b,
            "pathA": result.detection_a.gate_sequence,
            "pathB": result.detection_b.gate_sequence,
            "rawDistance": result.comparison.total_distance,
            "normalizedDistance": result.comparison.normalized_distance,
            "alignmentCount": len(result.comparison.alignment),
        }
        if result.comprehensive:
            payload["comprehensive"] = GeometryService._comprehensive_payload(result.comprehensive)
        return payload

    @staticmethod
    def _comprehensive_payload(comparison: ComprehensiveComparison) -> dict:
        """Serialize comprehensive path comparison metrics."""
        return {
            "levenshtein": {
                "totalDistance": comparison.levenshtein.total_distance,
                "normalizedDistance": comparison.levenshtein.normalized_distance,
                "alignmentCount": len(comparison.levenshtein.alignment),
            },
            "frechet": {
                "distance": comparison.frechet.distance,
                "optimalCoupling": [list(pair) for pair in comparison.frechet.optimal_coupling],
            },
            "dtw": {
                "totalCost": comparison.dtw.total_cost,
                "normalizedCost": comparison.dtw.normalized_cost,
                "warpingPath": [list(pair) for pair in comparison.dtw.warping_path],
                "compressionRatio": comparison.dtw.compression_ratio,
            },
            "signatureSimilarity": comparison.signature_similarity,
            "overallSimilarity": comparison.overall_similarity,
        }

    @staticmethod
    def validation_payload(report: Report, include_schema: bool = False) -> dict:
        config = report.config
        thresholds = config.thresholds
        gw_config = config.gromov_wasserstein
        payload = {
            "suiteVersion": report.suite_version,
            "timestamp": GeometryService._iso_timestamp(report.timestamp),
            "passed": report.passed,
            "config": {
                "includeFixtures": config.include_fixtures,
                "thresholds": {
                    "identityDistanceMax": thresholds.identity_distance_max,
                    "permutationDistanceMax": thresholds.permutation_distance_max,
                    "symmetryDeltaMax": thresholds.symmetry_delta_max,
                    "couplingMassErrorMax": thresholds.coupling_mass_error_max,
                    "traversalSelfCorrelationMin": thresholds.traversal_self_correlation_min,
                    "traversalPerturbedCorrelationMax": thresholds.traversal_perturbed_correlation_max,
                    "signatureSimilarityMin": thresholds.signature_similarity_min,
                    "frechetDistanceMax": thresholds.frechet_distance_max,
                    "dimensionConstraintCkaMin": thresholds.dimension_constraint_cka_min,
                    "dimensionConstraintGeodesicMeanAbsDiffMax": (
                        thresholds.dimension_constraint_geodesic_mean_abs_diff_max
                    ),
                    "dimensionConstraintGeodesicMaxAbsDiffMax": (
                        thresholds.dimension_constraint_geodesic_max_abs_diff_max
                    ),
                    "dimensionConstraintSpectralEigenMeanAbsDiffMax": (
                        thresholds.dimension_constraint_spectral_eigen_mean_abs_diff_max
                    ),
                    "dimensionConstraintSpectralEigenMaxAbsDiffMax": (
                        thresholds.dimension_constraint_spectral_eigen_max_abs_diff_max
                    ),
                    "dimensionConstraintSpectralEntropyAbsDiffMax": (
                        thresholds.dimension_constraint_spectral_entropy_abs_diff_max
                    ),
                    "dimensionConstraintHeatTraceMaxAbsDiffMax": (
                        thresholds.dimension_constraint_heat_trace_max_abs_diff_max
                    ),
                    "dimensionConstraintTopologyAbsDiffMax": (
                        thresholds.dimension_constraint_topology_abs_diff_max
                    ),
                },
                "gromovWasserstein": {
                    "sinkhornEpsilon": gw_config.sinkhorn_epsilon,
                    "sinkhornIterations": gw_config.sinkhorn_iterations,
                    "sinkhornThreshold": gw_config.sinkhorn_threshold,
                    "maxOuterIterations": gw_config.max_outer_iterations,
                    "minOuterIterations": gw_config.min_outer_iterations,
                    "convergenceThreshold": gw_config.convergence_threshold,
                    "relativeObjectiveThreshold": gw_config.relative_objective_threshold,
                    "useSquaredLoss": gw_config.use_squared_loss,
                    "numRestarts": gw_config.num_restarts,
                },
            },
            "gromovWasserstein": {
                "distanceIdentity": report.gromov_wasserstein.distance_identity,
                "distancePermutation": report.gromov_wasserstein.distance_permutation,
                "symmetryDelta": report.gromov_wasserstein.symmetry_delta,
                "maxRowMassError": report.gromov_wasserstein.max_row_mass_error,
                "maxColumnMassError": report.gromov_wasserstein.max_column_mass_error,
                "converged": report.gromov_wasserstein.converged,
                "iterations": report.gromov_wasserstein.iterations,
                "passed": report.gromov_wasserstein.passed,
            },
            "traversalCoherence": {
                "selfCorrelation": report.traversal_coherence.self_correlation,
                "perturbedCorrelation": report.traversal_coherence.perturbed_correlation,
                "transitionCount": report.traversal_coherence.transition_count,
                "pathCount": report.traversal_coherence.path_count,
                "passed": report.traversal_coherence.passed,
            },
            "pathSignature": {
                "signatureSimilarity": report.path_signature.signature_similarity,
                "signedArea": report.path_signature.signed_area,
                "signatureNorm": report.path_signature.signature_norm,
                "frechetDistance": report.path_signature.frechet_distance,
                "passed": report.path_signature.passed,
            },
            "spectralSignature": {
                "eigenvalueMin": report.spectral_signature.eigenvalue_min,
                "eigenvalueMax": report.spectral_signature.eigenvalue_max,
                "algebraicConnectivity": report.spectral_signature.algebraic_connectivity,
                "componentCount": report.spectral_signature.component_count,
                "heatTrace": report.spectral_signature.heat_trace,
                "heatTimes": report.spectral_signature.heat_times,
                "connected": report.spectral_signature.connected,
                "passed": report.spectral_signature.passed,
            },
            "spectralSignatureConnected": {
                "eigenvalueMin": report.spectral_signature_connected.eigenvalue_min,
                "eigenvalueMax": report.spectral_signature_connected.eigenvalue_max,
                "algebraicConnectivity": report.spectral_signature_connected.algebraic_connectivity,
                "componentCount": report.spectral_signature_connected.component_count,
                "heatTrace": report.spectral_signature_connected.heat_trace,
                "heatTimes": report.spectral_signature_connected.heat_times,
                "connected": report.spectral_signature_connected.connected,
                "passed": report.spectral_signature_connected.passed,
            },
            "dimensionConstraint": {
                "baseDimension": report.dimension_constraint.base_dimension,
                "paddedDimension": report.dimension_constraint.padded_dimension,
                "sampleCount": report.dimension_constraint.sample_count,
                "kNeighbors": report.dimension_constraint.k_neighbors,
                "gramCka": report.dimension_constraint.gram_cka,
                "geodesicDiff": {
                    "meanAbs": report.dimension_constraint.geodesic_mean_abs_diff,
                    "maxAbs": report.dimension_constraint.geodesic_max_abs_diff,
                },
                "spectral": {
                    "eigenMeanAbsDiff": report.dimension_constraint.spectral_eigen_mean_abs_diff,
                    "eigenMaxAbsDiff": report.dimension_constraint.spectral_eigen_max_abs_diff,
                    "spectralEntropyBase": report.dimension_constraint.spectral_entropy_base,
                    "spectralEntropyPadded": report.dimension_constraint.spectral_entropy_padded,
                    "heatTraceBase": report.dimension_constraint.heat_trace_base,
                    "heatTracePadded": report.dimension_constraint.heat_trace_padded,
                    "heatTimes": report.dimension_constraint.heat_times,
                },
                "topology": {
                    "bettiNumbersBase": report.dimension_constraint.betti_numbers_base,
                    "bettiNumbersPadded": report.dimension_constraint.betti_numbers_padded,
                    "componentCountBase": report.dimension_constraint.component_count_base,
                    "componentCountPadded": report.dimension_constraint.component_count_padded,
                    "cycleCountBase": report.dimension_constraint.cycle_count_base,
                    "cycleCountPadded": report.dimension_constraint.cycle_count_padded,
                    "persistenceEntropyBase": report.dimension_constraint.persistence_entropy_base,
                    "persistenceEntropyPadded": report.dimension_constraint.persistence_entropy_padded,
                    "maxPersistenceBase": report.dimension_constraint.max_persistence_base,
                    "maxPersistencePadded": report.dimension_constraint.max_persistence_padded,
                },
                "passed": report.dimension_constraint.passed,
            },
            "fixtures": GeometryService._fixtures_payload(report.fixtures)
            if report.fixtures
            else None,
        }
        if include_schema:
            payload = {"_schema": "mc.geometry.validation.v1", **payload}
        return payload

    @staticmethod
    def _iso_timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _fixtures_payload(fixtures) -> dict | None:
        if fixtures is None:
            return None
        return {
            "gromovWasserstein": {
                "pointsA": fixtures.gromov_wasserstein.points_a,
                "pointsB": fixtures.gromov_wasserstein.points_b,
                "permutation": fixtures.gromov_wasserstein.permutation,
                "sourceDistances": fixtures.gromov_wasserstein.source_distances,
                "targetDistances": fixtures.gromov_wasserstein.target_distances,
                "symmetrySourceDistances": fixtures.gromov_wasserstein.symmetry_source_distances,
                "symmetryTargetDistances": fixtures.gromov_wasserstein.symmetry_target_distances,
            },
            "traversalCoherence": {
                "anchorIds": fixtures.traversal_coherence.anchor_ids,
                "anchorGram": fixtures.traversal_coherence.anchor_gram,
                "perturbedGram": fixtures.traversal_coherence.perturbed_gram,
                "paths": [
                    {"anchorIds": path.anchor_ids} for path in fixtures.traversal_coherence.paths
                ],
            },
            "pathSignature": {
                "gateEmbeddings": fixtures.path_signature.gate_embeddings,
                "shiftedEmbeddings": fixtures.path_signature.shifted_embeddings,
                "path": {
                    "id": str(fixtures.path_signature.path.id),
                    "modelID": fixtures.path_signature.path.model_id,
                    "promptID": fixtures.path_signature.path.prompt_id,
                    "nodes": [
                        {
                            "gateID": node.gate_id,
                            "tokenIndex": node.token_index,
                            "entropy": node.entropy,
                        }
                        for node in fixtures.path_signature.path.nodes
                    ],
                },
                "projectionDim": fixtures.path_signature.projection_dim,
            },
            "spectralSignature": {
                "points": fixtures.spectral_signature.points,
                "kNeighbors": fixtures.spectral_signature.k_neighbors,
                "normalizedLaplacian": fixtures.spectral_signature.normalized_laplacian,
                "heatTimes": fixtures.spectral_signature.heat_times,
                "expectedComponentCount": fixtures.spectral_signature.expected_component_count,
                "expectedConnected": fixtures.spectral_signature.expected_connected,
            },
            "spectralSignatureConnected": {
                "points": fixtures.spectral_signature_connected.points,
                "kNeighbors": fixtures.spectral_signature_connected.k_neighbors,
                "normalizedLaplacian": fixtures.spectral_signature_connected.normalized_laplacian,
                "heatTimes": fixtures.spectral_signature_connected.heat_times,
                "expectedComponentCount": fixtures.spectral_signature_connected.expected_component_count,
                "expectedConnected": fixtures.spectral_signature_connected.expected_connected,
            },
            "dimensionConstraint": {
                "points": fixtures.dimension_constraint.points,
                "paddedDimension": fixtures.dimension_constraint.padded_dimension,
                "kNeighbors": fixtures.dimension_constraint.k_neighbors,
                "heatTimes": fixtures.dimension_constraint.heat_times,
            },
        }
