from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modelcypher.core.domain.gate_detector import Configuration as GateConfig
from modelcypher.core.domain.gate_detector import DetectionResult, GateDetector
from modelcypher.core.domain.geometry_validation_suite import Config as ValidationConfig
from modelcypher.core.domain.geometry_validation_suite import GeometryValidationSuite, Report
from modelcypher.core.domain.path_geometry import (
    ComprehensiveComparison,
    PathComparison,
    PathGeometry,
    PathSignature,
)


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
    comprehensive: Optional[ComprehensiveComparison] = None


class GeometryService:
    def __init__(self, detector: GateDetector | None = None) -> None:
        self.detector = detector or GateDetector()

    def validate(self, include_fixtures: bool = False) -> Report:
        base = ValidationConfig.default()
        config = ValidationConfig(
            include_fixtures=include_fixtures,
            thresholds=base.thresholds,
            gromov_wasserstein=base.gromov_wasserstein,
        )
        return GeometryValidationSuite.run(config)

    def detect_path(
        self,
        text: str,
        model_id: str,
        prompt_id: str,
        threshold: float | None = None,
        entropy_trace: list[float] | None = None,
    ) -> DetectionResult:
        detector = self._detector_for_threshold(threshold)
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
        threshold: float | None = None,
        comprehensive: bool = False,
    ) -> PathComparisonResult:
        detector = self._detector_for_threshold(threshold)
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
        return {
            "modelA": result.model_a,
            "modelB": result.model_b,
            "pathA": result.detection_a.gate_sequence,
            "pathB": result.detection_b.gate_sequence,
            "rawDistance": result.comparison.total_distance,
            "normalizedDistance": result.comparison.normalized_distance,
            "alignmentCount": len(result.comparison.alignment),
        }

    @staticmethod
    def validation_payload(report: Report, include_schema: bool = False) -> dict:
        payload = {
            "suiteVersion": report.suite_version,
            "timestamp": GeometryService._iso_timestamp(report.timestamp),
            "passed": report.passed,
            "config": report.config,
            "gromovWasserstein": report.gromov_wasserstein,
            "traversalCoherence": report.traversal_coherence,
            "pathSignature": report.path_signature,
            "fixtures": report.fixtures,
        }
        if include_schema:
            payload = {"_schema": "tc.geometry.validation.v1", **payload}
        return payload

    def _detector_for_threshold(self, threshold: float | None) -> GateDetector:
        if threshold is None:
            return self.detector
        config = GateConfig(
            detection_threshold=threshold,
            window_sizes=self.detector.config.window_sizes,
            stride=self.detector.config.stride,
            collapse_consecutive=self.detector.config.collapse_consecutive,
            max_gates_per_response=self.detector.config.max_gates_per_response,
        )
        return GateDetector(
            configuration=config,
            embedder=self.detector.embedder,
            gate_inventory=self.detector.gate_metadata.values(),
        )

    @staticmethod
    def _iso_timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
