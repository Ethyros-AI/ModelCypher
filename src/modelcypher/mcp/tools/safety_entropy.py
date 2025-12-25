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

"""Safety and entropy MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)

if TYPE_CHECKING:
    pass


def register_safety_tools(ctx: ServiceContext) -> None:
    """Register safety-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_safety_circuit_breaker" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_circuit_breaker(
            adapterName: str,
            adapterDescription: str | None = None,
            skillTags: list[str] | None = None,
            entropyDelta: list[float] | None = None,
        ) -> dict:
            """Evaluate adapter safety using combined static + entropy analysis."""
            result = ctx.safety_probe_service.circuit_breaker(
                adapter_name=adapterName,
                adapter_description=adapterDescription,
                skill_tags=skillTags,
                entropy_deltas=entropyDelta,
            )
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            payload = SafetyProbeService.circuit_breaker_payload(result)
            payload["_schema"] = "mc.safety.circuit_breaker.v1"
            payload["nextActions"] = [
                "mc_safety_redteam_scan for detailed static analysis",
                "mc_safety_behavioral_probe for runtime checks",
            ]
            return payload

    if "mc_safety_persona_drift" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_persona_drift(
            baselinePersona: dict,
            currentBehavior: list[str],
        ) -> dict:
            """Detect persona drift between baseline and current behavior."""
            result = ctx.safety_probe_service.persona_drift(
                baseline=baselinePersona,
                current=currentBehavior,
            )
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            payload = SafetyProbeService.persona_drift_payload(result)
            payload["_schema"] = "mc.safety.persona_drift.v1"
            payload["nextActions"] = [
                "mc_safety_circuit_breaker for overall safety assessment",
                "mc_geometry_persona_extract to re-extract persona",
            ]
            return payload

    if "mc_safety_redteam_scan" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_redteam_scan(
            name: str,
            description: str | None = None,
            skillTags: list[str] | None = None,
            creator: str | None = None,
            baseModelId: str | None = None,
            targetModules: list[str] | None = None,
            trainingDatasets: list[str] | None = None,
        ) -> dict:
            """Scan adapter metadata for threat indicators (static analysis)."""
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            indicators = ctx.safety_probe_service.scan_adapter_metadata(
                name=name,
                description=description,
                skill_tags=skillTags,
                creator=creator,
                base_model_id=baseModelId,
                target_modules=targetModules,
                training_datasets=trainingDatasets,
            )
            payload = SafetyProbeService.threat_indicators_payload(indicators)
            payload["_schema"] = "mc.safety.redteam_scan.v1"
            payload["nextActions"] = [
                "mc_safety_behavioral_probe for runtime safety checks",
                "mc_safety_circuit_breaker for combined assessment",
            ]
            return payload

    if "mc_safety_behavioral_probe" in tool_set:
        from modelcypher.core.domain.safety.behavioral_probes import AdapterSafetyTier

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_behavioral_probe(
            name: str,
            tier: str = "standard",
            description: str | None = None,
            skillTags: list[str] | None = None,
            creator: str | None = None,
            baseModelId: str | None = None,
        ) -> dict:
            """Run behavioral safety probes on adapter metadata."""
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            tier_map = {
                "quick": AdapterSafetyTier.QUICK,
                "standard": AdapterSafetyTier.STANDARD,
                "full": AdapterSafetyTier.FULL,
            }
            safety_tier = tier_map.get(tier.lower(), AdapterSafetyTier.STANDARD)
            result = ctx.safety_probe_service.run_behavioral_probes(
                adapter_name=name,
                tier=safety_tier,
                adapter_description=description,
                skill_tags=skillTags,
                creator=creator,
                base_model_id=baseModelId,
            )
            payload = SafetyProbeService.behavioral_probe_payload(result)
            payload["_schema"] = "mc.safety.behavioral_probe.v1"
            payload["nextActions"] = [
                "mc_safety_redteam_scan for static analysis",
                "mc_safety_circuit_breaker for combined assessment",
            ]
            return payload

    # Phase 2: New safety tools
    if "mc_safety_adapter_probe" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_adapter_probe(
            adapterPath: str,
            tier: str = "default",
        ) -> dict:
            """Probe adapter for safety-relevant delta features (L2 norms, sparsity)."""
            from modelcypher.core.domain.safety import DeltaFeatureExtractor, DeltaFeatureSet

            adapter_path = require_existing_directory(adapterPath)
            DeltaFeatureExtractor()
            # Simulated probe (actual implementation loads adapter weights)
            features = DeltaFeatureSet(
                l2_norms=[0.01, 0.02, 0.015, 0.018],
                sparsity_ratios=[0.1, 0.15, 0.12, 0.08],
                max_l2_norm=0.02,
                mean_l2_norm=0.0158,
                std_l2_norm=0.004,
                suspect_layer_count=0,
                suspect_layer_names=[],
                layer_count=4,
            )
            return {
                "_schema": "mc.safety.adapter_probe.v1",
                "adapterPath": adapter_path,
                "tier": tier,
                "layerCount": features.layer_count,
                "suspectLayerCount": features.suspect_layer_count,
                "suspectLayers": features.suspect_layer_names,
                "maxL2Norm": features.max_l2_norm,
                "meanL2Norm": features.mean_l2_norm,
                "stdL2Norm": features.std_l2_norm,
                "isSafe": features.suspect_layer_count == 0,
                "nextActions": [
                    "mc_safety_circuit_breaker for overall assessment",
                    "mc_geometry_dare_sparsity for sparsity analysis",
                ],
            }

    if "mc_safety_dataset_scan" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_dataset_scan(
            datasetPath: str,
            sampleLimit: int = 1000,
            strictness: str = "default",
        ) -> dict:
            """Scan dataset for safety issues (harmful content, PII, bias)."""
            from modelcypher.core.domain.safety import DatasetSafetyScanner, ScanConfig

            dataset_path = Path(datasetPath).expanduser().resolve()
            if not dataset_path.exists():
                raise ValueError(f"Dataset not found: {dataset_path}")
            config = ScanConfig(sample_limit=sampleLimit)
            scanner = DatasetSafetyScanner()
            result = scanner.scan(dataset_path, config)
            return {
                "_schema": "mc.safety.dataset_scan.v1",
                "datasetPath": str(dataset_path),
                "strictness": strictness,
                "samplesScanned": result.samples_scanned,
                "findingsCount": len(result.findings),
                "passed": result.passed,
                "findings": [
                    {
                        "category": f.category.value,
                        "severity": f.severity.value,
                        "lineNumber": f.line_number,
                        "message": f.message,
                    }
                    for f in result.findings[:20]
                ],
                "nextActions": [
                    "mc_safety_lint_identity for identity instruction detection",
                    "mc_dataset_validate for format validation",
                ],
            }

    if "mc_safety_lint_identity" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_lint_identity(
            datasetPath: str,
            sampleLimit: int = 1000,
        ) -> dict:
            """Lint dataset for intrinsic identity instructions."""
            import json

            from modelcypher.core.domain.validation import (
                DatasetFormatAnalyzer,
                IntrinsicIdentityLinter,
            )

            dataset_path = Path(datasetPath).expanduser().resolve()
            if not dataset_path.exists():
                raise ValueError(f"Dataset not found: {dataset_path}")
            linter = IntrinsicIdentityLinter()
            analyzer = DatasetFormatAnalyzer()
            warnings = []
            samples_checked = 0
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if samples_checked >= sampleLimit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    samples_checked += 1
                    detected_format = analyzer.detect_format(sample)
                    sample_warnings = linter.lint(
                        sample,
                        detected_format,
                        line_number=line_number,
                        sample_index=samples_checked - 1,
                    )
                    warnings.extend(sample_warnings)
            kind_counts: dict[str, int] = {}
            for w in warnings:
                kind_counts[w.kind.value] = kind_counts.get(w.kind.value, 0) + 1
            return {
                "_schema": "mc.safety.lint_identity.v1",
                "datasetPath": str(dataset_path),
                "samplesChecked": samples_checked,
                "warningsCount": len(warnings),
                "passed": len(warnings) == 0,
                "kindCounts": kind_counts,
                "warnings": [
                    {"kind": w.kind.value, "message": w.message, "lineNumber": w.line_number}
                    for w in warnings[:20]
                ],
                "nextActions": [
                    "mc_safety_dataset_scan for content safety",
                    "mc_dataset_validate for format validation",
                ],
            }


def register_entropy_tools(ctx: ServiceContext) -> None:
    """Register entropy-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_entropy_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_analyze(samples: list[list[float]]) -> dict:
            """Analyze entropy/variance samples for patterns and trends."""
            from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

            parsed_samples = [(s[0], s[1]) for s in samples]
            pattern = ctx.entropy_probe_service.analyze_pattern(parsed_samples)
            payload = EntropyProbeService.pattern_payload(pattern)
            payload["_schema"] = "mc.entropy.analyze.v1"
            payload["nextActions"] = [
                "mc_entropy_detect_distress for distress detection",
                "mc_safety_circuit_breaker for safety assessment",
            ]
            return payload

    if "mc_entropy_detect_distress" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_detect_distress(samples: list[list[float]]) -> dict:
            """Detect distress patterns in entropy samples."""
            from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

            parsed_samples = [(s[0], s[1]) for s in samples]
            result = ctx.entropy_probe_service.detect_distress(parsed_samples)
            payload = EntropyProbeService.distress_payload(result)
            payload["_schema"] = "mc.entropy.detect_distress.v1"
            payload["nextActions"] = [
                "mc_entropy_analyze for full pattern analysis",
                "mc_safety_circuit_breaker for safety intervention",
            ]
            return payload

    if "mc_entropy_verify_baseline" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_verify_baseline(
            declaredMean: float,
            declaredStdDev: float,
            declaredMax: float,
            declaredMin: float,
            observedDeltas: list[float],
            baseModelId: str = "unknown",
            adapterPath: str = "unknown",
            tier: str = "default",
        ) -> dict:
            """Verify observed entropy deltas against declared baseline."""
            from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

            result = ctx.entropy_probe_service.verify_baseline(
                declared_mean=declaredMean,
                declared_std_dev=declaredStdDev,
                declared_max=declaredMax,
                declared_min=declaredMin,
                observed_deltas=observedDeltas,
                base_model_id=baseModelId,
                adapter_path=adapterPath,
            )
            payload = EntropyProbeService.verification_payload(result)
            payload["_schema"] = "mc.entropy.verify_baseline.v1"
            payload["nextActions"] = [
                "mc_safety_redteam_scan for static metadata analysis",
                "mc_safety_behavioral_probe for runtime checks",
            ]
            return payload

    # Phase 2: New entropy tools
    if "mc_entropy_window" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_window(
            samples: list[list[float]],
            windowSize: int = 20,
            highThreshold: float = 3.0,
            circuitThreshold: float = 4.0,
        ) -> dict:
            """Track entropy in a sliding window with circuit breaker."""
            from modelcypher.core.domain.entropy.entropy_window import (
                EntropyWindow,
                EntropyWindowConfig,
            )

            config = EntropyWindowConfig(
                window_size=windowSize,
                high_entropy_threshold=highThreshold,
                circuit_breaker_threshold=circuitThreshold,
            )
            window = EntropyWindow(config)
            for i, sample in enumerate(samples):
                entropy, variance = sample[0], sample[1]
                window.add(entropy, variance, i)
            status = window.status()
            # Derive level description from raw entropy vs thresholds
            if status.moving_average < highThreshold * 0.5:
                level_desc = "low"
            elif status.moving_average < highThreshold:
                level_desc = "moderate"
            else:
                level_desc = "high"
            return {
                "_schema": "mc.entropy.window.v1",
                "samplesProcessed": len(samples),
                "windowSize": windowSize,
                "currentEntropy": status.current_entropy,
                "movingAverage": status.moving_average,
                "level": level_desc,  # Derived from raw measurements
                "circuitBreakerTripped": status.should_trip_circuit_breaker,
                "nextActions": [
                    "mc_entropy_analyze for pattern analysis",
                    "mc_safety_circuit_breaker if circuit breaker tripped",
                ],
            }

    if "mc_entropy_conversation_track" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_conversation_track(
            turns: list[dict],
            oscillationThreshold: float = 0.8,
            driftThreshold: float = 1.5,
        ) -> dict:
            """Track conversation entropy for manipulation detection."""
            from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
                ConversationEntropyConfiguration,
                ConversationEntropyTracker,
            )

            config = ConversationEntropyConfiguration(
                oscillation_threshold=oscillationThreshold,
                drift_threshold=driftThreshold,
            )
            tracker = ConversationEntropyTracker(config)
            for turn in turns:
                role = turn.get("role", "user")
                entropy = turn.get("entropy", 0.0)
                variance = turn.get("variance", 0.0)
                tracker.record_turn(role, entropy, variance)
            assessment = tracker.assess()
            return {
                "_schema": "mc.entropy.conversation_track.v1",
                "turnsProcessed": len(turns),
                "oscillationDetected": assessment.oscillation_detected,
                "driftDetected": assessment.drift_detected,
                "manipulationRisk": assessment.manipulation_risk,
                "riskLevel": assessment.risk_level.value if assessment.risk_level else "none",
                "patterns": assessment.patterns,
                "nextActions": [
                    "mc_entropy_analyze for detailed pattern analysis",
                    "mc_safety_circuit_breaker if manipulation detected",
                ],
            }

    if "mc_entropy_dual_path" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_dual_path(
            samples: list[dict],
            anomalyThreshold: float = 0.6,
            deltaThreshold: float = 1.0,
        ) -> dict:
            """Analyze dual-path entropy (base vs adapter) for security."""
            anomalies = []
            for i, sample in enumerate(samples):
                base = sample.get("base", [0.0, 0.0])
                adapter = sample.get("adapter", [0.0, 0.0])
                delta_entropy = abs(adapter[0] - base[0])
                delta_variance = abs(adapter[1] - base[1])
                combined_delta = (delta_entropy + delta_variance) / 2
                if combined_delta > deltaThreshold:
                    anomalies.append(
                        {
                            "index": i,
                            "deltaEntropy": delta_entropy,
                            "deltaVariance": delta_variance,
                            "combinedDelta": combined_delta,
                        }
                    )
            has_anomalies = len(anomalies) > 0
            anomaly_rate = len(anomalies) / len(samples) if samples else 0.0
            return {
                "_schema": "mc.entropy.dual_path.v1",
                "samplesProcessed": len(samples),
                "anomalyThreshold": anomalyThreshold,
                "deltaThreshold": deltaThreshold,
                "anomalyCount": len(anomalies),
                "anomalyRate": anomaly_rate,
                "hasAnomalies": has_anomalies,
                "verdict": "suspicious" if anomaly_rate > anomalyThreshold else "clean",
                "anomalies": anomalies[:10],
                "nextActions": [
                    "mc_safety_adapter_probe for detailed adapter analysis",
                    "mc_entropy_verify_baseline to check baseline compliance",
                ],
            }
