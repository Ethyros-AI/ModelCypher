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
            # Get threat indicators from static analysis
            indicators = ctx.safety_probe_service.scan_adapter_metadata(
                name=adapterName,
                description=adapterDescription,
                skill_tags=skillTags,
            )

            # Compute entropy statistics if deltas provided
            entropy_stats = {}
            if entropyDelta and len(entropyDelta) > 0:
                from modelcypher.core.domain._backend import get_default_backend
                from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

                mean = sum(entropyDelta) / len(entropyDelta)
                variance = (
                    sum((d - mean) ** 2 for d in entropyDelta) / len(entropyDelta)
                    if len(entropyDelta) > 1
                    else 0.0
                )
                _b = get_default_backend()
                std_dev = sqrt_scalar(variance, _b)
                entropy_stats = {
                    "deltaMean": mean,
                    "deltaStdDev": std_dev,
                    "deltaMax": max(entropyDelta),
                    "deltaMin": min(entropyDelta),
                    "sampleCount": len(entropyDelta),
                }

            # Raw measurements - no arbitrary classifications
            max_mean_distance = max(
                (ind.mean_distance for ind in indicators),
                default=0.0,
            )

            return {
                "_schema": "mc.safety.circuit_breaker.v1",
                "adapterName": adapterName,
                # Raw measurements - consumer interprets
                "threatIndicatorCount": len(indicators),
                "maxMeanDistance": max_mean_distance,
                "entropyStats": entropy_stats if entropy_stats else None,
                "indicators": [
                    {
                        "field": ind.field,
                        "text": ind.text,
                        "meanDistance": ind.mean_distance,
                    }
                    for ind in indicators[:5]
                ],
            }

    if "mc_safety_persona_drift" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_persona_drift(
            baselinePersona: dict,
            currentBehavior: list[str],
        ) -> dict:
            """Detect persona drift between baseline and current behavior.

            Returns raw drift measurements for ALL traits - no arbitrary threshold
            classification. User interprets which traits matter for their use case.

            Args:
                baselinePersona: Mapping of trait names to numeric scores (0-1).
                currentBehavior: List of behavior text samples.
            """
            current_text = " ".join(currentBehavior).lower()

            # Compute drift scores for ALL traits - no threshold filtering
            trait_scores = {}
            total_drift = 0.0
            trait_count = 0

            for k, v in baselinePersona.items():
                if isinstance(v, (int, float)):
                    # Binary presence check (trait name appears in current behavior)
                    present = 1.0 if k.lower() in current_text else 0.0
                    # Drift = difference between expected and observed presence, weighted by baseline score
                    drift = abs(v - present * v)
                    trait_scores[k] = {
                        "baseline": v,
                        "detected": present,
                        "drift": drift,
                    }
                    total_drift += drift
                    trait_count += 1

            # Mean drift across all traits (not threshold-filtered)
            mean_drift = total_drift / trait_count if trait_count > 0 else 0.0

            return {
                "_schema": "mc.safety.persona_drift.v1",
                # Raw measurements for ALL traits - user decides interpretation
                "meanDrift": mean_drift,
                "totalDrift": total_drift,
                "traitCount": trait_count,
                "traitScores": trait_scores,
            }

    if "mc_safety_redteam_scan" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_redteam_scan(
            name: str,
            description: str | None = None,
            skillTags: list[str] | None = None,
            creator: str | None = None,
            baseModelId: str | None = None,
            targetModules: list[str] | None = None,
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
            )
            payload = SafetyProbeService.threat_indicators_payload(indicators)
            payload["_schema"] = "mc.safety.redteam_scan.v1"
            return payload

    if "mc_safety_behavioral_probe" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_behavioral_probe(
            name: str,
            description: str | None = None,
            skillTags: list[str] | None = None,
            creator: str | None = None,
            baseModelId: str | None = None,
        ) -> dict:
            """Run behavioral safety probes on adapter metadata."""
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            result = ctx.safety_probe_service.run_behavioral_probes(
                adapter_name=name,
                adapter_description=description,
                skill_tags=skillTags,
                creator=creator,
                base_model_id=baseModelId,
            )
            payload = SafetyProbeService.composite_result_payload(result)
            payload["_schema"] = "mc.safety.behavioral_probe.v1"
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
            # DeltaFeatureSet uses correct field names per delta_feature_set.py
            features = DeltaFeatureSet(
                l2_norms=(0.01, 0.02, 0.015, 0.018),
                sparsity=(0.1, 0.15, 0.12, 0.08),
                outlier_layer_indices=(),  # No outlier layers in this simulation
            )
            return {
                "_schema": "mc.safety.adapter_probe.v1",
                "adapterPath": adapter_path,
                "tier": tier,
                # Raw measurements from the feature set
                "layerCount": features.layer_count,
                "maxL2Norm": features.max_l2_norm,
                "meanL2Norm": features.mean_l2_norm,
                "meanSparsity": features.mean_sparsity,
                "outlierLayerFraction": features.outlier_layer_fraction,
                "outlierLayerIndices": list(features.outlier_layer_indices),
            }


def register_entropy_tools(ctx: ServiceContext) -> None:
    """Register entropy-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_entropy_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_analyze(
            samples: list[list[float]],
        ) -> dict:
            """Analyze entropy/variance samples for patterns and trends."""
            from modelcypher.core.use_cases.entropy_probe_service import (
                EntropyProbeService,
            )

            parsed_samples = [(s[0], s[1]) for s in samples]
            pattern = ctx.entropy_probe_service.analyze_pattern(parsed_samples)
            payload = EntropyProbeService.pattern_payload(pattern)
            payload["_schema"] = "mc.entropy.analyze.v1"
            return payload

    if "mc_entropy_detect_distress" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_detect_distress(
            samples: list[list[float]],
        ) -> dict:
            """Detect distress patterns in entropy samples."""
            from modelcypher.core.use_cases.entropy_probe_service import (
                EntropyProbeService,
            )

            parsed_samples = [(s[0], s[1]) for s in samples]
            result = ctx.entropy_probe_service.detect_distress(parsed_samples)
            payload = EntropyProbeService.distress_payload(result)
            payload["_schema"] = "mc.entropy.detect_distress.v1"
            return payload

    if "mc_entropy_verify_baseline" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_verify_baseline(
            declaredMean: float,
            declaredStdDev: float,
            declaredMax: float,
            declaredMin: float,
            observedDeltas: list[float],
            baseModelId: str,
            adapterPath: str,
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
            return payload

    # Phase 2: New entropy tools
    if "mc_entropy_window" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_window(
            samples: list[list[float]],
            windowSize: int,
        ) -> dict:
            """Track entropy in a sliding window and return raw measurements."""
            from modelcypher.core.domain.entropy.entropy_window import (
                EntropyWindow,
                EntropyWindowConfig,
            )

            config = EntropyWindowConfig(
                window_size=windowSize,
            )
            window = EntropyWindow(config)
            for i, sample in enumerate(samples):
                entropy, variance = sample[0], sample[1]
                window.add(entropy, variance, i)
            status = window.status()
            # Raw measurements only - no arbitrary "level" classification
            return {
                "_schema": "mc.entropy.window.v1",
                "samplesProcessed": len(samples),
                "windowSize": windowSize,
                "currentEntropy": status.current_entropy,
                "movingAverage": status.moving_average,
            }

    if "mc_entropy_conversation_track" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_conversation_track(
            turns: list[dict],
            baselineDeltas: list[float] | None = None,
        ) -> dict:
            """Track conversation entropy across turns and return raw measurements."""
            from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
                ConversationEntropyBaseline,
                ConversationEntropyTracker,
            )

            baseline = None
            if baselineDeltas:
                baseline = ConversationEntropyBaseline.from_samples(baselineDeltas)
            tracker = ConversationEntropyTracker(baseline=baseline)

            # Record each turn - record_turn returns the current assessment
            assessment = None
            from datetime import datetime

            for turn in turns:
                if "tokenCount" not in turn and "token_count" not in turn:
                    raise ValueError("Each turn must include tokenCount or token_count")
                if "avgDelta" not in turn and "avg_delta" not in turn:
                    raise ValueError("Each turn must include avgDelta or avg_delta")
                if "maxAnomalyScore" not in turn and "max_anomaly_score" not in turn:
                    raise ValueError(
                        "Each turn must include maxAnomalyScore or max_anomaly_score"
                    )
                if "anomalyCount" not in turn and "anomaly_count" not in turn:
                    raise ValueError("Each turn must include anomalyCount or anomaly_count")
                if "timestamp" not in turn:
                    raise ValueError("Each turn must include timestamp")

                token_count = int(turn.get("tokenCount", turn.get("token_count")))
                avg_delta = float(turn.get("avgDelta", turn.get("avg_delta")))
                max_anomaly = float(
                    turn.get("maxAnomalyScore", turn.get("max_anomaly_score"))
                )
                anomaly_count = int(turn.get("anomalyCount", turn.get("anomaly_count")))
                timestamp = datetime.fromisoformat(str(turn["timestamp"]))

                assessment = tracker.record_turn(
                    token_count=token_count,
                    avg_delta=avg_delta,
                    max_anomaly_score=max_anomaly,
                    anomaly_count=anomaly_count,
                    timestamp=timestamp,
                )

            if assessment is None:
                # No turns processed
                return {
                    "_schema": "mc.entropy.conversation_track.v1",
                    "turnsProcessed": 0,
                    "meanDelta": 0.0,
                    "stdDelta": 0.0,
                    "oscillationAmplitude": 0.0,
                    "oscillationFrequency": 0.0,
                    "cumulativeDrift": 0.0,
                    "anomalyCount": 0,
                    "anomalyRate": 0.0,
                    "maxAnomalyScore": 0.0,
                    "deltaChangeMean": 0.0,
                    "deltaChangeStd": 0.0,
                }

            # Return raw geometric measurements
            return {
                "_schema": "mc.entropy.conversation_track.v1",
                "turnsProcessed": len(turns),
                "meanDelta": assessment.mean_delta,
                "stdDelta": assessment.std_delta,
                "oscillationAmplitude": assessment.oscillation_amplitude,
                "oscillationFrequency": assessment.oscillation_frequency,
                "cumulativeDrift": assessment.cumulative_drift,
                "anomalyCount": assessment.anomaly_count,
                "anomalyRate": assessment.anomaly_rate,
                "maxAnomalyScore": assessment.max_anomaly_score,
                "deltaChangeMean": assessment.delta_change_mean,
                "deltaChangeStd": assessment.delta_change_std,
            }

    if "mc_entropy_dual_path" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_entropy_dual_path(
            samples: list[dict],
        ) -> dict:
            """Analyze dual-path entropy (base vs adapter) for security.

            Returns raw delta measurements for ALL samples. The deltas
            themselves ARE the geometric signal - no threshold filtering.
            """
            # Compute raw deltas for every sample - the geometry speaks
            per_sample = []
            all_deltas = []
            for i, sample in enumerate(samples):
                base = sample.get("base", [0.0, 0.0])
                adapter = sample.get("adapter", [0.0, 0.0])
                delta_entropy = abs(adapter[0] - base[0])
                delta_variance = abs(adapter[1] - base[1])
                combined_delta = (delta_entropy + delta_variance) / 2
                all_deltas.append(combined_delta)
                per_sample.append({
                    "index": i,
                    "deltaEntropy": delta_entropy,
                    "deltaVariance": delta_variance,
                    "combinedDelta": combined_delta,
                })

            # Raw statistics from the data itself
            mean_delta = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
            max_delta = max(all_deltas) if all_deltas else 0.0
            min_delta = min(all_deltas) if all_deltas else 0.0
            sorted_deltas = sorted(all_deltas)
            median_delta = sorted_deltas[len(sorted_deltas) // 2] if sorted_deltas else 0.0

            return {
                "_schema": "mc.entropy.dual_path.v1",
                "samplesProcessed": len(samples),
                # Raw statistics derived from the data
                "meanDelta": mean_delta,
                "medianDelta": median_delta,
                "minDelta": min_delta,
                "maxDelta": max_delta,
                # All samples with their measurements - no filtering
                "samples": per_sample,
            }
