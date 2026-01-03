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

"""Thermodynamic analysis MCP tools.

Contains tools for:
- Thermodynamic job analysis
- Path analysis
- Path integration
- Entropy analysis
- Entropy measurement
- Entropy differential measurement
"""

from __future__ import annotations

from .common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)


def register_thermo_tools(ctx: ServiceContext) -> None:
    """Register thermodynamic analysis MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_thermo_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_analyze(jobId: str) -> dict:
            result = ctx.thermo_service.analyze(jobId)
            return {
                "_schema": "mc.thermo.analyze.v1",
                "jobId": result.job_id,
                "entropy": result.entropy,
                "temperature": result.temperature,
                "freeEnergy": result.free_energy,
            }

    if "mc_thermo_path" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_path(checkpoints: list[str]) -> dict:
            resolved = [require_existing_path(path) for path in checkpoints]
            result = ctx.thermo_service.path(resolved)
            return {
                "_schema": "mc.thermo.path.v1",
                "checkpoints": result.checkpoints,
                "pathLength": result.path_length,
                "curvature": result.curvature,
            }

    if "mc_thermo_path_integration" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_path_integration(
            prompt: str,
            model: str,
        ) -> dict:
            """
            Run thermodynamic path integration with gate detection.

            Returns ALL gates with their similarity scores. The similarity
            IS the geometry - no arbitrary threshold filtering.
            """
            model_path = require_existing_directory(model)
            # Return all gates - similarity scores speak for themselves
            result = ctx.thermo_service.path_integration(
                prompt=prompt,
                model_path=model_path,
            )
            measurement = result.measurement
            assessment = measurement.assessment
            return {
                "_schema": "mc.thermo.path_integration.v1",
                "modelId": result.model_id,
                "prompt": result.prompt,
                "responseText": result.response_text,
                "meanEntropy": measurement.mean_entropy,
                "entropyVariance": measurement.entropy_variance,
                "firstTokenEntropy": measurement.first_token_entropy,
                "entropyTrajectory": measurement.entropy_trajectory,
                "gateSequence": measurement.gate_sequence,
                "gateCount": measurement.gate_count,
                "gateDetails": [
                    {
                        "gateId": gate.gate_id,
                        "gateName": gate.gate_name,
                        "localEntropy": gate.local_entropy,
                        "similarity": gate.similarity,
                    }
                    for gate in measurement.gate_details
                ],
                "entropyPathCorrelation": measurement.entropy_path_correlation,
                "gateTransitionEntropies": [
                    {
                        "fromGate": item.from_gate,
                        "toGate": item.to_gate,
                        "entropyDelta": item.entropy_delta,
                    }
                    for item in measurement.gate_transition_entropies
                ],
                "assessment": {
                    "correlation": assessment.correlation,
                    "spikeRate": assessment.spike_rate,
                    "measurementCount": assessment.measurement_count,
                },
            }

    if "mc_thermo_entropy" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_entropy(jobId: str) -> dict:
            result = ctx.thermo_service.entropy(jobId)
            return {
                "_schema": "mc.thermo.entropy.v1",
                "jobId": result.job_id,
                "entropyHistory": result.entropy_history,
                "finalEntropy": result.final_entropy,
                "entropyDelta": result.entropy_delta,
                "entropyRatio": result.entropy_ratio,
            }

    if "mc_thermo_measure" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_measure(
            prompt: str,
            model: str,
        ) -> dict:
            """Measure entropy across linguistic modifiers for a prompt."""
            model_path = require_existing_directory(model)
            result = ctx.thermo_service.measure(prompt, model_path)

            return {
                "_schema": "mc.thermo.measure.v1",
                "basePrompt": result.base_prompt,
                "measurements": [
                    {
                        "modifier": m.modifier,
                        "meanEntropy": m.mean_entropy,
                        "deltaH": m.delta_h,
                    }
                    for m in result.measurements
                ],
                "statistics": {
                    "meanEntropy": result.statistics.mean_entropy,
                    "stdEntropy": result.statistics.std_entropy,
                    "minEntropy": result.statistics.min_entropy,
                    "maxEntropy": result.statistics.max_entropy,
                    "meanDeltaH": result.statistics.mean_delta_h,
                },
                "timestamp": result.timestamp.isoformat(),
            }

    if "mc_thermo_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_detect(
            prompt: str,
            model: str,
        ) -> dict:
            """Measure prompt entropy differential."""
            model_path = require_existing_directory(model)
            result = ctx.thermo_service.detect(prompt, model_path)

            return {
                "_schema": "mc.thermo.detect.v1",
                "prompt": result.prompt,
                "baselineEntropy": result.baseline_entropy,
                "intensityEntropy": result.intensity_entropy,
                "deltaH": result.delta_h,
                "processingTime": result.processing_time,
            }

    if "mc_thermo_detect_batch" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_detect_batch(
            promptsFile: str,
            model: str,
        ) -> dict:
            """Batch measure entropy differentials across multiple prompts."""
            model_path = require_existing_directory(model)
            prompts_path = require_existing_path(promptsFile)
            results = ctx.thermo_service.detect_batch(prompts_path, model_path)

            return {
                "_schema": "mc.thermo.detect_batch.v1",
                "promptsFile": promptsFile,
                "totalPrompts": len(results),
                "results": [
                    {
                        "prompt": r.prompt,
                        "baselineEntropy": r.baseline_entropy,
                        "intensityEntropy": r.intensity_entropy,
                        "deltaH": r.delta_h,
                        "processingTime": r.processing_time,
                    }
                    for r in results
                ],
            }
