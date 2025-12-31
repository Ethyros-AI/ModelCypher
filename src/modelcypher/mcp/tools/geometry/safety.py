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

"""Geometry safety MCP tools.

Contains tools for:
- Jailbreak entropy testing
- DARE sparsity analysis
- DoRA decomposition
"""

from __future__ import annotations

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
)


def _resolve_text_backbone(model):
    """Resolve the text backbone components from various model architectures."""
    embed_tokens = None
    layers = None
    norm = None

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
        layers = model.model.layers
        norm = getattr(model.model, "norm", None)
        return (embed_tokens, layers, norm)

    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "transformer"):
            transformer = lm.transformer
            embed_tokens = getattr(transformer, "embedding", None)
            if embed_tokens is not None:
                embed_tokens = getattr(embed_tokens, "word_embeddings", embed_tokens)
            layers = getattr(transformer, "encoder", None)
            if layers is not None:
                layers = getattr(layers, "layers", layers)
            norm = getattr(transformer, "output_layer_norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)
        if hasattr(lm, "model"):
            embed_tokens = getattr(lm.model, "embed_tokens", None)
            layers = getattr(lm.model, "layers", None)
            norm = getattr(lm.model, "norm", None)
            if embed_tokens is not None and layers is not None:
                return (embed_tokens, layers, norm)

    if hasattr(model, "embed_tokens") and hasattr(model, "layers"):
        embed_tokens = model.embed_tokens
        layers = model.layers
        norm = getattr(model, "norm", None)
        return (embed_tokens, layers, norm)

    return None


def _forward_text_backbone(input_ids, embed_tokens, layers, norm, target_layer, backend):
    """Forward pass through text backbone to extract hidden states."""
    hidden = embed_tokens(input_ids)

    for i, layer in enumerate(layers):
        if i > target_layer:
            break
        hidden = layer(hidden)

    if norm is not None and target_layer == len(layers) - 1:
        hidden = norm(hidden)

    return hidden


def register_geometry_safety_tools(ctx: ServiceContext) -> None:
    """Register geometry safety tools (jailbreak, DARE, DoRA)."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_safety_jailbreak_test" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_safety_jailbreak_test(
            modelPath: str,
            prompts: list[str] | None = None,
            promptsFile: str | None = None,
            adapterPath: str | None = None,
        ) -> dict:
            """Execute jailbreak entropy analysis to test model safety boundaries."""
            if not prompts and not promptsFile:
                raise ValueError("Provide either prompts list or promptsFile path")
            prompt_input: list[str] | str
            if promptsFile:
                prompt_input = promptsFile
            else:
                prompt_input = prompts or []
            result = ctx.geometry_safety_service.jailbreak_test(
                model_path=modelPath,
                prompts=prompt_input,
                adapter_path=adapterPath,
            )
            vulnerability_details = [
                {
                    "prompt": v.prompt[:100] + "..." if len(v.prompt) > 100 else v.prompt,
                    "vulnerabilityType": v.vulnerability_type,
                    "severity": v.severity,
                    "baselineEntropy": v.baseline_entropy,
                    "attackEntropy": v.attack_entropy,
                    "deltaH": v.delta_h,
                    "confidence": v.confidence,
                    "attackVector": v.attack_vector,
                    "mitigationHint": v.mitigation_hint,
                }
                for v in result.vulnerability_details
            ]
            return {
                "_schema": "mc.geometry.safety.jailbreak_test.v1",
                "modelPath": result.model_path,
                "adapterPath": result.adapter_path,
                "promptsTested": result.prompts_tested,
                "vulnerabilitiesFound": result.vulnerabilities_found,
                "riskScore": result.risk_score,
                "processingTime": result.processing_time,
                "vulnerabilityDetails": vulnerability_details or None,
            }

    if "mc_geometry_dare_sparsity" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dare_sparsity(checkpointPath: str, basePath: str | None = None) -> dict:
            analysis = ctx.geometry_adapter_service.analyze_dare(checkpointPath, basePath)
            per_layer = []
            for name, metrics in analysis.per_layer_sparsity.items():
                importance = max(0.0, min(1.0, metrics.essential_fraction))
                per_layer.append(
                    {
                        "layerName": name,
                        "sparsity": metrics.sparsity,
                        "importance": importance,
                    }
                )
            layer_ranking = [
                entry["layerName"]
                for entry in sorted(per_layer, key=lambda x: x["importance"], reverse=True)
            ]
            return {
                "_schema": "mc.geometry.dare_sparsity.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "effectiveSparsity": analysis.effective_sparsity,
                "perLayerSparsity": per_layer or None,
                "layerRanking": layer_ranking or None,
            }

    if "mc_geometry_dora_decomposition" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dora_decomposition(
            checkpointPath: str, basePath: str | None = None
        ) -> dict:
            result = ctx.geometry_adapter_service.analyze_dora(checkpointPath, basePath)
            stability_score = ctx.geometry_adapter_service.dora_stability_score(result)
            per_layer = []
            for name, metrics in result.per_layer_metrics.items():
                per_layer.append(
                    {
                        "layerName": name,
                        "magnitudeChange": metrics.relative_magnitude_change,
                        "directionalDrift": metrics.directional_drift,
                        "magnitudeRatio": metrics.magnitude_ratio,
                        "directionCosine": metrics.direction_cosine,
                    }
                )
            return {
                "_schema": "mc.geometry.dora_decomposition.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "magnitudeChangeRatio": result.overall_magnitude_change,
                "directionalDrift": result.overall_directional_drift,
                "magnitudeToDirectionRatio": result.magnitude_to_direction_ratio,
                "perLayerDecomposition": per_layer or None,
                "stabilityScore": stability_score,
            }
