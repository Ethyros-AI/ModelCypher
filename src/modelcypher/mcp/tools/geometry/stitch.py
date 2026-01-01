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

"""Geometry stitch and refinement MCP tools.

Contains tools for:
- Manifold stitching analysis
- Stitching application
- Affine stitching layer training
- Refinement density analysis
- Domain signal profile
"""

from __future__ import annotations

from pathlib import Path

from ..common import (
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)


def register_geometry_stitch_tools(ctx: ServiceContext) -> None:
    """Register geometry stitch and refinement tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_stitch_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_stitch_analyze(checkpoints: list[str]) -> dict:
            """Analyze manifold stitching between checkpoints."""
            validated_paths = [require_existing_directory(cp) for cp in checkpoints]
            result = ctx.geometry_stitch_service.analyze(validated_paths)
            return {
                "_schema": "mc.geometry.stitch.analyze.v1",
                "checkpoints": validated_paths,
                "manifoldDistance": result.manifold_distance,
                "stitchingPoints": [
                    {
                        "layerName": sp.layer_name,
                        "sourceDim": sp.source_dim,
                        "targetDim": sp.target_dim,
                        "qualityScore": sp.quality_score,
                    }
                    for sp in result.stitching_points
                ],
                "stitchConfig": result.recommended_config,
            }

    if "mc_geometry_stitch_apply" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_stitch_apply(
            source: str,
            target: str,
            outputPath: str,
        ) -> dict:
            """Apply stitching operation between checkpoints.

            Learning rate and convergence parameters are derived from the
            geometry of the anchor activations. No user parameters.
            """
            source_path = require_existing_directory(source)
            target_path = require_existing_directory(target)
            # Use defaults from Config - learning rate derived internally
            config = {"use_procrustes_warm_start": True}
            result = ctx.geometry_stitch_service.apply(source_path, target_path, outputPath, config)
            return {
                "_schema": "mc.geometry.stitch.apply.v1",
                "outputPath": result.output_path,
                "stitchedLayers": result.stitched_layers,
                "qualityScore": result.quality_score,
            }

    if "mc_geometry_stitch_train" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_stitch_train(
            anchorPairs: list[dict],
        ) -> dict:
            """Train an affine stitching layer from anchor pairs.

            Learning rate, weight decay, and convergence parameters are
            derived from the geometry of the anchor activations. No user
            parameters - the geometry determines optimal training.
            """
            from modelcypher.core.domain.geometry.affine_stitching_layer import (
                AffineStitchingLayer,
                AnchorPair,
            )

            if len(anchorPairs) < 5:
                raise ValueError("At least 5 anchor pairs required for training")
            parsed_pairs = []
            for pair in anchorPairs:
                source_act = pair.get("sourceActivation") or pair.get("source")
                target_act = pair.get("targetActivation") or pair.get("target")
                anchor_id = pair.get("anchorId") or pair.get("id")
                if source_act is None or target_act is None:
                    raise ValueError("Each anchor pair must have source and target activations")
                parsed_pairs.append(
                    AnchorPair(
                        source_activation=source_act,
                        target_activation=target_act,
                        anchor_id=anchor_id,
                    )
                )
            # Use default Config - parameters derived from geometry
            result = AffineStitchingLayer.train(parsed_pairs)
            if result is None:
                return {
                    "_schema": "mc.geometry.stitch.train.v1",
                    "status": "failed",
                    "error": "Training failed - insufficient data or convergence failure",
                }
            h4_metrics = result.h4_metrics()
            return {
                "_schema": "mc.geometry.stitch.train.v1",
                "status": "success",
                "converged": result.converged,
                "iterations": result.iterations,
                "forwardError": result.forward_error,
                "backwardError": result.backward_error,
                "sourceDimension": result.source_dimension,
                "targetDimension": result.target_dimension,
                "sampleCount": result.sample_count,
                "isPerfect": h4_metrics.is_perfect,
                "transferQuality": h4_metrics.transfer_quality,
                "weights": result.weights,
                "bias": result.bias,
            }

    if "mc_geometry_refinement_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refinement_analyze(
            baseModel: str,
            adaptedModel: str,
        ) -> dict:
            """Analyze refinement density between base and adapted models.

            Thresholds and blend coefficients are derived from the geometry -
            no configuration needed.
            """
            from modelcypher.core.domain.geometry.dare_sparsity import Configuration as DAREConfig
            from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer
            from modelcypher.core.domain.geometry.dora_decomposition import DoRADecomposition
            from modelcypher.core.domain.geometry.refinement_density import (
                RefinementDensityAnalyzer,
            )

            base_path = require_existing_directory(baseModel)
            adapted_path = require_existing_directory(adaptedModel)
            try:
                import mlx.core as mx
                from mlx_lm import load as mlx_load

                _, base_weights = mlx_load(base_path, lazy=True)
                _, adapted_weights = mlx_load(adapted_path, lazy=True)
                base_weights = dict(base_weights)
                adapted_weights = dict(adapted_weights)
                delta_weights = {}
                for name in base_weights:
                    if name not in adapted_weights:
                        continue
                    base = base_weights[name]
                    adapted = adapted_weights[name]
                    if base.shape != adapted.shape:
                        continue
                    delta = adapted - base
                    mx.eval(delta)
                    flat = delta.flatten().tolist()
                    if len(flat) > 10000:
                        import random

                        flat = random.sample(flat, 10000)
                    delta_weights[name] = flat
                sparsity_analysis = DARESparsityAnalyzer.analyze(
                    delta_weights, DAREConfig(compute_per_layer_metrics=True)
                )
                base_mx, adapted_mx = {}, {}
                for name in base_weights:
                    if name not in adapted_weights:
                        continue
                    base_mx[name] = base_weights[name]
                    adapted_mx[name] = adapted_weights[name]
                dora = DoRADecomposition()
                dora_result = dora.analyze_adapter(base_mx, adapted_mx)
                # Use default config - geometry determines everything
                analyzer = RefinementDensityAnalyzer()
                result = analyzer.analyze(
                    source_model=adapted_path,
                    target_model=base_path,
                    sparsity_analysis=sparsity_analysis,
                    dora_result=dora_result,
                )
                result_dict = result.to_dict()
                return {
                    "_schema": "mc.geometry.refinement.analyze.v1",
                    "sourceModel": result_dict.get("sourceModel"),
                    "targetModel": result_dict.get("targetModel"),
                    "meanCompositeScore": result_dict.get("meanCompositeScore"),
                    "stdCompositeScore": result_dict.get("stdCompositeScore"),
                    "maxCompositeScore": result_dict.get("maxCompositeScore"),
                    "derivedThresholds": result_dict.get("derivedThresholds"),
                    "layersAboveHardSwap": result_dict.get("layersAboveHardSwap"),
                    "layersAboveHighAlpha": result_dict.get("layersAboveHighAlpha"),
                    "hardSwapLayers": result_dict.get("hardSwapLayers"),
                    "alphaByLayer": result_dict.get("alphaByLayer"),
                    "layerScores": result_dict.get("layerScores"),
                }
            except ImportError as e:
                raise ValueError(f"MLX not available: {e}")

    if "mc_geometry_domain_profile" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_domain_profile(
            layerSignals: dict | None = None,
            modelId: str = "unknown",
            domain: str = "unknown",
            baselineDomain: str = "baseline",
            totalLayers: int = 32,
            promptCount: int = 0,
            maxTokensPerPrompt: int = 0,
            profilePath: str | None = None,
        ) -> dict:
            """Load or construct a domain signal profile."""
            import json

            from modelcypher.core.domain.geometry.domain_signal_profile import (
                DomainSignalProfile,
                LayerSignal,
            )

            if profilePath:
                path = Path(profilePath).expanduser().resolve()
                if not path.exists():
                    raise ValueError(f"Profile not found: {path}")
                data = json.loads(path.read_text())
                profile = DomainSignalProfile.from_dict(data)
            elif layerSignals:
                parsed_signals = {}
                for layer_idx, signals in layerSignals.items():
                    idx = int(layer_idx)
                    parsed_signals[idx] = LayerSignal(
                        sparsity=signals.get("sparsity"),
                        gradient_variance=signals.get("gradientVariance"),
                        gradient_snr=signals.get("gradientSNR"),
                        mean_gradient_norm=signals.get("meanGradientNorm"),
                        gradient_sample_count=signals.get("gradientSampleCount"),
                    )
                profile = DomainSignalProfile.create(
                    layer_signals=parsed_signals,
                    model_id=modelId,
                    domain=domain,
                    baseline_domain=baselineDomain,
                    total_layers=totalLayers,
                    prompt_count=promptCount,
                    max_tokens_per_prompt=maxTokensPerPrompt,
                )
            else:
                raise ValueError("Provide either profilePath or layerSignals")
            profile_dict = profile.to_dict()
            sparsity_values = [
                s.sparsity for s in profile.layer_signals.values() if s.sparsity is not None
            ]
            gradient_snr_values = [
                s.gradient_snr for s in profile.layer_signals.values() if s.gradient_snr is not None
            ]
            return {
                "_schema": "mc.geometry.domain.profile.v1",
                "modelId": profile.model_id,
                "domain": profile.domain,
                "baselineDomain": profile.baseline_domain,
                "totalLayers": profile.total_layers,
                "promptCount": profile.prompt_count,
                "maxTokensPerPrompt": profile.max_tokens_per_prompt,
                "generatedAt": profile_dict.get("generatedAt"),
                "layerSignals": profile_dict.get("layerSignals"),
                "summary": {
                    "layersWithSparsity": len(sparsity_values),
                    "meanSparsity": sum(sparsity_values) / len(sparsity_values)
                    if sparsity_values
                    else None,
                    "layersWithGradientSNR": len(gradient_snr_values),
                    "meanGradientSNR": sum(gradient_snr_values) / len(gradient_snr_values)
                    if gradient_snr_values
                    else None,
                },
            }
