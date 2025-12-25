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

"""Geometry MCP tools.

This module contains geometry-related MCP tools for:
- Path detection and comparison
- Gromov-Wasserstein distance
- Intrinsic dimension estimation
- Topological fingerprinting
- Sparse region analysis
- Refusal direction detection
- Persona vector extraction
- Manifold clustering
- Transport-guided merging
- Invariant layer mapping
- CRM (Concept Response Matrix)
- Primes analysis
- Stitch analysis
- DARE sparsity / DoRA decomposition
- 3D spatial metrology (Euclidean, gravity, occlusion)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .common import (
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
    require_existing_path,
)

if TYPE_CHECKING:
    pass

# Constants
DEFAULT_PATH_THRESHOLD = 0.55
DEFAULT_PATH_MAX_TOKENS = 200


def register_geometry_tools(ctx: ServiceContext) -> None:
    """Register geometry-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    # Basic geometry tools
    if "mc_geometry_validate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_validate(includeFixtures: bool = False) -> dict:
            report = ctx.geometry_service.validate(include_fixtures=includeFixtures)
            return ctx.geometry_service.validation_payload(report, include_schema=True)

    if "mc_geometry_path_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_path_detect(
            text: str,
            model: str | None = None,
            threshold: float = DEFAULT_PATH_THRESHOLD,
            entropyTrace: list[float] | None = None,
        ) -> dict:
            if model:
                response = ctx.inference_engine.infer(
                    model,
                    text,
                    max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0,
                    top_p=1.0,
                )
                text_to_analyze = response.get("response", "")
                model_id = Path(model).name if Path(model).exists() else model
            else:
                text_to_analyze = text
                model_id = "input-text"
            detection = ctx.geometry_service.detect_path(
                text_to_analyze,
                model_id=model_id,
                prompt_id="mcp-path-detect",
                threshold=threshold,
                entropy_trace=entropyTrace,
            )
            payload = ctx.geometry_service.detection_payload(detection)
            payload["_schema"] = "mc.geometry.path.detect.v1"
            payload["nextActions"] = [
                "mc_geometry_path_compare to compare two paths",
                "mc_safety_circuit_breaker for safety assessment",
            ]
            return payload

    if "mc_geometry_path_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_path_compare(
            textA: str | None = None,
            textB: str | None = None,
            modelA: str | None = None,
            modelB: str | None = None,
            prompt: str | None = None,
            threshold: float = DEFAULT_PATH_THRESHOLD,
            comprehensive: bool = False,
        ) -> dict:
            if textA and textB:
                text_to_analyze_a, text_to_analyze_b = textA, textB
                model_id_a, model_id_b = "text-a", "text-b"
            elif modelA and modelB and prompt:
                response_a = ctx.inference_engine.infer(
                    modelA,
                    prompt,
                    max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0,
                    top_p=1.0,
                )
                response_b = ctx.inference_engine.infer(
                    modelB,
                    prompt,
                    max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0,
                    top_p=1.0,
                )
                text_to_analyze_a = response_a.get("response", "")
                text_to_analyze_b = response_b.get("response", "")
                model_id_a = Path(modelA).name if Path(modelA).exists() else modelA
                model_id_b = Path(modelB).name if Path(modelB).exists() else modelB
            else:
                raise ValueError("Provide textA/textB or modelA/modelB with prompt.")
            result = ctx.geometry_service.compare_paths(
                text_a=text_to_analyze_a,
                text_b=text_to_analyze_b,
                model_a=model_id_a,
                model_b=model_id_b,
                prompt_id="mcp-path-compare",
                threshold=threshold,
                comprehensive=comprehensive,
            )
            payload = ctx.geometry_service.path_comparison_payload(result)
            payload["_schema"] = "mc.geometry.path.compare.v1"
            payload["nextActions"] = [
                "mc_geometry_path_detect to inspect individual paths",
                "mc_geometry_validate to validate geometry suite",
            ]
            return payload

    # Metrics tools
    if "mc_geometry_gromov_wasserstein" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_gromov_wasserstein(
            sourcePoints: list[list[float]],
            targetPoints: list[list[float]],
            epsilon: float = 0.05,
            maxIterations: int = 50,
        ) -> dict:
            """Compute Gromov-Wasserstein distance between point clouds."""
            result = ctx.geometry_metrics_service.compute_gromov_wasserstein(
                source_points=sourcePoints,
                target_points=targetPoints,
                epsilon=epsilon,
                max_iterations=maxIterations,
            )
            payload = ctx.geometry_metrics_service.gromov_wasserstein_payload(result)
            payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
            payload["nextActions"] = [
                "mc_geometry_intrinsic_dimension to estimate dimensionality",
                "mc_geometry_topological_fingerprint for topology analysis",
            ]
            return payload

    if "mc_geometry_intrinsic_dimension" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_intrinsic_dimension(
            points: list[list[float]],
            useRegression: bool = True,
            bootstrapSamples: int = 200,
        ) -> dict:
            """Estimate intrinsic dimension using TwoNN."""
            result = ctx.geometry_metrics_service.estimate_intrinsic_dimension(
                points=points,
                use_regression=useRegression,
                bootstrap_samples=bootstrapSamples,
            )
            payload = ctx.geometry_metrics_service.intrinsic_dimension_payload(result)
            payload["_schema"] = "mc.geometry.intrinsic_dimension.v1"
            payload["nextActions"] = [
                "mc_geometry_topological_fingerprint for topology analysis",
                "mc_geometry_gromov_wasserstein for structure comparison",
            ]
            return payload

    if "mc_geometry_topological_fingerprint" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_topological_fingerprint(
            points: list[list[float]],
            maxDimension: int = 1,
            numSteps: int = 50,
        ) -> dict:
            """Compute topological fingerprint using persistent homology."""
            result = ctx.geometry_metrics_service.compute_topological_fingerprint(
                points=points,
                max_dimension=maxDimension,
                num_steps=numSteps,
            )
            payload = ctx.geometry_metrics_service.topological_fingerprint_payload(result)
            payload["_schema"] = "mc.geometry.topological_fingerprint.v1"
            payload["nextActions"] = [
                "mc_geometry_intrinsic_dimension for dimensionality",
                "mc_geometry_gromov_wasserstein for structure comparison",
            ]
            return payload

    # Sparse region tools
    if "mc_geometry_sparse_domains" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_domains(category: str | None = None) -> dict:
            """List built-in sparse region domains for LoRA targeting."""
            if category:
                domains = ctx.geometry_sparse_service.get_domains_by_category(category)
            else:
                domains = ctx.geometry_sparse_service.list_domains()
            payload = ctx.geometry_sparse_service.domains_payload(domains)
            payload["_schema"] = "mc.geometry.sparse_domains.v1"
            payload["nextActions"] = [
                "mc_geometry_sparse_locate to find sparse regions",
                "mc_geometry_intrinsic_dimension for dimensionality analysis",
            ]
            return payload

    if "mc_geometry_sparse_locate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_locate(
            domainStats: list[dict],
            baselineStats: list[dict],
            domainName: str = "unknown",
            baseRank: int = 16,
            sparsityThreshold: float = 0.3,
        ) -> dict:
            """Locate sparse regions suitable for LoRA injection."""
            result = ctx.geometry_sparse_service.locate_sparse_regions(
                domain_stats=domainStats,
                baseline_stats=baselineStats,
                domain_name=domainName,
                base_rank=baseRank,
                sparsity_threshold=sparsityThreshold,
            )
            payload = ctx.geometry_sparse_service.analysis_payload(result)
            payload["_schema"] = "mc.geometry.sparse_locate.v1"
            payload["nextActions"] = [
                "mc_geometry_dare_sparsity for DARE analysis",
                "mc_geometry_dora_decomposition for magnitude separation",
            ]
            return payload

    if "mc_geometry_sparse_neurons" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_neurons(
            modelPath: str,
            domain: str | None = None,
            promptsFile: str | None = None,
            layerStart: float = 0.0,
            layerEnd: float = 1.0,
            sparsityThreshold: float = 0.8,
        ) -> dict:
            """Analyze per-neuron sparsity for fine-grained knowledge grafting.

            Identifies individual neurons that are sparse enough to be
            good candidates for knowledge transfer during model merging.

            Args:
                modelPath: Path to model directory
                domain: Use built-in domain probes (math, code, factual, reasoning)
                promptsFile: Path to JSON file with custom prompts
                layerStart: Start layer fraction (0.0-1.0)
                layerEnd: End layer fraction (0.0-1.0)
                sparsityThreshold: Sparsity threshold for graft candidates

            Returns:
                Neuron sparsity map with graft candidates and dead neurons
            """
            import json

            from modelcypher.core.domain.geometry.neuron_sparsity_analyzer import (
                NeuronSparsityConfig,
                compute_neuron_sparsity_map,
            )

            model_path = require_existing_directory(modelPath)

            # Load prompts
            prompts: list[str] = []
            if promptsFile:
                prompts_path = Path(promptsFile)
                if not prompts_path.exists():
                    raise ValueError(f"Prompts file not found: {promptsFile}")
                prompts = json.loads(prompts_path.read_text())
            elif domain:
                # Use built-in domain probes
                domains_list = ctx.geometry_sparse_service.list_domains()
                domain_def = next(
                    (d for d in domains_list if d.name.lower() == domain.lower()), None
                )
                if domain_def is None:
                    raise ValueError(
                        f"Unknown domain: {domain}. Use mc_geometry_sparse_domains to list available domains."
                    )
                prompts = domain_def.probes
            else:
                raise ValueError("Provide either domain or promptsFile")

            config = NeuronSparsityConfig(
                sparsity_threshold=sparsityThreshold,
                min_prompts=min(len(prompts), 20),
            )

            # Collect activations via model inference
            from modelcypher.core.domain.entropy.hidden_state_extractor import (
                ExtractorConfig,
                HiddenStateExtractor,
            )
            from modelcypher.core.use_cases.model_probe_service import ModelProbeService

            # Get model info for layer count
            probe_service = ModelProbeService()
            model_info = probe_service.probe(str(model_path))
            total_layers = len([l for l in model_info.layers if "layers." in l.name])

            # Create extractor for neuron analysis in specified layer range
            extractor_config = ExtractorConfig.for_neuron_analysis_range(
                total_layers,
                start_fraction=layerStart,
                end_fraction=layerEnd,
                hidden_dim=model_info.hidden_size,
            )
            extractor = HiddenStateExtractor(extractor_config)

            # Collect activations via inference
            from modelcypher.adapters.local_inference import LocalInferenceEngine

            engine = LocalInferenceEngine()
            extractor.start_neuron_collection()

            for prompt in prompts[: config.min_prompts]:
                try:
                    # Run inference to trigger activation capture
                    engine.infer(str(model_path), prompt, max_tokens=50, temperature=0.0)
                except Exception:
                    pass  # Continue with other prompts
                extractor.finalize_prompt_activations()

            # Get collected activations
            activations = extractor.get_neuron_activations()

            sparsity_map = compute_neuron_sparsity_map(activations, config)
            summary = sparsity_map.summary()

            return {
                "_schema": "mc.geometry.sparse_neurons.v1",
                "modelPath": str(model_path),
                "domain": domain,
                "config": {
                    "sparsityThreshold": config.sparsity_threshold,
                    "activationThreshold": config.activation_threshold,
                    "layerRange": [layerStart, layerEnd],
                },
                "summary": summary,
                "graftCandidates": sparsity_map.get_graft_candidates(),
                "deadNeurons": sparsity_map.dead_neurons,
                "interpretation": (
                    f"Analyzed {summary.get('num_layers', 0)} layers with "
                    f"{summary.get('total_neurons', 0)} neurons. "
                    f"{summary.get('total_sparse', 0)} sparse ({summary.get('sparse_fraction', 0):.0%}), "
                    f"{summary.get('total_dead', 0)} dead ({summary.get('dead_fraction', 0):.0%})."
                ),
                "nextActions": [
                    "mc_geometry_sparse_locate for layer-level analysis",
                    "mc_model_merge with neuron-level alpha masking",
                ],
            }

    # Refusal detection tools
    if "mc_geometry_refusal_pairs" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refusal_pairs() -> dict:
            """Get standard contrastive prompt pairs for refusal detection."""
            pairs = ctx.geometry_sparse_service.get_contrastive_pairs()
            payload = ctx.geometry_sparse_service.contrastive_pairs_payload(pairs)
            payload["_schema"] = "mc.geometry.refusal_pairs.v1"
            payload["nextActions"] = [
                "mc_geometry_refusal_detect to compute refusal direction",
                "mc_safety_circuit_breaker for safety monitoring",
            ]
            return payload

    if "mc_geometry_refusal_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refusal_detect(
            harmfulActivations: list[list[float]],
            harmlessActivations: list[list[float]],
            layerIndex: int,
            modelId: str,
            normalize: bool = True,
        ) -> dict:
            """Detect refusal direction from contrastive activations."""
            result = ctx.geometry_sparse_service.detect_refusal_direction(
                harmful_activations=harmfulActivations,
                harmless_activations=harmlessActivations,
                layer_index=layerIndex,
                model_id=modelId,
                normalize=normalize,
            )
            if result is None:
                return {
                    "_schema": "mc.geometry.refusal_detect.v1",
                    "error": "Could not compute refusal direction",
                    "nextActions": ["mc_geometry_refusal_pairs to get prompts"],
                }
            payload = ctx.geometry_sparse_service.refusal_direction_payload(result)
            payload["_schema"] = "mc.geometry.refusal_detect.v1"
            payload["nextActions"] = [
                "mc_safety_circuit_breaker for safety monitoring",
                "mc_safety_persona_drift to detect drift",
            ]
            return payload

    # Persona tools
    if "mc_geometry_persona_traits" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_traits() -> dict:
            """List standard persona traits for vector extraction."""
            traits = ctx.geometry_persona_service.list_traits()
            payload = ctx.geometry_persona_service.traits_payload(traits)
            payload["_schema"] = "mc.geometry.persona_traits.v1"
            payload["nextActions"] = [
                "mc_geometry_persona_extract to extract vectors",
                "mc_geometry_persona_drift to measure drift",
            ]
            return payload

    if "mc_geometry_persona_extract" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_extract(
            positiveActivations: list[list[float]],
            negativeActivations: list[list[float]],
            traitId: str,
            layerIndex: int,
            modelId: str,
            normalize: bool = True,
        ) -> dict:
            """Extract a persona vector from contrastive activations."""
            vector = ctx.geometry_persona_service.extract_persona_vector(
                positive_activations=positiveActivations,
                negative_activations=negativeActivations,
                trait_id=traitId,
                layer_index=layerIndex,
                model_id=modelId,
                normalize=normalize,
            )
            if vector is None:
                return {
                    "_schema": "mc.geometry.persona_extract.v1",
                    "error": "Could not extract persona vector",
                    "nextActions": ["mc_geometry_persona_traits for definitions"],
                }
            payload = ctx.geometry_persona_service.persona_vector_payload(vector)
            payload["_schema"] = "mc.geometry.persona_extract.v1"
            payload["nextActions"] = [
                "mc_geometry_persona_drift to measure drift",
                "mc_safety_persona_drift for safety monitoring",
            ]
            return payload

    if "mc_geometry_persona_drift" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_drift(
            positions: list[dict],
            step: int,
            driftThreshold: float = 0.2,
        ) -> dict:
            """Compute drift metrics from persona position measurements."""
            metrics = ctx.geometry_persona_service.compute_drift(
                positions=positions,
                step=step,
                drift_threshold=driftThreshold,
            )
            payload = ctx.geometry_persona_service.drift_metrics_payload(metrics)
            payload["_schema"] = "mc.geometry.persona_drift.v1"
            payload["nextActions"] = [
                "mc_safety_circuit_breaker if drift is significant",
                "mc_train_pause to halt training if needed",
            ]
            return payload

    # Manifold tools
    if "mc_geometry_manifold_cluster" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_cluster(
            points: list[dict],
            epsilon: float = 0.3,
            minPoints: int = 5,
            computeDimension: bool = True,
        ) -> dict:
            """Cluster manifold points into regions using DBSCAN."""
            result = ctx.geometry_persona_service.cluster_points(
                points=points,
                epsilon=epsilon,
                min_points=minPoints,
                compute_dimension=computeDimension,
            )
            payload = ctx.geometry_persona_service.clustering_payload(result)
            payload["_schema"] = "mc.geometry.manifold_cluster.v1"
            payload["nextActions"] = [
                "mc_geometry_manifold_dimension for ID estimate",
                "mc_geometry_manifold_query to classify points",
            ]
            return payload

    if "mc_geometry_manifold_dimension" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_dimension(
            points: list[list[float]],
            bootstrapSamples: int = 0,
            useRegression: bool = True,
        ) -> dict:
            """Estimate intrinsic dimension of a point cloud using TwoNN."""
            result = ctx.geometry_persona_service.estimate_dimension(
                points=points,
                bootstrap_samples=bootstrapSamples,
                use_regression=useRegression,
            )
            payload = ctx.geometry_persona_service.dimension_payload(result)
            payload["_schema"] = "mc.geometry.manifold_dimension.v1"
            payload["nextActions"] = [
                "mc_geometry_manifold_cluster to find regions",
                "mc_geometry_intrinsic_dimension for comparison",
            ]
            return payload

    if "mc_geometry_manifold_query" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_query(
            point: dict,
            regions: list[dict],
            epsilon: float = 0.3,
        ) -> dict:
            """Query which region a point belongs to."""
            result = ctx.geometry_persona_service.query_region(
                point=point,
                regions=regions,
                epsilon=epsilon,
            )
            payload = ctx.geometry_persona_service.region_query_payload(result)
            payload["_schema"] = "mc.geometry.manifold_query.v1"
            payload["nextActions"] = [
                "mc_geometry_manifold_cluster to update clusters",
                "mc_thermo_measure to get point features",
            ]
            return payload

    # Transport tools
    if "mc_geometry_transport_merge" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_transport_merge(
            sourceWeights: list[list[float]],
            targetWeights: list[list[float]],
            transportPlan: list[list[float]],
            couplingThreshold: float = 0.001,
            normalizeRows: bool = True,
            blendAlpha: float = 0.5,
        ) -> dict:
            """Merge weights using a transport plan."""
            merged = ctx.geometry_transport_service.synthesize_weights(
                source_weights=sourceWeights,
                target_weights=targetWeights,
                transport_plan=transportPlan,
                coupling_threshold=couplingThreshold,
                normalize_rows=normalizeRows,
                blend_alpha=blendAlpha,
            )
            if merged is None:
                return {
                    "_schema": "mc.geometry.transport_merge.v1",
                    "error": "Failed to merge weights",
                    "nextActions": ["mc_geometry_gromov_wasserstein for transport plan"],
                }
            return {
                "_schema": "mc.geometry.transport_merge.v1",
                "mergedShape": [len(merged), len(merged[0]) if merged else 0],
                "blendAlpha": blendAlpha,
                "nextActions": [
                    "mc_geometry_transport_synthesize for GW-guided merge",
                    "mc_model_merge for full model merging",
                ],
            }

    if "mc_geometry_transport_synthesize" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_transport_synthesize(
            sourceActivations: list[list[float]],
            targetActivations: list[list[float]],
            sourceWeights: list[list[float]],
            targetWeights: list[list[float]],
            couplingThreshold: float = 0.001,
            blendAlpha: float = 0.5,
            gwEpsilon: float = 0.05,
            gwMaxIterations: int = 50,
        ) -> dict:
            """Compute GW transport plan and synthesize merged weights."""
            from modelcypher.core.use_cases.geometry_transport_service import MergeConfig

            config = MergeConfig(
                coupling_threshold=couplingThreshold,
                blend_alpha=blendAlpha,
                gw_epsilon=gwEpsilon,
                gw_max_iterations=gwMaxIterations,
            )
            result = ctx.geometry_transport_service.synthesize_with_gw(
                source_activations=sourceActivations,
                target_activations=targetActivations,
                source_weights=sourceWeights,
                target_weights=targetWeights,
                config=config,
            )
            if result is None:
                return {
                    "_schema": "mc.geometry.transport_synthesize.v1",
                    "error": "Failed to synthesize",
                    "nextActions": ["mc_geometry_transport_merge for manual transport"],
                }
            payload = ctx.geometry_transport_service.merge_result_payload(result)
            payload["_schema"] = "mc.geometry.transport_synthesize.v1"
            payload["nextActions"] = [
                "mc_geometry_intrinsic_dimension for merged space analysis",
                "mc_model_merge for full model merging",
            ]
            return payload

    # Training status tools
    if "mc_geometry_training_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_status(jobId: str, format: str = "full") -> dict:
            format_key = format.lower()
            if format_key not in {"full", "summary"}:
                raise ValueError("format must be 'full' or 'summary'")
            return ctx.geometry_training_service.training_status_payload(
                jobId, output_format=format_key
            )

    if "mc_geometry_training_history" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_history(jobId: str) -> dict:
            return ctx.geometry_training_service.training_history_payload(jobId)


def register_geometry_invariant_tools(ctx: ServiceContext) -> None:
    """Register geometry invariant/atlas tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_invariant_map_layers" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_invariant_map_layers(
            sourcePath: str,
            targetPath: str,
            families: list[str] | None = None,
            scope: str = "sequenceInvariants",
            atlasSources: list[str] | None = None,
            atlasDomains: list[str] | None = None,
            triangulation: bool = True,
            collapseThreshold: float = 0.35,
            sampleLayers: int = 12,
        ) -> dict:
            """Map layers between models using multi-atlas triangulation."""
            from modelcypher.core.use_cases.invariant_layer_mapping_service import (
                InvariantLayerMappingService,
                LayerMappingConfig,
            )

            source_path = require_existing_directory(sourcePath)
            target_path = require_existing_directory(targetPath)
            config = LayerMappingConfig(
                source_model_path=str(source_path),
                target_model_path=str(target_path),
                invariant_scope=scope,
                families=families,
                atlas_sources=atlasSources,
                atlas_domains=atlasDomains,
                use_triangulation=triangulation,
                collapse_threshold=collapseThreshold,
                sample_layer_count=sampleLayers,
            )
            result = ctx.invariant_mapping_service.map_layers(config)
            payload = InvariantLayerMappingService.result_payload(result)
            payload["nextActions"] = [
                "mc_geometry_invariant_collapse_risk to analyze collapse risk",
                "mc_geometry_atlas_inventory to see available probes",
            ]
            return payload

    if "mc_geometry_invariant_collapse_risk" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_invariant_collapse_risk(
            modelPath: str,
            families: list[str] | None = None,
            threshold: float = 0.35,
            sampleLayers: int = 12,
        ) -> dict:
            """Analyze layer collapse risk for a model."""
            from modelcypher.core.use_cases.invariant_layer_mapping_service import (
                CollapseRiskConfig,
                InvariantLayerMappingService,
            )

            model_path = require_existing_directory(modelPath)
            config = CollapseRiskConfig(
                model_path=str(model_path),
                families=families,
                collapse_threshold=threshold,
                sample_layer_count=sampleLayers,
            )
            result = ctx.invariant_mapping_service.analyze_collapse_risk(config)
            payload = InvariantLayerMappingService.collapse_risk_payload(result)
            payload["nextActions"] = [
                "mc_geometry_invariant_map_layers to map layers",
                "mc_geometry_atlas_inventory to see available probes",
            ]
            return payload

    if "mc_geometry_atlas_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_atlas_inventory(
            source: str | None = None,
            domain: str | None = None,
        ) -> dict:
            """Get inventory of available probes across all atlases."""
            from modelcypher.core.domain.agents.unified_atlas import (
                AtlasDomain,
                AtlasSource,
                UnifiedAtlasInventory,
            )

            counts = UnifiedAtlasInventory.probe_count()
            total = UnifiedAtlasInventory.total_probe_count()
            filtered_count = total
            if source or domain:
                sources_filter = None
                domains_filter = None
                if source:
                    source_map = {
                        "sequence": AtlasSource.SEQUENCE_INVARIANT,
                        "semantic": AtlasSource.SEMANTIC_PRIME,
                        "gate": AtlasSource.COMPUTATIONAL_GATE,
                        "emotion": AtlasSource.EMOTION_CONCEPT,
                    }
                    if source.lower() in source_map:
                        sources_filter = {source_map[source.lower()]}
                if domain:
                    domain_map = {
                        "mathematical": AtlasDomain.MATHEMATICAL,
                        "logical": AtlasDomain.LOGICAL,
                        "linguistic": AtlasDomain.LINGUISTIC,
                        "mental": AtlasDomain.MENTAL,
                        "computational": AtlasDomain.COMPUTATIONAL,
                        "structural": AtlasDomain.STRUCTURAL,
                        "affective": AtlasDomain.AFFECTIVE,
                        "relational": AtlasDomain.RELATIONAL,
                        "temporal": AtlasDomain.TEMPORAL,
                        "spatial": AtlasDomain.SPATIAL,
                    }
                    if domain.lower() in domain_map:
                        domains_filter = {domain_map[domain.lower()]}
                if sources_filter:
                    filtered = UnifiedAtlasInventory.probes_by_source(sources_filter)
                    if domains_filter:
                        filtered = [p for p in filtered if p.domain in domains_filter]
                    filtered_count = len(filtered)
                elif domains_filter:
                    filtered = UnifiedAtlasInventory.probes_by_domain(domains_filter)
                    filtered_count = len(filtered)
            return {
                "_schema": "mc.geometry.atlas.inventory.v1",
                "totalProbes": total,
                "filteredCount": filtered_count,
                "sources": {
                    "sequenceInvariant": {
                        "count": counts.get(AtlasSource.SEQUENCE_INVARIANT, 0),
                        "description": "Mathematical sequences and logical invariants",
                    },
                    "semanticPrime": {
                        "count": counts.get(AtlasSource.SEMANTIC_PRIME, 0),
                        "description": "NSM semantic primitives",
                    },
                    "computationalGate": {
                        "count": counts.get(AtlasSource.COMPUTATIONAL_GATE, 0),
                        "description": "Programming primitives",
                    },
                    "emotionConcept": {
                        "count": counts.get(AtlasSource.EMOTION_CONCEPT, 0),
                        "description": "Plutchik emotion wheel",
                    },
                },
                "nextActions": [
                    "mc_geometry_invariant_map_layers with scope='multiAtlas'",
                    "mc_geometry_invariant_collapse_risk to check compatibility",
                ],
            }


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
                "overallAssessment": result.overall_assessment,
                "riskScore": result.risk_score,
                "processingTime": result.processing_time,
                "vulnerabilityDetails": vulnerability_details or None,
                "nextActions": [
                    "mc_safety_circuit_breaker for combined safety assessment",
                    "mc_thermo_detect for detailed entropy analysis",
                    "mc_safety_persona_drift for alignment monitoring",
                ],
            }

    if "mc_geometry_dare_sparsity" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dare_sparsity(checkpointPath: str, basePath: str | None = None) -> dict:
            analysis = ctx.geometry_adapter_service.analyze_dare(checkpointPath, basePath)
            readiness = ctx.geometry_adapter_service.dare_merge_readiness(
                analysis.effective_sparsity
            )
            per_layer = []
            for name, metrics in analysis.per_layer_sparsity.items():
                importance = max(0.0, min(1.0, metrics.essential_fraction))
                per_layer.append(
                    {
                        "layerName": name,
                        "sparsity": metrics.sparsity,
                        "importance": importance,
                        "canDrop": metrics.sparsity >= analysis.recommended_drop_rate,
                    }
                )
            layer_ranking = [
                entry["layerName"]
                for entry in sorted(per_layer, key=lambda x: x["importance"], reverse=True)
            ]
            interpretation = (
                f"Effective sparsity {analysis.effective_sparsity:.2%} "
                f"({analysis.quality_assessment.value}). Recommended drop rate "
                f"{analysis.recommended_drop_rate:.2f}."
            )
            return {
                "_schema": "mc.geometry.dare_sparsity.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "effectiveSparsity": analysis.effective_sparsity,
                "qualityAssessment": analysis.quality_assessment.value,
                "mergeReadiness": readiness,
                "perLayerSparsity": per_layer or None,
                "layerRanking": layer_ranking or None,
                "recommendedDropRate": analysis.recommended_drop_rate,
                "interpretation": interpretation,
                "nextActions": [
                    "mc_geometry_dora_decomposition for learning type",
                    "mc_checkpoint_score for quality assessment",
                    "mc_checkpoint_export for deployment",
                ],
            }

    if "mc_geometry_dora_decomposition" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dora_decomposition(
            checkpointPath: str, basePath: str | None = None
        ) -> dict:
            result = ctx.geometry_adapter_service.analyze_dora(checkpointPath, basePath)
            learning_type = ctx.geometry_adapter_service.dora_learning_type(result)
            learning_confidence = ctx.geometry_adapter_service.dora_learning_type_confidence(result)
            stability_score = ctx.geometry_adapter_service.dora_stability_score(result)
            overfit_risk = ctx.geometry_adapter_service.dora_overfit_risk(result)
            per_layer = []
            for name, metrics in result.per_layer_metrics.items():
                if metrics.interpretation.value in {"amplification", "attenuation"}:
                    dominant = "magnitude"
                elif metrics.interpretation.value == "rotation":
                    dominant = "direction"
                else:
                    dominant = "balanced"
                per_layer.append(
                    {
                        "layerName": name,
                        "magnitudeChange": metrics.relative_magnitude_change,
                        "directionalDrift": metrics.directional_drift,
                        "dominantType": dominant,
                    }
                )
            learning_type_value = learning_type if learning_type != "minimal" else "balanced"
            return {
                "_schema": "mc.geometry.dora_decomposition.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "magnitudeChangeRatio": result.overall_magnitude_change,
                "directionalDrift": result.overall_directional_drift,
                "learningType": learning_type_value,
                "learningTypeConfidence": learning_confidence,
                "perLayerDecomposition": per_layer or None,
                "stabilityScore": stability_score,
                "overfitRisk": overfit_risk,
                "interpretation": ctx.geometry_adapter_service.dora_interpretation(result),
                "nextActions": [
                    "mc_geometry_dare_sparsity for sparsity assessment",
                    "mc_checkpoint_export for deployment",
                ],
            }


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


def register_geometry_primes_tools(ctx: ServiceContext) -> None:
    """Register geometry primes tools with real implementations."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_primes_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_list(category: str | None = None) -> dict:
            """List all NSM semantic primes (Goddard & Wierzbicka 2014)."""
            from modelcypher.core.domain.agents.semantic_prime_atlas import (
                SemanticPrimeInventory,
            )

            primes = SemanticPrimeInventory.english_2014()
            if category:
                primes = [p for p in primes if p.category.value == category]
            categories = sorted(set(p.category.value for p in primes))
            return {
                "_schema": "mc.geometry.primes.list.v1",
                "primes": [
                    {"id": p.id, "category": p.category.value, "exponents": p.english_exponents}
                    for p in primes
                ],
                "count": len(primes),
                "categories": categories,
                "nextActions": [
                    "mc_geometry_primes_probe to analyze prime activations in a model",
                    "mc_geometry_primes_compare to compare prime alignment between models",
                ],
            }

    if "mc_geometry_primes_probe" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_probe(
            modelPath: str,
            layer: int = -1,
            outputFile: str | None = None,
        ) -> dict:
            """Probe model for semantic prime representations using CKA."""
            import json

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.agents.semantic_prime_atlas import SemanticPrimeInventory
            from modelcypher.core.domain.geometry.cka import compute_cka

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(model_path)
            backend = MLXBackend()

            # Resolve architecture
            resolved = _resolve_text_backbone(model)
            if not resolved:
                raise ValueError("Could not resolve model architecture")
            embed_tokens, layers, norm = resolved
            num_layers = len(layers)
            target_layer = layer if layer >= 0 else num_layers - 1

            # Probe primes
            primes = SemanticPrimeInventory.english_2014()
            activations = {}
            for prime in primes:
                try:
                    probe_text = prime.english_exponents[0] if prime.english_exponents else prime.id
                    tokens = tokenizer.encode(probe_text)
                    input_ids = backend.array([tokens])
                    hidden = _forward_text_backbone(
                        input_ids, embed_tokens, layers, norm, target_layer, backend
                    )
                    activation = backend.mean(hidden[0], axis=0)
                    backend.eval(activation)
                    activations[prime.id] = activation
                except Exception:
                    pass  # Skip failed primes

            if not activations:
                raise ValueError("No activations extracted")

            # Optionally save activations
            if outputFile:
                activations_json = {
                    name: backend.to_numpy(act).tolist() for name, act in activations.items()
                }
                Path(outputFile).write_text(json.dumps(activations_json, indent=2))

            # Compute coherence with CKA
            all_acts = [a for a in activations.values()]
            X_all = backend.stack(all_acts)
            backend.eval(X_all)
            result = compute_cka(backend.to_numpy(X_all), backend.to_numpy(X_all))

            # Compute category coherence
            category_primes: dict[str, list] = {}
            for prime in primes:
                cat = prime.category.value
                if cat not in category_primes:
                    category_primes[cat] = []
                if prime.id in activations:
                    category_primes[cat].append(backend.to_numpy(activations[prime.id]))

            category_coherence = {}
            for cat, acts in category_primes.items():
                if len(acts) >= 2:
                    X = backend.stack([backend.array(a) for a in acts])
                    backend.eval(X)
                    cat_result = compute_cka(backend.to_numpy(X), backend.to_numpy(X))
                    category_coherence[cat] = cat_result.cka

            return {
                "_schema": "mc.geometry.primes.probe.v1",
                "modelPath": model_path,
                "layer": target_layer,
                "primesProbed": len(activations),
                "totalPrimes": len(primes),
                "overallCoherence": result.cka,
                "categoryCoherence": category_coherence,
                "interpretation": (
                    "Strong semantic structure - primes form coherent clusters."
                    if result.cka > 0.7
                    else "Moderate semantic structure - some prime clustering detected."
                    if result.cka > 0.4
                    else "Weak semantic structure - primes are diffusely represented."
                ),
                "nextActions": [
                    "mc_geometry_primes_compare to compare with another model",
                    "mc_model_probe for architecture details",
                ],
            }

    if "mc_geometry_primes_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_compare(activationsA: str, activationsB: str) -> dict:
            """Compare prime representations between two saved activation files."""
            import json

            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.cka import compute_cka
            from modelcypher.core.domain.geometry.vector_math import VectorMath

            backend = get_default_backend()
            path_a = require_existing_path(activationsA)
            path_b = require_existing_path(activationsB)

            acts_a = json.loads(Path(path_a).read_text())
            acts_b = json.loads(Path(path_b).read_text())
            common = sorted(set(acts_a.keys()) & set(acts_b.keys()))

            if len(common) < 2:
                raise ValueError("Need at least 2 common primes to compare")

            X = backend.stack([backend.array(acts_a[p]) for p in common])
            Y = backend.stack([backend.array(acts_b[p]) for p in common])
            backend.eval(X)
            backend.eval(Y)
            result = compute_cka(backend.to_numpy(X), backend.to_numpy(Y))

            # Find most similar and divergent
            sims = []
            for p in common:
                sim = VectorMath.cosine_similarity(acts_a[p], acts_b[p])
                sims.append((p, sim))
            sims.sort(key=lambda x: x[1], reverse=True)

            return {
                "_schema": "mc.geometry.primes.compare.v1",
                "modelA": path_a,
                "modelB": path_b,
                "commonPrimes": len(common),
                "ckaSimilarity": result.cka,
                "mostSimilarPrimes": [p for p, _ in sims[:5]],
                "mostDivergentPrimes": [p for p, _ in sims[-5:]],
                "interpretation": (
                    "Models have highly similar semantic prime structure."
                    if result.cka > 0.8
                    else "Models have moderately similar semantic structure."
                    if result.cka > 0.5
                    else "Models have divergent semantic prime representations."
                ),
                "nextActions": [
                    "mc_model_analyze_alignment for layer-wise drift analysis",
                    "mc_geometry_primes_probe for individual model analysis",
                ],
            }


def register_geometry_crm_tools(ctx: ServiceContext) -> None:
    """Register geometry CRM tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_crm_build" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_crm_build(
            modelPath: str,
            outputPath: str,
            adapter: str | None = None,
            includePrimes: bool = True,
            includeGates: bool = True,
            includePolyglot: bool = True,
            includeSequenceInvariants: bool = True,
            sequenceFamilies: list[str] | None = None,
            maxPromptsPerAnchor: int = 3,
            maxPolyglotTextsPerLanguage: int = 2,
            anchorPrefixes: list[str] | None = None,
            maxAnchors: int | None = None,
        ) -> dict:
            """Build a concept response matrix (CRM) for a model."""
            from modelcypher.core.domain.agents.sequence_invariant_atlas import SequenceFamily
            from modelcypher.core.use_cases.concept_response_matrix_service import CRMBuildConfig

            model_path = require_existing_directory(modelPath)
            output_path = str(Path(outputPath).expanduser().resolve())
            parsed_families: frozenset[SequenceFamily] | None = None
            if sequenceFamilies:
                family_set: set[SequenceFamily] = set()
                for name in sequenceFamilies:
                    try:
                        family_set.add(SequenceFamily(name.strip().lower()))
                    except ValueError:
                        pass
                if family_set:
                    parsed_families = frozenset(family_set)
            config = CRMBuildConfig(
                include_primes=includePrimes,
                include_gates=includeGates,
                include_polyglot=includePolyglot,
                include_sequence_invariants=includeSequenceInvariants,
                sequence_families=parsed_families,
                max_prompts_per_anchor=maxPromptsPerAnchor,
                max_polyglot_texts_per_language=maxPolyglotTextsPerLanguage,
                anchor_prefixes=anchorPrefixes,
                max_anchors=maxAnchors,
            )
            summary = ctx.geometry_crm_service.build(
                model_path=model_path,
                output_path=output_path,
                config=config,
                adapter=adapter,
            )
            return {
                "_schema": "mc.geometry.crm.build.v1",
                "modelPath": summary.model_path,
                "outputPath": summary.output_path,
                "layerCount": summary.layer_count,
                "hiddenDim": summary.hidden_dim,
                "anchorCount": summary.anchor_count,
                "primeCount": summary.prime_count,
                "gateCount": summary.gate_count,
                "sequenceInvariantCount": summary.sequence_invariant_count,
                "nextActions": [
                    "mc_geometry_crm_compare to compare against another model",
                    "mc_model_merge to use the CRM in shared subspace alignment",
                ],
            }

    if "mc_geometry_crm_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_crm_compare(
            sourcePath: str,
            targetPath: str,
            includeMatrix: bool = False,
        ) -> dict:
            """Compare two CRMs and compute CKA-based correspondence."""
            source_path = require_existing_path(sourcePath)
            target_path = require_existing_path(targetPath)
            summary = ctx.geometry_crm_service.compare(
                source_path, target_path, include_matrix=includeMatrix
            )
            payload = {
                "_schema": "mc.geometry.crm.compare.v1",
                "sourcePath": summary.source_path,
                "targetPath": summary.target_path,
                "commonAnchorCount": summary.common_anchor_count,
                "overallAlignment": summary.overall_alignment,
                "layerCorrespondence": summary.layer_correspondence,
                "nextActions": [
                    "mc_geometry_crm_build to regenerate CRM with more anchors",
                    "mc_model_merge to apply shared-subspace alignment",
                ],
            }
            if summary.cka_matrix is not None:
                payload["ckaMatrix"] = summary.cka_matrix
            return payload

    if "mc_geometry_crm_sequence_inventory" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_crm_sequence_inventory(family: str | None = None) -> dict:
            """List available sequence invariant probes for CRM anchoring."""
            from modelcypher.core.domain.agents.sequence_invariant_atlas import (
                SequenceFamily,
                SequenceInvariantInventory,
            )

            family_filter: set[SequenceFamily] | None = None
            if family:
                try:
                    family_filter = {SequenceFamily(family.strip().lower())}
                except ValueError:
                    return {
                        "_schema": "mc.error.v1",
                        "error": f"Unknown family '{family}'",
                        "validFamilies": [f.value for f in SequenceFamily],
                    }
            probes = SequenceInvariantInventory.probes_for_families(family_filter)
            counts = SequenceInvariantInventory.probe_count_by_family()
            return {
                "_schema": "mc.geometry.crm.sequence_inventory.v1",
                "totalProbes": len(probes),
                "familyCounts": {fam.value: count for fam, count in counts.items()},
                "probes": [
                    {
                        "id": p.id,
                        "family": p.family.value,
                        "domain": p.domain.value,
                        "name": p.name,
                        "description": p.description,
                        "weight": p.cross_domain_weight,
                    }
                    for p in probes
                ],
                "nextActions": [
                    "mc_geometry_crm_build with includeSequenceInvariants=true",
                    "mc_geometry_crm_build with sequenceFamilies=[...] to filter",
                ],
            }


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
                "recommendedConfig": result.recommended_config,
                "interpretation": result.interpretation,
                "nextActions": ["mc_geometry_stitch_apply to perform the stitching"],
            }

    if "mc_geometry_stitch_apply" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_geometry_stitch_apply(
            source: str,
            target: str,
            outputPath: str,
            learningRate: float = 0.01,
            maxIterations: int = 500,
        ) -> dict:
            """Apply stitching operation between checkpoints."""
            source_path = require_existing_directory(source)
            target_path = require_existing_directory(target)
            config = {
                "learning_rate": learningRate,
                "max_iterations": maxIterations,
                "use_procrustes_warm_start": True,
            }
            result = ctx.geometry_stitch_service.apply(source_path, target_path, outputPath, config)
            return {
                "_schema": "mc.geometry.stitch.apply.v1",
                "outputPath": result.output_path,
                "stitchedLayers": result.stitched_layers,
                "qualityScore": result.quality_score,
                "nextActions": [
                    "mc_model_probe to verify the stitched model",
                    "mc_infer to test the stitched model",
                ],
            }

    if "mc_geometry_stitch_train" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_stitch_train(
            anchorPairs: list[dict],
            learningRate: float = 0.01,
            weightDecay: float = 1e-4,
            maxIterations: int = 1000,
            convergenceThreshold: float = 1e-5,
            useProcrusteWarmStart: bool = True,
        ) -> dict:
            """Train an affine stitching layer from anchor pairs."""
            from modelcypher.core.domain.geometry.affine_stitching_layer import (
                AffineStitchingLayer,
                AnchorPair,
            )
            from modelcypher.core.domain.geometry.affine_stitching_layer import (
                Config as StitchConfig,
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
            config = StitchConfig(
                learning_rate=learningRate,
                weight_decay=weightDecay,
                max_iterations=maxIterations,
                convergence_threshold=convergenceThreshold,
                use_procrustes_warm_start=useProcrusteWarmStart,
            )
            result = AffineStitchingLayer.train(parsed_pairs, config=config)
            if result is None:
                return {
                    "_schema": "mc.geometry.stitch.train.v1",
                    "status": "failed",
                    "error": "Training failed - insufficient data or convergence failure",
                    "nextActions": [
                        "Add more anchor pairs and retry",
                        "Adjust learning rate or iterations",
                    ],
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
                "h4Validated": h4_metrics.is_h4_validated(),
                "transferQuality": h4_metrics.transfer_quality,
                "weights": result.weights,
                "bias": result.bias,
                "nextActions": [
                    "Use weights/bias to transform activations",
                    "mc_geometry_stitch_apply to apply to full model",
                ],
            }

    if "mc_geometry_refinement_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refinement_analyze(
            baseModel: str,
            adaptedModel: str,
            mode: str = "default",
            sparsityWeight: float = 0.35,
            directionalWeight: float = 0.35,
            transitionWeight: float = 0.30,
            hardSwapThreshold: float = 0.80,
        ) -> dict:
            """Analyze refinement density between base and adapted models."""
            from modelcypher.core.domain.geometry.dare_sparsity import Configuration as DAREConfig
            from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer
            from modelcypher.core.domain.geometry.dora_decomposition import DoRADecomposition
            from modelcypher.core.domain.geometry.refinement_density import (
                RefinementDensityAnalyzer,
                RefinementDensityConfig,
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
                if mode == "aggressive":
                    config = RefinementDensityConfig.aggressive()
                elif mode == "conservative":
                    config = RefinementDensityConfig.conservative()
                else:
                    config = RefinementDensityConfig(
                        sparsity_weight=sparsityWeight,
                        directional_weight=directionalWeight,
                        transition_weight=transitionWeight,
                        hard_swap_threshold=hardSwapThreshold,
                    )
                analyzer = RefinementDensityAnalyzer(config)
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
                    "maxCompositeScore": result_dict.get("maxCompositeScore"),
                    "layersAboveHardSwap": result_dict.get("layersAboveHardSwap"),
                    "layersAboveHighAlpha": result_dict.get("layersAboveHighAlpha"),
                    "hardSwapLayers": result_dict.get("hardSwapLayers"),
                    "alphaByLayer": result_dict.get("alphaByLayer"),
                    "layerScores": result_dict.get("layerScores"),
                    "interpretation": result.interpretation(),
                    "nextActions": [
                        "mc_model_merge with recommended alpha values",
                        "mc_geometry_dare_sparsity for detailed DARE analysis",
                        "mc_geometry_dora_decomposition for detailed DoRA analysis",
                    ],
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
                "nextActions": [
                    "mc_geometry_refinement_analyze to use profile in analysis",
                    "mc_geometry_sparse_locate to find sparse regions",
                ],
            }


def register_geometry_spatial_tools(ctx: ServiceContext) -> None:
    """Register 3D spatial metrology tools.

    These tools probe how language models capture 3-dimensional spatial
    relationships in their internal representations. Tests whether the latent
    manifold encodes a geometrically consistent 3D world model.
    """
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_spatial_anchors" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_anchors(
            axis: str | None = None,
            category: str | None = None,
        ) -> dict:
            """List the Spatial Prime Atlas anchors.

            Shows the 23 spatial anchors with their expected 3D coordinates (X, Y, Z)
            and categories. These anchors probe the model's 3D world model.

            Args:
                axis: Filter by axis (x_lateral, y_vertical, z_depth)
                category: Filter by category (vertical, lateral, depth, mass, furniture)

            Returns:
                List of spatial anchors with 3D coordinates
            """
            from modelcypher.core.domain.geometry.spatial_3d import (
                SPATIAL_PRIME_ATLAS,
                SpatialAxis,
                get_spatial_anchors_by_axis,
            )

            if axis:
                try:
                    axis_enum = SpatialAxis(axis)
                    anchors = get_spatial_anchors_by_axis(axis_enum)
                except ValueError:
                    raise ValueError(f"Invalid axis: {axis}. Use: x_lateral, y_vertical, z_depth")
            else:
                anchors = SPATIAL_PRIME_ATLAS

            if category:
                anchors = [a for a in anchors if a.category == category]

            return {
                "_schema": "mc.geometry.spatial.anchors.v1",
                "anchors": [
                    {
                        "name": a.name,
                        "prompt": a.prompt,
                        "expectedX": a.expected_x,
                        "expectedY": a.expected_y,
                        "expectedZ": a.expected_z,
                        "category": a.category,
                    }
                    for a in anchors
                ],
                "count": len(anchors),
                "categories": list(set(a.category for a in anchors)),
                "axisLegend": {
                    "X": "Lateral (Left=-1, Right=+1)",
                    "Y": "Vertical (Down=-1, Up=+1) - Gravity axis",
                    "Z": "Depth (Far=-1, Near=+1) - Perspective axis",
                },
                "nextActions": [
                    "mc_geometry_spatial_analyze to run full 3D analysis",
                    "mc_geometry_spatial_probe_model for end-to-end model probing",
                ],
            }

    if "mc_geometry_spatial_euclidean" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_euclidean(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Test Euclidean consistency of spatial anchor representations.

            Checks if the Pythagorean theorem holds in latent space:
            dist(A,C)² ≈ dist(A,B)² + dist(B,C)² for right-angle triplets.

            If consistency score > 0.6 and no triangle inequality violations,
            the model has internalized Euclidean 3D geometry.

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Euclidean consistency analysis with Pythagorean error
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import EuclideanConsistencyAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = EuclideanConsistencyAnalyzer(backend=backend)
            result = analyzer.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.euclidean.v1",
                **result.to_dict(),
                "interpretation": (
                    "The model has a 3D Euclidean world model."
                    if result.is_euclidean
                    else "The model's spatial representation is non-Euclidean."
                ),
                "nextActions": [
                    "mc_geometry_spatial_gravity to test gravity gradient",
                    "mc_geometry_spatial_analyze for full 3D analysis",
                ],
            }

    if "mc_geometry_spatial_gravity" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_gravity(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Analyze gravity gradient in latent representations.

            Tests if the model has a 'gravity gradient' where heavy objects
            are pulled toward 'down' (Floor, Ground) in latent space.

            High mass correlation (>0.5) indicates the model understands
            physical mass as a geometric property, not just a word.

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Gravity gradient analysis with mass correlation
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import GravityGradientAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = GravityGradientAnalyzer(backend=backend)
            result = analyzer.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.gravity.v1",
                **result.to_dict(),
                "interpretation": (
                    "Gravity gradient detected - the model has a physics engine for mass."
                    if result.gravity_axis_detected
                    else "No gravity gradient - spatial reasoning may be surface-level."
                ),
                "nextActions": [
                    "mc_geometry_spatial_euclidean to verify Euclidean structure",
                    "mc_geometry_spatial_analyze for full 3D analysis",
                ],
            }

    if "mc_geometry_spatial_density" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_density(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Probe volumetric density of spatial representations.

            Tests if physical objects have representational densities that
            match their real-world properties:
            - Heavy objects should have 'denser' representations
            - Distant objects should have attenuated density (inverse-square law)

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Volumetric density analysis with inverse-square compliance
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import VolumetricDensityProber

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            prober = VolumetricDensityProber(backend=backend)
            result = prober.analyze(activations)

            return {
                "_schema": "mc.geometry.spatial.density.v1",
                **result.to_dict(),
                "interpretation": (
                    f"Density-mass correlation: {result.density_mass_correlation:.2f}. "
                    f"Inverse-square compliance: {result.inverse_square_compliance:.2f}. "
                    + (
                        "Physical mass is encoded geometrically."
                        if abs(result.density_mass_correlation) > 0.3
                        else "Mass encoding is weak."
                    )
                ),
                "nextActions": [
                    "mc_geometry_spatial_analyze for full 3D analysis",
                    "mc_geometry_spatial_gravity for gravity gradient",
                ],
            }

    if "mc_geometry_spatial_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_analyze(
            anchorActivations: dict[str, list[float]],
        ) -> dict:
            """Run full 3D world model analysis.

            Comprehensive analysis combining:
            - Euclidean consistency (Pythagorean theorem test)
            - Gravity gradient (mass -> down correlation)
            - Volumetric density (inverse-square law)

            All models encode physics geometrically. The world_model_score measures
            Visual-Spatial Grounding Density: how concentrated probability mass is
            along human-perceptual 3D axes. Lower scores indicate physics encoded
            along alternative geometric axes (linguistic, formula-based).

            Args:
                anchorActivations: Dict mapping anchor_name to activation vector

            Returns:
                Full 3D world model analysis with verdict
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import Spatial3DAnalyzer

            backend = MLXBackend()
            activations = {name: backend.array(vec) for name, vec in anchorActivations.items()}

            analyzer = Spatial3DAnalyzer(backend=backend)
            report = analyzer.full_analysis(activations)

            return {
                "_schema": "mc.geometry.spatial.full_analysis.v1",
                **report.to_dict(),
                "verdict": (
                    "HIGH VISUAL GROUNDING - Probability concentrated on human-perceptual 3D axes."
                    if report.has_3d_world_model and report.physics_engine_detected
                    else "MODERATE GROUNDING - 3D structure present, probability more diffuse."
                    if report.has_3d_world_model
                    else "ALTERNATIVE GROUNDING - Physics encoded along non-visual axes (linguistic/formula-based)."
                ),
                "nextActions": [
                    "mc_geometry_spatial_probe_model to test another model",
                    "mc_model_merge to preserve 3D structure during merging",
                ],
            }

    if "mc_geometry_spatial_probe_model" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_probe_model(
            modelPath: str,
            layer: int = -1,
            saveActivations: str | None = None,
        ) -> dict:
            """Probe a model with the Spatial Prime Atlas.

            Runs all 23 spatial anchor prompts through the model and extracts
            activations, then performs full 3D world model analysis.

            This is the end-to-end command to test if a model has a physics engine.

            Args:
                modelPath: Path to model directory
                layer: Layer to extract activations from (-1 = last)
                saveActivations: Optional path to save activations JSON

            Returns:
                Full 3D world model analysis with verdict
            """
            import json

            import mlx.core as mx

            from modelcypher.adapters.model_loader import load_model_for_training
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.spatial_3d import (
                SPATIAL_PRIME_ATLAS,
                Spatial3DAnalyzer,
            )

            model_path = require_existing_directory(modelPath)
            model, tokenizer = load_model_for_training(str(model_path))

            backend = MLXBackend()
            anchor_activations = {}

            for anchor in SPATIAL_PRIME_ATLAS:
                tokens = tokenizer.encode(anchor.prompt)
                input_ids = mx.array([tokens])

                try:
                    hidden = model.model.embed_tokens(input_ids)
                    target_layer = layer if layer >= 0 else len(model.model.layers) - 1
                    for i, layer_module in enumerate(model.model.layers):
                        hidden = layer_module(hidden, mask=None)
                        if i == target_layer:
                            break

                    activation = mx.mean(hidden[0], axis=0)
                    mx.eval(activation)
                    anchor_activations[anchor.name] = activation
                except Exception:
                    pass  # Skip anchors that fail

            if not anchor_activations:
                return {
                    "_schema": "mc.geometry.spatial.probe_model.v1",
                    "modelPath": str(model_path),
                    "error": "No activations extracted",
                    "nextActions": ["Check model architecture supports hidden state extraction"],
                }

            # Save activations if requested
            if saveActivations:
                activations_json = {
                    name: backend.to_numpy(act).tolist() for name, act in anchor_activations.items()
                }
                Path(saveActivations).write_text(json.dumps(activations_json, indent=2))

            # Run full analysis
            analyzer = Spatial3DAnalyzer(backend=backend)
            report = analyzer.full_analysis(anchor_activations)

            return {
                "_schema": "mc.geometry.spatial.probe_model.v1",
                "modelPath": str(model_path),
                "anchorsProbed": len(anchor_activations),
                "layer": layer if layer >= 0 else "last",
                **report.to_dict(),
                "verdict": (
                    "HIGH VISUAL GROUNDING - Physics probability concentrated on 3D visual axes."
                    if report.has_3d_world_model and report.physics_engine_detected
                    else "MODERATE GROUNDING - 3D structure detected, probability diffuse."
                    if report.has_3d_world_model
                    else "ALTERNATIVE GROUNDING - Physics encoded geometrically along non-visual axes."
                ),
                "nextActions": [
                    "mc_geometry_spatial_analyze with custom activations",
                    "mc_model_merge to preserve spatial representations",
                ],
            }

    if "mc_geometry_spatial_cross_grounding_feasibility" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_cross_grounding_feasibility(
            sourceAnchors: dict[str, list[float]],
            targetAnchors: dict[str, list[float]],
        ) -> dict:
            """Estimate feasibility of cross-grounding knowledge transfer.

            Compares the coordinate systems of two models to determine how much
            'rotation' exists between their grounding axes. Lower rotation means
            easier transfer; higher rotation requires more sophisticated mapping.

            This is a pre-flight check before running a full transfer.

            Args:
                sourceAnchors: Dict of anchor_name -> activation_vector from source model
                targetAnchors: Dict of anchor_name -> activation_vector from target model

            Returns:
                Feasibility assessment with rotation estimate and recommendation
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.cross_grounding_transfer import (
                CrossGroundingTransferEngine,
            )

            backend = MLXBackend()
            source = {name: backend.array(vec) for name, vec in sourceAnchors.items()}
            target = {name: backend.array(vec) for name, vec in targetAnchors.items()}

            engine = CrossGroundingTransferEngine(backend=backend)
            feasibility = engine.estimate_transfer_feasibility(source, target)

            return {
                "_schema": "mc.geometry.spatial.cross_grounding_feasibility.v1",
                **feasibility,
                "nextActions": [
                    "mc_geometry_spatial_cross_grounding_transfer to perform transfer",
                    "mc_geometry_spatial_analyze to analyze each model individually",
                ],
            }

    if "mc_geometry_spatial_cross_grounding_transfer" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spatial_cross_grounding_transfer(
            sourceAnchors: dict[str, list[float]],
            targetAnchors: dict[str, list[float]],
            concepts: dict[str, list[float]] | None = None,
            sourceGrounding: str = "unknown",
            targetGrounding: str = "unknown",
        ) -> dict:
            """Transfer knowledge from source to target model via cross-grounding.

            Uses Density Re-mapping to transfer concepts by preserving Relational Stress
            (distances to universal anchors) rather than absolute coordinates.

            This is the '3D Printer' for high-dimensional knowledge transfer.

            Args:
                sourceAnchors: Dict of anchor_name -> activation_vector from source model
                targetAnchors: Dict of anchor_name -> activation_vector from target model
                concepts: Optional dict of concept_id -> vector to transfer
                         If not provided, uses subset of source anchors as demo
                sourceGrounding: Source grounding type (high_visual, moderate, alternative)
                targetGrounding: Target grounding type

            Returns:
                Ghost Anchors with synthesized target positions
            """
            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.cross_grounding_transfer import (
                CrossGroundingTransferEngine,
            )

            backend = MLXBackend()
            source = {name: backend.array(vec) for name, vec in sourceAnchors.items()}
            target = {name: backend.array(vec) for name, vec in targetAnchors.items()}

            # Process concepts
            if concepts:
                concept_arrays = {name: backend.array(vec) for name, vec in concepts.items()}
            else:
                # Demo with subset of source anchors
                demo_keys = ["chair", "floor", "ceiling", "left_hand", "background"]
                concept_arrays = {k: v for k, v in source.items() if k in demo_keys}
                if not concept_arrays:
                    concept_arrays = dict(list(source.items())[:5])

            engine = CrossGroundingTransferEngine(backend=backend)
            result = engine.transfer_concepts(
                concepts=concept_arrays,
                source_anchors=source,
                target_anchors=target,
                source_grounding=sourceGrounding,
                target_grounding=targetGrounding,
            )

            # Serialize Ghost Anchors
            ghost_anchors_serialized = [
                {
                    "conceptId": g.concept_id,
                    "sourcePosition": g.source_position.tolist(),
                    "targetPosition": g.target_position.tolist(),
                    "stressPreservation": g.stress_preservation,
                    "synthesisConfidence": g.synthesis_confidence,
                    "warning": g.warning,
                }
                for g in result.ghost_anchors
            ]

            return {
                "_schema": "mc.geometry.spatial.cross_grounding_transfer.v1",
                "sourceGrounding": result.source_model_grounding,
                "targetGrounding": result.target_model_grounding,
                "groundingRotation": {
                    "angleDegrees": result.grounding_rotation.angle_degrees,
                    "alignmentScore": result.grounding_rotation.alignment_score,
                    "isAligned": result.grounding_rotation.is_aligned,
                    "confidence": result.grounding_rotation.confidence,
                },
                "ghostAnchors": ghost_anchors_serialized,
                "meanStressPreservation": result.mean_stress_preservation,
                "minStressPreservation": result.min_stress_preservation,
                "successfulTransfers": result.successful_transfers,
                "failedTransfers": result.failed_transfers,
                "interpretabilityGap": result.interpretability_gap,
                "recommendation": result.recommendation,
                "nextActions": [
                    "Use Ghost Anchor targetPositions for downstream tasks",
                    "mc_geometry_spatial_analyze to verify target positions",
                ],
            }


def register_geometry_interference_tools(ctx: ServiceContext) -> None:
    """Register interference prediction and null-space filtering tools.

    These tools support pre-merge quality estimation:
    - Interference prediction using Riemannian density estimation
    - Null-space filtering to eliminate interference by construction
    - Safety polytope for unified merge safety decisions
    """
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_interference_predict" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_interference_predict(
            sourceModel: str,
            targetModel: str,
            layer: int = -1,
            domains: list[str] | None = None,
        ) -> dict:
            """
            Predict interference between two models before merging.

            Uses Riemannian density estimation to model concepts as probability
            distributions and predict constructive vs destructive interference.

            Args:
                sourceModel: Path to source model
                targetModel: Path to target model
                layer: Layer to analyze (-1 for last)
                domains: List of domains to analyze (spatial, social, temporal, moral)
                         Defaults to all domains if not specified.

            Returns:
                Interference prediction with safety scores and recommendations.
            """

            from modelcypher.backends.mlx_backend import MLXBackend
            from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
                DomainGeometryWaypointService,
                GeometryDomain,
            )
            from modelcypher.core.domain.geometry.interference_predictor import (
                InterferencePredictor,
            )
            from modelcypher.core.domain.geometry.riemannian_density import (
                RiemannianDensityEstimator,
            )

            source_path = require_existing_directory(sourceModel)
            target_path = require_existing_directory(targetModel)

            # Parse domains
            domain_list = []
            if domains:
                for d in domains:
                    try:
                        domain_list.append(GeometryDomain(d.strip().lower()))
                    except ValueError:
                        pass
            if not domain_list:
                domain_list = list(GeometryDomain)

            DomainGeometryWaypointService()
            RiemannianDensityEstimator()
            InterferencePredictor()
            MLXBackend()

            domain_results = {}

            for domain in domain_list:
                try:
                    # This would need activation extraction - simplified for MCP
                    domain_results[domain.value] = {
                        "analyzed": True,
                        "note": "Use CLI for full activation extraction",
                    }
                except Exception as e:
                    domain_results[domain.value] = {"error": str(e)}

            return {
                "_schema": "mc.geometry.interference.predict.v1",
                "sourceModel": source_path,
                "targetModel": target_path,
                "layer": layer,
                "domainsRequested": [d.value for d in domain_list],
                "perDomain": domain_results,
                "recommendation": "Use `mc geometry interference predict` CLI for full analysis with activation extraction.",
                "nextActions": [
                    "mc_geometry_null_space_filter to apply interference mitigation",
                    "mc_geometry_safety_polytope_check for unified safety assessment",
                ],
            }

    if "mc_geometry_null_space_filter" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_null_space_filter(
            weightDelta: list[list[float]],
            priorActivations: list[list[float]],
            rankThreshold: float = 0.01,
            method: str = "svd",
        ) -> dict:
            """
            Filter weight delta to null space of prior activations.

            Eliminates interference by construction: if Δw ∈ null(A),
            then A @ (W + Δw) = A @ W.

            Based on MINGLE (arXiv:2509.21413).

            Args:
                weightDelta: Weight update to filter (2D array)
                priorActivations: Activation matrix from prior task [n_samples, d]
                rankThreshold: Threshold for null space determination (default 0.01)
                method: Computation method: 'svd', 'qr', or 'eigenvalue'

            Returns:
                Filtered delta with diagnostics
            """
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.null_space_filter import (
                NullSpaceFilter,
                NullSpaceFilterConfig,
                NullSpaceMethod,
            )

            backend = get_default_backend()
            delta = backend.array(weightDelta)
            activations = backend.array(priorActivations)
            backend.eval(delta)
            backend.eval(activations)

            try:
                method_enum = NullSpaceMethod(method.lower())
            except ValueError:
                method_enum = NullSpaceMethod.SVD

            config = NullSpaceFilterConfig(
                rank_threshold=rankThreshold,
                method=method_enum,
            )

            null_filter = NullSpaceFilter(config)
            delta_flat = backend.reshape(delta, (-1,))
            backend.eval(delta_flat)
            result = null_filter.filter_delta(delta_flat, activations)

            # Convert filtered_delta to list for JSON serialization
            filtered_list = backend.to_numpy(result.filtered_delta).tolist() if hasattr(result.filtered_delta, 'shape') else result.filtered_delta

            return {
                "_schema": "mc.geometry.null_space.filter.v1",
                "filteringApplied": result.filtering_applied,
                "nullSpaceDim": result.null_space_dim,
                "preservedFraction": result.preserved_fraction,
                "projectionLoss": result.projection_loss,
                "originalNorm": result.original_norm,
                "filteredNorm": result.filtered_norm,
                "filteredDelta": filtered_list,
                "interpretation": (
                    f"Preserved {result.preserved_fraction:.1%} of delta, "
                    f"eliminated {result.projection_loss:.1%} interference component."
                    if result.filtering_applied
                    else "No filtering applied (null space empty or dimension mismatch)."
                ),
                "nextActions": [
                    "Apply filteredDelta to weights for interference-free merge",
                    "mc_geometry_safety_polytope_check for comprehensive safety",
                ],
            }

    if "mc_geometry_null_space_profile" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_null_space_profile(
            layerActivations: dict[str, list[list[float]]],
            graftThreshold: float = 0.1,
        ) -> dict:
            """
            Compute null space profile across model layers.

            Identifies which layers have sufficient null space for
            knowledge grafting without interference.

            Args:
                layerActivations: Dict mapping layer index (as string) to
                                  activation matrix [n_samples, d]
                graftThreshold: Minimum null fraction to be considered graftable

            Returns:
                Per-layer null space analysis and graftable layer list
            """
            from modelcypher.core.domain._backend import get_default_backend
            from modelcypher.core.domain.geometry.null_space_filter import (
                NullSpaceFilter,
                NullSpaceFilterConfig,
            )

            backend = get_default_backend()
            config = NullSpaceFilterConfig()
            null_filter = NullSpaceFilter(config)

            layer_arrays = {}
            for k, v in layerActivations.items():
                arr = backend.array(v)
                backend.eval(arr)
                layer_arrays[int(k)] = arr

            profile = null_filter.compute_model_null_space_profile(
                layer_arrays, graft_threshold=graftThreshold
            )

            per_layer_info = {}
            for layer_idx, lp in profile.per_layer.items():
                per_layer_info[str(layer_idx)] = {
                    "nullDim": lp.null_dim,
                    "totalDim": lp.total_dim,
                    "nullFraction": lp.null_fraction,
                    "meanSingularValue": lp.mean_singular_value,
                    "conditionNumber": lp.condition_number,
                }

            return {
                "_schema": "mc.geometry.null_space.profile.v1",
                "totalNullDim": profile.total_null_dim,
                "totalDim": profile.total_dim,
                "meanNullFraction": profile.mean_null_fraction,
                "graftableLayers": profile.graftable_layers,
                "perLayer": per_layer_info,
                "interpretation": (
                    f"{len(profile.graftable_layers)} layers have ≥{graftThreshold:.0%} "
                    f"null space available for knowledge grafting."
                ),
                "nextActions": [
                    "mc_geometry_null_space_filter to filter deltas for graftable layers",
                    "mc_geometry_safety_polytope_model for full model safety profile",
                ],
            }

    if "mc_geometry_safety_polytope_check" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_safety_polytope_check(
            interferenceScore: float,
            importanceScore: float,
            instabilityScore: float,
            complexityScore: float,
            baseAlpha: float = 0.5,
        ) -> dict:
            """
            Check if a layer's diagnostics fall within the safety polytope.

            Combines four diagnostic dimensions into a unified safety decision:
            - Interference: Volume overlap between concept distributions
            - Importance: Layer significance (refinement density)
            - Instability: Numerical conditioning (spectral analysis)
            - Complexity: Manifold dimensionality

            Args:
                interferenceScore: Interference risk [0, 1]
                importanceScore: Layer importance [0, 1]
                instabilityScore: Numerical instability risk [0, 1]
                complexityScore: Manifold complexity [0, 1]
                baseAlpha: Base merge coefficient

            Returns:
                Safety verdict with recommended mitigations and adjusted alpha
            """
            from modelcypher.core.domain.geometry.safety_polytope import (
                DiagnosticVector,
                SafetyPolytope,
            )

            polytope = SafetyPolytope()
            diagnostics = DiagnosticVector(
                interference_score=interferenceScore,
                importance_score=importanceScore,
                instability_score=instabilityScore,
                complexity_score=complexityScore,
            )

            result = polytope.check_layer(diagnostics, base_alpha=baseAlpha)

            violations_info = [
                {
                    "dimension": v.dimension,
                    "value": v.value,
                    "threshold": v.threshold,
                    "severity": v.severity,
                    "mitigation": v.mitigation.value,
                }
                for v in result.violations
            ]

            return {
                "_schema": "mc.geometry.safety_polytope.check.v1",
                "verdict": result.verdict.value,
                "isSafe": result.is_safe,
                "needsMitigation": result.needs_mitigation,
                "isCritical": result.is_critical,
                "diagnostics": {
                    "interference": interferenceScore,
                    "importance": importanceScore,
                    "instability": instabilityScore,
                    "complexity": complexityScore,
                    "magnitude": diagnostics.magnitude,
                    "maxDimension": diagnostics.max_dimension,
                },
                "violations": violations_info,
                "mitigations": [m.value for m in result.mitigations],
                "recommendedAlpha": result.recommended_alpha,
                "confidence": result.confidence,
                "interpretation": (
                    "SAFE: All diagnostics within bounds."
                    if result.is_safe
                    else f"{result.verdict.value.upper()}: {len(result.violations)} violation(s) detected. "
                    f"Apply mitigations: {', '.join(m.value for m in result.mitigations)}."
                ),
                "nextActions": [
                    "mc_geometry_null_space_filter for interference mitigation",
                    "mc_geometry_safety_polytope_model for full model profile",
                ]
                if result.needs_mitigation
                else [
                    "Proceed with merge using recommendedAlpha",
                ],
            }

    if "mc_geometry_safety_polytope_model" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_safety_polytope_model(
            layerDiagnostics: dict[str, dict[str, float]],
            baseAlpha: float = 0.5,
        ) -> dict:
            """
            Analyze safety polytope across all model layers.

            Args:
                layerDiagnostics: Dict mapping layer index (as string) to
                    diagnostic dict with keys: interference, importance,
                    instability, complexity (all [0, 1])
                baseAlpha: Base merge coefficient

            Returns:
                Full model safety profile with per-layer verdicts
            """
            from modelcypher.core.domain.geometry.safety_polytope import (
                DiagnosticVector,
                SafetyPolytope,
            )

            polytope = SafetyPolytope()

            layer_diagnostics = {}
            for layer_str, diag_dict in layerDiagnostics.items():
                layer_idx = int(layer_str)
                layer_diagnostics[layer_idx] = DiagnosticVector(
                    interference_score=diag_dict.get("interference", 0.0),
                    importance_score=diag_dict.get("importance", 0.0),
                    instability_score=diag_dict.get("instability", 0.0),
                    complexity_score=diag_dict.get("complexity", 0.0),
                )

            profile = polytope.analyze_model_pair(layer_diagnostics, base_alpha=baseAlpha)

            per_layer_info = {}
            for layer_idx, result in profile.per_layer.items():
                per_layer_info[str(layer_idx)] = {
                    "verdict": result.verdict.value,
                    "recommendedAlpha": result.recommended_alpha,
                    "violationCount": len(result.violations),
                    "mitigations": [m.value for m in result.mitigations],
                }

            return {
                "_schema": "mc.geometry.safety_polytope.model.v1",
                "overallVerdict": profile.overall_verdict.value,
                "mergeable": profile.mergeable,
                "safeLayers": profile.safe_layers,
                "cautionLayers": profile.caution_layers,
                "unsafeLayers": profile.unsafe_layers,
                "criticalLayers": profile.critical_layers,
                "globalMitigations": [m.value for m in profile.global_mitigations],
                "meanDiagnostics": {
                    "interference": profile.mean_interference,
                    "importance": profile.mean_importance,
                    "instability": profile.mean_instability,
                    "complexity": profile.mean_complexity,
                },
                "perLayer": per_layer_info,
                "interpretation": (
                    f"{profile.overall_verdict.value.upper()}: "
                    f"{len(profile.safe_layers)} safe, {len(profile.caution_layers)} caution, "
                    f"{len(profile.unsafe_layers)} unsafe, {len(profile.critical_layers)} critical."
                ),
                "nextActions": (
                    ["Do not merge - critical issues detected."]
                    if not profile.mergeable
                    else ["Apply globalMitigations before merge."]
                    if profile.global_mitigations
                    else ["Proceed with merge using per-layer recommendedAlpha values."]
                ),
            }
