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
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .common import (
    READ_ONLY_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
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
                    model, text,
                    max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0, top_p=1.0,
                )
                text_to_analyze = response.get("response", "")
                model_id = Path(model).name if Path(model).exists() else model
            else:
                text_to_analyze = text
                model_id = "input-text"
            detection = ctx.geometry_service.detect_path(
                text_to_analyze, model_id=model_id,
                prompt_id="mcp-path-detect", threshold=threshold,
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
                    modelA, prompt, max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0, top_p=1.0,
                )
                response_b = ctx.inference_engine.infer(
                    modelB, prompt, max_tokens=DEFAULT_PATH_MAX_TOKENS,
                    temperature=0.0, top_p=1.0,
                )
                text_to_analyze_a = response_a.get("response", "")
                text_to_analyze_b = response_b.get("response", "")
                model_id_a = Path(modelA).name if Path(modelA).exists() else modelA
                model_id_b = Path(modelB).name if Path(modelB).exists() else modelB
            else:
                raise ValueError("Provide textA/textB or modelA/modelB with prompt.")
            result = ctx.geometry_service.compare_paths(
                text_a=text_to_analyze_a, text_b=text_to_analyze_b,
                model_a=model_id_a, model_b=model_id_b,
                prompt_id="mcp-path-compare", threshold=threshold,
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
                source_points=sourcePoints, target_points=targetPoints,
                epsilon=epsilon, max_iterations=maxIterations,
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
                points=points, use_regression=useRegression,
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
                points=points, max_dimension=maxDimension, num_steps=numSteps,
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
                domain_stats=domainStats, baseline_stats=baselineStats,
                domain_name=domainName, base_rank=baseRank,
                sparsity_threshold=sparsityThreshold,
            )
            payload = ctx.geometry_sparse_service.analysis_payload(result)
            payload["_schema"] = "mc.geometry.sparse_locate.v1"
            payload["nextActions"] = [
                "mc_geometry_dare_sparsity for DARE analysis",
                "mc_geometry_dora_decomposition for magnitude separation",
            ]
            return payload

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
                layer_index=layerIndex, model_id=modelId, normalize=normalize,
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
                trait_id=traitId, layer_index=layerIndex,
                model_id=modelId, normalize=normalize,
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
                positions=positions, step=step, drift_threshold=driftThreshold,
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
                points=points, epsilon=epsilon,
                min_points=minPoints, compute_dimension=computeDimension,
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
                points=points, bootstrap_samples=bootstrapSamples,
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
                point=point, regions=regions, epsilon=epsilon,
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
                source_weights=sourceWeights, target_weights=targetWeights,
                transport_plan=transportPlan, coupling_threshold=couplingThreshold,
                normalize_rows=normalizeRows, blend_alpha=blendAlpha,
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
                coupling_threshold=couplingThreshold, blend_alpha=blendAlpha,
                gw_epsilon=gwEpsilon, gw_max_iterations=gwMaxIterations,
            )
            result = ctx.geometry_transport_service.synthesize_with_gw(
                source_activations=sourceActivations,
                target_activations=targetActivations,
                source_weights=sourceWeights, target_weights=targetWeights,
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
            return ctx.geometry_training_service.training_status_payload(jobId, output_format=format_key)

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
                InvariantLayerMappingService, LayerMappingConfig,
            )
            source_path = require_existing_directory(sourcePath)
            target_path = require_existing_directory(targetPath)
            config = LayerMappingConfig(
                source_model_path=str(source_path),
                target_model_path=str(target_path),
                invariant_scope=scope, families=families,
                atlas_sources=atlasSources, atlas_domains=atlasDomains,
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
                InvariantLayerMappingService, CollapseRiskConfig,
            )
            model_path = require_existing_directory(modelPath)
            config = CollapseRiskConfig(
                model_path=str(model_path), families=families,
                collapse_threshold=threshold, sample_layer_count=sampleLayers,
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
                AtlasSource, AtlasDomain, UnifiedAtlasInventory,
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
