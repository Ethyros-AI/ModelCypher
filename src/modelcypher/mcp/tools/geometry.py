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
            readiness = ctx.geometry_adapter_service.dare_merge_readiness(analysis.effective_sparsity)
            per_layer = []
            for name, metrics in analysis.per_layer_sparsity.items():
                importance = max(0.0, min(1.0, metrics.essential_fraction))
                per_layer.append({
                    "layerName": name,
                    "sparsity": metrics.sparsity,
                    "importance": importance,
                    "canDrop": metrics.sparsity >= analysis.recommended_drop_rate,
                })
            layer_ranking = [entry["layerName"] for entry in sorted(per_layer, key=lambda x: x["importance"], reverse=True)]
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
        def mc_geometry_dora_decomposition(checkpointPath: str, basePath: str | None = None) -> dict:
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
                per_layer.append({
                    "layerName": name,
                    "magnitudeChange": metrics.relative_magnitude_change,
                    "directionalDrift": metrics.directional_drift,
                    "dominantType": dominant,
                })
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


def register_geometry_primes_tools(ctx: ServiceContext) -> None:
    """Register geometry primes tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_geometry_primes_list" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_list() -> dict:
            """List all semantic prime anchors."""
            primes = ctx.geometry_primes_service.list_primes()
            return {
                "_schema": "mc.geometry.primes.list.v1",
                "primes": [
                    {"id": p.id, "name": p.name, "category": p.category, "exponents": p.exponents}
                    for p in primes
                ],
                "count": len(primes),
                "nextActions": [
                    "mc_geometry_primes_probe to analyze prime activations in a model",
                    "mc_geometry_primes_compare to compare prime alignment between models",
                ],
            }

    if "mc_geometry_primes_probe" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_probe(modelPath: str) -> dict:
            """Probe model for prime activation patterns."""
            model_path = require_existing_directory(modelPath)
            activations = ctx.geometry_primes_service.probe(model_path)
            return {
                "_schema": "mc.geometry.primes.probe.v1",
                "modelPath": model_path,
                "activations": [
                    {"primeId": a.prime_id, "activationStrength": a.activation_strength, "layerActivations": a.layer_activations}
                    for a in activations
                ],
                "count": len(activations),
                "nextActions": [
                    "mc_geometry_primes_compare to compare with another model",
                    "mc_model_probe for architecture details",
                ],
            }

    if "mc_geometry_primes_compare" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_primes_compare(modelA: str, modelB: str) -> dict:
            """Compare prime alignment between two models."""
            path_a = require_existing_directory(modelA)
            path_b = require_existing_directory(modelB)
            result = ctx.geometry_primes_service.compare(path_a, path_b)
            return {
                "_schema": "mc.geometry.primes.compare.v1",
                "modelA": path_a,
                "modelB": path_b,
                "alignmentScore": result.alignment_score,
                "divergentPrimes": result.divergent_primes,
                "convergentPrimes": result.convergent_primes,
                "interpretation": result.interpretation,
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
                model_path=model_path, output_path=output_path, config=config, adapter=adapter,
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
            summary = ctx.geometry_crm_service.compare(source_path, target_path, include_matrix=includeMatrix)
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
                SequenceFamily, SequenceInvariantInventory,
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
                    {"id": p.id, "family": p.family.value, "domain": p.domain.value, "name": p.name, "description": p.description, "weight": p.cross_domain_weight}
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
                    {"layerName": sp.layer_name, "sourceDim": sp.source_dim, "targetDim": sp.target_dim, "qualityScore": sp.quality_score}
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
            config = {"learning_rate": learningRate, "max_iterations": maxIterations, "use_procrustes_warm_start": True}
            result = ctx.geometry_stitch_service.apply(source_path, target_path, outputPath, config)
            return {
                "_schema": "mc.geometry.stitch.apply.v1",
                "outputPath": result.output_path,
                "stitchedLayers": result.stitched_layers,
                "qualityScore": result.quality_score,
                "nextActions": ["mc_model_probe to verify the stitched model", "mc_infer to test the stitched model"],
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
                AffineStitchingLayer, AnchorPair, Config as StitchConfig,
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
                parsed_pairs.append(AnchorPair(source_activation=source_act, target_activation=target_act, anchor_id=anchor_id))
            config = StitchConfig(
                learning_rate=learningRate, weight_decay=weightDecay,
                max_iterations=maxIterations, convergence_threshold=convergenceThreshold,
                use_procrustes_warm_start=useProcrusteWarmStart,
            )
            result = AffineStitchingLayer.train(parsed_pairs, config=config)
            if result is None:
                return {
                    "_schema": "mc.geometry.stitch.train.v1",
                    "status": "failed",
                    "error": "Training failed - insufficient data or convergence failure",
                    "nextActions": ["Add more anchor pairs and retry", "Adjust learning rate or iterations"],
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
                "nextActions": ["Use weights/bias to transform activations", "mc_geometry_stitch_apply to apply to full model"],
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
            from modelcypher.core.domain.geometry.dare_sparsity import DARESparsityAnalyzer, Configuration as DAREConfig
            from modelcypher.core.domain.geometry.dora_decomposition import DoRADecomposition
            from modelcypher.core.domain.geometry.refinement_density import RefinementDensityAnalyzer, RefinementDensityConfig

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
                sparsity_analysis = DARESparsityAnalyzer.analyze(delta_weights, DAREConfig(compute_per_layer_metrics=True))
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
                        sparsity_weight=sparsityWeight, directional_weight=directionalWeight,
                        transition_weight=transitionWeight, hard_swap_threshold=hardSwapThreshold,
                    )
                analyzer = RefinementDensityAnalyzer(config)
                result = analyzer.analyze(
                    source_model=adapted_path, target_model=base_path,
                    sparsity_analysis=sparsity_analysis, dora_result=dora_result,
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
            from modelcypher.core.domain.geometry.domain_signal_profile import DomainSignalProfile, LayerSignal

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
                    layer_signals=parsed_signals, model_id=modelId, domain=domain,
                    baseline_domain=baselineDomain, total_layers=totalLayers,
                    prompt_count=promptCount, max_tokens_per_prompt=maxTokensPerPrompt,
                )
            else:
                raise ValueError("Provide either profilePath or layerSignals")
            profile_dict = profile.to_dict()
            sparsity_values = [s.sparsity for s in profile.layer_signals.values() if s.sparsity is not None]
            gradient_snr_values = [s.gradient_snr for s in profile.layer_signals.values() if s.gradient_snr is not None]
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
                    "meanSparsity": sum(sparsity_values) / len(sparsity_values) if sparsity_values else None,
                    "layersWithGradientSNR": len(gradient_snr_values),
                    "meanGradientSNR": sum(gradient_snr_values) / len(gradient_snr_values) if gradient_snr_values else None,
                },
                "nextActions": [
                    "mc_geometry_refinement_analyze to use profile in analysis",
                    "mc_geometry_sparse_locate to find sparse regions",
                ],
            }
