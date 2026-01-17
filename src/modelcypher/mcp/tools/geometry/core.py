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

"""Core geometry MCP tools.

Contains the main geometry tools for:
- Path detection and comparison
- Concept detection
- Cross-cultural analysis
- Gromov-Wasserstein distance
- Intrinsic dimension estimation
- Topological fingerprinting
- Spectral signature
- Dimension-constraint invariance
- Sparse region analysis
- Refusal direction detection
- Persona vector extraction
- Manifold clustering
- Transport-guided merging
- Training status
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    require_existing_directory,
)

if TYPE_CHECKING:
    pass

def register_geometry_tools(ctx: ServiceContext) -> None:
    """Register geometry-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    # Basic geometry tools
    if "mc_geometry_validate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_validate(includeFixtures: bool = False) -> dict:
            """Run geometry validation suite and return raw diagnostics.

            Args:
                includeFixtures: Include built-in fixtures in the report.

            Returns:
                Validation payload with measurements and schema.
            """
            report = ctx.geometry_service.validate(include_fixtures=includeFixtures)
            return ctx.geometry_service.validation_payload(report, include_schema=True)

    if "mc_geometry_path_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_path_detect(
            text: str,
            model: str | None = None,
            entropyTrace: list[float] | None = None,
        ) -> dict:
            """Detect path geometry in a response or provided text.

            If `model` is set, `text` is treated as a prompt and the model
            response is analyzed. Otherwise, `text` is analyzed directly.
            """
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.use_cases.geometry_service import GeometryService

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Path detection requires embeddings."
                )

            service = GeometryService(embedder=embedder)

            if model:
                response = ctx.inference_engine.infer(model, text)
                text_to_analyze = response.get("response", "")
                model_id = Path(model).name if Path(model).exists() else model
            else:
                text_to_analyze = text
                model_id = "input-text"
            detection = service.detect_path(
                text_to_analyze,
                model_id=model_id,
                prompt_id="mcp-path-detect",
                entropy_trace=entropyTrace,
            )
            payload = service.detection_payload(detection)
            payload["_schema"] = "mc.geometry.path.detect.v1"
            return payload

    if "mc_geometry_path_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_path_compare(
            textA: str | None = None,
            textB: str | None = None,
            modelA: str | None = None,
            modelB: str | None = None,
            prompt: str | None = None,
            comprehensive: bool = False,
        ) -> dict:
            """Compare path geometry between two texts or model responses."""
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.use_cases.geometry_service import GeometryService

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Path detection requires embeddings."
                )

            service = GeometryService(embedder=embedder)

            if textA and textB:
                text_to_analyze_a, text_to_analyze_b = textA, textB
                model_id_a, model_id_b = "text-a", "text-b"
            elif modelA and modelB and prompt:
                response_a = ctx.inference_engine.infer(modelA, prompt)
                response_b = ctx.inference_engine.infer(modelB, prompt)
                text_to_analyze_a = response_a.get("response", "")
                text_to_analyze_b = response_b.get("response", "")
                model_id_a = Path(modelA).name if Path(modelA).exists() else modelA
                model_id_b = Path(modelB).name if Path(modelB).exists() else modelB
            else:
                raise ValueError("Provide textA/textB or modelA/modelB with prompt.")
            result = service.compare_paths(
                text_a=text_to_analyze_a,
                text_b=text_to_analyze_b,
                model_a=model_id_a,
                model_b=model_id_b,
                prompt_id="mcp-path-compare",
                comprehensive=comprehensive,
            )
            payload = service.path_comparison_payload(result)
            payload["_schema"] = "mc.geometry.path.compare.v1"
            return payload

    if "mc_geometry_concept_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_concept_detect(
            text: str,
            model: str | None = None,
        ) -> dict:
            """Detect concept sequence in text or model response.

            Detection is derived from probe embedding geometry. If `model` is set,
            `text` is treated as a prompt and the response is analyzed.
            """
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes
            from modelcypher.core.domain.geometry.concept_detector import ConceptDetector

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Concept detection requires embeddings."
                )

            probes = get_atlas_probes()
            if not probes:
                raise ValueError(
                    "No atlas probes registered for concept detection. "
                    "Call register_default_atlas_inventories() before use."
                )
            detector = ConceptDetector(embedder, probes)

            if model:
                response = ctx.inference_engine.infer(model, text)
                text_to_analyze = response.get("response", "")
                model_id = Path(model).name if Path(model).exists() else model
            else:
                text_to_analyze = text
                model_id = "input-text"

            detection = detector.detect(
                response=text_to_analyze,
                model_id=model_id,
                prompt_id="mcp-concept-detect",
            )

            payload = {
                "_schema": "mc.geometry.concept.detect.v1",
                "modelId": detection.model_id,
                "promptId": detection.prompt_id,
                "responseText": detection.response_text,
                "conceptSequence": detection.concept_sequence,
                "detectedConcepts": [
                    {
                        "conceptId": concept.concept_id,
                        "category": concept.category,
                        "similarity": concept.similarity,
                        "characterSpan": {
                            "lowerBound": concept.character_span[0],
                            "upperBound": concept.character_span[1],
                        },
                        "triggerText": concept.trigger_text,
                        "crossModalSimilarity": concept.cross_modal_similarity,
                    }
                    for concept in detection.detected_concepts
                ],
                "meanSimilarity": detection.mean_similarity,
                "meanCrossModalSimilarity": detection.mean_cross_modal_similarity,
            }
            return payload

    if "mc_geometry_concept_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_concept_compare(
            textA: str | None = None,
            textB: str | None = None,
            modelA: str | None = None,
            modelB: str | None = None,
            prompt: str | None = None,
        ) -> dict:
            """Compare concept sequences between two texts or model responses.

            Detection is derived from probe embedding geometry. Provide textA/textB
            or modelA/modelB with a prompt.
            """
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.domain.geometry.atlas_registry import get_atlas_probes
            from modelcypher.core.domain.geometry.concept_detector import ConceptDetector

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Concept detection requires embeddings."
                )

            probes = get_atlas_probes()
            if not probes:
                raise ValueError(
                    "No atlas probes registered for concept detection. "
                    "Call register_default_atlas_inventories() before use."
                )
            detector = ConceptDetector(embedder, probes)

            if textA and textB:
                text_to_analyze_a = textA
                text_to_analyze_b = textB
                model_id_a = "text-a"
                model_id_b = "text-b"
            elif modelA and modelB and prompt:
                response_a = ctx.inference_engine.infer(modelA, prompt)
                response_b = ctx.inference_engine.infer(modelB, prompt)
                text_to_analyze_a = response_a.get("response", "")
                text_to_analyze_b = response_b.get("response", "")
                model_id_a = Path(modelA).name if Path(modelA).exists() else modelA
                model_id_b = Path(modelB).name if Path(modelB).exists() else modelB
            else:
                raise ValueError("Provide textA/textB or modelA/modelB with prompt.")

            result_a = detector.detect(text_to_analyze_a, model_id_a, prompt_id="mcp-concept-a")
            result_b = detector.detect(text_to_analyze_b, model_id_b, prompt_id="mcp-concept-b")
            comparison = ConceptDetector.compare_results(result_a, result_b)

            payload = {
                "_schema": "mc.geometry.concept.compare.v1",
                "modelA": comparison.model_a,
                "modelB": comparison.model_b,
                "conceptPathA": list(comparison.concept_path_a),
                "conceptPathB": list(comparison.concept_path_b),
                "alignedConcepts": list(comparison.aligned_concepts),
                "uniqueToA": list(comparison.unique_to_a),
                "uniqueToB": list(comparison.unique_to_b),
                "alignmentRatio": comparison.alignment_ratio,
                "cka": comparison.cka,
                "cosineSimilarity": comparison.cosine_similarity,
            }
            return payload

    if "mc_geometry_cross_cultural_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_cross_cultural_analyze(
            gramA: list,
            gramB: list,
            primeIds: list[str],
            primeCategories: dict[str, str] | None = None,
        ) -> dict:
            """Analyze cross-cultural geometry from Gram matrices and prime IDs."""
            from modelcypher.core.domain.geometry.cross_cultural_geometry import (
                CrossCulturalGeometry,
            )

            def _flatten_gram(matrix: list) -> list[float]:
                if not matrix:
                    return []
                if isinstance(matrix[0], list):
                    return [float(value) for row in matrix for value in row]
                return [float(value) for value in matrix]

            flat_a = _flatten_gram(gramA)
            flat_b = _flatten_gram(gramB)
            prime_categories = primeCategories or {}

            if not primeIds:
                raise ValueError("primeIds is required and must be non-empty")
            n = len(primeIds)
            if len(flat_a) != n * n or len(flat_b) != n * n:
                raise ValueError(
                    f"Gram sizes must match primeIds length (expected {n*n}, got {len(flat_a)} and {len(flat_b)})"
                )

            result = CrossCulturalGeometry.analyze(flat_a, flat_b, primeIds, prime_categories)
            if result is None:
                raise ValueError("Cross-cultural analysis failed; check gram sizes and inputs.")

            alignment = CrossCulturalGeometry.analyze_alignment(flat_a, flat_b, n)

            return {
                "_schema": "mc.geometry.cross_cultural.analyze.v2",
                "primeIds": list(primeIds),
                "gramRoughnessA": result.gram_roughness_a,
                "gramRoughnessB": result.gram_roughness_b,
                "mergedGramRoughness": result.merged_gram_roughness,
                "roughnessReduction": result.roughness_reduction,
                "rowCorrelations": result.row_correlations,
                "rowSharpnessA": result.row_sharpness_a,
                "rowSharpnessB": result.row_sharpness_b,
                "rowSharpnessRatio": result.row_sharpness_ratio,
                "categoryDivergence": result.category_divergence,
                "alignment": {
                    "cka": alignment.cka,
                    "rawPearson": alignment.raw_pearson,
                    "alignmentGap": alignment.alignment_gap,
                }
                if alignment
                else None,
            }

    # Metrics tools
    if "mc_geometry_gromov_wasserstein" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_gromov_wasserstein(
            sourcePoints: list[list[float]],
            targetPoints: list[list[float]],
        ) -> dict:
            """Compute Gromov-Wasserstein distance between point clouds.
            """
            result = ctx.geometry_metrics_service.compute_gromov_wasserstein(
                source_points=sourcePoints,
                target_points=targetPoints,
            )
            payload = ctx.geometry_metrics_service.gromov_wasserstein_payload(result)
            payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
            return payload

    if "mc_geometry_intrinsic_dimension" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_intrinsic_dimension(
            points: list[list[float]],
        ) -> dict:
            """Estimate intrinsic dimension using TwoNN.

            All parameters are derived from data:
            - k_neighbors: Connectivity-based (Berry & Sauer 2016)
            - Method: Always regression (Facco et al., more robust)
            - Bootstrap resamples: Derived from sample size
            """
            result = ctx.geometry_metrics_service.estimate_intrinsic_dimension(
                points=points,
            )
            payload = ctx.geometry_metrics_service.intrinsic_dimension_payload(result)
            payload["_schema"] = "mc.geometry.intrinsic_dimension.v1"
            return payload

    if "mc_geometry_topological_fingerprint" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_topological_fingerprint(
            points: list[list[float]],
        ) -> dict:
            """Compute topological fingerprint using persistent homology.

            All parameters are derived from the geometry of the data.
            """
            result = ctx.geometry_metrics_service.compute_topological_fingerprint(
                points=points,
            )
            payload = ctx.geometry_metrics_service.topological_fingerprint_payload(result)
            payload["_schema"] = "mc.geometry.topological_fingerprint.v1"
            return payload

    if "mc_geometry_spectral_signature" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spectral_signature(
            points: list[list[float]],
        ) -> dict:
            """Compute geodesic spectral signature from a point cloud.

            All parameters are derived from the geometry of the data:
            - k_neighbors: derived from graph connectivity requirements
            - kernel_bandwidth: derived from median neighbor distance
            - heat_trace_times: derived from eigenvalue spectrum
            - normalized_laplacian: always True
            """
            result = ctx.geometry_metrics_service.compute_spectral_signature(
                points=points,
            )
            payload = ctx.geometry_metrics_service.spectral_signature_payload(result)
            payload["_schema"] = "mc.geometry.spectral_signature.v1"
            return payload

    if "mc_geometry_dimension_constraint_invariance" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dimension_constraint_invariance(
            points: list[list[float]],
            paddedDimension: int,
        ) -> dict:
            """Measure invariance under zero-padding dimension constraints.

            All parameters are derived from the geometry of the data.
            No configuration is accepted or needed.
            """
            result = ctx.geometry_metrics_service.compute_dimension_constraint_invariance(
                points=points,
                padded_dimension=paddedDimension,
            )
            payload = ctx.geometry_metrics_service.dimension_constraint_invariance_payload(result)
            payload["_schema"] = "mc.geometry.dimension_constraint_invariance.v1"
            return payload

    # Sparse region tools
    if "mc_geometry_sparse_domains" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_domains() -> dict:
            """List built-in sparse region domains for LoRA targeting."""
            domains = ctx.geometry_sparse_service.list_domains()
            payload = ctx.geometry_sparse_service.domains_payload(domains)
            payload["_schema"] = "mc.geometry.sparse_domains.v1"
            return payload

    if "mc_geometry_sparse_locate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_locate(
            domainStats: list[dict],
            baselineStats: list[dict],
            domainName: str,
        ) -> dict:
            """Locate sparse regions in activation statistics.

            All parameters (sparsity threshold, alignment) are derived from the data.
            No configuration is accepted or needed.
            """
            result = ctx.geometry_sparse_service.locate_sparse_regions(
                domain_stats=domainStats,
                baseline_stats=baselineStats,
                domain_name=domainName,
            )
            payload = ctx.geometry_sparse_service.analysis_payload(result)
            payload["_schema"] = "mc.geometry.sparse_locate.v1"
            return payload

    if "mc_geometry_sparse_neurons" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_neurons(
            modelPath: str,
        ) -> dict:
            """Analyze per-neuron sparsity for fine-grained knowledge grafting.

            Identifies individual neurons that meet the derived sparsity
            threshold for knowledge transfer during model merging.

            Args:
                modelPath: Path to model directory
            Returns:
                Neuron sparsity map with graft candidates and dead neurons
            """
            from modelcypher.core.domain.geometry.neuron_sparsity_analyzer import (
                compute_neuron_sparsity_map,
            )
            from modelcypher.core.domain.geometry.sparse_region_domains import (
                SparseRegionDomains,
            )

            model_path = require_existing_directory(modelPath)
            layer_start = 0.0
            layer_end = 1.0
            domain_name = "all"

            prompts: list[str] = []
            for domain in SparseRegionDomains.all_built_in:
                prompts.extend(domain.probe_prompts)
            if prompts:
                prompts = list(dict.fromkeys(prompts))
            if not prompts:
                raise ValueError("Built-in sparse domain prompts are unavailable")

            # Collect activations via model inference
            from modelcypher.core.domain.entropy.hidden_state_extractor import (
                HiddenStateExtractor,
            )

            # Get model info for layer count
            probe_service = ctx.model_probe_service
            model_info = probe_service.probe(str(model_path))
            total_layers = len([l for l in model_info.layers if "layers." in l.name])

            # Create extractor for neuron analysis in specified layer range
            # Caller specifies layer range - geometry determines which layers matter
            start_layer = int(total_layers * layer_start)
            end_layer = int(total_layers * layer_end)
            target_layers = set(range(start_layer, end_layer + 1))
            extractor = HiddenStateExtractor(
                target_layers=target_layers,
                expected_hidden_dim=model_info.hidden_size,
            )
            extractor.enable_neuron_collection()

            # Collect activations via inference
            engine = ctx.inference_engine
            extractor.start_neuron_collection()

            for prompt in prompts:
                try:
                    # Run inference to trigger activation capture
                    engine.infer(str(model_path), prompt)
                except Exception:
                    pass  # Continue with other prompts
                extractor.finalize_prompt_activations()

            # Get collected activations
            activations = extractor.get_neuron_activations()

            sparsity_map = compute_neuron_sparsity_map(activations)
            summary = sparsity_map.summary()

            return {
                "_schema": "mc.geometry.sparse_neurons.v1",
                "modelPath": str(model_path),
                "domain": domain_name,
                "layerRange": [layer_start, layer_end],
                "summary": summary,
                "graftCandidates": sparsity_map.get_graft_candidates(),
                "deadNeurons": sparsity_map.dead_neurons,
            }

    # Refusal detection tools
    if "mc_geometry_refusal_pairs" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refusal_pairs() -> dict:
            """Get standard contrastive prompt pairs for refusal detection."""
            pairs = ctx.geometry_sparse_service.get_contrastive_pairs()
            payload = ctx.geometry_sparse_service.contrastive_pairs_payload(pairs)
            payload["_schema"] = "mc.geometry.refusal_pairs.v1"
            return payload

    if "mc_geometry_refusal_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_refusal_detect(
            harmfulActivations: list[list[float]],
            harmlessActivations: list[list[float]],
        ) -> dict:
            """Detect refusal direction from contrastive activations."""
            layer_index = -1
            model_id = "unknown"
            result = ctx.geometry_sparse_service.detect_refusal_direction(
                harmful_activations=harmfulActivations,
                harmless_activations=harmlessActivations,
                layer_index=layer_index,
                model_id=model_id,
            )
            if result is None:
                return {
                    "_schema": "mc.geometry.refusal_detect.v1",
                    "error": "Could not compute refusal direction",
                }
            payload = ctx.geometry_sparse_service.refusal_direction_payload(result)
            payload["_schema"] = "mc.geometry.refusal_detect.v1"
            return payload

    # Persona tools
    if "mc_geometry_persona_traits" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_traits() -> dict:
            """List standard persona traits for vector extraction."""
            traits = ctx.geometry_persona_service.list_traits()
            payload = ctx.geometry_persona_service.traits_payload(traits)
            payload["_schema"] = "mc.geometry.persona_traits.v1"
            return payload

    if "mc_geometry_persona_extract" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_extract(
            positiveActivations: list[list[float]],
            negativeActivations: list[list[float]],
            traitId: str,
        ) -> dict:
            """Extract a persona vector from contrastive activations.

            All parameters are derived from the data at runtime.
            """
            vector = ctx.geometry_persona_service.extract_persona_vector(
                positive_activations=positiveActivations,
                negative_activations=negativeActivations,
                trait_id=traitId,
                layer_index=-1,
                model_id="unknown",
            )
            if vector is None:
                return {
                    "_schema": "mc.geometry.persona_extract.v1",
                    "error": "Could not extract persona vector",
                }
            payload = ctx.geometry_persona_service.persona_vector_payload(vector)
            payload["_schema"] = "mc.geometry.persona_extract.v1"
            return payload

    if "mc_geometry_persona_drift" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_persona_drift(
            positions: list[dict],
            step: int,
        ) -> dict:
            """Compute drift metrics from persona position measurements.

            Returns raw drift measurements. User interprets based on their
            model's baseline characteristics.
            """
            # Compute raw drift without threshold classification
            metrics = ctx.geometry_persona_service.compute_drift(
                positions=positions,
                step=step,
                drift_threshold=0.0,  # No classification - return raw values
            )
            payload = ctx.geometry_persona_service.drift_metrics_payload(metrics)
            payload["_schema"] = "mc.geometry.persona_drift.v1"
            return payload

    # Manifold tools
    if "mc_geometry_manifold_cluster" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_cluster(
            points: list[dict],
        ) -> dict:
            """Cluster manifold points into regions using DBSCAN.

            All clustering parameters are derived from the geometry of the data.
            No configuration is accepted or needed.
            """
            result = ctx.geometry_persona_service.cluster_points(
                points=points,
            )
            payload = ctx.geometry_persona_service.clustering_payload(result)
            payload["_schema"] = "mc.geometry.manifold_cluster.v1"
            return payload

    if "mc_geometry_manifold_dimension" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_dimension(
            points: list[list[float]],
        ) -> dict:
            """Estimate intrinsic dimension of a point cloud using TwoNN.

            All parameters are derived from data - no configuration needed.
            """
            result = ctx.geometry_persona_service.estimate_dimension(
                points=points,
            )
            payload = ctx.geometry_persona_service.dimension_payload(result)
            payload["_schema"] = "mc.geometry.manifold_dimension.v1"
            return payload

    if "mc_geometry_manifold_query" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_query(
            point: dict,
            regions: list[dict],
        ) -> dict:
            """Query which region a point belongs to.

            Distance thresholds are derived from region geometry (radii).
            No configuration is accepted or needed.
            """
            result = ctx.geometry_persona_service.query_region(
                point=point,
                regions=regions,
            )
            payload = ctx.geometry_persona_service.region_query_payload(result)
            payload["_schema"] = "mc.geometry.manifold_query.v1"
            return payload

    # Training status tools
    if "mc_geometry_training_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_status(jobId: str, format: str = "full") -> dict:
            """Return geometry training status metrics for a given job."""
            format_key = format.lower()
            if format_key not in {"full", "summary"}:
                raise ValueError("format must be 'full' or 'summary'")
            return ctx.geometry_training_service.training_status_payload(
                jobId, output_format=format_key
            )

    if "mc_geometry_training_history" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_history(jobId: str) -> dict:
            """Return geometry training history metrics for a given job."""
            return ctx.geometry_training_service.training_history_payload(jobId)
