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

# Constants
DEFAULT_PATH_MAX_TOKENS = 200


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
                entropy_trace=entropyTrace,
            )
            payload = ctx.geometry_service.detection_payload(detection)
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
                comprehensive=comprehensive,
            )
            payload = ctx.geometry_service.path_comparison_payload(result)
            payload["_schema"] = "mc.geometry.path.compare.v1"
            return payload

    if "mc_geometry_concept_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_concept_detect(
            text: str,
            model: str | None = None,
        ) -> dict:
            """Detect concept sequence in text or model response.

            All parameters (threshold, window sizes, stride) are derived from
            concept embedding geometry. If `model` is set, `text` is treated
            as a prompt and the response is analyzed.
            """
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.domain.geometry.concept_detector import (
                create_default_detector,
            )

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Concept detection requires embeddings."
                )

            detector = create_default_detector(embedder)

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
                        "category": concept.category.value,
                        "confidence": concept.confidence,
                        "characterSpan": {
                            "lowerBound": concept.character_span[0],
                            "upperBound": concept.character_span[1],
                        },
                        "triggerText": concept.trigger_text,
                        "crossModalConfidence": concept.cross_modal_confidence,
                    }
                    for concept in detection.detected_concepts
                ],
                "meanConfidence": detection.mean_confidence,
                "meanCrossModalConfidence": detection.mean_cross_modal_confidence,
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

            All parameters (threshold, window sizes, stride) are derived from
            concept embedding geometry. Provide textA/textB or modelA/modelB
            with a prompt.
            """
            from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
            from modelcypher.core.domain.geometry.concept_detector import (
                ConceptDetector,
                create_default_detector,
            )

            embedder = EmbeddingDefaults.make_default_embedder()
            if embedder is None:
                raise ValueError(
                    "No embedding provider available. Concept detection requires embeddings."
                )

            detector = create_default_detector(embedder)

            if textA and textB:
                text_to_analyze_a = textA
                text_to_analyze_b = textB
                model_id_a = "text-a"
                model_id_b = "text-b"
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
                "_schema": "mc.geometry.cross_cultural.analyze.v1",
                "gramRoughnessA": result.gram_roughness_a,
                "gramRoughnessB": result.gram_roughness_b,
                "mergedGramRoughness": result.merged_gram_roughness,
                "roughnessReduction": result.roughness_reduction,
                "complementarityScore": result.complementarity_score,
                "convergentPrimes": result.convergent_primes,
                "divergentPrimes": result.divergent_primes,
                "complementaryPrimes": [
                    {
                        "primeId": item.prime_id,
                        "sharperModel": item.sharper_model.value,
                        "sharpnessRatio": item.sharpness_ratio,
                    }
                    for item in result.complementary_primes
                ],
                "categoryDivergence": result.category_divergence,
                "mergeQualityScore": result.merge_quality_score,
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
            epsilon: float = 0.05,
            maxIterations: int = 50,
        ) -> dict:
            """Compute Gromov-Wasserstein distance between point clouds.
            """
            result = ctx.geometry_metrics_service.compute_gromov_wasserstein(
                source_points=sourcePoints,
                target_points=targetPoints,
                epsilon=epsilon,
                max_iterations=maxIterations,
            )
            payload = ctx.geometry_metrics_service.gromov_wasserstein_payload(result)
            payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
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
            return payload

    if "mc_geometry_spectral_signature" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_spectral_signature(
            points: list[list[float]],
            kNeighbors: int | None = None,
            kernelBandwidth: float | None = None,
            normalizedLaplacian: bool = True,
            heatTimes: list[float] | None = None,
            maxEigenvalues: int | None = None,
        ) -> dict:
            """Compute geodesic spectral signature from a point cloud."""
            result = ctx.geometry_metrics_service.compute_spectral_signature(
                points=points,
                k_neighbors=kNeighbors,
                kernel_bandwidth=kernelBandwidth,
                normalized_laplacian=normalizedLaplacian,
                heat_times=heatTimes,
            )
            payload = ctx.geometry_metrics_service.spectral_signature_payload(
                result, max_eigenvalues=maxEigenvalues
            )
            payload["_schema"] = "mc.geometry.spectral_signature.v1"
            return payload

    if "mc_geometry_dimension_constraint_invariance" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dimension_constraint_invariance(
            points: list[list[float]],
            paddedDimension: int,
            kNeighbors: int | None = None,
            heatTimes: list[float] | None = None,
        ) -> dict:
            """Measure invariance under zero-padding dimension constraints."""
            result = ctx.geometry_metrics_service.compute_dimension_constraint_invariance(
                points=points,
                padded_dimension=paddedDimension,
                k_neighbors=kNeighbors,
                heat_times=heatTimes,
            )
            payload = ctx.geometry_metrics_service.dimension_constraint_invariance_payload(result)
            payload["_schema"] = "mc.geometry.dimension_constraint_invariance.v1"
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
            return payload

    if "mc_geometry_sparse_neurons" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_sparse_neurons(
            modelPath: str,
            domain: str | None = None,
            promptsFile: str | None = None,
            layerStart: float = 0.0,
            layerEnd: float = 1.0,
            sparsityThreshold: float | None = None,
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
                sparsityThreshold: Sparsity threshold for graft candidates.
                    If None (default), derived from distribution as mean + 2σ.

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

            # All thresholds derived from data distribution when None
            config = NeuronSparsityConfig(
                sparsity_threshold=sparsityThreshold,  # None = mean + 2σ
                dead_neuron_threshold=None,  # mean + 3σ
                activation_threshold=None,  # noise floor from data
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
                }
            payload = ctx.geometry_persona_service.persona_vector_payload(vector)
            payload["_schema"] = "mc.geometry.persona_extract.v1"
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
            return payload

    if "mc_geometry_manifold_dimension" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_manifold_dimension(
            points: list[list[float]],
            useRegression: bool = True,
        ) -> dict:
            """Estimate intrinsic dimension of a point cloud using TwoNN."""
            result = ctx.geometry_persona_service.estimate_dimension(
                points=points,
                use_regression=useRegression,
            )
            payload = ctx.geometry_persona_service.dimension_payload(result)
            payload["_schema"] = "mc.geometry.manifold_dimension.v1"
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
                }
            return {
                "_schema": "mc.geometry.transport_merge.v1",
                "mergedShape": [len(merged), len(merged[0]) if merged else 0],
                "blendAlpha": blendAlpha,
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
                }
            payload = ctx.geometry_transport_service.merge_result_payload(result)
            payload["_schema"] = "mc.geometry.transport_synthesize.v1"
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
