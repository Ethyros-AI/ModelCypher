"""Merge entropy validation MCP tools.

Provides entropy-aware guidance for model merging:
- Pre-merge: Profile models to identify critical layers
- Guidance: Get per-layer alpha/sigma recommendations
- Validation: Check knowledge retention after merge
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .common import (
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
)

if TYPE_CHECKING:
    pass


def register_merge_entropy_tools(ctx: ServiceContext) -> None:
    """Register merge entropy validation MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_merge_entropy_profile" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_entropy_profile(
            model: str,
            numLayers: int = 32,
        ) -> dict:
            """Profile model entropy characteristics for merge planning.

            Analyzes a model's per-layer entropy to identify:
            - Critical layers near phase boundaries (need careful blending)
            - Dominant thermodynamic phase (ordered/critical/disordered)
            - Overall merge risk level

            Args:
                model: Model name or path to profile
                numLayers: Number of layers in the model (default: 32)

            Returns:
                Profile with entropy stats, phase classification, and merge risk
            """
            from pathlib import Path
            from modelcypher.core.domain.merging.entropy_merge_validator import (
                EntropyMergeValidator,
            )

            validator = EntropyMergeValidator()

            # Require real model path - no simulated data
            model_path = Path(model).expanduser()
            if not model_path.exists():
                return {
                    "_schema": "mc.merge.entropy.profile.v1",
                    "error": f"Model path not found: {model}",
                    "hint": "Provide a valid local model path for entropy profiling",
                    "nextActions": [
                        "mc_model_download to fetch model locally",
                        "Verify the model path is correct",
                    ],
                }

            profile = validator.create_profile(str(model_path), num_layers=numLayers)

            # Get top critical layers (limit to 5 for compact response)
            critical_layers = [
                name for name, p in profile.layer_profiles.items()
                if p.is_critical
            ][:5]

            return {
                "_schema": "mc.merge.entropy.profile.v1",
                "modelName": profile.model_name,
                "meanEntropy": round(profile.mean_entropy, 3),
                "entropyVariance": round(profile.entropy_variance, 4),
                "dominantPhase": profile.dominant_phase.value,
                "criticalLayerCount": profile.critical_layer_count,
                "topCriticalLayers": critical_layers,
                "mergeRisk": profile.merge_risk_level,
                "interpretation": (
                    f"{profile.model_name}: {profile.dominant_phase.value} phase, "
                    f"{profile.critical_layer_count} critical layers, "
                    f"{profile.merge_risk_level} merge risk"
                ),
                "nextActions": [
                    "mc_merge_entropy_guide to compare with target model",
                    "mc_model_merge with alpha adjusted for critical layers",
                ],
            }

    if "mc_merge_entropy_guide" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_entropy_guide(
            source: str,
            target: str,
            numLayers: int = 32,
        ) -> dict:
            """Generate entropy-aware merge recommendations.

            Compares source and target models to compute:
            - Per-layer alpha adjustments based on phase
            - Per-layer smoothing sigma recommendations
            - Critical layer warnings

            Args:
                source: Source model name/path
                target: Target model name/path
                numLayers: Number of layers (must match for both)

            Returns:
                Recommendations with alpha adjustments and smoothing sigmas
            """
            from pathlib import Path
            from modelcypher.core.domain.merging.entropy_merge_validator import (
                EntropyMergeValidator,
            )

            validator = EntropyMergeValidator()

            # Require real model paths - no simulated data
            source_path = Path(source).expanduser()
            if not source_path.exists():
                return {
                    "_schema": "mc.merge.entropy.guide.v1",
                    "error": f"Source model path not found: {source}",
                    "hint": "Provide valid local model paths for entropy-guided merging",
                    "nextActions": [
                        "mc_model_download to fetch model locally",
                        "Verify the model path is correct",
                    ],
                }

            target_path = Path(target).expanduser()
            if not target_path.exists():
                return {
                    "_schema": "mc.merge.entropy.guide.v1",
                    "error": f"Target model path not found: {target}",
                    "hint": "Provide valid local model paths for entropy-guided merging",
                    "nextActions": [
                        "mc_model_download to fetch model locally",
                        "Verify the model path is correct",
                    ],
                }

            source_profile = validator.create_profile(str(source_path), num_layers=numLayers)
            target_profile = validator.create_profile(str(target_path), num_layers=numLayers)

            alpha_adj = validator.compute_alpha_adjustments(source_profile, target_profile)
            sigmas = validator.compute_smoothing_sigmas(source_profile, target_profile)

            # Only include non-default adjustments (compact response)
            recommendations = {}
            for layer_name in alpha_adj:
                adj = alpha_adj[layer_name]
                sigma = sigmas.get(layer_name, 1.0)
                if adj < 1.0 or sigma > 1.0:  # Non-default values
                    recommendations[layer_name] = {
                        "alphaAdjust": round(adj, 2),
                        "smoothingSigma": round(sigma, 1),
                    }

            # Limit to top 5 for compact response
            top_recommendations = dict(list(recommendations.items())[:5])

            # Compute global recommendation
            if alpha_adj:
                global_alpha = sum(alpha_adj.values()) / len(alpha_adj)
            else:
                global_alpha = 1.0

            critical_count = source_profile.critical_layer_count + target_profile.critical_layer_count

            return {
                "_schema": "mc.merge.entropy.guide.v1",
                "sourceRisk": source_profile.merge_risk_level,
                "targetRisk": target_profile.merge_risk_level,
                "criticalLayerCount": critical_count,
                "globalAlphaAdjust": round(global_alpha, 2),
                "recommendations": top_recommendations,
                "interpretation": (
                    f"Source: {source_profile.merge_risk_level} risk, "
                    f"Target: {target_profile.merge_risk_level} risk. "
                    f"{len(recommendations)} layers need adjustment. "
                    f"Suggested global alpha: {global_alpha:.2f}"
                ),
                "nextActions": [
                    f"mc_model_merge --alpha {global_alpha:.2f} for basic merge",
                    "mc_merge_entropy_validate after merge to check stability",
                ],
            }

    if "mc_merge_entropy_validate" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_entropy_validate(
            sourceEntropies: dict,
            targetEntropies: dict,
            mergedEntropies: dict,
            sourceModel: str = "source",
            targetModel: str = "target",
        ) -> dict:
            """Validate merge stability via entropy comparison.

            Compares entropy characteristics before/after merge to assess:
            - Knowledge retention score (0-1)
            - Layer stability (stable/marginal/unstable/critical)
            - Critical/unstable layer identification

            Args:
                sourceEntropies: Dict of layer_name -> entropy for source model
                targetEntropies: Dict of layer_name -> entropy for target model
                mergedEntropies: Dict of layer_name -> entropy for merged model
                sourceModel: Name of source model (for reporting)
                targetModel: Name of target model (for reporting)

            Returns:
                Validation result with stability assessment and recommendations
            """
            from modelcypher.core.domain.merging.entropy_merge_validator import (
                EntropyMergeValidator,
            )

            validator = EntropyMergeValidator()
            validation = validator.validate_merge(
                source_entropies=sourceEntropies,
                target_entropies=targetEntropies,
                merged_entropies=mergedEntropies,
                source_model=sourceModel,
                target_model=targetModel,
            )

            # Limit lists to 5 items for compact response
            critical = validation.critical_layer_names[:5]
            unstable = validation.unstable_layer_names[:5]

            return {
                "_schema": "mc.merge.entropy.validate.v1",
                "overallStability": validation.overall_stability.value,
                "knowledgeRetention": round(validation.mean_knowledge_retention, 3),
                "isSafe": validation.is_safe,
                "criticalLayers": critical,
                "unstableLayers": unstable,
                "totalLayersValidated": len(validation.layer_validations),
                "interpretation": (
                    f"Merge {validation.overall_stability.value}: "
                    f"{validation.mean_knowledge_retention:.0%} knowledge retention. "
                    f"{len(validation.critical_layer_names)} critical, "
                    f"{len(validation.unstable_layer_names)} unstable layers."
                ),
                "nextActions": (
                    ["mc_merge_perplexity to verify model quality"]
                    if validation.is_safe
                    else [
                        "mc_merge_diagnose to investigate layer issues",
                        "mc_model_merge with lower alpha for unstable layers",
                    ]
                ),
            }

    if "mc_model_validate_knowledge" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_validate_knowledge(
            sourceModel: str,
            mergedModel: str,
            domains: list[str] | None = None,
            useVariations: bool = False,
        ) -> dict:
            """Validate knowledge transfer after model merge.

            Runs knowledge probes against source and merged models to measure
            how much knowledge was retained during the merge process.

            Args:
                sourceModel: Path to original source model
                mergedModel: Path to merged model
                domains: Optional list of domains to test (math, code, factual, reasoning)
                useVariations: Whether to test probe variations for robustness

            Returns:
                Knowledge retention report with per-domain scores and overall status
            """
            from modelcypher.core.domain.merging.knowledge_transfer_validator import (
                KnowledgeDomain,
                KnowledgeValidationConfig,
                KnowledgeProbeCorpus,
                run_knowledge_probes,
                compute_retention_by_domain,
                KnowledgeTransferReport,
            )

            # Parse domain filters
            domain_filter: set[KnowledgeDomain] | None = None
            if domains:
                domain_filter = set()
                for d in domains:
                    try:
                        domain_filter.add(KnowledgeDomain(d.lower()))
                    except ValueError:
                        pass

            # Create config and corpus
            config = KnowledgeValidationConfig(use_variations=useVariations)
            corpus = KnowledgeProbeCorpus()

            # Get probes for requested domains
            if domain_filter:
                probes = []
                for domain in domain_filter:
                    probes.extend(corpus.get_probes(domain))
            else:
                probes = corpus.all_probes

            # Create generators for each model
            def source_generate(prompt: str) -> str:
                result = ctx.inference_engine.infer(
                    sourceModel, prompt, max_tokens=100, temperature=0.0
                )
                return result.get("response", "")

            def merged_generate(prompt: str) -> str:
                result = ctx.inference_engine.infer(
                    mergedModel, prompt, max_tokens=100, temperature=0.0
                )
                return result.get("response", "")

            # Run probes on both models
            source_results = run_knowledge_probes(source_generate, probes, config)
            merged_results = run_knowledge_probes(merged_generate, probes, config)

            # Compute retention by domain
            retention = compute_retention_by_domain(source_results, merged_results)

            # Create report
            report = KnowledgeTransferReport(per_domain=retention)

            # Build response
            per_domain_summary = {}
            for domain, result in report.per_domain.items():
                per_domain_summary[domain.value] = {
                    "sourcePassRate": round(result.source_pass_rate, 3),
                    "mergedPassRate": round(result.merged_pass_rate, 3),
                    "retentionScore": round(result.retention_score, 3),
                    "probesTested": result.probes_tested,
                }

            failed_domains = report.get_failed_domains(threshold=0.8)

            return {
                "_schema": "mc.model.validate_knowledge.v1",
                "sourceModel": sourceModel,
                "mergedModel": mergedModel,
                "overallRetention": round(report.overall_retention, 3),
                "status": report.status.value,
                "perDomain": per_domain_summary,
                "failedDomains": [d.value for d in failed_domains],
                "probesTested": sum(r.probes_tested for r in report.per_domain.values()),
                "interpretation": (
                    f"Knowledge retention: {report.overall_retention:.0%}. "
                    f"Status: {report.status.value}. "
                    f"{len(failed_domains)} domains below threshold."
                ),
                "nextActions": (
                    ["mc_infer to test specific capabilities"]
                    if report.status.value in ("excellent", "acceptable")
                    else [
                        "mc_model_merge with higher source alpha",
                        "mc_merge_diagnose to identify degraded layers",
                    ]
                ),
            }

    if "mc_model_vocab_compare" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_vocab_compare(
            modelA: str,
            modelB: str,
        ) -> dict:
            """Compare vocabularies between two models for cross-vocabulary merging.

            Analyzes tokenizer overlap to determine merge strategy:
            - High overlap (>90%): FVT (Fast Vocabulary Transfer) only
            - Medium overlap (50-90%): FVT + Procrustes verification
            - Low overlap (<50%): Procrustes + Affine transformation

            Args:
                modelA: Path to first model
                modelB: Path to second model

            Returns:
                Vocabulary alignment report with overlap stats and merge recommendations
            """
            try:
                from transformers import AutoTokenizer
            except ImportError:
                return {
                    "_schema": "mc.model.vocab_compare.v1",
                    "error": "transformers package not installed",
                    "nextActions": ["pip install transformers"],
                }

            from modelcypher.core.domain.vocabulary import compare_tokenizers

            # Load tokenizers
            try:
                tokenizer_a = AutoTokenizer.from_pretrained(modelA, trust_remote_code=True)
            except Exception as e:
                return {
                    "_schema": "mc.model.vocab_compare.v1",
                    "error": f"Failed to load tokenizer for model A: {e}",
                    "modelA": modelA,
                }

            try:
                tokenizer_b = AutoTokenizer.from_pretrained(modelB, trust_remote_code=True)
            except Exception as e:
                return {
                    "_schema": "mc.model.vocab_compare.v1",
                    "error": f"Failed to load tokenizer for model B: {e}",
                    "modelB": modelB,
                }

            # Compare vocabularies
            result = compare_tokenizers(tokenizer_a, tokenizer_b)

            # Determine if cross-vocab merging is needed
            needs_bridge = result.overlap_ratio < 0.95
            method = result.recommended_method

            return {
                "_schema": "mc.model.vocab_compare.v1",
                "modelA": modelA,
                "modelB": modelB,
                "sourceVocabSize": result.source_vocab_size,
                "targetVocabSize": result.target_vocab_size,
                "overlapCount": result.overlap_count,
                "overlapRatio": round(result.overlap_ratio, 4),
                "approximateCount": result.approximate_count,
                "unmappedCount": result.unmapped_count,
                "coverage": round(result.coverage, 4),
                "recommendedMethod": method,
                "mergeFeasibility": result.merge_feasibility,
                "needsBridge": needs_bridge,
                "interpretation": (
                    f"Vocabulary overlap: {result.overlap_ratio:.1%}. "
                    f"Coverage: {result.coverage:.1%}. "
                    f"Recommended method: {method}. "
                    f"Feasibility: {result.merge_feasibility}."
                ),
                "nextActions": (
                    ["mc_model_merge to merge with same-vocabulary pipeline"]
                    if not needs_bridge
                    else [
                        f"mc_unified_merge with vocab bridge ({method})",
                        "mc_model_vocab_compare with different model pair",
                    ]
                ),
            }
