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

"""Merge entropy validation MCP tools.

Provides entropy-aware metrics for model merging:
- Pre-merge: Profile models to measure layer entropy
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

            Analyzes a model's per-layer entropy using Entropy-Lens projection.
            Returns raw entropy statistics only.

            Args:
                model: Model name or path to profile
                numLayers: Number of layers in the model (default: 32)

            Returns:
                Profile with entropy stats per layer
            """
            from pathlib import Path

            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.domain.merging.entropy_merge_validator import (
                EntropyMergeValidator,
            )

            validator = EntropyMergeValidator()
            model_loader = MLXModelLoader()

            # Require real model path - no simulated data
            model_path = Path(model).expanduser()
            if not model_path.exists():
                return {
                    "_schema": "mc.merge.entropy.profile.v1",
                    "error": f"Model path not found: {model}",
                    "hint": "Provide a valid local model path for entropy profiling",
                }

            profile = validator.create_profile(
                str(model_path), model_loader=model_loader, num_layers=numLayers
            )

            # Sort by entropy (highest first)
            sorted_layers = sorted(
                profile.layer_profiles.values(),
                key=lambda lp: lp.mean_entropy,
                reverse=True,
            )
            top_entropy_layers = [lp.layer_name for lp in sorted_layers[:5]]

            return {
                "_schema": "mc.merge.entropy.profile.v1",
                "modelName": profile.model_name,
                "meanEntropy": round(profile.mean_entropy, 3),
                "entropyVariance": round(profile.entropy_variance, 4),
                "layerCount": len(profile.layer_profiles),
                "topEntropyLayers": top_entropy_layers,
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

            Compares entropy characteristics before/after merge to report:
            - Knowledge retention score (0-1)
            - Entropy ratio statistics

            Args:
                sourceEntropies: Dict of layer_name -> entropy for source model
                targetEntropies: Dict of layer_name -> entropy for target model
                mergedEntropies: Dict of layer_name -> entropy for merged model
                sourceModel: Name of source model (for reporting)
                targetModel: Name of target model (for reporting)

            Returns:
                Validation result with raw entropy metrics
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

            sorted_layers = sorted(
                validation.layer_validations.values(),
                key=lambda v: v.entropy_ratio,
                reverse=True,
            )
            top_layers = [
                {
                    "layerName": v.layer_name,
                    "entropyRatio": round(v.entropy_ratio, 4),
                    "entropyDelta": round(v.entropy_delta, 4),
                    "knowledgeRetentionScore": round(v.knowledge_retention_score, 4),
                }
                for v in sorted_layers[:5]
            ]

            return {
                "_schema": "mc.merge.entropy.validate.v1",
                "knowledgeRetention": round(validation.mean_knowledge_retention, 3),
                "meanEntropyRatio": round(validation.mean_entropy_ratio, 3),
                "maxEntropyRatio": round(validation.max_entropy_ratio, 3),
                "entropyRatioStd": round(validation.entropy_ratio_std, 3),
                "totalLayersValidated": len(validation.layer_validations),
                "topEntropyRatioLayers": top_layers,
            }

    if "mc_model_validate_knowledge" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_validate_knowledge(
            sourceModel: str,
            mergedModel: str,
        ) -> dict:
            """Validate knowledge transfer after model merge.

            Runs knowledge probes against source and merged models to measure
            how much knowledge was retained during the merge process.

            Args:
                sourceModel: Path to original source model
                mergedModel: Path to merged model

            Returns:
                Knowledge retention report with per-domain scores and overall retention
            """
            from modelcypher.core.domain.merging.knowledge_transfer_validator import (
                KnowledgeProbeCorpus,
                KnowledgeTransferReport,
                compute_retention_by_domain,
                run_knowledge_probes,
            )

            # Use the full probe corpus for geometry-derived validation
            corpus = KnowledgeProbeCorpus()
            probes = corpus.get_probes()

            # Create generators for each model
            def source_generate(prompt: str) -> str:
                result = ctx.inference_engine.infer(sourceModel, prompt)
                return result.get("response", "")

            def merged_generate(prompt: str) -> str:
                result = ctx.inference_engine.infer(mergedModel, prompt)
                return result.get("response", "")

            # Run probes on both models
            source_results = run_knowledge_probes(source_generate, probes)
            merged_results = run_knowledge_probes(merged_generate, probes)

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

            return {
                "_schema": "mc.model.validate_knowledge.v1",
                "sourceModel": sourceModel,
                "mergedModel": mergedModel,
                "overallRetention": round(report.overall_retention, 3),
                "perDomain": per_domain_summary,
                "probesTested": sum(r.probes_tested for r in report.per_domain.values()),
            }

    if "mc_model_vocab_compare" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_vocab_compare(
            modelA: str,
            modelB: str,
        ) -> dict:
            """Compare vocabularies between two models for cross-vocabulary merging.

            Analyzes tokenizer overlap and reports raw alignment statistics.

            Args:
                modelA: Path to first model
                modelB: Path to second model

            Returns:
                Vocabulary alignment report with overlap statistics
            """
            try:
                from transformers import AutoTokenizer
            except ImportError:
                return {
                    "_schema": "mc.model.vocab_compare.v1",
                    "error": "transformers package not installed",
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

            # Determine if exact token IDs fully match
            needs_bridge = result.overlap_ratio < 1.0

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
                "needsBridge": needs_bridge,
            }
