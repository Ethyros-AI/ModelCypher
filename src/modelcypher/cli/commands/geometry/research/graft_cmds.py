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

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.output import write_output
from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

from .common import get_context, load_model_and_provider

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("graft-boundary")
    def graft_boundary(
        ctx: typer.Context,
        source_path: str = typer.Argument(..., help="Path to source model directory"),
        target_path: str = typer.Argument(..., help="Path to target model directory"),
        output_path: str | None = typer.Option(
            None, "--output-path", "-o", help="Save results to JSON file"
        ),
    ) -> None:
        """Analyze graft boundary by correlating density with null space.

        Analyzes the relationship between concept density and null space availability
        without introducing thresholds.

        Key insight: Sparse concepts (low density) should have more null space
        available, making grafting safer. Dense concepts have less null space,
        making grafting risky.

        Outputs:
        - Per-density-bracket analysis
        - Null space correlation per layer
        - Derived graft mask (which layers/concepts to graft)
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
            ModelDensityProfile,
        )
        from modelcypher.core.domain.geometry.knowledge_diff import (
            KnowledgeDiffer,
            compute_graft_mask,
        )
        from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
        from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

        # Load target model (primary model for null space analysis)
        logger.info("Loading target model: %s", target_path)
        target_model, target_tokenizer, target_backend, target_provider, target_num_layers = (
            load_model_and_provider(target_path)
        )

        # Load source model
        logger.info("Loading source model: %s", source_path)
        _, _, source_backend, source_provider, source_num_layers = load_model_and_provider(
            source_path
        )

        num_layers = min(source_num_layers, target_num_layers)
        resolved_layers = list(range(num_layers))

        probes = UnifiedAtlasInventory.all_probes()

        # Step 1: Compute knowledge diff
        logger.info("Computing knowledge diff...")
        source_analyzer = KnowledgeDensityAnalyzer(backend=source_backend)
        source_profile = source_analyzer.analyze_model(
            probes, source_provider, resolved_layers
        )
        source_profile = ModelDensityProfile(
            model_path=source_path,
            layers=source_profile.layers,
            layer_profiles=source_profile.layer_profiles,
            domain_densities=source_profile.domain_densities,
            overall_density=source_profile.overall_density,
        )

        target_analyzer = KnowledgeDensityAnalyzer(backend=target_backend)
        target_profile = target_analyzer.analyze_model(
            probes, target_provider, resolved_layers
        )
        target_profile = ModelDensityProfile(
            model_path=target_path,
            layers=target_profile.layers,
            layer_profiles=target_profile.layer_profiles,
            domain_densities=target_profile.domain_densities,
            overall_density=target_profile.overall_density,
        )

        differ = KnowledgeDiffer()
        diff = differ.diff(source_profile, target_profile)

        # Step 2: Compute geodesic orthogonal space profile for target model
        # Geodesic math is accurate for high-D manifolds (8kD+)
        # Flat-space SVD-based methods are only accurate up to 3D
        logger.info("Computing geodesic orthogonal space profile for target...")
        geo_filter = GeodesicNullSpaceFilter(backend=target_backend)

        # Collect activations and compute geodesic profile per layer
        layer_geo_profile: dict[int, dict] = {}
        orthogonal_fractions = []
        graftable_layers = []

        for layer_idx in resolved_layers:
            activations = []
            for probe in probes:
                texts = list(probe.support_texts or [])
                if texts:
                    acts = target_provider.get_activations(texts, layer_idx)
                    activations.extend(acts)
            if not activations:
                continue

            act_array = target_backend.stack(
                [target_backend.array(a) for a in activations], axis=0
            )
            target_backend.eval(act_array)

            total_dim = int(act_array.shape[1])

            # Probe with a unit vector to measure orthogonal space
            probe_vec = target_backend.ones((total_dim,), dtype="float32")
            norm_arr = geodesic_norms(
                target_backend.reshape(probe_vec, (1, -1)), target_backend
            )
            target_backend.eval(norm_arr)
            norm_val = float(target_backend.to_scalar(norm_arr))
            probe_vec = probe_vec / norm_val
            target_backend.eval(probe_vec)

            result = geo_filter.filter_delta(probe_vec, act_array)
            orthogonal_frac = result.preserved_fraction
            orthogonal_fractions.append(orthogonal_frac)

            layer_geo_profile[layer_idx] = {
                "orthogonal_dim": result.orthogonal_dim,
                "total_dim": total_dim,
                "orthogonal_fraction": orthogonal_frac,
                "mean_geodesic_distance": result.mean_geodesic_distance,
                "k_neighbors": result.k_neighbors,
            }

        # Determine graftable layers (above mean orthogonal fraction)
        mean_orthogonal_frac = sum(orthogonal_fractions) / len(orthogonal_fractions) if orthogonal_fractions else 0.0
        for layer_idx, info in layer_geo_profile.items():
            if info["orthogonal_fraction"] >= mean_orthogonal_frac:
                graftable_layers.append(layer_idx)

        # Correlate geodesic orthogonal space with density (raw measurements)
        layer_orthogonal_density_correlation = []
        for layer_idx in resolved_layers:
            if layer_idx not in layer_geo_profile:
                continue

            geo_info = layer_geo_profile[layer_idx]

            # Get concepts at this layer
            layer_concepts = [
                opp for opp in diff.ranked_opportunities
                if opp.layer == layer_idx
            ]

            if not layer_concepts:
                continue

            mean_density = sum(c.target_density for c in layer_concepts) / len(layer_concepts)
            mean_opp = sum(c.opportunity_score for c in layer_concepts) / len(layer_concepts)

            layer_orthogonal_density_correlation.append({
                "layer": layer_idx,
                "orthogonalFraction": geo_info["orthogonal_fraction"],
                "orthogonalDim": geo_info["orthogonal_dim"],
                "totalDim": geo_info["total_dim"],
                "meanGeodesicDistance": geo_info["mean_geodesic_distance"],
                "meanTargetDensity": mean_density,
                "meanOpportunity": mean_opp,
                "conceptCount": len(layer_concepts),
                "isGraftable": layer_idx in graftable_layers,
            })

        # Generate graft mask
        graft_mask = compute_graft_mask(diff)

        # Build payload
        payload = {
            "_schema": "mc.geometry.research.graft_boundary.geodesic.v1",
            "sourcePath": source_path,
            "targetPath": target_path,
            "layers": resolved_layers,
            "orthogonalSpaceCorrelation": layer_orthogonal_density_correlation,
            "graftableLayers": graftable_layers,
            "meanOrthogonalFraction": mean_orthogonal_frac,
            "totalConcepts": diff.total_concepts,
            "positiveOpportunityCount": diff.positive_opportunity_count,
            "graftMaskSummary": {
                "totalProbes": len(graft_mask),
                "probesWithGraft": sum(
                    1
                    for probe_layers in graft_mask.values()
                    if any(probe_layers.values())
                ),
            },
        }

        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Results saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "GRAFT BOUNDARY ANALYSIS (GEODESIC)",
                f"Source: {source_path}",
                f"Target: {target_path}",
                f"Layers: {', '.join(str(layer) for layer in resolved_layers)}",
                "",
                f"Graftable Layers (by geodesic orthogonal space): {graftable_layers}",
                f"Mean Orthogonal Fraction: {mean_orthogonal_frac:.3f}",
            ]

            lines.append("")
            lines.append("GEODESIC ORTHOGONAL / DENSITY CORRELATION BY LAYER:")
            lines.append("-" * 60)
            for corr in layer_orthogonal_density_correlation:
                graftable = "GRAFTABLE" if corr["isGraftable"] else "limited"
                lines.append(
                    f"  L{corr['layer']}: ortho_frac={corr['orthogonalFraction']:.3f}, "
                    f"density={corr['meanTargetDensity']:.3f}, "
                    f"opportunity={corr['meanOpportunity']:.3f} [{graftable}]"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    @app.command("zero-shot-transfer")
    def zero_shot_transfer(
        ctx: typer.Context,
        source_path: str = typer.Argument(
            ..., help="Path to source model directory (knowledge donor)"
        ),
        target_path: str = typer.Argument(
            ..., help="Path to target model directory (graft recipient)"
        ),
        output_path: str | None = typer.Option(
            None, "--output-path", "-o", help="Path to save results JSON"
        ),
    ) -> None:
        """Validate zero-shot transfer by analyzing graft candidates.

        Identifies sparse concepts in target that can receive knowledge from source.
        Creates a validation plan for testing transfer without fine-tuning.

        Success criteria:
        - Sparse concept activations shift toward source after grafting
        - Dense concept activations remain stable
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.cka import HSICEstimator, compute_cka
        from modelcypher.core.domain.geometry.knowledge_density import (
            KnowledgeDensityAnalyzer,
            ModelDensityProfile,
        )
        from modelcypher.core.domain.geometry.knowledge_diff import KnowledgeDiffer

        # Load models using helper
        logger.info("Loading target model: %s", target_path)
        target_model, target_tokenizer, b, target_provider, target_n_layers = (
            load_model_and_provider(target_path)
        )

        logger.info("Loading source model: %s", source_path)
        _, _, _, source_provider, source_n_layers = (
            load_model_and_provider(source_path)
        )

        # Determine layers to analyze
        num_layers = min(target_n_layers, source_n_layers)

        resolved_layers = list(range(num_layers))

        logger.info("Analyzing layers: %s", resolved_layers)

        # Load unified atlas probes
        probes = UnifiedAtlasInventory.all_probes()

        # Analyze density for target
        density_analyzer = KnowledgeDensityAnalyzer(backend=b)
        differ = KnowledgeDiffer()

        target_profile = density_analyzer.analyze_model(
            probes=probes,
            activation_provider=target_provider,
            layers=resolved_layers,
        )

        source_profile = density_analyzer.analyze_model(
            probes=probes,
            activation_provider=source_provider,
            layers=resolved_layers,
        )

        # Set model paths
        target_profile = ModelDensityProfile(
            model_path=target_path,
            layers=target_profile.layers,
            layer_profiles=target_profile.layer_profiles,
            domain_densities=target_profile.domain_densities,
            overall_density=target_profile.overall_density,
        )
        source_profile = ModelDensityProfile(
            model_path=source_path,
            layers=source_profile.layers,
            layer_profiles=source_profile.layer_profiles,
            domain_densities=source_profile.domain_densities,
            overall_density=source_profile.overall_density,
        )

        # Compute knowledge diff
        diff = differ.diff(source_profile, target_profile)

        # Transfer candidates: concepts where geometry indicates source can help target
        # The opportunity_score IS the geometric signal: positive = source denser than target
        # No arbitrary thresholds - the geometry determines the boundary
        transfer_candidates = [
            opp
            for opp in diff.ranked_opportunities
            if opp.opportunity_score > 0  # Natural geometric boundary
        ]

        # Stability checks: concepts where target is already denser (negative opportunity)
        stability_checks = [
            opp for opp in diff.ranked_opportunities
            if opp.opportunity_score <= 0  # Target is already at or above source density
        ]

        # Group transfer candidates by layer
        candidates_by_layer: dict[int, list] = {}
        for cand in transfer_candidates:
            if cand.layer not in candidates_by_layer:
                candidates_by_layer[cand.layer] = []
            candidates_by_layer[cand.layer].append({
                "probeId": cand.probe_id,
                "name": cand.name,
                "domain": cand.domain,
                "targetDensity": cand.target_density,
                "sourceDensity": cand.source_density,
                "opportunityScore": cand.opportunity_score,
            })

        # Group stability checks by layer
        stability_by_layer: dict[int, list] = {}
        for check in stability_checks:
            if check.layer not in stability_by_layer:
                stability_by_layer[check.layer] = []
            stability_by_layer[check.layer].append({
                "probeId": check.probe_id,
                "name": check.name,
                "domain": check.domain,
                "targetDensity": check.target_density,
                "sourceDensity": check.source_density,
            })

        # Compute CKA similarity between source and target at each layer
        # This measures how similar the representations are currently
        layer_cka: dict[int, float] = {}
        for layer in resolved_layers:
            # Get activations for a sample of probes
            sample_probes = probes
            sample_texts = []
            for probe in sample_probes:
                if probe.support_texts:
                    sample_texts.extend(list(probe.support_texts))

            if sample_texts:
                try:
                    target_acts = target_provider.get_activations(sample_texts, layer)
                    source_acts = source_provider.get_activations(sample_texts, layer)

                    if target_acts and source_acts:
                        target_arr = b.stack([b.array(a) for a in target_acts], axis=0)
                        source_arr = b.stack([b.array(a) for a in source_acts], axis=0)
                        b.eval(target_arr)
                        b.eval(source_arr)

                        # Truncate to shared dimension
                        min_dim = min(target_arr.shape[1], source_arr.shape[1])
                        target_arr = target_arr[:, :min_dim]
                        source_arr = source_arr[:, :min_dim]

                        result = compute_cka(
                            target_arr,
                            source_arr,
                            backend=b,
                            estimator=HSICEstimator.AUTO,
                            feature_bias_correction=True,
                        )
                        layer_cka[layer] = (
                            float(result.cka_corrected)
                            if result.cka_corrected is not None
                            else float(result.cka)
                        )
                except Exception as e:
                    logger.debug("CKA failed for layer %d: %s", layer, e)
                    layer_cka[layer] = 0.0
            else:
                layer_cka[layer] = 0.0

        # Build validation plan
        validation_plan = {
            "preGraftChecks": [
                "Measure baseline perplexity on sparse concept prompts",
                "Measure baseline perplexity on dense concept prompts",
                "Record activation norms for transfer candidates",
            ],
            "graftOperation": {
                "targetLayers": list(candidates_by_layer.keys()),
                "conceptsToGraft": len(transfer_candidates),
                "expectedBehavior": "Sparse concept activations shift toward source pattern",
            },
            "postGraftChecks": [
                "Measure post-graft perplexity on sparse concept prompts (should improve)",
                "Measure post-graft perplexity on dense concept prompts (should be stable)",
                "Compare activation shift magnitude for sparse vs dense concepts",
            ],
        }

        payload = {
            "_schema": "mc.geometry.research.zero_shot_transfer.v1",
            "sourcePath": source_path,
            "targetPath": target_path,
            "layers": resolved_layers,
            "transferCandidates": {
                "total": len(transfer_candidates),
                "byLayer": {str(k): v for k, v in candidates_by_layer.items()},
                "topOpportunities": [
                    {
                        "probeId": c.probe_id,
                        "name": c.name,
                        "domain": c.domain,
                        "layer": c.layer,
                        "targetDensity": c.target_density,
                        "sourceDensity": c.source_density,
                        "opportunityScore": c.opportunity_score,
                    }
                    for c in transfer_candidates[:10]
                ],
            },
            "stabilityChecks": {
                "total": len(stability_checks),
                "byLayer": {str(k): len(v) for k, v in stability_by_layer.items()},
            },
            "layerCkaSimilarity": {str(k): v for k, v in layer_cka.items()},
            "validationPlan": validation_plan,
            "summary": {
                # Raw measurements only - no interpretation strings (per CLAUDE.md "No Vibes")
                "transferCandidateCount": len(transfer_candidates),
                "stabilityCheckCount": len(stability_checks),
                "stabilityToTransferRatio": (
                    len(stability_checks) / len(transfer_candidates)
                    if len(transfer_candidates) > 0
                    else float("inf")
                ),
                "meanLayerCka": (
                    sum(layer_cka.values()) / len(layer_cka)
                    if layer_cka
                    else 0.0
                ),
            },
        }

        if output_path:
            Path(output_path).write_text(json.dumps(payload, indent=2))
            logger.info("Results saved to %s", output_path)

        if context.output_format == "text":
            lines = [
                "ZERO-SHOT TRANSFER VALIDATION PLAN",
                f"Source (donor): {source_path}",
                f"Target (recipient): {target_path}",
                "",
                "TRANSFER CANDIDATES (sparse in target, dense in source):",
                "-" * 60,
                f"  Total candidates: {len(transfer_candidates)}",
            ]

            for layer, cands in sorted(candidates_by_layer.items()):
                lines.append(f"  Layer {layer}: {len(cands)} concepts")

            lines.append("")
            lines.append("TOP 10 TRANSFER OPPORTUNITIES:")
            lines.append("-" * 60)
            for c in transfer_candidates[:10]:
                lines.append(
                    f"  {c.name} (L{c.layer}): "
                    f"target={c.target_density:.3f}, source={c.source_density:.3f}, "
                    f"opportunity={c.opportunity_score:.3f}"
                )

            lines.append("")
            lines.append("STABILITY CHECKS (should not change):")
            lines.append("-" * 60)
            lines.append(f"  Total dense concepts: {len(stability_checks)}")
            for layer, checks in sorted(stability_by_layer.items()):
                lines.append(f"  Layer {layer}: {len(checks)} concepts to protect")

            lines.append("")
            lines.append("LAYER CKA SIMILARITY:")
            lines.append("-" * 60)
            for layer, cka in sorted(layer_cka.items()):
                # Raw values only - no interpretation (per CLAUDE.md "No Vibes")
                lines.append(f"  Layer {layer}: CKA={cka:.4f}")

            lines.append("")
            lines.append("SUMMARY STATISTICS:")
            lines.append("-" * 60)
            mean_cka = sum(layer_cka.values()) / len(layer_cka) if layer_cka else 0.0
            ratio = (
                len(stability_checks) / len(transfer_candidates)
                if len(transfer_candidates) > 0
                else float("inf")
            )
            lines.append(f"  Transfer candidates: {len(transfer_candidates)}")
            lines.append(f"  Stability checks: {len(stability_checks)}")
            lines.append(f"  Stability/transfer ratio: {ratio:.2f}")
            lines.append(f"  Mean layer CKA: {mean_cka:.4f}")
            lines.append(f"  Target layers: {list(candidates_by_layer.keys())}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)
