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

"""Curvature profiling CLI commands.

These commands compute and manage curvature profiles for model family baselines.
Curvature profiles complement knowledge density profiles by capturing HOW
representations are encoded (geometry) rather than WHAT (semantics).
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from modelcypher.cli.output import write_output

from .common import cleanup_memory, get_context, load_model_and_provider

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command("curvature-profile")
    def curvature_profile(
        ctx: typer.Context,
        model_path: str = typer.Argument(..., help="Path to the model directory"),
        output_path: str | None = typer.Option(
            None, "--save", "-s", help="Save profile to JSON file"
        ),
    ) -> None:
        """Compute curvature profile for a model.

        Extracts geometric curvature measurements across all layers:
        - Sectional curvature (via Christoffel symbols → Riemann tensor)
        - Ollivier-Ricci curvature (via optimal transport on k-NN graph)
        - Intrinsic dimension (via Two-NN estimator)

        All parameters are derived from data - no configuration needed.

        Use --output to save profile for family baseline aggregation.
        """
        context = get_context(ctx)

        from datetime import datetime

        from modelcypher.core.domain.atlas.unified_atlas import UnifiedAtlasInventory
        from modelcypher.core.domain.geometry.curvature_profile import (
            CurvatureProfile,
            LayerCurvature,
            parse_model_info,
        )
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )
        from modelcypher.core.domain.geometry.manifold_curvature import (
            SectionalCurvatureEstimator,
        )
        from modelcypher.core.domain.geometry.ollivier_ricci import (
            OllivierRicciCurvature,
        )

        # Parse model info
        model_family, model_size = parse_model_info(model_path)
        logger.info(f"Profiling {model_family} {model_size} model: {model_path}")

        # Load model
        model, tokenizer, backend, provider, num_layers = load_model_and_provider(model_path)

        layer_indices = list(range(num_layers))

        probes = UnifiedAtlasInventory.all_probes()
        probe_texts = [f"The concept of {p.name}." for p in probes]

        logger.info(f"Computing curvature for {len(layer_indices)} layers with {len(probe_texts)} probes")

        # Initialize estimators (all parameters derived from data)
        sectional_estimator = SectionalCurvatureEstimator()
        orc_estimator = OllivierRicciCurvature(backend=backend)

        layer_curvatures: list[LayerCurvature] = []

        for layer_idx in layer_indices:
            logger.info(f"Processing layer {layer_idx}/{num_layers}...")

            try:
                # Collect activations for this layer
                activations = _collect_layer_activations(
                    provider, probe_texts, layer_idx, backend
                )

                if len(activations) < 10:
                    logger.warning(f"Layer {layer_idx}: insufficient activations ({len(activations)})")
                    layer_curvatures.append(LayerCurvature(layer_idx=layer_idx))
                    continue

                stacked = backend.stack(activations, axis=0)
                backend.eval(stacked)

                # Compute sectional curvature (k derived from data)
                try:
                    profile = sectional_estimator.estimate_manifold_profile(stacked)
                    sectional_mean = profile.global_mean
                    sectional_std = profile.global_variance ** 0.5
                    sectional_min = min(lc.min_sectional for lc in profile.local_curvatures)
                    sectional_max = max(lc.max_sectional for lc in profile.local_curvatures)
                    dominant_sign = profile.dominant_sign.value
                except Exception as e:
                    logger.debug(f"Sectional curvature failed: {e}")
                    sectional_mean = sectional_std = sectional_min = sectional_max = 0.0
                    dominant_sign = "unknown"

                # Compute Ollivier-Ricci curvature (k derived from data)
                try:
                    orc_result = orc_estimator.compute(stacked)
                    ricci_mean = orc_result.mean_edge_curvature
                    ricci_std = orc_result.std_edge_curvature
                except Exception as e:
                    logger.debug(f"Ollivier-Ricci failed: {e}")
                    ricci_mean = ricci_std = 0.0

                # Compute intrinsic dimension - all params derived from data
                try:
                    id_estimator = IntrinsicDimension(backend)
                    id_result = id_estimator.compute(stacked, with_ci=False)
                    intrinsic_dim = id_result.intrinsic_dimension
                    intrinsic_unc = 0.0  # CI not computed for speed
                except Exception as e:
                    logger.debug(f"Intrinsic dimension failed: {e}")
                    intrinsic_dim = intrinsic_unc = 0.0

                layer_curvatures.append(LayerCurvature(
                    layer_idx=layer_idx,
                    sectional_mean=sectional_mean,
                    sectional_std=sectional_std,
                    sectional_min=sectional_min,
                    sectional_max=sectional_max,
                    dominant_sign=dominant_sign,
                    ollivier_ricci_mean=ricci_mean,
                    ollivier_ricci_std=ricci_std,
                    intrinsic_dimension=intrinsic_dim,
                    intrinsic_dimension_uncertainty=intrinsic_unc,
                ))

                logger.debug(
                    f"Layer {layer_idx}: sectional={sectional_mean:.4f}, "
                    f"ricci={ricci_mean:.4f}, dim={intrinsic_dim:.1f}"
                )

            except Exception as e:
                logger.warning(f"Layer {layer_idx} failed: {e}")
                layer_curvatures.append(LayerCurvature(layer_idx=layer_idx))

        # Compute global statistics
        valid_sectional = [lc.sectional_mean for lc in layer_curvatures if lc.sectional_mean != 0]
        valid_ricci = [lc.ollivier_ricci_mean for lc in layer_curvatures if lc.ollivier_ricci_mean != 0]
        valid_dim = [lc.intrinsic_dimension for lc in layer_curvatures if lc.intrinsic_dimension != 0]

        global_sectional_mean = sum(valid_sectional) / len(valid_sectional) if valid_sectional else 0.0
        global_sectional_std = _compute_std(valid_sectional) if len(valid_sectional) > 1 else 0.0
        global_ricci_mean = sum(valid_ricci) / len(valid_ricci) if valid_ricci else 0.0
        global_ricci_std = _compute_std(valid_ricci) if len(valid_ricci) > 1 else 0.0
        global_dim_mean = sum(valid_dim) / len(valid_dim) if valid_dim else 0.0

        # Build profile
        profile = CurvatureProfile(
            model_path=model_path,
            model_family=model_family,
            model_size=model_size,
            layer_curvatures=layer_curvatures,
            total_layers=num_layers,
            global_sectional_mean=global_sectional_mean,
            global_sectional_std=global_sectional_std,
            global_ollivier_ricci_mean=global_ricci_mean,
            global_ollivier_ricci_std=global_ricci_std,
            global_intrinsic_dimension_mean=global_dim_mean,
            extraction_date=datetime.now().isoformat(),
            extraction_config={
                "num_probes": len(probe_texts),
                "layers": layer_indices,
            },
        )

        # Save if output path specified
        if output_path:
            profile.save(output_path)

        # Clean up
        cleanup_memory()

        # Output
        if context.output_format == "text":
            lines = [
                f"CURVATURE PROFILE: {Path(model_path).name}",
                f"Family: {model_family}, Size: {model_size}",
                f"Layers analyzed: {len(layer_curvatures)}/{num_layers}",
                "",
                "GLOBAL STATISTICS:",
                f"  Sectional curvature: mean={global_sectional_mean:.4f}, std={global_sectional_std:.4f}",
                f"  Ollivier-Ricci:      mean={global_ricci_mean:.4f}, std={global_ricci_std:.4f}",
                f"  Intrinsic dimension: mean={global_dim_mean:.1f}",
                "",
                "PER-LAYER CURVATURE:",
                "-" * 70,
            ]

            for lc in layer_curvatures:
                lines.append(
                    f"  L{lc.layer_idx:2d}: sectional={lc.sectional_mean:7.4f} ({lc.dominant_sign:8s}), "
                    f"ricci={lc.ollivier_ricci_mean:7.4f}, dim={lc.intrinsic_dimension:5.1f}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(profile.to_dict(), context.output_format, context.pretty)

    @app.command("curvature-baseline")
    def curvature_baseline(
        ctx: typer.Context,
        profile_dir: str = typer.Argument(..., help="Directory containing curvature profiles"),
        family: str = typer.Option(
            None, "--family", "-f", help="Filter by model family (qwen, llama, etc.)"
        ),
        output_path: str | None = typer.Option(
            None, "--save", "-s", help="Save baseline to JSON file"
        ),
    ) -> None:
        """Build a family baseline from multiple curvature profiles.

        Aggregates curvature profiles from models in the same family to create
        a baseline for z-score comparisons. This enables baseline-relative
        evaluation (no hardcoded thresholds).

        Example:
            mc geometry research curvature-baseline ./profiles --family qwen -o qwen_baseline.json
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.curvature_profile import (
            CurvatureProfile,
            build_family_baseline,
        )

        profile_path = Path(profile_dir)
        if not profile_path.exists():
            write_output({"error": f"Profile directory not found: {profile_dir}"}, "json", True)
            raise typer.Exit(1)

        # Load all profiles
        profiles: list[CurvatureProfile] = []
        for json_file in profile_path.glob("*.json"):
            try:
                profile = CurvatureProfile.load(json_file)
                if family is None or profile.model_family == family:
                    profiles.append(profile)
                    logger.info(f"Loaded profile: {json_file.name} ({profile.model_family})")
            except Exception as e:
                logger.debug(f"Skipping {json_file}: {e}")

        if not profiles:
            write_output(
                {"error": f"No profiles found for family '{family or 'any'}' in {profile_dir}"},
                "json",
                True,
            )
            raise typer.Exit(1)

        # Determine family
        detected_family = family or profiles[0].model_family
        logger.info(f"Building baseline for {detected_family} from {len(profiles)} profiles")

        # Build baseline
        baseline = build_family_baseline(profiles, detected_family)

        # Save if output path specified
        if output_path:
            baseline.save(output_path)

        # Output
        if context.output_format == "text":
            lines = [
                f"FAMILY BASELINE: {detected_family}",
                f"Contributing models: {len(baseline.contributing_models)}",
                "",
                "LAYER POSITION STATISTICS:",
                "-" * 70,
                "Position  Sectional(mean±std)     Ricci(mean±std)        Dimension",
                "-" * 70,
            ]

            for i, pos in enumerate(baseline.layer_positions):
                sec_mean = baseline.sectional_mean_by_position[i] if i < len(baseline.sectional_mean_by_position) else 0
                sec_std = baseline.sectional_std_by_position[i] if i < len(baseline.sectional_std_by_position) else 0
                ric_mean = baseline.ollivier_ricci_mean_by_position[i] if i < len(baseline.ollivier_ricci_mean_by_position) else 0
                ric_std = baseline.ollivier_ricci_std_by_position[i] if i < len(baseline.ollivier_ricci_std_by_position) else 0
                dim = baseline.intrinsic_dimension_by_position[i] if i < len(baseline.intrinsic_dimension_by_position) else 0

                lines.append(
                    f"  {pos:5.2f}    {sec_mean:6.4f}±{sec_std:5.4f}        "
                    f"{ric_mean:6.4f}±{ric_std:5.4f}        {dim:5.1f}"
                )

            lines.append("")
            lines.append("Contributing models:")
            for model in baseline.contributing_models:
                lines.append(f"  - {Path(model).name}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(baseline.to_dict(), context.output_format, context.pretty)

    @app.command("curvature-compare")
    def curvature_compare(
        ctx: typer.Context,
        source_profile: str = typer.Argument(..., help="Source model curvature profile JSON"),
        target_profile: str = typer.Argument(..., help="Target model curvature profile JSON"),
        baseline_path: str | None = typer.Option(
            None, "--baseline", "-b", help="Family baseline for z-score computation"
        ),
    ) -> None:
        """Compare curvature profiles of two models.

        Computes curvature diffs and z-scores for merge planning.
        Uses family baseline for z-score comparison when provided.

        Example:
            mc geometry research curvature-compare source.json target.json -b qwen_baseline.json
        """
        context = get_context(ctx)

        from modelcypher.core.domain.geometry.curvature_profile import (
            CurvatureProfile,
            FamilyBaseline,
            compute_curvature_alignment,
        )

        # Load profiles
        src = CurvatureProfile.load(source_profile)
        tgt = CurvatureProfile.load(target_profile)

        # Load baseline if provided
        baseline = FamilyBaseline.load(baseline_path) if baseline_path else None

        # Compute curvature deltas
        alignment = compute_curvature_alignment(src, tgt, baseline)

        # Output
        if context.output_format == "text":
            lines = [
                "CURVATURE COMPARISON",
                f"Source: {Path(source_profile).name} ({src.model_family} {src.model_size})",
                f"Target: {Path(target_profile).name} ({tgt.model_family} {tgt.model_size})",
                "",
                f"Aligned: {alignment.aligned}",
                "",
                "COMPONENT DIFFS:",
                f"  Sectional curvature:   {alignment.sectional_diff:.6f} (z={alignment.sectional_z_score:.2f})",
                f"  Ollivier-Ricci:        {alignment.ollivier_ricci_diff:.6f} (z={alignment.ollivier_ricci_z_score:.2f})",
                f"  Intrinsic dimension:   {alignment.intrinsic_dimension_diff:.6f} (z={alignment.intrinsic_dimension_z_score:.2f})",
                "",
                f"Baseline: {alignment.baseline_family} ({alignment.baseline_model_count} models)",
            ]

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(alignment.to_dict(), context.output_format, context.pretty)


    @app.command("curvature-alignment")
    def curvature_alignment(
        ctx: typer.Context,
        source_profile: str = typer.Argument(..., help="Path to source curvature profile JSON"),
        target_profile: str = typer.Argument(..., help="Path to target curvature profile JSON"),
        output_path: str | None = typer.Option(
            None, "--save", "-s", help="Save alignment plan to JSON file"
        ),
    ) -> None:
        """Compute alignment guidance from curvature profiles.

        Analyzes curvature differences between source and target models to
        determine the transformation needed for alignment.

        The output includes:
        - Per-layer dimension scales and curvature deltas
        - Normalized Ollivier-Ricci mismatch ratios
        - Layer correspondence by curvature similarity
        """
        from pathlib import Path

        from modelcypher.cli.output import write_output
        from modelcypher.core.domain.geometry.curvature_alignment import (
            compute_alignment_guidance,
            compute_layer_correspondence_by_curvature,
        )
        from modelcypher.core.domain.geometry.curvature_profile import CurvatureProfile

        context = get_context(ctx)

        # Load profiles
        src = CurvatureProfile.load(source_profile)
        tgt = CurvatureProfile.load(target_profile)

        # Compute alignment plan
        plan = compute_alignment_guidance(src, tgt)

        # Compute layer correspondence by curvature similarity
        correspondence = compute_layer_correspondence_by_curvature(src, tgt)

        # Build output
        result = {
            "source_model": plan.source_model,
            "target_model": plan.target_model,
            "mean_dimension_scale": plan.mean_dimension_scale,
            "mean_intrinsic_dimension_diff": plan.mean_intrinsic_dimension_diff,
            "mean_ollivier_ricci_delta": plan.mean_ollivier_ricci_delta,
            "mean_ollivier_ricci_relative_diff": plan.mean_ollivier_ricci_relative_diff,
            "layer_correspondence": {str(k): v for k, v in correspondence.items()},
            "layer_guidance": [
                {
                    "layer_idx": g.layer_idx,
                    "dimension_scale": g.dimension_scale,
                    "intrinsic_dimension_diff": g.intrinsic_dimension_diff,
                    "ollivier_ricci_delta": g.ollivier_ricci_delta,
                    "ollivier_ricci_relative_diff": g.ollivier_ricci_relative_diff,
                }
                for g in plan.layer_guidance
            ],
        }

        # Save to file if requested
        if output_path:
            import json
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps(result, indent=2))
            logger.info("Alignment plan saved to %s", output_path)

        # Text output
        if context.output_format == "text":
            lines = [
                "CURVATURE-GUIDED ALIGNMENT PLAN",
                f"Source: {Path(source_profile).name} ({src.model_family} {src.model_size})",
                f"Target: {Path(target_profile).name} ({tgt.model_family} {tgt.model_size})",
                "",
                f"MEAN DIMENSION SCALE: {plan.mean_dimension_scale:.3f}",
                f"MEAN INTRINSIC DIM DIFF: {plan.mean_intrinsic_dimension_diff:.3f}",
                f"MEAN OLLIVIER-RICCI DELTA: {plan.mean_ollivier_ricci_delta:.3f}",
                f"MEAN OLLIVIER-RICCI REL DIFF: {plan.mean_ollivier_ricci_relative_diff:.3f}",
                "",
            ]

            lines.append("PER-LAYER GUIDANCE:")
            for g in plan.layer_guidance:
                lines.append(
                    f"  Layer {g.layer_idx}: dim_scale={g.dimension_scale:.3f}, "
                    f"dim_diff={g.intrinsic_dimension_diff:.3f}, "
                    f"ricci_delta={g.ollivier_ricci_delta:.3f}, "
                    f"ricci_rel={g.ollivier_ricci_relative_diff:.3f}"
                )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(result, context.output_format, context.pretty)


def _collect_layer_activations(provider, probe_texts, layer_idx, backend):
    """Collect activations from a specific layer for given probes."""
    # Use the batch API - provider.get_activations returns list of list[float]
    raw_activations = provider.get_activations(probe_texts, layer_idx)

    # Convert to backend arrays
    activations = []
    for raw in raw_activations:
        if raw:  # Skip empty activations
            arr = backend.array(raw)
            activations.append(arr)

    return activations


def _compute_std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
