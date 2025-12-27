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

"""CLI commands for merge analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from modelcypher.core.domain._backend import get_default_backend

from modelcypher.cli.composition import get_domain_geometry_waypoint_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
        DomainGeometryWaypointService,
        GeometryDomain,
    )

app = typer.Typer(help="Merge analysis for model compatibility")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("predict")
def predict_interference(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    target_path: str = typer.Argument(..., help="Path to target model"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
    domains: str | None = typer.Option(
        None,
        "--domains",
        help="Comma-separated domains (spatial,social,temporal,moral). Default: all",
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save report to file"
    ),
) -> None:
    """Analyze merge requirements between source and target models."""
    context = _context(ctx)

    from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
        GeometryDomain,
    )
    from modelcypher.core.domain.geometry.interference_predictor import (
        InterferencePredictor,
        TransformationType,
    )
    from modelcypher.core.domain.geometry.riemannian_density import (
        RiemannianDensityEstimator,
    )

    typer.echo("Predicting interference...")
    typer.echo(f"  Source: {source_path}")
    typer.echo(f"  Target: {target_path}")

    # Parse domains
    domain_list = None
    if domains:
        domain_list = []
        for d in domains.split(","):
            try:
                domain_list.append(GeometryDomain(d.strip().lower()))
            except ValueError:
                typer.echo(
                    f"Invalid domain: {d}. Valid: spatial, social, temporal, moral", err=True
                )
                raise typer.Exit(1)
    else:
        domain_list = list(GeometryDomain)

    # Extract activations for both models
    waypoint_service = get_domain_geometry_waypoint_service()
    density_estimator = RiemannianDensityEstimator()
    predictor = InterferencePredictor()

    # Collect activations per domain
    source_activations: dict[str, dict[str, Any]] = {}
    target_activations: dict[str, dict[str, Any]] = {}

    for domain in domain_list:
        typer.echo(f"  Extracting {domain.value} activations...")
        try:
            source_acts = _extract_domain_activations(source_path, domain, layer, waypoint_service)
            target_acts = _extract_domain_activations(target_path, domain, layer, waypoint_service)
            source_activations[domain.value] = source_acts
            target_activations[domain.value] = target_acts
        except Exception as e:
            logger.warning(f"Failed to extract {domain.value}: {e}")

    # Predict interference per domain
    domain_results: dict[str, dict] = {}

    for domain_name, source_acts in source_activations.items():
        target_acts = target_activations.get(domain_name, {})
        if not source_acts or not target_acts:
            continue

        # Estimate volumes
        source_volumes = {}
        target_volumes = {}
        common_concepts = set(source_acts.keys()) & set(target_acts.keys())

        for concept_id in common_concepts:
            src_arr = source_acts[concept_id]
            tgt_arr = target_acts[concept_id]

            # Need multiple samples for volume estimation
            if src_arr.ndim == 1:
                src_arr = src_arr.reshape(1, -1)
            if tgt_arr.ndim == 1:
                tgt_arr = tgt_arr.reshape(1, -1)

            # Enable store_raw_activations for cross-dimensional CKA comparison
            source_volumes[concept_id] = density_estimator.estimate_concept_volume(
                f"source:{concept_id}", src_arr, store_raw_activations=True
            )
            target_volumes[concept_id] = density_estimator.estimate_concept_volume(
                f"target:{concept_id}", tgt_arr, store_raw_activations=True
            )

        # Analyze merge requirements for this domain
        domain_analysis = {
            "concepts_analyzed": len(common_concepts),
            "transformation_counts": {t.value: 0 for t in TransformationType},
            "overlap_scores": [],
            "alignment_scores": [],
            "transformation_details": [],
        }

        for concept_id in common_concepts:
            result = predictor.predict(source_volumes[concept_id], target_volumes[concept_id])
            # Count transformations needed
            for t in result.transformations:
                domain_analysis["transformation_counts"][t.value] += 1
            domain_analysis["overlap_scores"].append(result.overlap_score)
            domain_analysis["alignment_scores"].append(result.alignment_score)
            # Record transformation details for concepts needing significant work
            if result.transformations:
                domain_analysis["transformation_details"].append(
                    {
                        "concept": concept_id,
                        "transformations": [t.value for t in result.transformations],
                        "overlap": result.overlap_score,
                        "alignment": result.alignment_score,
                        "descriptions": result.transformation_descriptions,
                    }
                )

        # Compute domain-level metrics
        if domain_analysis["overlap_scores"]:
            backend = get_default_backend()
            overlap_arr = backend.array(domain_analysis["overlap_scores"])
            align_arr = backend.array(domain_analysis["alignment_scores"])
            domain_analysis["mean_overlap"] = float(backend.mean(overlap_arr))
            domain_analysis["mean_alignment"] = float(backend.mean(align_arr))
        else:
            domain_analysis["mean_overlap"] = 0.0
            domain_analysis["mean_alignment"] = 1.0

        del domain_analysis["overlap_scores"]  # Don't need raw lists in output
        del domain_analysis["alignment_scores"]
        domain_results[domain_name] = domain_analysis

    # Compute global transformation counts
    global_transformation_counts = {t.value: 0 for t in TransformationType}
    all_overlap_scores = []
    all_alignment_scores = []
    total_transformations_needed = 0

    for domain_name, dr in domain_results.items():
        all_overlap_scores.append(dr["mean_overlap"])
        all_alignment_scores.append(dr["mean_alignment"])
        for ttype, count in dr.get("transformation_counts", {}).items():
            global_transformation_counts[ttype] = global_transformation_counts.get(ttype, 0) + count
            total_transformations_needed += count

    if all_overlap_scores:
        backend = get_default_backend()
        mean_overlap = float(backend.mean(backend.array(all_overlap_scores)))
        mean_alignment = float(backend.mean(backend.array(all_alignment_scores)))
    else:
        mean_overlap = 0.0
        mean_alignment = 1.0

    # Generate transformation summary (what will be applied, not verdicts)
    summary_parts = []
    if global_transformation_counts.get("procrustes_rotation", 0) > 0:
        summary_parts.append(f"Procrustes rotation for {global_transformation_counts['procrustes_rotation']} concept pairs")
    if global_transformation_counts.get("curvature_correction", 0) > 0:
        summary_parts.append(f"Curvature correction for {global_transformation_counts['curvature_correction']} pairs")
    if global_transformation_counts.get("alpha_scaling", 0) > 0:
        summary_parts.append(f"Alpha scaling for {global_transformation_counts['alpha_scaling']} overlapping regions")
    if global_transformation_counts.get("boundary_smoothing", 0) > 0:
        summary_parts.append(f"Boundary smoothing for {global_transformation_counts['boundary_smoothing']} edges")

    if not summary_parts:
        transformation_summary = "Minimal transformation needed. Direct merge."
    else:
        transformation_summary = ". ".join(summary_parts) + "."

    payload = {
        "_schema": "mc.geometry.merge_analysis.v1",
        "sourceModel": source_path,
        "targetModel": target_path,
        "layer": layer,
        "domainsAnalyzed": [d.value for d in domain_list],
        "perDomain": domain_results,
        "globalMetrics": {
            "meanOverlap": mean_overlap,
            "meanAlignment": mean_alignment,
            "transformationCounts": global_transformation_counts,
            "totalTransformationsNeeded": total_transformations_needed,
        },
        "transformationSummary": transformation_summary,
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Report saved to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "MERGE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Target: {Path(target_path).name}",
            f"Layer: {layer if layer != -1 else 'last'}",
            "",
            "-" * 50,
            "TRANSFORMATIONS REQUIRED:",
            f"  {transformation_summary}",
            "-" * 50,
            "",
            "Per-Domain Analysis:",
        ]

        for domain_name, dr in domain_results.items():
            lines.append(f"  {domain_name.upper()}:")
            lines.append(f"    Concepts: {dr['concepts_analyzed']}")
            lines.append(f"    Mean Overlap: {dr['mean_overlap']:.2f}")
            lines.append(f"    Mean Alignment: {dr['mean_alignment']:.2f}")
            for ttype, count in dr.get("transformation_counts", {}).items():
                if count > 0:
                    lines.append(f"    {ttype}: {count}")

            if dr.get("transformation_details"):
                lines.append(f"    Concepts Needing Transformation ({len(dr['transformation_details'])}):")
                for td in dr["transformation_details"][:3]:
                    lines.append(f"      - {td['concept']}: {', '.join(td['transformations'])}")

        lines.extend(
            [
                "",
                "Summary:",
                f"  {transformation_summary}",
                "",
            ]
        )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _extract_domain_activations(
    model_path: str,
    domain: "GeometryDomain",
    layer: int,
    service: "DomainGeometryWaypointService",
) -> dict[str, Any]:
    """Extract activations for probes in a specific domain."""

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.domain_geometry_waypoints import GeometryDomain

    backend = MLXBackend()
    model, tokenizer = load_model_for_training(model_path)

    # Get probes for this domain
    if domain == GeometryDomain.SPATIAL:
        from modelcypher.core.domain.agents.spatial_atlas import SpatialConceptInventory

        probes = [(p.id, p.prompt) for p in SpatialConceptInventory.all_concepts()]
    elif domain == GeometryDomain.SOCIAL:
        from modelcypher.core.domain.agents.social_atlas import SocialConceptInventory

        probes = [(p.id, p.prompt) for p in SocialConceptInventory.all_concepts()]
    elif domain == GeometryDomain.TEMPORAL:
        from modelcypher.core.domain.agents.temporal_atlas import TemporalConceptInventory

        probes = [(p.id, p.prompt) for p in TemporalConceptInventory.all_concepts()]
    elif domain == GeometryDomain.MORAL:
        from modelcypher.core.domain.agents.moral_atlas import MoralConceptInventory

        probes = [(p.id, p.prompt) for p in MoralConceptInventory.all_concepts()]
    else:
        return {}

    return service._extract_activations(model, tokenizer, layer, probes, backend)


@app.command("volume")
def compute_volume(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model"),
    concept: str = typer.Argument(..., help="Concept to analyze"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze"),
    samples: int = typer.Option(10, "--samples", help="Number of prompt variations"),
) -> None:
    """
    Compute ConceptVolume for a single concept.

    Shows the distributional properties of a concept in the model's
    latent space: centroid, covariance, geodesic radius, curvature.
    """
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.riemannian_density import (
        RiemannianDensityEstimator,
    )

    typer.echo(f"Computing volume for concept: {concept}")

    backend = MLXBackend()
    model, tokenizer = load_model_for_training(model_path)

    # Generate prompt variations
    base_prompts = [
        f"The word {concept} represents",
        f"The concept of {concept} means",
        f"{concept.capitalize()} is defined as",
        f"When we say {concept}, we mean",
        f"The meaning of {concept} is",
    ]
    # Duplicate to get more samples
    prompts = base_prompts * (samples // len(base_prompts) + 1)
    prompts = prompts[:samples]

    # Extract activations
    activations = []
    for prompt in prompts:
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])

            # Forward pass
            if hasattr(model, "model"):
                hidden = model.model.embed_tokens(input_ids)
                num_layers = len(model.model.layers)
                target_layer = layer if layer >= 0 else num_layers - 1

                for i, layer_module in enumerate(model.model.layers):
                    try:
                        result = layer_module(hidden, mask=None)
                    except TypeError:
                        result = layer_module(hidden)

                    if isinstance(result, tuple):
                        hidden = result[0]
                    else:
                        hidden = result

                    if i == target_layer:
                        break

                backend.eval(hidden)
                act = backend.mean(hidden[0], axis=0)
                backend.eval(act)
                activations.append(backend.to_numpy(act))

        except Exception as e:
            logger.warning(f"Failed to extract: {e}")

    if not activations:
        typer.echo("Failed to extract any activations", err=True)
        raise typer.Exit(1)

    # Estimate volume - use backend for stacking activations
    act_array = backend.stack([backend.array(a) for a in activations])
    # Convert to numpy for RiemannianDensityEstimator which may expect numpy
    act_array_np = backend.to_numpy(act_array)
    estimator = RiemannianDensityEstimator()
    volume = estimator.estimate_concept_volume(concept, act_array_np)

    # Compute eigenvalues using backend.eigh (returns eigenvalues, eigenvectors)
    cov_arr = backend.array(volume.covariance)
    eigenvalues, _ = backend.eigh(cov_arr)
    backend.eval(eigenvalues)
    eigenvalues_list = backend.to_numpy(eigenvalues).tolist()
    top_eigenvalues = sorted(eigenvalues_list, reverse=True)[:5]

    payload = {
        "_schema": "mc.geometry.interference.volume.v1",
        "model": model_path,
        "concept": concept,
        "layer": layer,
        "samples": len(activations),
        "dimension": volume.dimension,
        "geodesicRadius": float(volume.geodesic_radius),
        "effectiveRadius": float(volume.effective_radius),
        "volume": float(volume.volume),
        "topEigenvalues": [float(e) for e in top_eigenvalues],
        "curvature": {
            "available": volume.local_curvature is not None,
            "meanSectional": float(volume.local_curvature.mean_sectional)
            if volume.local_curvature
            else None,
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 50,
            f"CONCEPT VOLUME: {concept}",
            "=" * 50,
            "",
            f"Model: {Path(model_path).name}",
            f"Layer: {layer if layer != -1 else 'last'}",
            f"Samples: {len(activations)}",
            "",
            f"Dimension: {volume.dimension}",
            f"Geodesic Radius: {volume.geodesic_radius:.4f}",
            f"Effective Radius: {volume.effective_radius:.4f}",
            f"Volume: {volume.volume:.2e}",
            "",
            "Top Eigenvalues:",
        ]
        for i, ev in enumerate(top_eigenvalues):
            lines.append(f"  PC{i + 1}: {ev:.6f}")

        if volume.local_curvature:
            lines.extend(
                [
                    "",
                    f"Mean Sectional Curvature: {volume.local_curvature.mean_sectional:.6f}",
                ]
            )

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("null-space")
def null_space_filter(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model for activation extraction"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
    samples: int = typer.Option(20, "--samples", help="Number of activation samples"),
    rank_threshold: float = typer.Option(0.01, "--rank-threshold", help="Threshold for null space"),
) -> None:
    """
    Analyze null space availability for interference-free merging.

    Computes the null space profile of a model's activations to identify
    which layers have space for knowledge grafting without interference.

    Based on MINGLE (arXiv:2509.21413).
    """
    context = _context(ctx)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.null_space_filter import (
        NullSpaceFilter,
        NullSpaceFilterConfig,
    )

    typer.echo(f"Analyzing null space for: {model_path}")

    backend = MLXBackend()
    model, tokenizer = load_model_for_training(model_path)

    # Generate sample prompts for activation extraction
    sample_prompts = [
        "The concept of justice represents",
        "A chair is used for",
        "Yesterday I went to",
        "The number five is",
        "My friend told me",
    ] * (samples // 5 + 1)
    sample_prompts = sample_prompts[:samples]

    # Extract activations per layer
    layer_activations: dict[int, list[Any]] = {}

    typer.echo(f"  Extracting activations from {samples} prompts...")

    for prompt in sample_prompts:
        try:
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])

            if hasattr(model, "model"):
                hidden = model.model.embed_tokens(input_ids)
                num_layers = len(model.model.layers)
                target_layer = layer if layer >= 0 else num_layers - 1

                for i, layer_module in enumerate(model.model.layers):
                    try:
                        result = layer_module(hidden, mask=None)
                    except TypeError:
                        result = layer_module(hidden)

                    if isinstance(result, tuple):
                        hidden = result[0]
                    else:
                        hidden = result

                    if i not in layer_activations:
                        layer_activations[i] = []

                    backend.eval(hidden)
                    act = backend.mean(hidden[0], axis=0)
                    backend.eval(act)
                    layer_activations[i].append(backend.to_numpy(act))

                    if i == target_layer:
                        break

        except Exception as e:
            logger.warning(f"Failed to extract: {e}")

    # Compute null space profile
    config = NullSpaceFilterConfig(rank_threshold=rank_threshold)
    filter = NullSpaceFilter(config)

    # Stack activations using backend and convert to numpy for NullSpaceFilter
    layer_arrays = {
        layer_idx: backend.to_numpy(
            backend.stack([backend.array(a) for a in acts])
        )
        for layer_idx, acts in layer_activations.items()
    }

    profile = filter.compute_model_null_space_profile(layer_arrays)

    payload = {
        "_schema": "mc.geometry.interference.null_space.v1",
        "model": model_path,
        "samples": samples,
        "rankThreshold": rank_threshold,
        "totalNullDim": profile.total_null_dim,
        "totalDim": profile.total_dim,
        "meanNullFraction": profile.mean_null_fraction,
        "graftableLayers": profile.graftable_layers,
        "perLayer": {
            str(layer_idx): {
                "nullDim": lp.null_dim,
                "totalDim": lp.total_dim,
                "nullFraction": lp.null_fraction,
                "conditionNumber": lp.condition_number,
            }
            for layer_idx, lp in profile.per_layer.items()
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "NULL SPACE ANALYSIS",
            "=" * 60,
            "",
            f"Model: {Path(model_path).name}",
            f"Samples: {samples}",
            f"Rank Threshold: {rank_threshold}",
            "",
            "-" * 40,
            "Summary",
            "-" * 40,
            f"Total Null Dim: {profile.total_null_dim}",
            f"Total Dim: {profile.total_dim}",
            f"Mean Null Fraction: {profile.mean_null_fraction:.1%}",
            f"Graftable Layers: {len(profile.graftable_layers)}",
            "",
            "-" * 40,
            "Per-Layer Analysis",
            "-" * 40,
        ]

        for layer_idx, lp in sorted(profile.per_layer.items()):
            graft_marker = " [GRAFTABLE]" if layer_idx in profile.graftable_layers else ""
            lines.append(f"  Layer {layer_idx}: {lp.null_fraction:.1%} null{graft_marker}")

        lines.extend(
            [
                "",
                "Interpretation:",
                f"  {len(profile.graftable_layers)} layers have ≥10% null space",
                "  available for interference-free knowledge grafting.",
                "",
            ]
        )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("safety-polytope")
def safety_polytope_check(
    ctx: typer.Context,
    interference: float = typer.Argument(..., help="Interference score [0-1]"),
    importance: float = typer.Argument(..., help="Importance score [0-1]"),
    instability: float = typer.Argument(..., help="Instability score [0-1]"),
    complexity: float = typer.Argument(..., help="Complexity score [0-1]"),
    baseline_file: str = typer.Option(
        ..., "--baseline-file", help="JSON file with layer diagnostics for bound derivation"
    ),
    base_alpha: float | None = typer.Option(
        None, "--base-alpha", help="Base merge alpha used for recommendations"
    ),
) -> None:
    """
    Check if diagnostics fall within the safety polytope.

    The safety polytope is a 4D decision boundary that combines:
    - Interference: Volume overlap risk
    - Importance: Layer significance
    - Instability: Numerical conditioning
    - Complexity: Manifold dimensionality

    Bounds are derived from the baseline diagnostics file (no fixed thresholds).
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.safety_polytope import (
        DiagnosticVector,
        PolytopeBounds,
        SafetyPolytope,
    )

    baseline_path = Path(baseline_file)
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if isinstance(baseline_data, dict) and "layerDiagnostics" in baseline_data:
        baseline_data = baseline_data["layerDiagnostics"]
    if isinstance(baseline_data, dict):
        baseline_items = list(baseline_data.values())
    else:
        baseline_items = list(baseline_data or [])
    if not baseline_items:
        raise typer.BadParameter("baseline-file must contain layer diagnostics")

    baseline_diagnostics = [
        DiagnosticVector(
            interference_score=item.get("interference", 0.0),
            importance_score=item.get("importance", 0.0),
            instability_score=item.get("instability", 0.0),
            complexity_score=item.get("complexity", 0.0),
        )
        for item in baseline_items
    ]
    bounds = PolytopeBounds.from_baseline_metrics(
        interference_samples=[diag.interference_score for diag in baseline_diagnostics],
        importance_samples=[diag.importance_score for diag in baseline_diagnostics],
        instability_samples=[diag.instability_score for diag in baseline_diagnostics],
        complexity_samples=[diag.complexity_score for diag in baseline_diagnostics],
        magnitude_samples=[diag.magnitude for diag in baseline_diagnostics],
    )
    polytope = SafetyPolytope(bounds=bounds)
    diagnostics = DiagnosticVector(
        interference_score=interference,
        importance_score=importance,
        instability_score=instability,
        complexity_score=complexity,
    )

    # Alpha derived from diagnostics + base alpha (if provided)
    result = polytope.analyze_layer(diagnostics, base_alpha=base_alpha)

    payload = {
        "_schema": "mc.geometry.interference.safety_polytope.v1",
        "diagnostics": {
            "interference": interference,
            "importance": importance,
            "instability": instability,
            "complexity": complexity,
            "magnitude": diagnostics.magnitude,
            "maxDimension": diagnostics.max_dimension,
        },
        "bounds": {
            "interference": bounds.interference_threshold,
            "importance": bounds.importance_threshold,
            "instability": bounds.instability_threshold,
            "complexity": bounds.complexity_threshold,
            "magnitude": bounds.magnitude_threshold,
            "highInstability": bounds.high_instability_threshold,
            "highInterference": bounds.high_interference_threshold,
        },
        "triggers": [
            {
                "dimension": trigger.dimension,
                "value": trigger.value,
                "threshold": trigger.threshold,
                "intensity": trigger.intensity,
                "transformation": trigger.transformation.value,
            }
            for trigger in result.triggers
        ],
        "transformations": [t.value for t in result.transformations],
        "recommendedAlpha": result.recommended_alpha,
        "confidence": result.confidence,
        "transformationEffort": result.transformation_effort,
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "SAFETY POLYTOPE CHECK",
            "=" * 60,
            "",
            "Diagnostics:",
            f"  Interference: {interference:.3f}",
            f"  Importance:   {importance:.3f}",
            f"  Instability:  {instability:.3f}",
            f"  Complexity:   {complexity:.3f}",
            f"  Magnitude:    {diagnostics.magnitude:.3f}",
            "",
            f"Confidence: {result.confidence:.1%}",
        ]
        if result.recommended_alpha is not None:
            lines.append(f"Recommended Alpha: {result.recommended_alpha:.3f}")

        if result.triggers:
            lines.extend(
                [
                    "",
                    "-" * 40,
                    "Triggers",
                    "-" * 40,
                ]
            )
            for v in result.triggers:
                lines.append(
                    f"  {v.dimension}: {v.value:.3f} > {v.threshold:.3f} (intensity: {v.intensity:.2f})"
                )
                lines.append(f"    Transformation: {v.transformation.value}")

        if result.transformations:
            lines.extend(
                [
                    "",
                    "-" * 40,
                    "Recommended Transformations",
                    "-" * 40,
                ]
            )
            for m in result.transformations:
                lines.append(f"  • {m.value}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
