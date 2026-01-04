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
from typing import TYPE_CHECKING, Any

import typer

from modelcypher.cli.composition import get_domain_geometry_waypoint_service
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_model_path
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    power_iteration_eigh,
)
from modelcypher.core.support.array_utils import array_to_list

if TYPE_CHECKING:
    from modelcypher.core.domain.domains import AtlasDomain
    from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
        DomainGeometryWaypointService,
    )

app = typer.Typer(help="Merge analysis for model alignment")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("predict")
def predict_interference(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    target_path: str = typer.Argument(..., help="Path to target model"),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save report to file"
    ),
) -> None:
    """Analyze merge effort and interference between source and target models."""
    context = _context(ctx)

    # Validate model paths early
    validate_model_path(source_path, context=context)
    validate_model_path(target_path, context=context)

    from modelcypher.core.domain.domains import AtlasDomain
    from modelcypher.core.domain.geometry.interference_predictor import (
        MergeAnalyzer,
    )
    from modelcypher.core.domain.geometry.riemannian_density import (
        RiemannianDensityEstimator,
    )

    typer.echo("Predicting interference...")
    typer.echo(f"  Source: {source_path}")
    typer.echo(f"  Target: {target_path}")

    domain_list = list(AtlasDomain)
    layer = -1

    # Extract activations for both models
    waypoint_service = get_domain_geometry_waypoint_service()
    density_estimator = RiemannianDensityEstimator()
    predictor = MergeAnalyzer()

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

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.cka import compute_cka_backend, HSICEstimator

    backend = get_default_backend()

    for domain_name, source_acts in source_activations.items():
        target_acts = target_activations.get(domain_name, {})
        if not source_acts or not target_acts:
            continue

        # Find common concepts between source and target
        common_concepts = sorted(set(source_acts.keys()) & set(target_acts.keys()))
        if len(common_concepts) < 2:
            logger.warning(f"Need at least 2 common concepts for CKA, got {len(common_concepts)}")
            continue

        # Stack all concept activations into domain-level matrices
        # Each row = one concept's activation, giving us n_concepts samples for CKA
        source_stacked = []
        target_stacked = []
        for concept_id in common_concepts:
            src_arr = source_acts[concept_id]
            tgt_arr = target_acts[concept_id]
            # Flatten to 1D if needed
            if src_arr.ndim > 1:
                src_arr = backend.reshape(src_arr, (-1,))
            if tgt_arr.ndim > 1:
                tgt_arr = backend.reshape(tgt_arr, (-1,))
            source_stacked.append(src_arr)
            target_stacked.append(tgt_arr)

        # Stack into [n_concepts, hidden_dim] - may have different hidden dims
        source_matrix = backend.stack(source_stacked, axis=0)
        target_matrix = backend.stack(target_stacked, axis=0)
        backend.eval(source_matrix, target_matrix)

        # No normalization - CKA is scale-invariant via Gram matrix normalization
        # Compute domain-level CKA (dimension-agnostic)
        domain_cka = compute_cka_backend(
            source_matrix,
            target_matrix,
            backend=backend,
            estimator=HSICEstimator.BIASED,
            feature_bias_correction=False,
        )
        logger.info(f"Domain {domain_name} CKA: {domain_cka:.4f} ({len(common_concepts)} concepts)")

        # Also create per-concept volumes for detailed analysis
        source_volumes = {}
        target_volumes = {}

        for concept_id in common_concepts:
            src_arr = source_acts[concept_id]
            tgt_arr = target_acts[concept_id]

            # Need 2D for volume estimation
            if src_arr.ndim == 1:
                src_arr = backend.reshape(src_arr, (1, -1))
            if tgt_arr.ndim == 1:
                tgt_arr = backend.reshape(tgt_arr, (1, -1))

            # Note: Per-concept CKA will return 0.0 (only 1 sample per concept)
            # But we use domain-level CKA above for the actual alignment metric
            source_volumes[concept_id] = density_estimator.estimate_concept_volume(
                f"source:{concept_id}", src_arr, store_raw_activations=True
            )
            target_volumes[concept_id] = density_estimator.estimate_concept_volume(
                f"target:{concept_id}", tgt_arr, store_raw_activations=True
            )

        # Analyze merge requirements for this domain
        domain_analysis = {
            "concepts_analyzed": len(common_concepts),
            "overlap_scores": [],
            "subspace_alignments": [],
            "curvature_scores": [],
            "distance_scores": [],
        }

        for concept_id in common_concepts:
            result = predictor.analyze(source_volumes[concept_id], target_volumes[concept_id])
            domain_analysis["overlap_scores"].append(result.overlap_score)
            domain_analysis["subspace_alignments"].append(result.subspace_alignment)
            domain_analysis["curvature_scores"].append(result.curvature_divergence)
            domain_analysis["distance_scores"].append(result.distance_score)

        # Compute domain-level metrics
        # Use domain-level CKA for alignment (dimension-agnostic, works for cross-architecture)
        # Per-concept CKA is 0.0 because each concept has only 1 sample
        if domain_analysis["overlap_scores"]:
            overlap_arr = backend.array(domain_analysis["overlap_scores"])
            curvature_arr = backend.array(domain_analysis["curvature_scores"])
            distance_arr = backend.array(domain_analysis["distance_scores"])
            domain_analysis["mean_overlap"] = float(backend.mean(overlap_arr))
            domain_analysis["mean_curvature_divergence"] = float(backend.mean(curvature_arr))
            domain_analysis["mean_distance"] = float(backend.mean(distance_arr))
        else:
            domain_analysis["mean_overlap"] = 0.0
            domain_analysis["mean_curvature_divergence"] = 0.0
            domain_analysis["mean_distance"] = 0.0

        # Use domain-level CKA for alignment (computed above)
        domain_analysis["domain_cka"] = domain_cka
        eps = float(machine_epsilon(backend, backend.array([1.0])))
        domain_analysis["domain_aligned"] = abs(domain_cka - 1.0) <= eps

        del domain_analysis["overlap_scores"]  # Don't need raw lists in output
        del domain_analysis["subspace_alignments"]
        del domain_analysis["curvature_scores"]
        del domain_analysis["distance_scores"]
        domain_results[domain_name] = domain_analysis

    all_overlap_scores = []
    all_domain_cka = []
    all_curvature_scores = []
    all_distance_scores = []

    for domain_name, dr in domain_results.items():
        all_overlap_scores.append(dr["mean_overlap"])
        all_domain_cka.append(dr["domain_cka"])
        all_curvature_scores.append(dr["mean_curvature_divergence"])
        all_distance_scores.append(dr["mean_distance"])

    if all_overlap_scores:
        backend = get_default_backend()
        mean_overlap = float(backend.mean(backend.array(all_overlap_scores)))
        mean_cka = float(backend.mean(backend.array(all_domain_cka)))
        mean_curvature = float(backend.mean(backend.array(all_curvature_scores)))
        mean_distance = float(backend.mean(backend.array(all_distance_scores)))
    else:
        mean_overlap = 0.0
        mean_cka = 1.0
        mean_curvature = 0.0
        mean_distance = 0.0

    payload = {
        "_schema": "mc.geometry.merge_analysis.v1",
        "sourceModel": source_path,
        "targetModel": target_path,
        "layer": layer,
        "domainsAnalyzed": [d.value for d in domain_list],
        "perDomain": domain_results,
        "globalMetrics": {
            "meanOverlap": mean_overlap,
            "meanCka": mean_cka,
            "meanCurvatureDivergence": mean_curvature,
            "meanDistance": mean_distance,
        },
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
            "Per-Domain Analysis:",
        ]

        for domain_name, dr in domain_results.items():
            lines.append(f"  {domain_name.upper()}:")
            lines.append(f"    Concepts: {dr['concepts_analyzed']}")
            lines.append(f"    Mean Overlap: {dr['mean_overlap']:.2f}")
            lines.append(f"    Domain CKA: {dr['domain_cka']:.4f}")
            lines.append(f"    Domain Aligned: {dr['domain_aligned']}")
            lines.append(f"    Mean Curvature Divergence: {dr['mean_curvature_divergence']:.2f}")
            lines.append(f"    Mean Distance: {dr['mean_distance']:.2f}")

        lines.extend(
            [
                "",
            ]
        )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _extract_domain_activations(
    model_path: str,
    domain: "AtlasDomain",
    layer: int,
    service: "DomainGeometryWaypointService",
) -> dict[str, Any]:
    """Extract activations for probes in a specific domain.

    Uses the UnifiedAtlas to get probes for ALL domains, not just a hardcoded subset.
    """

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

    backend = get_default_backend()
    model, tokenizer = load_model_for_training(model_path)

    # Get ALL probes for this domain from the UnifiedAtlasInventory
    all_probes = UnifiedAtlasInventory.all_probes()

    # Filter to probes matching this domain
    domain_probes = [p for p in all_probes if p.domain == domain]

    if not domain_probes:
        logger.warning(f"No probes found for domain {domain.value}")
        return {}

    # Convert to (id, prompt) format - use support_texts[0] if available, else description
    probes = []
    for p in domain_probes:
        if p.support_texts:
            prompt = p.support_texts[0]
        else:
            prompt = f"The concept of {p.name} means"
        probes.append((p.id, prompt))

    return service._extract_activations(model, tokenizer, layer, probes, backend)


@app.command("volume")
def compute_volume(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model"),
    concept: str = typer.Argument(..., help="Concept to analyze"),
) -> None:
    """
    Compute ConceptVolume for a single concept.

    Shows the distributional properties of a concept in the model's
    latent space: centroid, covariance, geodesic radius, curvature.
    """
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.riemannian_density import (
        RiemannianDensityEstimator,
    )

    typer.echo(f"Computing volume for concept: {concept}")
    layer = -1

    backend = get_default_backend()
    model, tokenizer = load_model_for_training(model_path)

    # Generate prompt variations
    base_prompts = [
        f"The word {concept} represents",
        f"The concept of {concept} means",
        f"{concept.capitalize()} is defined as",
        f"When we say {concept}, we mean",
        f"The meaning of {concept} is",
    ]
    prompts = base_prompts

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
                activations.append(act)

        except Exception as e:
            logger.warning(f"Failed to extract: {e}")

    if not activations:
        typer.echo("Failed to extract any activations", err=True)
        raise typer.Exit(1)

    # Estimate volume - use backend for stacking activations
    act_array = backend.stack(activations)
    estimator = RiemannianDensityEstimator()
    volume = estimator.estimate_concept_volume(concept, act_array)

    # Compute eigenvalues (geodesic - GPU-only)
    cov_arr = backend.array(volume.covariance)
    n_cov = int(cov_arr.shape[0])
    eigenvalues, _ = power_iteration_eigh(backend, cov_arr, k=n_cov)
    backend.eval(eigenvalues)
    eigenvalues_list = array_to_list(backend, eigenvalues)
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
) -> None:
    """
    Analyze null space availability for interference-free merging.

    Computes the null space profile of a model's activations to identify
    which layers have space for knowledge grafting without interference.

    Rank threshold is derived from the spectral gap of the activation matrix.
    No user parameters for thresholds.

    Based on MINGLE (arXiv:2509.21413).
    """
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
    from modelcypher.core.domain.geometry.vector_math import geodesic_norms

    typer.echo(f"Analyzing geodesic orthogonal space for: {model_path}")
    layer = -1

    backend = get_default_backend()
    model, tokenizer = load_model_for_training(model_path)

    # Use fixed set of diverse prompts for activation extraction
    sample_prompts = [
        "The concept of justice represents",
        "A chair is used for",
        "Yesterday I went to",
        "The number five is",
        "My friend told me",
        "The theory of relativity explains",
        "A computer program can",
        "When the sun rises",
        "Mathematics helps us understand",
        "The color blue reminds me of",
        "In the beginning there was",
        "Scientific discovery requires",
        "The nature of consciousness is",
        "A good story needs",
        "The purpose of education is",
        "Music affects our emotions by",
        "The structure of DNA contains",
        "Economic systems depend on",
        "Language allows us to",
        "The universe began with",
    ]

    # Extract activations per layer
    layer_activations: dict[int, list[Any]] = {}

    typer.echo(f"  Extracting activations from {len(sample_prompts)} prompts...")

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
                    layer_activations[i].append(act)

                    if i == target_layer:
                        break

        except Exception as e:
            logger.warning(f"Failed to extract: {e}")

    # Compute geodesic orthogonal profile - accurate for high-D manifolds (8kD+)
    # Euclidean SVD-based methods are only accurate up to 3D
    geo_filter = GeodesicNullSpaceFilter(backend)

    # Stack activations and compute geodesic profile per layer
    per_layer_info: dict[int, dict] = {}
    orthogonal_fractions = []
    graftable_layers = []

    for layer_idx, acts in layer_activations.items():
        arr = backend.stack(acts)
        backend.eval(arr)

        total_dim = int(arr.shape[1])

        # Probe with a unit vector to measure orthogonal space
        probe = backend.ones((total_dim,), dtype="float32")
        norm_arr = geodesic_norms(backend.reshape(probe, (1, -1)), backend)
        backend.eval(norm_arr)
        norm_val = float(backend.to_scalar(norm_arr))
        probe = probe / norm_val
        backend.eval(probe)

        result = geo_filter.filter_delta(probe, arr)
        orthogonal_frac = result.preserved_fraction
        orthogonal_fractions.append(orthogonal_frac)

        per_layer_info[layer_idx] = {
            "orthogonalDim": result.orthogonal_dim,
            "totalDim": total_dim,
            "orthogonalFraction": orthogonal_frac,
            "meanGeodesicDistance": result.mean_geodesic_distance,
            "kNeighbors": result.k_neighbors,
        }

    # Determine graftable layers (above mean orthogonal fraction)
    mean_frac = sum(orthogonal_fractions) / len(orthogonal_fractions) if orthogonal_fractions else 0.0
    for layer_idx, info in per_layer_info.items():
        if info["orthogonalFraction"] >= mean_frac:
            graftable_layers.append(layer_idx)

    total_orthogonal_dim = sum(info["orthogonalDim"] for info in per_layer_info.values())
    total_dim = sum(info["totalDim"] for info in per_layer_info.values())

    payload = {
        "_schema": "mc.geometry.interference.geodesic_orthogonal.v1",
        "model": model_path,
        "samples": len(sample_prompts),
        "totalOrthogonalDim": total_orthogonal_dim,
        "totalDim": total_dim,
        "meanOrthogonalFraction": mean_frac,
        "graftableLayers": graftable_layers,
        "perLayer": {
            str(layer_idx): info
            for layer_idx, info in per_layer_info.items()
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "GEODESIC ORTHOGONAL SPACE ANALYSIS",
            "=" * 60,
            "",
            f"Model: {Path(model_path).name}",
            f"Samples: {len(sample_prompts)}",
            "Method: Geodesic (accurate for high-D manifolds)",
            "",
            "-" * 40,
            "Summary",
            "-" * 40,
            f"Total Orthogonal Dim: {total_orthogonal_dim}",
            f"Total Dim: {total_dim}",
            f"Mean Orthogonal Fraction: {mean_frac:.1%}",
            f"Graftable Layers: {len(graftable_layers)}",
            "",
            "-" * 40,
            "Per-Layer Analysis",
            "-" * 40,
        ]

        for layer_idx, info in sorted(per_layer_info.items()):
            graft_marker = " [GRAFTABLE]" if layer_idx in graftable_layers else ""
            lines.append(f"  Layer {layer_idx}: {info['orthogonalFraction']:.1%} orthogonal{graft_marker}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
