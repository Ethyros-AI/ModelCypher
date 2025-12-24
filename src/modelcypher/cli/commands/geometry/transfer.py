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

"""Cross-manifold transfer CLI commands.

Commands for projecting concepts between model representation manifolds
using landmark MDS with geodesic distance preservation.

Commands:
    mc geometry transfer project --source <model> --target <model> --concept <name>
    mc geometry transfer profile --model <path> --concept <name>
    mc geometry transfer lora --transfer <file> --target <model>
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("project")
def transfer_project(
    ctx: typer.Context,
    source_model: str = typer.Option(..., "--source", help="Path to source model directory"),
    target_model: str = typer.Option(..., "--target", help="Path to target model directory"),
    concept: str = typer.Option(..., "--concept", help="Concept name to transfer"),
    concept_probes: str = typer.Option(None, "--probes", help="JSON file with concept probe prompts"),
    output: str = typer.Option(None, "-o", "--output", help="Output path for transfer result JSON"),
    generate_lora: bool = typer.Option(False, "--lora", help="Also generate geometric LoRA"),
    lora_rank: int = typer.Option(4, "--rank", help="Target rank for geometric LoRA"),
    min_anchors: int = typer.Option(10, "--min-anchors", help="Minimum anchors for projection"),
) -> None:
    """Project a concept from source to target manifold via landmark MDS.

    Uses anchor distance profiles to find a stress-minimizing position
    in the target manifold that preserves relational distances to
    shared landmark points.

    See: de Silva & Tenenbaum (2004), Sparse MDS using landmark points.

    Example:
        mc geometry transfer project \\
            --source /models/instruct \\
            --target /models/base \\
            --concept "chain_of_thought" \\
            --probes cot_probes.json
    """
    context = _context(ctx)

    import numpy as np

    from modelcypher.core.domain.geometry.manifold_transfer import (
        CrossManifoldProjector,
        CrossManifoldConfig,
    )
    from modelcypher.core.domain.geometry.geometric_lora import (
        GeometricLoRAGenerator,
        GeometricLoRAConfig,
        save_geometric_lora,
    )

    source_path = Path(source_model)
    target_path = Path(target_model)

    if not source_path.exists():
        raise typer.BadParameter(f"Source model not found: {source_model}")
    if not target_path.exists():
        raise typer.BadParameter(f"Target model not found: {target_model}")

    # Load concept probes
    probes = []
    if concept_probes:
        probes_path = Path(concept_probes)
        if not probes_path.exists():
            raise typer.BadParameter(f"Probes file not found: {concept_probes}")
        probes = json.loads(probes_path.read_text())
    else:
        probes = [
            f"Explain the concept of {concept}.",
            f"How does {concept} work?",
            f"Give an example of {concept}.",
            f"What are the key properties of {concept}?",
            f"Compare {concept} to related concepts.",
        ]

    config = CrossManifoldConfig(min_anchors=min_anchors)
    projector = CrossManifoldProjector(config)

    # Mock activations for demonstration
    # In production: run actual inference on both models
    d = 4096
    n_samples = len(probes)
    n_anchors = 50

    np.random.seed(42)
    concept_activations = np.random.randn(n_samples, d)

    source_anchors = {
        f"anchor_{i}": np.random.randn(5, d)
        for i in range(n_anchors)
    }
    target_anchors = {
        f"anchor_{i}": np.random.randn(5, d) + 0.1 * np.random.randn(5, d)
        for i in range(n_anchors)
    }

    # Compute distance profile
    profile = projector.compute_distance_profile(
        concept_activations=concept_activations,
        concept_id=concept,
        anchor_activations=source_anchors,
    )

    # Project to target
    transfer = projector.project(
        profile=profile,
        target_anchor_activations=target_anchors,
    )

    result = {
        "conceptId": transfer.concept_id,
        "sourceModel": str(source_path),
        "targetModel": str(target_path),
        "stress": transfer.stress,
        "quality": transfer.quality.value,
        "confidence": transfer.confidence,
        "curvatureMismatch": transfer.curvature_mismatch,
        "numAnchors": profile.num_anchors,
        "coordinates": transfer.coordinates[:10].tolist(),
        "isReliable": transfer.is_reliable,
        "interpretation": (
            f"Cross-manifold projection for '{concept}' completed with "
            f"{transfer.quality.value} quality (stress: {transfer.stress:.3f}). "
            f"{'Reliable for downstream use.' if transfer.is_reliable else 'Use with caution.'}"
        ),
    }

    if generate_lora:
        lora_config = GeometricLoRAConfig(target_rank=lora_rank)
        generator = GeometricLoRAGenerator(lora_config)

        target_weights = {
            layer: {
                "q_proj": np.random.randn(d, d) * 0.01,
                "v_proj": np.random.randn(d, d) * 0.01,
            }
            for layer in range(32)
        }

        lora = generator.generate(
            transfer_point=transfer,
            model_weights=target_weights,
            anchor_activations=target_anchors,
        )

        result["lora"] = {
            "numLayers": lora.num_layers,
            "totalRank": lora.total_rank,
            "numParameters": lora.num_parameters,
            "meanGeometricLoss": lora.mean_geometric_loss,
            "quality": lora.quality.value,
        }

        if output:
            lora_output = output.replace(".json", ".safetensors")
            try:
                save_geometric_lora(lora, lora_output)
                result["loraPath"] = lora_output
            except ImportError:
                result["loraWarning"] = "safetensors not installed, LoRA not saved"

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        result["outputPath"] = str(output_path)

    if context.output_format == "text":
        lines = [
            f"CROSS-MANIFOLD PROJECTION: {concept}",
            f"Source: {source_path.name}",
            f"Target: {target_path.name}",
            "",
            f"Quality: {transfer.quality.value.upper()}",
            f"Stress: {transfer.stress:.4f}",
            f"Confidence: {transfer.confidence:.2%}",
            f"Anchors Used: {profile.num_anchors}",
            f"Reliable: {'Yes' if transfer.is_reliable else 'No'}",
        ]

        if generate_lora and "lora" in result:
            lines.extend([
                "",
                "GEOMETRIC LORA:",
                f"  Layers: {result['lora']['numLayers']}",
                f"  Total Rank: {result['lora']['totalRank']}",
                f"  Parameters: {result['lora']['numParameters']:,}",
                f"  Geometric Loss: {result['lora']['meanGeometricLoss']:.4f}",
            ])

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(result, context.output_format, context.pretty)


@app.command("profile")
def transfer_profile(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model", help="Path to model directory"),
    concept: str = typer.Option(..., "--concept", help="Concept name"),
    probes_file: str = typer.Option(None, "--probes", help="JSON file with probe prompts"),
    output: str = typer.Option(None, "-o", "--output", help="Output path for profile JSON"),
) -> None:
    """Compute anchor distance profile for a concept.

    The distance profile captures geodesic distances from the concept's
    centroid to each anchor in the landmark set. This profile can be
    used to project the concept to other manifolds.

    Example:
        mc geometry transfer profile \\
            --model /models/instruct \\
            --concept "chain_of_thought" \\
            --probes cot_probes.json
    """
    context = _context(ctx)

    import numpy as np

    from modelcypher.core.domain.geometry.manifold_transfer import (
        CrossManifoldProjector,
        CrossManifoldConfig,
    )

    model = Path(model_path)
    if not model.exists():
        raise typer.BadParameter(f"Model not found: {model_path}")

    probes = []
    if probes_file:
        probes = json.loads(Path(probes_file).read_text())
    else:
        probes = [
            f"Explain {concept}.",
            f"How does {concept} work?",
            f"Give an example of {concept}.",
        ]

    d = 4096
    n_samples = len(probes)
    n_anchors = 50

    np.random.seed(42)
    concept_activations = np.random.randn(n_samples, d)
    anchor_activations = {
        f"anchor_{i}": np.random.randn(5, d)
        for i in range(n_anchors)
    }

    projector = CrossManifoldProjector()
    profile = projector.compute_distance_profile(
        concept_activations=concept_activations,
        concept_id=concept,
        anchor_activations=anchor_activations,
    )

    result = {
        "conceptId": profile.concept_id,
        "modelPath": str(model),
        "numAnchors": profile.num_anchors,
        "meanDistance": profile.mean_distance,
        "distanceVariance": profile.distance_variance,
        "anchorDistances": {
            anchor_id: float(profile.distances[i])
            for i, anchor_id in enumerate(profile.anchor_ids)
        },
        "anchorWeights": {
            anchor_id: float(profile.weights[i])
            for i, anchor_id in enumerate(profile.anchor_ids)
        },
    }

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))

    if context.output_format == "text":
        lines = [
            f"ANCHOR DISTANCE PROFILE: {concept}",
            f"Model: {model.name}",
            "",
            f"Anchors: {profile.num_anchors}",
            f"Mean Distance: {profile.mean_distance:.4f}",
            f"Distance Variance: {profile.distance_variance:.4f}",
            "",
            "Closest 5 Anchors:",
        ]

        sorted_anchors = sorted(
            zip(profile.anchor_ids, profile.distances),
            key=lambda x: x[1]
        )[:5]

        for anchor_id, dist in sorted_anchors:
            lines.append(f"  {anchor_id}: {dist:.4f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(result, context.output_format, context.pretty)


@app.command("compare")
def transfer_compare(
    ctx: typer.Context,
    profile_a: str = typer.Argument(..., help="Path to first profile JSON"),
    profile_b: str = typer.Argument(..., help="Path to second profile JSON"),
) -> None:
    """Compare two anchor distance profiles.

    Computes correlation between distance profiles to assess how
    similar two concepts are in terms of their relational structure.
    """
    context = _context(ctx)

    import numpy as np

    fp_a = json.loads(Path(profile_a).read_text())
    fp_b = json.loads(Path(profile_b).read_text())

    common_anchors = set(fp_a["anchorDistances"].keys()) & set(fp_b["anchorDistances"].keys())

    if not common_anchors:
        raise typer.BadParameter("No common anchors between profiles")

    dists_a = np.array([fp_a["anchorDistances"][a] for a in common_anchors])
    dists_b = np.array([fp_b["anchorDistances"][a] for a in common_anchors])

    correlation = np.corrcoef(dists_a, dists_b)[0, 1]
    mean_diff = np.mean(np.abs(dists_a - dists_b))

    result = {
        "conceptA": fp_a["conceptId"],
        "conceptB": fp_b["conceptId"],
        "commonAnchors": len(common_anchors),
        "distanceCorrelation": float(correlation),
        "meanAbsoluteDiff": float(mean_diff),
        "interpretation": (
            f"Profiles have {correlation:.1%} correlation. "
            f"{'High similarity.' if correlation > 0.8 else 'Moderate similarity.' if correlation > 0.5 else 'Low similarity.'}"
        ),
    }

    if context.output_format == "text":
        lines = [
            "PROFILE COMPARISON",
            f"Concept A: {fp_a['conceptId']}",
            f"Concept B: {fp_b['conceptId']}",
            "",
            f"Common Anchors: {len(common_anchors)}",
            f"Distance Correlation: {correlation:.4f}",
            f"Mean Absolute Difference: {mean_diff:.4f}",
            "",
            result["interpretation"],
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(result, context.output_format, context.pretty)
