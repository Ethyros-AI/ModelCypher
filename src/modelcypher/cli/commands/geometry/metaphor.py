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

"""Metaphor Geometry CLI commands.

Measures how language models encode conceptual metaphor mappings (Lakoff & Johnson, 1980).
Based on Conceptual Metaphor Theory (CMT), tracks layer-wise source→target domain
convergence and tests cross-model invariance (Platonic Representation Hypothesis).

Commands:
    mc geometry metaphor list
    mc geometry metaphor trajectory /path/to/model --metaphor TIME_IS_MONEY
    mc geometry metaphor convergence /path/to/model
    mc geometry metaphor invariance model_a model_b --metaphor TIME_IS_MONEY
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array

app = typer.Typer(no_args_is_help=True)
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _extract_metaphor_activations(
    model_path: str,
    metaphor_id: str,
) -> dict[int, tuple["Array", "Array"]]:
    """Extract source and target domain activations for a metaphor.

    Args:
        model_path: Path to the model directory.
        metaphor_id: CMT mapping ID (e.g., "cmt_time_is_money").
    Returns:
        Dictionary mapping layer_index to (source_activations, target_activations).
    """
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )

    mapping = ConceptualMetaphorInventory.get_by_id(metaphor_id)
    if not mapping:
        raise typer.BadParameter(f"Unknown metaphor ID: {metaphor_id}")

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)

    if not resolved:
        raise typer.BadParameter(f"Could not resolve architecture for model at {model_path}")

    embed_tokens, layers, norm = resolved
    num_layers = len(layers)

    target_layers = list(range(num_layers))

    typer.echo(f"Architecture resolved: {num_layers} layers, probing {len(target_layers)} layers")

    backend = get_default_backend()
    layer_activations: dict[int, tuple["Array", "Array"]] = {}

    # Get activations for source domain exemplars
    typer.echo(f"Probing {len(mapping.source_exemplars)} source exemplars...")
    source_acts_by_layer: dict[int, list["Array"]] = {l: [] for l in target_layers}

    for exemplar in mapping.source_exemplars:
        try:
            tokens = tokenizer.encode(exemplar)
            input_ids = backend.array([tokens])

            for target_layer in target_layers:
                hidden = forward_through_backbone(
                    input_ids,
                    embed_tokens,
                    layers,
                    norm,
                    target_layer=target_layer,
                    backend=backend,
                )
                activation = backend.mean(hidden[0], axis=0)
                backend.async_eval(activation)
                source_acts_by_layer[target_layer].append(activation)
        except Exception as e:
            logger.warning(f"Failed source exemplar {exemplar}: {e}")

    # Get activations for target domain exemplars
    typer.echo(f"Probing {len(mapping.target_exemplars)} target exemplars...")
    target_acts_by_layer: dict[int, list["Array"]] = {l: [] for l in target_layers}

    for exemplar in mapping.target_exemplars:
        try:
            tokens = tokenizer.encode(exemplar)
            input_ids = backend.array([tokens])

            for target_layer in target_layers:
                hidden = forward_through_backbone(
                    input_ids,
                    embed_tokens,
                    layers,
                    norm,
                    target_layer=target_layer,
                    backend=backend,
                )
                activation = backend.mean(hidden[0], axis=0)
                backend.async_eval(activation)
                target_acts_by_layer[target_layer].append(activation)
        except Exception as e:
            logger.warning(f"Failed target exemplar {exemplar}: {e}")

    # Stack activations per layer
    for target_layer in target_layers:
        if source_acts_by_layer[target_layer] and target_acts_by_layer[target_layer]:
            # Sync all pending
            backend.eval(*source_acts_by_layer[target_layer], *target_acts_by_layer[target_layer])

            source_stacked = backend.stack(source_acts_by_layer[target_layer])
            target_stacked = backend.stack(target_acts_by_layer[target_layer])
            layer_activations[target_layer] = (source_stacked, target_stacked)

    if not layer_activations:
        raise typer.BadParameter("No activations extracted from model")

    typer.echo(f"Extracted activations for {len(layer_activations)} layers.")
    return layer_activations


@app.command("list")
def metaphor_list(
    ctx: typer.Context,
) -> None:
    """
    List available Conceptual Metaphor Theory (CMT) mappings.

    Shows all 8 CMT metaphors from Lakoff & Johnson (1980) with their
    source and target domains.

    Families:
    - time_as_resource: TIME IS MONEY
    - argument_as_conflict: ARGUMENT IS WAR
    - life_as_journey: LIFE IS A JOURNEY
    - ideas_as_objects: IDEAS ARE FOOD
    - emotions_as_substances: EMOTIONS ARE FLUIDS
    - mind_as_space: MIND IS A CONTAINER
    - understanding_as_perception: UNDERSTANDING IS SEEING
    - relationships_as_journeys: LOVE IS A JOURNEY
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )

    mappings = ConceptualMetaphorInventory.ALL_MAPPINGS

    payload = {
        "_schema": "mc.geometry.metaphor.list.v1",
        "mappings": [
            {
                "id": m.id,
                "name": m.name,
                "family": m.family.value,
                "source_domain": m.source_domain,
                "target_domain": m.target_domain,
                "source_exemplar_count": len(m.source_exemplars),
                "target_exemplar_count": len(m.target_exemplars),
                "bridging_expression_count": len(m.bridging_expressions),
            }
            for m in mappings
        ],
        "count": len(mappings),
    }

    if context.output_format == "text":
        lines = [
            "CONCEPTUAL METAPHOR THEORY (CMT) ATLAS",
            f"Total: {len(mappings)} metaphors",
            "",
            f"{'ID':<25} {'Name':<25} {'Source → Target':<30}",
            "-" * 80,
        ]
        for m in mappings:
            lines.append(
                f"{m.id:<25} {m.name:<25} {m.source_domain} → {m.target_domain}"
            )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("trajectory")
def metaphor_trajectory(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    metaphor: str = typer.Option(
        "cmt_time_is_money", "--metaphor", "-m", help="CMT mapping ID"
    ),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save trajectory to JSON"),
) -> None:
    """
    Collect metaphor trajectory through model layers.

    Tracks CKA between source and target domain activations at each layer.
    The convergence layer is where CKA peaks - where the model maps source
    concepts to target concepts.

    Example:
        mc geometry metaphor trajectory /path/to/model --metaphor cmt_time_is_money
    """
    context = _context(ctx)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )
    from modelcypher.core.domain.geometry.metaphor_trajectory import (
        MetaphorTrajectoryCollector,
        compute_convergence_profile,
        convergence_profile_to_dict,
        trajectory_to_dict,
    )

    mapping = ConceptualMetaphorInventory.get_by_id(metaphor)
    if not mapping:
        typer.echo(f"Unknown metaphor ID: {metaphor}", err=True)
        raise typer.Exit(1)

    # Extract activations
    layer_activations = _extract_metaphor_activations(model_path, metaphor)

    # Collect trajectory
    backend = get_default_backend()
    collector = MetaphorTrajectoryCollector(backend)
    model_id = Path(model_path).name

    trajectory = collector.collect_from_activations(mapping, model_id, layer_activations)
    profile = compute_convergence_profile(trajectory)

    payload = {
        "_schema": "mc.geometry.metaphor.trajectory.v1",
        "model_path": model_path,
        "trajectory": trajectory_to_dict(trajectory),
        "profile": convergence_profile_to_dict(profile),
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Saved trajectory to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"METAPHOR TRAJECTORY: {trajectory.metaphor_name}",
            "=" * 60,
            "",
            f"Model: {model_id}",
            f"Source Domain: {trajectory.source_domain}",
            f"Target Domain: {trajectory.target_domain}",
            "",
            f"Convergence Layer: {trajectory.convergence_layer}",
            f"Peak CKA: {trajectory.peak_cka:.4f}",
            f"Layers Analyzed: {trajectory.layer_count}",
            "",
            "-" * 40,
            "LAYER-WISE CKA",
            "-" * 40,
        ]
        for point in trajectory.points:
            bar = "█" * int(point.cka_source_target * 20)
            lines.append(
                f"  Layer {point.layer_index:3d}: {point.cka_source_target:.4f} {bar}"
            )

        lines.extend(
            [
                "",
                "-" * 40,
                "CONVERGENCE PROFILE",
                "-" * 40,
                f"  Early Layer CKA: {profile.early_layer_cka:.4f}",
                f"  Mid Layer CKA: {profile.mid_layer_cka:.4f}",
                f"  Late Layer CKA: {profile.late_layer_cka:.4f}",
                f"  Trajectory Monotonicity: {profile.trajectory_monotonicity:.4f}",
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("convergence")
def metaphor_convergence(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save to JSON"),
) -> None:
    """
    Analyze convergence patterns for all CMT metaphors.

    Runs trajectory analysis for all 8 CMT metaphors and reports
    convergence layers and peak CKA for each.

    Example:
        mc geometry metaphor convergence /path/to/model
    """
    context = _context(ctx)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )
    from modelcypher.core.domain.geometry.metaphor_trajectory import (
        MetaphorTrajectoryCollector,
        compute_convergence_profile,
        convergence_profile_to_dict,
        trajectory_to_dict,
    )

    backend = get_default_backend()
    collector = MetaphorTrajectoryCollector(backend)
    model_id = Path(model_path).name

    results = []
    for mapping in ConceptualMetaphorInventory.ALL_MAPPINGS:
        typer.echo(f"\nAnalyzing: {mapping.name}")
        try:
            layer_activations = _extract_metaphor_activations(model_path, mapping.id)
            trajectory = collector.collect_from_activations(mapping, model_id, layer_activations)
            profile = compute_convergence_profile(trajectory)

            results.append({
                "metaphor_id": mapping.id,
                "metaphor_name": mapping.name,
                "convergence_layer": trajectory.convergence_layer,
                "peak_cka": trajectory.peak_cka,
                "trajectory_monotonicity": profile.trajectory_monotonicity,
            })
        except Exception as e:
            typer.echo(f"  Warning: Failed {mapping.id}: {e}", err=True)
            results.append({
                "metaphor_id": mapping.id,
                "metaphor_name": mapping.name,
                "error": str(e),
            })

    # Compute aggregate statistics
    valid_results = [r for r in results if "convergence_layer" in r]
    if valid_results:
        mean_peak_cka = sum(r["peak_cka"] for r in valid_results) / len(valid_results)
        convergence_layers = [r["convergence_layer"] for r in valid_results]
        mean_convergence = sum(convergence_layers) / len(convergence_layers)
    else:
        mean_peak_cka = 0.0
        mean_convergence = 0.0

    payload = {
        "_schema": "mc.geometry.metaphor.convergence.v1",
        "model_path": model_path,
        "model_id": model_id,
        "results": results,
        "aggregate": {
            "mean_peak_cka": mean_peak_cka,
            "mean_convergence_layer": mean_convergence,
            "metaphors_analyzed": len(valid_results),
            "metaphors_failed": len(results) - len(valid_results),
        },
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"\nSaved results to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"METAPHOR CONVERGENCE ANALYSIS: {model_id}",
            "=" * 60,
            "",
            f"{'Metaphor':<30} {'Conv. Layer':>12} {'Peak CKA':>10}",
            "-" * 60,
        ]
        for r in results:
            if "error" in r:
                lines.append(f"{r['metaphor_name']:<30} {'ERROR':>12} {'-':>10}")
            else:
                lines.append(
                    f"{r['metaphor_name']:<30} {r['convergence_layer']:>12} {r['peak_cka']:>10.4f}"
                )

        lines.extend(
            [
                "",
                "-" * 60,
                "AGGREGATE",
                f"  Mean Convergence Layer: {mean_convergence:.1f}",
                f"  Mean Peak CKA: {mean_peak_cka:.4f}",
                f"  Metaphors Analyzed: {len(valid_results)}/{len(results)}",
            ]
        )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("invariance")
def metaphor_invariance(
    ctx: typer.Context,
    model_a_path: str = typer.Argument(..., help="Path to first model"),
    model_b_path: str = typer.Argument(..., help="Path to second model"),
    metaphor: str = typer.Option(
        "cmt_time_is_money", "--metaphor", "-m", help="CMT mapping ID"
    ),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save to JSON"),
) -> None:
    """
    Test metaphor geometry invariance between two models.

    Compares metaphor trajectories between models to test the
    Platonic Representation Hypothesis: do models converge to
    similar metaphor geometry?

    Measurements:
    - trajectory_cka: CKA between layer-wise CKA trajectories
    - convergence_layer_delta: Difference in convergence layers
    - peak_cka_delta: Difference in peak CKA values

    Example:
        mc geometry metaphor invariance /model_a /model_b --metaphor cmt_time_is_money
    """
    context = _context(ctx)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )
    from modelcypher.core.domain.geometry.metaphor_invariance import (
        MetaphorInvarianceAnalyzer,
        invariance_result_to_dict,
    )
    from modelcypher.core.domain.geometry.metaphor_trajectory import (
        MetaphorTrajectoryCollector,
    )

    mapping = ConceptualMetaphorInventory.get_by_id(metaphor)
    if not mapping:
        typer.echo(f"Unknown metaphor ID: {metaphor}", err=True)
        raise typer.Exit(1)

    backend = get_default_backend()
    collector = MetaphorTrajectoryCollector(backend)

    # Collect trajectories for both models
    typer.echo(f"\n--- Model A: {Path(model_a_path).name} ---")
    layer_acts_a = _extract_metaphor_activations(model_a_path, metaphor)
    trajectory_a = collector.collect_from_activations(
        mapping, Path(model_a_path).name, layer_acts_a
    )

    typer.echo(f"\n--- Model B: {Path(model_b_path).name} ---")
    layer_acts_b = _extract_metaphor_activations(model_b_path, metaphor)
    trajectory_b = collector.collect_from_activations(
        mapping, Path(model_b_path).name, layer_acts_b
    )

    # Compare trajectories
    analyzer = MetaphorInvarianceAnalyzer(backend)
    result = analyzer.compare_metaphor_geometry(trajectory_a, trajectory_b)

    payload = {
        "_schema": "mc.geometry.metaphor.invariance.v1",
        **invariance_result_to_dict(result),
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"\nSaved results to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"METAPHOR INVARIANCE: {result.metaphor_name}",
            "=" * 60,
            "",
            f"Model A: {result.model_a}",
            f"Model B: {result.model_b}",
            "",
            "-" * 40,
            "TRAJECTORY COMPARISON",
            "-" * 40,
            f"  Trajectory CKA: {result.trajectory_cka:.4f}",
            "",
            "-" * 40,
            "CONVERGENCE LAYERS",
            "-" * 40,
            f"  Model A: {result.convergence_layer_a}",
            f"  Model B: {result.convergence_layer_b}",
            f"  Delta (normalized): {result.convergence_layer_delta_normalized:.4f}",
            "",
            "-" * 40,
            "PEAK CKA",
            "-" * 40,
            f"  Model A: {result.peak_cka_a:.4f}",
            f"  Model B: {result.peak_cka_b:.4f}",
            f"  Delta: {result.peak_cka_delta:.4f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("invariance-batch")
def metaphor_invariance_batch(
    ctx: typer.Context,
    model_a_path: str = typer.Argument(..., help="Path to first model"),
    model_b_path: str = typer.Argument(..., help="Path to second model"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="Save to JSON"),
) -> None:
    """
    Test metaphor invariance across all CMT metaphors.

    Runs invariance analysis for all 8 CMT metaphors between two models
    and computes aggregate statistics for Platonic hypothesis testing.

    Example:
        mc geometry metaphor invariance-batch /model_a /model_b
    """
    context = _context(ctx)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.conceptual_metaphor_atlas import (
        ConceptualMetaphorInventory,
    )
    from modelcypher.core.domain.geometry.metaphor_invariance import (
        MetaphorInvarianceAnalyzer,
        PlatonicMetaphorValidator,
        invariance_result_to_dict,
    )
    from modelcypher.core.domain.geometry.metaphor_trajectory import (
        MetaphorTrajectoryCollector,
    )

    backend = get_default_backend()
    collector = MetaphorTrajectoryCollector(backend)

    # Collect trajectories for all metaphors
    trajectories = {
        Path(model_a_path).name: [],
        Path(model_b_path).name: [],
    }

    for mapping in ConceptualMetaphorInventory.ALL_MAPPINGS:
        typer.echo(f"\nAnalyzing: {mapping.name}")

        # Model A
        try:
            typer.echo(f"  Model A: {Path(model_a_path).name}")
            layer_acts_a = _extract_metaphor_activations(model_a_path, mapping.id)
            trajectory_a = collector.collect_from_activations(
                mapping, Path(model_a_path).name, layer_acts_a
            )
            trajectories[Path(model_a_path).name].append(trajectory_a)
        except Exception as e:
            typer.echo(f"  Warning: Failed Model A for {mapping.id}: {e}", err=True)

        # Model B
        try:
            typer.echo(f"  Model B: {Path(model_b_path).name}")
            layer_acts_b = _extract_metaphor_activations(model_b_path, mapping.id)
            trajectory_b = collector.collect_from_activations(
                mapping, Path(model_b_path).name, layer_acts_b
            )
            trajectories[Path(model_b_path).name].append(trajectory_b)
        except Exception as e:
            typer.echo(f"  Warning: Failed Model B for {mapping.id}: {e}", err=True)

    # Run Platonic hypothesis validation
    validator = PlatonicMetaphorValidator(backend)
    result = validator.validate_cross_architecture(trajectories)

    payload = {
        "_schema": "mc.geometry.metaphor.invariance_batch.v1",
        "model_a": Path(model_a_path).name,
        "model_b": Path(model_b_path).name,
        **result,
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"\nSaved results to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 60,
            "PLATONIC METAPHOR INVARIANCE",
            "=" * 60,
            "",
            f"Model A: {Path(model_a_path).name}",
            f"Model B: {Path(model_b_path).name}",
            "",
            "-" * 40,
            "AGGREGATE STATISTICS",
            "-" * 40,
            f"  Mean Trajectory CKA: {result['mean_trajectory_cka']:.4f}",
            f"  Std Trajectory CKA: {result['std_trajectory_cka']:.4f}",
            f"  Mean Convergence Delta: {result['mean_convergence_delta']:.4f}",
            f"  Mean Peak CKA Delta: {result['mean_peak_cka_delta']:.4f}",
            "",
            f"  Models Compared: {result['model_count']}",
            f"  Metaphors Analyzed: {result['metaphor_count']}",
            f"  Pairs Compared: {result['pair_count']}",
            "",
            "-" * 40,
            "PER-METAPHOR TRAJECTORY CKA",
            "-" * 40,
        ]
        for metaphor_id, cka in result.get("per_metaphor_cka", {}).items():
            lines.append(f"  {metaphor_id}: {cka:.4f}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
