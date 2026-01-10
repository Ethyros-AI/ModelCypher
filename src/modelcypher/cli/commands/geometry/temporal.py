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

"""
CLI commands for Temporal Topology analysis.

Probes the emergent temporal manifold in language models - the geometric structure
encoding direction (past→future), duration (moment→eternity), and causality axes.

Implements the "Latent Chronologist" hypothesis testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.commands.geometry.helpers import (
    forward_through_backbone,
    resolve_model_backbone,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_file_exists, validate_model_path

app = typer.Typer(help="Temporal topology analysis commands")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("anchors")
def temporal_anchors(
    ctx: typer.Context,
) -> None:
    """
    List the Temporal Prime Atlas anchors.

    Shows the 25 temporal anchors with their axis assignments and ordering levels.
    These anchors probe the model's temporal geometry.

    Axes:
    - direction: past → yesterday → today → tomorrow → future
    - duration: moment → hour → day → year → century
    - causality: because → causes → leads to → therefore → results in
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents.temporal_atlas import TemporalConceptInventory

    anchors = TemporalConceptInventory.all_concepts()

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "TEMPORAL PRIME ATLAS",
            "=" * 70,
            "",
            f"{'Concept':<15} {'Axis':<12} {'Level':<6} {'Category':<25}",
            "-" * 70,
        ]
        for a in anchors:
            lines.append(f"{a.id:<15} {a.axis.value:<12} {a.level:<6} {a.category.value:<25}")
        lines.append("")
        lines.append(f"Total: {len(anchors)} anchors")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "_schema": "mc.geometry.temporal.anchors.v1",
        "anchors": [
            {
                "concept": a.id,
                "axis": a.axis.value,
                "level": a.level,
                "category": a.category.value,
                "prompt": a.prompt,
            }
            for a in anchors
        ],
        "total": len(anchors),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def temporal_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    output_file: str = typer.Option(None, "--output-file", "-o", help="File to save activations"),
) -> None:
    """
    Probe a model for temporal topology structure.

    Tests the "Latent Chronologist" hypothesis by measuring:
    - Axis orthogonality (Direction ⊥ Duration ⊥ Causality)
    - Gradient consistency (monotonic orderings)
    - Arrow of Time detection (past→future direction)
    - Temporal Manifold Score (TMS)
    """
    context = _context(ctx)
    validate_model_path(model_path, context=context)

    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.agents.temporal_atlas import TemporalConceptInventory
    from modelcypher.core.domain.geometry.temporal_topology import TemporalTopologyAnalyzer
    from modelcypher.core.support.array_utils import array_to_list

    typer.echo(f"Loading model from {model_path}...")
    model, tokenizer = load_model_for_training(model_path)

    model_type = getattr(model, "model_type", "unknown")
    resolved = resolve_model_backbone(model, model_type)

    if not resolved:
        typer.echo("Error: Could not resolve architecture.", err=True)
        raise typer.Exit(1)

    embed_tokens, layers, norm = resolved
    if embed_tokens is None or layers is None:
        typer.echo("Error: Could not find embedding or layers.", err=True)
        raise typer.Exit(1)

    num_layers = len(layers)
    target_layer = num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = get_default_backend()
    anchor_activations = {}

    anchors = TemporalConceptInventory.all_concepts()
    typer.echo(f"Probing {len(anchors)} temporal anchors...")

    for anchor in anchors:
        try:
            tokens = tokenizer.encode(anchor.prompt)
            input_ids = backend.array([tokens])

            hidden = forward_through_backbone(
                input_ids,
                embed_tokens,
                layers,
                norm,
                target_layer=target_layer,
                backend=backend,
            )

            activation = backend.mean(hidden[0], axis=0)
            backend.eval(activation)
            anchor_activations[anchor.id] = activation

        except Exception as e:
            typer.echo(f"  Warning: Failed anchor {anchor.id}: {e}", err=True)

    if not anchor_activations:
        typer.echo("Error: No activations extracted.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(anchor_activations)} activations.")

    # Save activations if requested
    if output_file:
        activations_json = {
            name: array_to_list(backend, act) for name, act in anchor_activations.items()
        }
        Path(output_file).write_text(json.dumps(activations_json, indent=2))
        typer.echo(f"Saved activations to {output_file}")

    # Run analysis
    typer.echo("Running temporal topology analysis...")
    analyzer = TemporalTopologyAnalyzer(anchor_activations)
    report = analyzer.analyze()

    # Compute composite score from components
    components = report.temporal_manifold_components
    temporal_manifold_score = (
        components.orthogonality_score * 0.4 +
        components.gradient_score * 0.3 +
        components.arrow_score * 0.3
    )
    payload = {
        "_schema": "mc.geometry.temporal.probe_model.v1",
        "model_path": model_path,
        "anchors_probed": report.anchors_probed,
        "layer": target_layer,
        "temporal_manifold_score": temporal_manifold_score,
        "temporal_manifold_components": {
            "orthogonality_score": components.orthogonality_score,
            "gradient_score": components.gradient_score,
            "arrow_score": components.arrow_score,
        },
        "axis_orthogonality": {
            "direction_duration": report.axis_orthogonality.direction_duration,
            "direction_causality": report.axis_orthogonality.direction_causality,
            "duration_causality": report.axis_orthogonality.duration_causality,
            "mean": report.axis_orthogonality.mean_orthogonality,
        },
        "gradient_consistency": {
            "direction": {
                "correlation": report.gradient_consistency.direction_correlation,
                "monotonic": report.gradient_consistency.direction_monotonic,
            },
            "duration": {
                "correlation": report.gradient_consistency.duration_correlation,
                "monotonic": report.gradient_consistency.duration_monotonic,
            },
            "causality": {
                "correlation": report.gradient_consistency.causality_correlation,
                "monotonic": report.gradient_consistency.causality_monotonic,
            },
        },
        "arrow_of_time": {
            "detected": report.arrow_of_time.arrow_detected,
            "correlation": report.arrow_of_time.direction_correlation,
            "past_anchors": report.arrow_of_time.past_anchors,
            "future_anchors": report.arrow_of_time.future_anchors,
        },
        "principal_components_variance": report.principal_components_variance,
    }

    if context.output_format == "text":
        lines = [
            "=" * 60,
            f"TEMPORAL TOPOLOGY ANALYSIS: {Path(model_path).name}",
            "=" * 60,
            "",
            f"Anchors Probed: {report.anchors_probed}/23",
            f"Layer Analyzed: {target_layer}",
            "",
            f"Temporal Manifold Score: {report.temporal_manifold_score:.4f}",
            "",
            "-" * 40,
            "Axis Orthogonality:",
            f"  Direction ⊥ Duration:  {report.axis_orthogonality.direction_duration:.2%}",
            f"  Direction ⊥ Causality: {report.axis_orthogonality.direction_causality:.2%}",
            f"  Duration ⊥ Causality:  {report.axis_orthogonality.duration_causality:.2%}",
            f"  Mean:                  {report.axis_orthogonality.mean_orthogonality:.2%}",
            "",
            "Gradient Consistency:",
            f"  Direction: {'MONOTONIC' if report.gradient_consistency.direction_monotonic else 'non-monotonic'} (r={report.gradient_consistency.direction_correlation:.2f})",
            f"  Duration:  {'MONOTONIC' if report.gradient_consistency.duration_monotonic else 'non-monotonic'} (r={report.gradient_consistency.duration_correlation:.2f})",
            f"  Causality: {'MONOTONIC' if report.gradient_consistency.causality_monotonic else 'non-monotonic'} (r={report.gradient_consistency.causality_correlation:.2f})",
            "",
            "Arrow of Time:",
            f"  Detected: {'YES' if report.arrow_of_time.arrow_detected else 'NO'}",
            f"  Correlation: {report.arrow_of_time.direction_correlation:.2f}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze")
def temporal_analyze(
    ctx: typer.Context,
    activations_file: str = typer.Argument(..., help="JSON file with anchor activations"),
) -> None:
    """
    Analyze temporal topology from pre-computed activations.

    Input format: JSON with {anchor_concept: [activation_vector]} mappings.
    """
    context = _context(ctx)
    validate_file_exists(activations_file, description="Activations file", context=context)

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.temporal_topology import TemporalTopologyAnalyzer

    path = Path(activations_file)
    with open(path) as f:
        raw_activations = json.load(f)

    # Convert to backend arrays for analyzer
    backend = get_default_backend()
    activations = {name: backend.array(vec) for name, vec in raw_activations.items()}

    analyzer = TemporalTopologyAnalyzer(activations)
    report = analyzer.analyze()

    payload = {
        "_schema": "mc.geometry.temporal.analyze.v1",
        "source_file": activations_file,
        "anchors_analyzed": report.anchors_probed,
        "temporal_manifold_score": report.temporal_manifold_score,
        "axis_orthogonality": {
            "direction_duration": report.axis_orthogonality.direction_duration,
            "direction_causality": report.axis_orthogonality.direction_causality,
            "duration_causality": report.axis_orthogonality.duration_causality,
            "mean": report.axis_orthogonality.mean_orthogonality,
        },
        "gradient_consistency": {
            "direction": {
                "correlation": report.gradient_consistency.direction_correlation,
                "monotonic": report.gradient_consistency.direction_monotonic,
            },
            "duration": {
                "correlation": report.gradient_consistency.duration_correlation,
                "monotonic": report.gradient_consistency.duration_monotonic,
            },
            "causality": {
                "correlation": report.gradient_consistency.causality_correlation,
                "monotonic": report.gradient_consistency.causality_monotonic,
            },
        },
        "arrow_of_time": {
            "detected": report.arrow_of_time.arrow_detected,
            "correlation": report.arrow_of_time.direction_correlation,
        },
    }

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
