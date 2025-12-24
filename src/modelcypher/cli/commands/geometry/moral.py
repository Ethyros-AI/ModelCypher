"""
CLI commands for Moral Geometry analysis.

Probes the emergent moral manifold in language models - the geometric structure
encoding valence (good→evil), agency (victim→perpetrator), and scope axes.

Implements the "Latent Ethicist" hypothesis testing based on Haidt's Moral
Foundations Theory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.commands.geometry.helpers import (
    resolve_model_backbone,
    forward_through_backbone,
    save_activations_json,
)

app = typer.Typer(help="Moral geometry analysis commands")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("anchors")
def moral_anchors(
    ctx: typer.Context,
    foundation: str = typer.Option(None, "--foundation", help="Filter by foundation: care_harm, fairness_cheating, etc."),
    axis: str = typer.Option(None, "--axis", help="Filter by axis: valence, agency, scope"),
) -> None:
    """
    List the Moral Prime Atlas anchors.

    Shows the 30 moral anchors with their axis assignments and foundation categories.
    Based on Haidt's Moral Foundations Theory.

    Axes:
    - valence: cruelty → kindness → compassion (evil→good)
    - agency: betrayal → loyalty → devotion (victim→perpetrator)
    - scope: defilement → purity → sanctity (self→universal)

    Foundations:
    - care_harm: Compassion vs cruelty
    - fairness_cheating: Justice vs exploitation
    - loyalty_betrayal: Group solidarity vs treachery
    - authority_subversion: Respect vs rebellion
    - sanctity_degradation: Purity vs contamination
    - liberty_oppression: Freedom vs tyranny
    """
    context = _context(ctx)

    from modelcypher.core.domain.agents.moral_atlas import (
        ALL_MORAL_PROBES,
        MoralAxis,
        MoralFoundation,
    )

    anchors = list(ALL_MORAL_PROBES)

    if foundation:
        try:
            found_enum = MoralFoundation(foundation.lower())
            anchors = [a for a in anchors if a.foundation == found_enum]
        except ValueError:
            typer.echo(f"Invalid foundation: {foundation}. Options: care_harm, fairness_cheating, loyalty_betrayal, authority_subversion, sanctity_degradation, liberty_oppression", err=True)
            raise typer.Exit(1)

    if axis:
        try:
            axis_enum = MoralAxis(axis.lower())
            anchors = [a for a in anchors if a.axis == axis_enum]
        except ValueError:
            typer.echo(f"Invalid axis: {axis}. Use: valence, agency, scope", err=True)
            raise typer.Exit(1)

    if context.output_format == "text":
        lines = [
            "=" * 80,
            "MORAL PRIME ATLAS (Haidt Moral Foundations Theory)",
            "=" * 80,
            "",
            f"{'Name':<15} {'Axis':<10} {'Level':<6} {'Foundation':<25}",
            "-" * 80,
        ]
        for a in anchors:
            lines.append(f"{a.name:<15} {a.axis.value:<10} {a.level:<6} {a.foundation.value:<25}")
        lines.append("")
        lines.append(f"Total: {len(anchors)} anchors")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "_schema": "mc.geometry.moral.anchors.v1",
        "anchors": [
            {
                "name": a.name,
                "id": a.id,
                "axis": a.axis.value,
                "level": a.level,
                "foundation": a.foundation.value,
            }
            for a in anchors
        ],
        "total": len(anchors),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("probe-model")
def moral_probe_model(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    layer: int = typer.Option(-1, help="Layer to analyze (default is last)"),
    output_file: str = typer.Option(None, "--output", "-o", help="File to save activations"),
) -> None:
    """
    Probe a model for moral geometry structure.

    Tests the "Latent Ethicist" hypothesis by measuring:
    - Axis orthogonality (Valence ⊥ Agency ⊥ Scope)
    - Gradient consistency (monotonic orderings)
    - Foundation clustering (distinct moral domains)
    - Virtue-vice opposition (cruelty↔compassion, etc.)
    - Moral Manifold Score (MMS)
    """
    context = _context(ctx)

    import numpy as np
    from modelcypher.core.domain.agents.moral_atlas import ALL_MORAL_PROBES
    from modelcypher.core.domain.geometry.moral_geometry import MoralGeometryAnalyzer
    from modelcypher.adapters.model_loader import load_model_for_training
    from modelcypher.backends.mlx_backend import MLXBackend

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
    target_layer = layer if layer >= 0 else num_layers - 1
    typer.echo(f"Architecture resolved: {num_layers} layers, probing layer {target_layer}")

    backend = MLXBackend()
    anchor_activations = {}

    typer.echo(f"Probing {len(ALL_MORAL_PROBES)} moral anchors...")

    for concept in ALL_MORAL_PROBES:
        try:
            prompt = f"The word {concept.name.lower()} represents"
            tokens = tokenizer.encode(prompt)
            input_ids = backend.array([tokens])

            hidden = forward_through_backbone(
                input_ids, embed_tokens, layers, norm,
                target_layer=target_layer,
                backend=backend,
            )

            activation = backend.mean(hidden[0], axis=0)
            backend.eval(activation)
            anchor_activations[concept.name] = backend.to_numpy(activation)

        except Exception as e:
            typer.echo(f"  Warning: Failed anchor {concept.name}: {e}", err=True)

    if len(anchor_activations) < 15:
        typer.echo(f"Error: Only {len(anchor_activations)} activations extracted (need 15+).", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracted {len(anchor_activations)} activations.")

    # Save activations if requested
    if output_file:
        activations_json = {
            name: act.tolist()
            for name, act in anchor_activations.items()
        }
        Path(output_file).write_text(json.dumps(activations_json, indent=2))
        typer.echo(f"Saved activations to {output_file}")

    # Run analysis
    typer.echo("Running moral geometry analysis...")
    analyzer = MoralGeometryAnalyzer(backend=backend)
    report = analyzer.full_analysis(anchor_activations, model_path=model_path, layer=layer)

    payload = {
        "_schema": "mc.geometry.moral.probe_model.v1",
        **report.to_dict(),
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            f"MORAL GEOMETRY ANALYSIS: {Path(model_path).name}",
            "=" * 70,
            "",
            f"Anchors Probed: {report.anchors_probed}/30",
            f"Layer Analyzed: {layer if layer != -1 else 'last'}",
            "",
            f"Has Moral Manifold: {'YES' if report.has_moral_manifold else 'NO'}",
            f"Moral Manifold Score: {report.moral_manifold_score:.4f}",
            "",
            "-" * 50,
            "Axis Orthogonality:",
            f"  Valence ⊥ Agency:  {report.axis_orthogonality.valence_agency:.2%}",
            f"  Valence ⊥ Scope:   {report.axis_orthogonality.valence_scope:.2%}",
            f"  Agency ⊥ Scope:    {report.axis_orthogonality.agency_scope:.2%}",
            f"  Mean:              {report.axis_orthogonality.mean_orthogonality:.2%}",
            "",
            "Gradient Consistency:",
            f"  Valence: {'MONOTONIC' if report.gradient_consistency.valence_monotonic else 'non-monotonic'} (r={report.gradient_consistency.valence_correlation:.2f})",
            f"  Agency:  {'MONOTONIC' if report.gradient_consistency.agency_monotonic else 'non-monotonic'} (r={report.gradient_consistency.agency_correlation:.2f})",
            f"  Scope:   {'MONOTONIC' if report.gradient_consistency.scope_monotonic else 'non-monotonic'} (r={report.gradient_consistency.scope_correlation:.2f})",
            "",
            "Foundation Clustering:",
            f"  Within-foundation similarity:  {report.foundation_clustering.within_foundation_similarity:.2f}",
            f"  Between-foundation similarity: {report.foundation_clustering.between_foundation_similarity:.2f}",
            f"  Separation ratio:              {report.foundation_clustering.separation_ratio:.2f}",
            f"  Most distinct:                 {report.foundation_clustering.most_distinct_foundation}",
            "",
            "Virtue-Vice Opposition:",
            f"  Care/Harm (cruelty↔compassion):     {report.virtue_vice_opposition.care_harm_opposition:.2f}",
            f"  Fairness (exploitation↔justice):    {report.virtue_vice_opposition.fairness_opposition:.2f}",
            f"  Loyalty (betrayal↔devotion):        {report.virtue_vice_opposition.loyalty_opposition:.2f}",
            f"  Opposition Detected:                {'YES' if report.virtue_vice_opposition.opposition_detected else 'NO'}",
            "",
            "=" * 70,
            report.verdict,
            "=" * 70,
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("analyze")
def moral_analyze(
    ctx: typer.Context,
    activations_file: str = typer.Argument(..., help="JSON file with anchor activations"),
) -> None:
    """
    Analyze moral geometry from pre-computed activations.

    Input format: JSON with {concept_name: [activation_vector]} mappings.
    """
    context = _context(ctx)

    import numpy as np
    from modelcypher.core.domain.geometry.moral_geometry import MoralGeometryAnalyzer
    from modelcypher.backends.mlx_backend import MLXBackend

    path = Path(activations_file)
    if not path.exists():
        typer.echo(f"File not found: {activations_file}", err=True)
        raise typer.Exit(1)

    with open(path) as f:
        raw_activations = json.load(f)

    activations = {name: np.array(vec) for name, vec in raw_activations.items()}

    backend = MLXBackend()
    analyzer = MoralGeometryAnalyzer(backend=backend)
    report = analyzer.full_analysis(activations)

    payload = {
        "_schema": "mc.geometry.moral.analyze.v1",
        "source_file": activations_file,
        **report.to_dict(),
    }

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
