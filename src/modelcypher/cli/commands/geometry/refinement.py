"""Geometry refinement density CLI commands.

Provides commands for analyzing refinement density between base and adapted
models using DARE sparsity, DoRA directional drift, and transition CKA.

Commands:
    mc geometry refinement analyze <base_model> <adapted_model>
    mc geometry refinement summary <result_file>
"""

from __future__ import annotations

import json
from pathlib import Path


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.geometry.refinement_density import (
    RefinementDensityAnalyzer,
    RefinementDensityConfig,
)
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("analyze")
def geometry_refinement_analyze(
    ctx: typer.Context,
    base_model: str = typer.Argument(..., help="Path to base (target) model"),
    adapted_model: str = typer.Argument(..., help="Path to adapted (source/refined) model"),
    source_crm: str | None = typer.Option(None, "--source-crm", help="Path to source CRM file"),
    target_crm: str | None = typer.Option(None, "--target-crm", help="Path to target CRM file"),
    sparsity_weight: float = typer.Option(0.35, "--sparsity-weight", help="Weight for DARE sparsity contribution"),
    directional_weight: float = typer.Option(0.35, "--directional-weight", help="Weight for DoRA directional drift"),
    transition_weight: float = typer.Option(0.30, "--transition-weight", help="Weight for transition CKA"),
    hard_swap_threshold: float = typer.Option(0.80, "--hard-swap-threshold", help="Score threshold for hard swap"),
    mode: str = typer.Option("default", "--mode", help="Analysis mode: default, aggressive, conservative"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Write JSON result to file"),
) -> None:
    """Analyze refinement density between base and adapted models.

    Combines DARE sparsity, DoRA directional drift, and transition CKA
    to produce per-layer refinement scores and merge recommendations.

    Example:
        mc geometry refinement analyze ./base-model ./finetuned-model
        mc geometry refinement analyze ./base ./adapted --mode aggressive
    """
    context = _context(ctx)

    try:
        from modelcypher.core.domain.geometry.dare_sparsity import (
            DARESparsityAnalyzer,
            Configuration as DAREConfig,
        )
        from modelcypher.core.domain.geometry.dora_decomposition import (
            DoRADecomposition,
        )
        from modelcypher.core.domain.geometry.concept_response_matrix import (
            ConceptResponseMatrix,
        )
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        # Load models
        _, base_weights = mlx_load(base_model, lazy=True)
        _, adapted_weights = mlx_load(adapted_model, lazy=True)

        base_weights = dict(base_weights)
        adapted_weights = dict(adapted_weights)

        # Compute delta weights for DARE
        delta_weights = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base = base_weights[name]
            adapted = adapted_weights[name]
            if base.shape != adapted.shape:
                continue
            delta = adapted - base
            mx.eval(delta)
            delta_weights[name] = delta.flatten().tolist()

        sparsity_analysis = DARESparsityAnalyzer.analyze(
            delta_weights, DAREConfig(compute_per_layer_metrics=True)
        )

        # Compute DoRA decomposition
        base_mx = {}
        adapted_mx = {}
        for name in base_weights:
            if name not in adapted_weights:
                continue
            base_mx[name] = base_weights[name]
            adapted_mx[name] = adapted_weights[name]

        dora = DoRADecomposition()
        dora_result = dora.analyze_adapter(base_mx, adapted_mx)

        # Load CRMs if provided
        transition_experiment = None
        if source_crm and target_crm:
            source = ConceptResponseMatrix.load(source_crm)
            target = ConceptResponseMatrix.load(target_crm)
            transition_experiment = source.compute_transition_alignment(target)

        # Configure analyzer
        if mode == "aggressive":
            config = RefinementDensityConfig.aggressive()
        elif mode == "conservative":
            config = RefinementDensityConfig.conservative()
        else:
            config = RefinementDensityConfig(
                sparsity_weight=sparsity_weight,
                directional_weight=directional_weight,
                transition_weight=transition_weight,
                hard_swap_threshold=hard_swap_threshold,
            )

        analyzer = RefinementDensityAnalyzer(config)
        result = analyzer.analyze(
            source_model=adapted_model,
            target_model=base_model,
            sparsity_analysis=sparsity_analysis,
            dora_result=dora_result,
            transition_experiment=transition_experiment,
        )

        # Write to file if requested
        if output_file:
            Path(output_file).write_text(dump_json(result.to_dict()))

        payload = result.to_dict()
        payload["outputFile"] = output_file
        payload["nextActions"] = [
            "mc model merge with --use-refinement-density to apply recommendations",
            "mc geometry adapter sparsity for detailed DARE analysis",
            "mc geometry adapter decomposition for detailed DoRA analysis",
        ]

        if context.output_format == "text":
            lines = [
                "REFINEMENT DENSITY ANALYSIS",
                f"Source (refined): {adapted_model}",
                f"Target (base): {base_model}",
                "",
                f"Mean Composite Score: {result.mean_composite_score:.3f}",
                f"Max Composite Score: {result.max_composite_score:.3f}",
                "",
                f"Layers Above Hard Swap Threshold: {result.layers_above_hard_swap}",
                f"Layers Above High Alpha Threshold: {result.layers_above_high_alpha}",
                f"Layers Above Medium Alpha Threshold: {result.layers_above_medium_alpha}",
                "",
            ]

            if result.hard_swap_layers:
                lines.append(f"Recommended Hard Swap Layers: {result.hard_swap_layers}")

            components = []
            if result.has_sparsity_data:
                components.append("DARE")
            if result.has_directional_data:
                components.append("DoRA")
            if result.has_transition_data:
                components.append("Transition-CKA")
            lines.append(f"Data Sources: {', '.join(components) or 'None'}")

            if output_file:
                lines.append(f"\nOutput: {output_file}")

            # Show top layers by refinement
            sorted_scores = sorted(
                result.layer_scores.items(),
                key=lambda x: x[1].composite_score,
                reverse=True,
            )
            if sorted_scores:
                lines.append("\nTop Refined Layers:")
                for idx, score in sorted_scores[:5]:
                    lines.append(
                        f"  Layer {idx}: {score.composite_score:.3f} "
                        f"({score.refinement_level.value}) -> {score.merge_recommendation.value}"
                    )

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("summary")
def geometry_refinement_summary(
    ctx: typer.Context,
    result_file: str = typer.Argument(..., help="Path to refinement analysis JSON"),
) -> None:
    """Display summary of a saved refinement density analysis.

    Example:
        mc geometry refinement summary ./analysis.json
    """
    context = _context(ctx)

    try:
        data = json.loads(Path(result_file).read_text())

        payload = {
            "sourceModel": data.get("sourceModel"),
            "targetModel": data.get("targetModel"),
            "meanCompositeScore": data.get("meanCompositeScore"),
            "maxCompositeScore": data.get("maxCompositeScore"),
            "layersAboveHardSwap": data.get("layersAboveHardSwap"),
            "hardSwapLayers": data.get("hardSwapLayers"),
            "alphaByLayer": data.get("alphaByLayer"),
        }

        if context.output_format == "text":
            lines = [
                "REFINEMENT DENSITY SUMMARY",
                f"Source: {data.get('sourceModel')}",
                f"Target: {data.get('targetModel')}",
                f"Mean Score: {data.get('meanCompositeScore', 0):.3f}",
                f"Max Score: {data.get('maxCompositeScore', 0):.3f}",
                f"Hard Swap Candidates: {data.get('layersAboveHardSwap', 0)}",
            ]

            hard_swap = data.get("hardSwapLayers", [])
            if hard_swap:
                lines.append(f"Hard Swap Layers: {hard_swap}")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc
