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

"""Merge models via null-space knowledge transplant.

Usage:
    mc merge -s SOURCE -t TARGET -o OUTPUT

There is exactly ONE correct way to merge high-dimensional Legos:
1. Find geometric correspondence (CKA alignment)
2. Project source knowledge into target's null space
3. Add (not blend) the projected knowledge

This command runs the complete pipeline automatically; geometry decides grafts and alignment.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.commands.model import prevent_sleep
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.validation import validate_model_path
from modelcypher.utils.errors import ErrorDetail

# Support BOTH syntaxes:
#   mc merge -s SOURCE -t TARGET -o OUTPUT  (preferred, direct)
#   mc merge run -s SOURCE -t TARGET -o OUTPUT  (legacy, still works)
app = typer.Typer(invoke_without_command=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _run_dry_run(
    ctx: typer.Context,
    source: str,
    target: str,
    output_dir: str,
) -> None:
    """Show what a merge would do without actually running it."""
    from modelcypher.core.use_cases.model_probe_service import ModelProbeService

    context = _context(ctx)
    service = ModelProbeService()

    # Probe both models
    source_info = service.probe(source)
    target_info = service.probe(target)

    # Calculate estimated output size (roughly same as target since we're adding to it)
    output_path = Path(output_dir)

    payload = {
        "_schema": "mc.merge.dry_run.v1",
        "source": {
            "path": source,
            "architecture": source_info.architecture,
            "parameters": source_info.parameter_count,
            "vocabSize": source_info.vocab_size,
            "hiddenSize": source_info.hidden_size,
            "layers": len(source_info.layers),
            "quantization": source_info.quantization,
        },
        "target": {
            "path": target,
            "architecture": target_info.architecture,
            "parameters": target_info.parameter_count,
            "vocabSize": target_info.vocab_size,
            "hiddenSize": target_info.hidden_size,
            "layers": len(target_info.layers),
            "quantization": target_info.quantization,
        },
        "outputDir": str(output_path),
        "outputExists": output_path.exists(),
        "sameArchitecture": source_info.architecture == target_info.architecture,
        "sameVocab": source_info.vocab_size == target_info.vocab_size,
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "MERGE DRY RUN (no changes will be made)",
            "=" * 70,
            "",
            "SOURCE MODEL (knowledge donor)",
            f"  Path: {source}",
            f"  Architecture: {source_info.architecture}",
            f"  Parameters: {source_info.parameter_count:,}",
            f"  Vocab Size: {source_info.vocab_size:,}",
            f"  Hidden Size: {source_info.hidden_size}",
            f"  Layers: {len(source_info.layers)}",
        ]
        if source_info.quantization:
            lines.append(f"  Quantization: {source_info.quantization}")

        lines.extend([
            "",
            "TARGET MODEL (receives knowledge)",
            f"  Path: {target}",
            f"  Architecture: {target_info.architecture}",
            f"  Parameters: {target_info.parameter_count:,}",
            f"  Vocab Size: {target_info.vocab_size:,}",
            f"  Hidden Size: {target_info.hidden_size}",
            f"  Layers: {len(target_info.layers)}",
        ])
        if target_info.quantization:
            lines.append(f"  Quantization: {target_info.quantization}")

        lines.extend([
            "",
            "OUTPUT",
            f"  Directory: {output_path}",
            f"  Exists: {'Yes' if output_path.exists() else 'No'}",
        ])

        # Compatibility notes
        lines.extend(["", "COMPATIBILITY"])
        if source_info.architecture == target_info.architecture:
            lines.append("  Same architecture: Yes (within-cluster merge)")
        else:
            lines.append("  Same architecture: No (cross-architecture merge)")
        if source_info.vocab_size == target_info.vocab_size:
            lines.append("  Same vocabulary: Yes")
        else:
            lines.append(f"  Same vocabulary: No ({source_info.vocab_size:,} vs {target_info.vocab_size:,})")

        lines.extend(["", "=" * 70])
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


def _run_merge(
    ctx: typer.Context,
    source: str,
    target: str,
    output_dir: str,
    output_file: str | None,
    dry_run: bool = False,
    probe_mode: str = "atlas",
) -> None:
    """Core merge logic shared by callback and run command.

    Scale is always 1.0 for single merges - the null-space projection
    already ensures safe knowledge addition. No user-configurable knobs.
    """
    from modelcypher.cli.composition import get_merge_pipeline_service

    context = _context(ctx)

    # Validate model paths early
    validate_model_path(source, context=context)
    validate_model_path(target, context=context)

    # Dry-run mode: show what would happen without merging
    if dry_run:
        _run_dry_run(ctx, source, target, output_dir)
        return

    service = get_merge_pipeline_service()

    try:
        with prevent_sleep():
            # delta_scale=1.0 always - null-space projection handles safety
            result = service.run(
                source_path=source,
                target_path=target,
                output_dir=output_dir,
                probe_mode=probe_mode,
                delta_scale=1.0,
            )

        # Build output payload
        payload = {
            "_schema": "mc.merge.pipeline.v1",
            "pipelineId": result.pipeline_id,
            "timestamp": result.timestamp,
            "sourceModel": result.source_model,
            "targetModel": result.target_model,
            "outputDir": result.output_dir,
            "preMerge": {
                "domainsAnalyzed": result.pre_merge.domains_analyzed,
                "meanOverlap": result.pre_merge.mean_overlap,
                "meanSubspaceAlignment": result.pre_merge.mean_subspace_alignment,
                "meanCurvatureDivergence": result.pre_merge.mean_curvature_divergence,
                "meanDistance": result.pre_merge.mean_distance,
                "alignedPairs": result.pre_merge.aligned_pairs,
            },
            "mergeResult": {
                "layerCount": result.merge_result.get("layer_count"),
                "weightCount": result.merge_result.get("weight_count"),
                "meanPreservedFraction": result.merge_result.get("mean_preserved_fraction"),
            },
            "postMerge": {
                "layersTransplanted": result.post_merge.layers_transplanted,
                "weightsTransplanted": result.post_merge.weights_transplanted,
                "meanPreservedFraction": result.post_merge.mean_preserved_fraction,
                "meanCkaAfter": result.post_merge.mean_cka_after,
            },
            "timing": {
                "preMergeDurationS": round(result.pre_merge_duration_s, 2),
                "mergeDurationS": round(result.merge_duration_s, 2),
                "validationDurationS": round(result.validation_duration_s, 2),
            },
        }

        # Save full result if requested
        if output_file:
            full_result = {
                "_schema": "mc.merge.pipeline.full.v1",
                "pipelineId": result.pipeline_id,
                "timestamp": result.timestamp,
                "sourceModel": result.source_model,
                "targetModel": result.target_model,
                "outputDir": result.output_dir,
                "preMerge": asdict(result.pre_merge),
                "mergeResult": result.merge_result,
                "postMerge": asdict(result.post_merge),
                "timing": {
                    "preMergeDurationS": result.pre_merge_duration_s,
                    "mergeDurationS": result.merge_duration_s,
                    "validationDurationS": result.validation_duration_s,
                },
            }
            Path(output_file).write_text(json.dumps(full_result, indent=2, default=str))
            typer.echo(f"Full result saved to {output_file}")

        # Text output
        if context.output_format == "text":
            lines = [
                "=" * 70,
                "MERGE PIPELINE RESULT",
                "=" * 70,
                f"Pipeline ID: {result.pipeline_id}",
                f"Source: {result.source_model}",
                f"Target: {result.target_model}",
                f"Output: {result.output_dir}",
                "",
                "PRE-MERGE ANALYSIS",
                f"  Domains: {', '.join(result.pre_merge.domains_analyzed)}",
                f"  Mean Overlap: {result.pre_merge.mean_overlap:.4f}",
                f"  Mean Subspace Alignment: {result.pre_merge.mean_subspace_alignment:.4f}",
                f"  Mean Curvature Divergence: {result.pre_merge.mean_curvature_divergence:.4f}",
                f"  Mean Distance: {result.pre_merge.mean_distance:.4f}",
                f"  Aligned Pairs: {result.pre_merge.aligned_pairs}",
                "",
                "MERGE RESULT",
                f"  Layers: {result.merge_result.get('layer_count')}",
                f"  Weights: {result.merge_result.get('weight_count')}",
                f"  Mean Preserved Fraction: {result.merge_result.get('mean_preserved_fraction', 0):.4f}",
                "",
                "POST-MERGE VALIDATION",
                f"  Mean Preserved Fraction: {result.post_merge.mean_preserved_fraction:.4f}",
                f"  Mean CKA After: {result.post_merge.mean_cka_after:.4f}",
                f"  Layers Transplanted: {result.post_merge.layers_transplanted}",
                f"  Weights Transplanted: {result.post_merge.weights_transplanted}",
            ]

            lines.extend([
                "",
                "TIMING",
                f"  Pre-merge: {result.pre_merge_duration_s:.2f}s",
                f"  Merge: {result.merge_duration_s:.2f}s",
                f"  Validation: {result.validation_duration_s:.2f}s",
                "=" * 70,
            ])

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-1100",
            title="Pipeline failed",
            detail=str(e),
            hint="Check model paths and merge parameters",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict())
        raise typer.Exit(code=1)


@app.callback()
def merge_callback(
    ctx: typer.Context,
    source: str | None = typer.Option(None, "--source", "-s", help="Path to source model (knowledge donor)"),
    target: str | None = typer.Option(None, "--target", "-t", help="Path to target model (receives knowledge)"),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Output directory for merged model"),
    output_file: str | None = typer.Option(None, "--output-file", "-f", help="Save full pipeline result to JSON file"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would happen without actually merging"),
    probe_mode: str = typer.Option("atlas", "--probe-mode", "-p", help="Probe mode: 'atlas' (963 conceptual) or 'token' (49K+ vocab for 100%% dim coverage)"),
) -> None:
    """Merge two models via null-space knowledge transplant.

    Takes knowledge from SOURCE and adds it to TARGET without destroying
    TARGET's existing capabilities. The result is a denser model.

    All geometric parameters (scale, alignment) are auto-derived.
    The math determines how knowledge is added.

    Examples:
        mc merge -s ./qwen -t ./smol -o ./merged
        mc merge -s ./qwen -t ./smol -o ./merged --probe-mode token
    """
    # If a subcommand was invoked (like 'run'), don't do anything here
    if ctx.invoked_subcommand is not None:
        return

    # If options were provided directly, run the merge
    if source and target and output_dir:
        _run_merge(ctx, source, target, output_dir, output_file, dry_run=dry_run, probe_mode=probe_mode)
    elif source or target or output_dir:
        # Partial options provided - show error
        missing = []
        if not source:
            missing.append("--source/-s")
        if not target:
            missing.append("--target/-t")
        if not output_dir:
            missing.append("--output-dir/-o")
        typer.echo(f"Error: Missing required options: {', '.join(missing)}", err=True)
        raise typer.Exit(code=1)
    # else: no options, show help (handled by Typer's no_args_is_help behavior)


@app.command()
def run(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", "-s", help="Path to source model (knowledge donor)"),
    target: str = typer.Option(..., "--target", "-t", help="Path to target model (receives knowledge)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for merged model"),
    output_file: str | None = typer.Option(
        None,
        "--output-file",
        "-f",
        help="Save full pipeline result to JSON file",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would happen without actually merging"),
    probe_mode: str = typer.Option("atlas", "--probe-mode", "-p", help="Probe mode: 'atlas' (963 conceptual) or 'token' (49K+ vocab for 100%% dim coverage)"),
) -> None:
    """Merge two models via null-space knowledge transplant.

    Takes knowledge from SOURCE and adds it to TARGET without destroying
    TARGET's existing capabilities. The result is a denser model.

    All geometric parameters (scale, alignment) are auto-derived.
    The math determines how knowledge is added.

    Examples:
        mc merge run -s ./qwen -t ./smol -o ./merged
        mc merge run -s ./qwen -t ./smol -o ./merged --probe-mode token
    """
    _run_merge(ctx, source, target, output_dir, output_file, dry_run=dry_run, probe_mode=probe_mode)


@app.command()
def batch(
    ctx: typer.Context,
    sources: list[str] = typer.Option(..., "--source", "-s", help="Source model paths (repeat for multiple: -s A -s B -s C)"),
    target: str = typer.Option(..., "--target", "-t", help="Target model (receives all knowledge)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for merged model"),
    accumulative: bool = typer.Option(True, "--accumulative/--sequential", help="Accumulative (add all to target) vs sequential merging"),
    fast_mode: bool = typer.Option(True, "--fast/--precise", help="Fast mode skips CKA precision checks (safe: CKA=1.0 is invariant)"),
) -> None:
    """Merge multiple source models into a single target (N→1 merging).

    This is optimized for dumping knowledge from many models into one compact
    target (e.g., LFM2). The target is loaded and probed ONCE, then reused
    for all source merges.

    CKA = 1.0 is an invariant (not a target). All models encode the same
    geometric shape. The alignment transform achieves CKA = 1.0 by construction.

    Accumulative mode (default) projects all sources into the ORIGINAL target's
    null-space. This preserves target behavior while adding all source knowledge.

    Scale is automatically computed for each merge based on measured delta
    magnitude and remaining budget (1% of weight norm). The math determines
    the safe injection amount - no user-configurable knobs.

    Examples:
        mc merge batch -s ./model1 -s ./model2 -s ./model3 -t ./lfm2 -o ./merged
        mc merge batch -s ./qwen -s ./llama -s ./mistral -t ./smol -o ./super_merged
    """
    from modelcypher.adapters.hf_hub import HuggingFaceModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    for source in sources:
        if not validate_model_path(source):
            typer.echo(f"Error: Source path not found: {source}", err=True)
            raise typer.Exit(code=1)
    if not validate_model_path(target):
        typer.echo(f"Error: Target path not found: {target}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"BATCH MERGE: {len(sources)} sources → {target}")
    typer.echo(f"  Mode: {'accumulative' if accumulative else 'sequential'}")
    typer.echo(f"  Fast mode: {fast_mode} (CKA=1.0 is invariant)")
    typer.echo("  Scale: auto-derived from deviation budget")

    with prevent_sleep():
        model_loader = HuggingFaceModelLoader()
        merger = UnifiedGeometricMerger(model_loader=model_loader, backend=backend)

        # Always use auto_scale - math determines safe injection amount
        result = merger.merge_batch(
            source_paths=sources,
            target_path=target,
            output_dir=output_dir,
            accumulative=accumulative,
            fast_mode=fast_mode,
            auto_scale=True,  # Always enabled - no user knob
        )

    typer.echo(f"BATCH MERGE: Complete. Output saved to {output_dir}")
    typer.echo(f"  Total layers: {result.total_layers}")
    typer.echo(f"  Sources merged: {len(sources)}")


@app.command()
def budget(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to original baseline model"),
    current: str = typer.Option(..., "--current", "-c", help="Path to current (merged) model"),
) -> None:
    """Check deviation budget status before merging.

    Compares current model weights against baseline to show how much
    deviation budget remains (threshold = 1% of baseline weight norm).

    Examples:
        mc merge budget --baseline ./original --current ./merged2
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.deviation_budget import DeviationBudget

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    if not validate_model_path(baseline):
        typer.echo(f"Error: Baseline path not found: {baseline}", err=True)
        raise typer.Exit(code=1)
    if not validate_model_path(current):
        typer.echo(f"Error: Current path not found: {current}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"BUDGET CHECK: {current} vs baseline {baseline}")

    # Load weights
    model_loader = MLXModelLoader()

    typer.echo("  Loading baseline weights...")
    baseline_weights, _ = model_loader.load_weights(baseline)

    typer.echo("  Loading current weights...")
    current_weights, _ = model_loader.load_weights(current)

    # Check budget
    budget_tracker = DeviationBudget(backend=backend)
    budget_tracker.record_baseline(baseline_weights)
    status = budget_tracker.check_merge_budget(current_weights)

    # Output results
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("DEVIATION BUDGET STATUS")
    typer.echo("=" * 60)
    typer.echo(f"  Current deviation: {status.current_deviation:.1f} L2")
    typer.echo(f"  Threshold: {status.threshold:.1f} L2")
    typer.echo(f"  Budget used: {status.budget_used_percent:.1f}%")
    typer.echo(f"  Remaining budget: {status.threshold - status.current_deviation:.1f} L2")
    typer.echo("")

    if status.is_safe:
        if status.budget_used_percent > 70:
            typer.echo("  Status: WARNING - Approaching threshold")
        else:
            typer.echo("  Status: SAFE")
    else:
        typer.echo("  Status: DANGER - Budget exceeded!")

    typer.echo(f"  Recommendation: {status.recommendation}")
    typer.echo("")

    # Report remaining budget (actual scale depends on next source model)
    remaining = status.threshold - status.current_deviation
    if remaining > 0:
        typer.echo(f"  Remaining budget: {remaining:.1f} L2")
        typer.echo("  Note: Use --auto-scale with 'mc merge batch' to automatically")
        typer.echo("        compute delta_scale based on actual source model delta.")
    else:
        typer.echo("  Cannot recommend delta_scale - budget already exceeded")

    typer.echo("=" * 60)


@app.command("multi-channel")
def multi_channel(
    ctx: typer.Context,
    channels: list[str] = typer.Option(
        ...,
        "--channel",
        "-c",
        help="Channel in format 'name:path' (repeat for multiple: -c spatial:/path/to/world -c text:/path/to/llm)",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Target model (receives all knowledge)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for merged model"),
    routing_mode: str = typer.Option(
        "uniform",
        "--routing",
        "-r",
        help="Routing mode: 'uniform' (equal weight), 'identity' (no mixing), 'diagonal_weighted' (norm-based)",
    ),
    fast_mode: bool = typer.Option(True, "--fast/--precise", help="Fast mode skips CKA precision checks"),
) -> None:
    """Merge multiple channels simultaneously via Birkhoff routing.

    This is the preferred method for multi-modal merging (e.g., world model +
    vision-language model + text model → unified model).

    Unlike 'batch' (sequential), this method:
    1. Probes all channels simultaneously
    2. Projects all channels into target's null-space (shared basis)
    3. Combines channels via doubly stochastic routing (spectral norm <= 1.0)
    4. Applies geometric addition (not interpolation)

    Mathematical Foundation (from mHC-null-space connection):
        W' = W_target + sum_j H[i,j] * P_null(A_target) @ delta_W_j

    Properties:
    - CKA = 1.0 per channel (geometry preserved)
    - Spectral norm <= 1.0 (stable combination)
    - No interference (channels add, not blend)

    Examples:
        mc merge multi-channel -c spatial:/path/to/world -c text:/path/to/llm -t ./lfm2 -o ./merged
        mc merge multi-channel -c spatial:./world -c temporal:./video -c text:./llm -t ./target -o ./out
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger

    context = _context(ctx)
    backend = get_default_backend()

    # Parse channel arguments (format: "name:path")
    channel_paths: dict[str, str] = {}
    for channel_spec in channels:
        if ":" not in channel_spec:
            typer.echo(f"Error: Invalid channel format '{channel_spec}'. Use 'name:path' format.", err=True)
            raise typer.Exit(code=1)
        name, path = channel_spec.split(":", 1)
        if not name or not path:
            typer.echo(f"Error: Invalid channel format '{channel_spec}'. Both name and path required.", err=True)
            raise typer.Exit(code=1)
        channel_paths[name] = path

    # Validate paths
    for name, path in channel_paths.items():
        if not validate_model_path(path):
            typer.echo(f"Error: Channel '{name}' path not found: {path}", err=True)
            raise typer.Exit(code=1)
    if not validate_model_path(target):
        typer.echo(f"Error: Target path not found: {target}", err=True)
        raise typer.Exit(code=1)

    # Validate routing mode
    valid_modes = ["uniform", "identity", "diagonal_weighted"]
    if routing_mode not in valid_modes:
        typer.echo(f"Error: Invalid routing mode '{routing_mode}'. Use one of: {', '.join(valid_modes)}", err=True)
        raise typer.Exit(code=1)

    channel_names = list(channel_paths.keys())
    typer.echo(f"MULTI-CHANNEL MERGE: {len(channel_names)} channels → {target}")
    typer.echo(f"  Channels: {', '.join(channel_names)}")
    typer.echo(f"  Routing mode: {routing_mode}")
    typer.echo(f"  Fast mode: {fast_mode}")

    with prevent_sleep():
        model_loader = MLXModelLoader()
        merger = UnifiedGeometricMerger(model_loader=model_loader, backend=backend)

        result = merger.merge_multi_channel(
            channel_paths=channel_paths,
            target_path=target,
            output_dir=output_dir,
            routing_mode=routing_mode,
            fast_mode=fast_mode,
        )

    typer.echo(f"MULTI-CHANNEL MERGE: Complete. Output saved to {output_dir}")
    typer.echo(f"  Layers merged: {result.transplant_metrics.get('layers_merged', 0)}")
    typer.echo(f"  Channels: {len(channel_names)}")
    typer.echo(f"  Mean preserved fraction: {result.mean_preserved_fraction:.4f}")
