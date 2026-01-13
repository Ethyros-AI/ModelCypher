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
    from modelcypher.cli.composition import get_model_probe_service

    context = _context(ctx)
    service = get_model_probe_service()

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
) -> None:
    """Core merge logic shared by callback and run command.

    Scale is always 1.0 for single merges - the null-space projection
    already ensures safe knowledge addition. No user-configurable knobs.
    Always uses atlas probes with geometry-derived count.
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
            # probe_mode="atlas" always - geometry-derived probe count
            result = service.run(
                source_path=source,
                target_path=target,
                output_dir=output_dir,
                probe_mode="atlas",
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
) -> None:
    """Merge two models via null-space knowledge transplant.

    Takes knowledge from SOURCE and adds it to TARGET without destroying
    TARGET's existing capabilities. The result is a denser model.

    Uses semantic concept probes from the atlas system to align manifolds.
    All geometric parameters (scale, alignment) are auto-derived.

    Examples:
        mc merge -s ./qwen -t ./smol -o ./merged
    """
    # If a subcommand was invoked (like 'run'), don't do anything here
    if ctx.invoked_subcommand is not None:
        return

    # If options were provided directly, run the merge
    if source and target and output_dir:
        _run_merge(ctx, source, target, output_dir, output_file, dry_run=dry_run)
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
) -> None:
    """Merge two models via null-space knowledge transplant.

    Takes knowledge from SOURCE and adds it to TARGET without destroying
    TARGET's existing capabilities. The result is a denser model.

    Uses semantic concept probes from the atlas system to align manifolds.
    All geometric parameters (scale, alignment) are auto-derived.

    Examples:
        mc merge run -s ./qwen -t ./smol -o ./merged
    """
    _run_merge(ctx, source, target, output_dir, output_file, dry_run=dry_run)


@app.command()
def batch(
    ctx: typer.Context,
    sources: list[str] = typer.Option(..., "--source", "-s", help="Source model paths (repeat for multiple: -s A -s B -s C)"),
    target: str = typer.Option(..., "--target", "-t", help="Target model (receives all knowledge)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for merged model"),
    accumulative: bool = typer.Option(True, "--accumulative/--sequential", help="Accumulative (add all to target) vs sequential merging"),
    fast_mode: bool = typer.Option(True, "--fast/--precise", help="Fast mode skips geodesic CKA diagnostics (alignment is closed-form)"),
    detect_outliers: bool = typer.Option(False, "--detect-outliers", help="Analyze concept alignment before merging (shows which models disagree)"),
    consensus_mode: bool = typer.Option(False, "--consensus/--no-consensus", help="Use consensus-based correction: fix misaligned concepts before adding"),
) -> None:
    """Merge multiple source models into a single target (N→1 merging).

    This is optimized for dumping knowledge from many models into one compact
    target (e.g., LFM2). The target is loaded and probed ONCE, then reused
    for all source merges.

    Linear alignment is closed-form. Geodesic CKA reports manifold overlap
    and can be < 1.0 when probes miss shared structure.

    Accumulative mode (default) projects all sources into the ORIGINAL target's
    null-space. This preserves target behavior while adding all source knowledge.

    Consensus mode (--consensus) enables two-phase merging:
    1. CORRECTION: Fix concepts where target disagrees with source consensus
    2. ADDITION: Add source-only knowledge via null-space projection

    Scale is automatically computed for each merge based on measured delta
    magnitude and remaining budget (1% of weight norm). The math determines
    the safe injection amount - no user-configurable knobs.

    Examples:
        mc merge batch -s ./model1 -s ./model2 -s ./model3 -t ./lfm2 -o ./merged
        mc merge batch -s ./qwen -s ./llama -s ./mistral -t ./smol -o ./super_merged --consensus
        mc merge batch -s ./m1 -s ./m2 -t ./target -o ./out --detect-outliers
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
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
    typer.echo(f"  Fast mode: {fast_mode} (geodesic diagnostics skipped)")
    if consensus_mode:
        typer.echo("  Consensus: ENABLED (correction + addition)")
    typer.echo("  Scale: auto-derived from deviation budget")

    # Outlier detection (optional analysis before merge)
    if detect_outliers:
        from modelcypher.core.domain.geometry.outlier_detector import OutlierDetector, OutlierResult

        typer.echo("")
        typer.echo("=" * 60)
        typer.echo("OUTLIER DETECTION ANALYSIS")
        typer.echo("=" * 60)
        typer.echo("  Analyzing alignment across all models...")
        typer.echo("  (Models that disagree with consensus may have learned concepts wrong)")
        typer.echo("")

        # Outlier detection must happen in Gram space (dimension-invariant)
        # We need to probe models first and compare their relational structure
        typer.echo("  Note: Full outlier detection requires probing (runs during merge).")
        typer.echo("  Pre-merge check uses Gram matrix comparison on a small probe set.")
        typer.echo("")

        # Load models and run quick probe to build Gram matrices
        from modelcypher.core.domain.geometry.cka import (
            compute_cka_from_grams,
            geodesic_squared_distances,
            _shared_rbf_sigma,
            _rbf_gram_from_sq_distances,
        )
        from modelcypher.adapters.mlx_model_loader import MLXModelLoader
        from modelcypher.core.domain.agents.probe_loader import ProbeLoader

        model_loader = MLXModelLoader()
        all_paths = [target] + list(sources)

        # Load probes for quick Gram comparison
        # Use sqrt(available) probes - balances coverage vs speed
        probe_loader = ProbeLoader()
        available_probes = probe_loader.load_probes()
        n_probes = max(len(all_paths) + 2, int(len(available_probes) ** 0.5))
        n_probes = min(n_probes, 128)  # Cap for pre-merge speed
        probes = available_probes[:n_probes]

        # Collect Gram matrices for each model
        gram_matrices = []
        for path in all_paths:
            model, tokenizer = model_loader.load_model(path)

            # Get activations for probes (middle layer)
            from modelcypher.core.use_cases.merge.helpers import get_hidden_state
            activations = []
            for probe in probes:
                try:
                    act = get_hidden_state(model, tokenizer, probe.text, backend)
                    if act is not None:
                        activations.append(act)
                except Exception:
                    continue

            # Need at least n_models + 1 valid probes for meaningful Gram comparison
            min_valid = len(all_paths) + 1
            if len(activations) < min_valid:
                typer.echo(f"  Warning: Too few valid probes ({len(activations)} < {min_valid}) for {path}")
                gram_matrices.append(None)
                continue

            # Build geodesic RBF Gram matrix (n_probes × n_probes, dimension-invariant)
            # Uses k-NN graph shortest paths for proper manifold distances
            X = backend.stack(activations, axis=0)
            backend.eval(X)
            sq_dist = geodesic_squared_distances(X, backend)
            # Use median heuristic for sigma (data-derived, no hardcoded values)
            sigma = _shared_rbf_sigma(sq_dist, sq_dist, backend)
            K = _rbf_gram_from_sq_distances(sq_dist, sigma, backend)
            backend.eval(K)
            gram_matrices.append(K)

            # Clean up model
            del model
            if hasattr(backend, "clear_cache"):
                backend.clear_cache()

        # Compare Gram matrices using geodesic CKA (overlap diagnostic)
        detector = OutlierDetector(backend)
        n_models = len(gram_matrices)
        valid_grams = [(i, K) for i, K in enumerate(gram_matrices) if K is not None]

        if len(valid_grams) < 2:
            typer.echo("  Error: Need at least 2 valid Gram matrices for comparison.")
            result_detect = None
        else:
            # Compute pairwise CKA distances (1 - CKA = distance)
            pairwise_cka = {}
            for i, (idx_i, K_i) in enumerate(valid_grams):
                for j, (idx_j, K_j) in enumerate(valid_grams):
                    if i < j:
                        cka_val = compute_cka_from_grams(K_i, K_j, backend)
                        pairwise_cka[(idx_i, idx_j)] = cka_val

            # Compute mean CKA for each model (higher = more aligned with others)
            mean_cka = []
            for idx, _ in valid_grams:
                cka_sum = 0.0
                count = 0
                for (i, j), cka in pairwise_cka.items():
                    if i == idx or j == idx:
                        cka_sum += cka
                        count += 1
                mean_cka.append((idx, cka_sum / count if count > 0 else 0.0))

            # Outliers have LOW mean CKA (far from others in representation space)
            # Convert to "errors" (1 - mean_cka) for the detector
            cka_errors = [1.0 - cka for _, cka in mean_cka]
            result_detect = detector.detect_from_gpa(cka_errors)

            # Map back to original indices
            valid_indices = [idx for idx, _ in valid_grams]
            result_detect = OutlierResult(
                consensus_indices=tuple(valid_indices[i] for i in result_detect.consensus_indices),
                outlier_indices=tuple(valid_indices[i] for i in result_detect.outlier_indices),
                errors=tuple(cka_errors),
                threshold=result_detect.threshold,
                mean_error=result_detect.mean_error,
                std_error=result_detect.std_error,
            )

        typer.echo(f"  Models analyzed: {len(all_paths)}")

        if result_detect is None:
            typer.echo("  Could not perform outlier detection (insufficient valid probes).")
        else:
            typer.echo(f"  Consensus models: {len(result_detect.consensus_indices)}")
            typer.echo(f"  Outlier models: {len(result_detect.outlier_indices)}")
            typer.echo(f"  Detection threshold (1-CKA): {result_detect.threshold:.4f}")
            typer.echo("")

            # Show pairwise CKA values for transparency
            typer.echo("  Pairwise CKA (representation similarity):")
            for (i, j), cka in pairwise_cka.items():
                path_i = all_paths[i].split("/")[-1]
                path_j = all_paths[j].split("/")[-1]
                typer.echo(f"    {path_i} <-> {path_j}: CKA={cka:.4f}")
            typer.echo("")

            if result_detect.outlier_indices:
                typer.echo("  OUTLIERS DETECTED (low CKA with others):")
                for idx in result_detect.outlier_indices:
                    model_path = all_paths[idx]
                    # Find mean CKA for this model
                    model_mean_cka = mean_cka[valid_indices.index(idx)][1] if idx in valid_indices else 0.0
                    role = "TARGET" if idx == 0 else f"SOURCE-{idx}"
                    typer.echo(f"    [{role}] {model_path.split('/')[-1]} (mean CKA: {model_mean_cka:.4f})")
                typer.echo("")
                if 0 in result_detect.outlier_indices:
                    typer.echo("  WARNING: Target model is an outlier!")
                    typer.echo("           Consider using --consensus to correct misaligned concepts.")
            else:
                typer.echo("  All models share similar representation structure (in consensus).")

        typer.echo("=" * 60)
        typer.echo("")

    with prevent_sleep():
        from modelcypher.cli.composition import _get_registry
        registry = _get_registry()
        model_loader = MLXModelLoader()
        merger = UnifiedGeometricMerger(
            model_loader=model_loader,
            backend=backend,
            activation_provider=registry.activation_provider,
            activation_store=registry.activation_store,
        )

        # Always use auto_scale - math determines safe injection amount
        result = merger.merge_batch(
            source_paths=sources,
            target_path=target,
            output_dir=output_dir,
            accumulative=accumulative,
            fast_mode=fast_mode,
            auto_scale=True,  # Always enabled - no user knob
            consensus_mode=consensus_mode,
        )

    typer.echo(f"BATCH MERGE: Complete. Output saved to {output_dir}")
    typer.echo(f"  Total layers: {result.layer_count}")
    typer.echo(f"  Sources merged: {len(sources)}")
    if consensus_mode:
        typer.echo("  Mode: Consensus (correction applied before addition)")


@app.command()
def deviation(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to original baseline model"),
    current: str = typer.Option(..., "--current", "-c", help="Path to current (merged) model"),
) -> None:
    """Measure deviation from baseline (informational only).

    The geometry handles safety by construction via null-space projection.
    This command measures and reports deviation for transparency - it does
    NOT gate operations or recommend actions.

    Examples:
        mc merge deviation --baseline ./original --current ./merged2
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.deviation_budget import DeviationTracker

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    if not validate_model_path(baseline):
        typer.echo(f"Error: Baseline path not found: {baseline}", err=True)
        raise typer.Exit(code=1)
    if not validate_model_path(current):
        typer.echo(f"Error: Current path not found: {current}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"DEVIATION MEASUREMENT: {current} vs baseline {baseline}")

    # Load weights
    model_loader = MLXModelLoader()

    typer.echo("  Loading baseline weights...")
    baseline_weights, _ = model_loader.load_weights(baseline)

    typer.echo("  Loading current weights...")
    current_weights, _ = model_loader.load_weights(current)

    # Measure deviation
    tracker = DeviationTracker(backend=backend)
    tracker.record_baseline(baseline_weights)
    measurement = tracker.measure(current_weights)

    # Output results
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("DEVIATION MEASUREMENT (informational only)")
    typer.echo("=" * 60)
    typer.echo(f"  Deviation from baseline: {measurement.deviation:.1f} L2")
    typer.echo(f"  Baseline weight norm: {measurement.baseline_norm:.1f} L2")
    typer.echo(f"  Deviation percent: {measurement.deviation_percent:.2f}%")
    typer.echo(f"  Condition number: {measurement.condition_number:.1f}")
    typer.echo("")
    typer.echo("  Note: The geometry handles safety by construction via null-space")
    typer.echo("        projection. This measurement is for transparency only.")
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
    - Geodesic CKA per channel is reported as overlap diagnostics
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


@app.command("bridge")
def bridge(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Path to source encoder (or model with embedding layer)"),
    target: str = typer.Argument(..., help="Path to target encoder (or model with embedding layer)"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for bridge file (safetensors format)"),
    n_samples: int = typer.Option(
        100,
        "--samples",
        "-n",
        help="Number of probe samples for bridge generation",
    ),
    probe_sources: str | None = typer.Option(
        None,
        "--probe-sources",
        help="Comma-separated atlas sources (e.g., 'semantic_prime,computational_gate'). Default: all sources",
    ),
    source_name: str | None = typer.Option(None, "--source-name", help="Optional name for source encoder"),
    target_name: str | None = typer.Option(None, "--target-name", help="Optional name for target encoder"),
) -> None:
    """Generate a cross-modal bridge between two encoders.

    Creates a linear transform that maps embeddings from SOURCE space to TARGET.
    Geodesic CKA reports manifold overlap for the aligned embeddings.

    Uses semantic concept probes from the atlas system (4596 probes across 23
    categories). These structured concepts span the semantic manifold systematically,
    improving probe coverage for geodesic alignment diagnostics.

    The bridge is saved in safetensors format and includes:
    - Forward transform (source → target)
    - Inverse transform (target → source)
    - Scale ratio for magnitude normalization
    - Metadata (dimensions, names, CKA achieved)

    Mathematical Foundation:
        F = pinv(source) @ target  (closed-form linear alignment)
        Geodesic CKA reports overlap on the aligned probes.

    Examples:
        mc merge bridge /path/to/clip /path/to/lfm2 -o clip_to_lfm2.safetensors
        mc merge bridge /path/to/whisper /path/to/lfm2 -o audio_to_lfm2.safetensors --samples 200
        mc merge bridge ./model_a ./model_b -o bridge.safetensors --probe-sources semantic_prime,emotion_concept
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.bridge.generator import BridgeGenerator

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    if not validate_model_path(source):
        typer.echo(f"Error: Source path not found: {source}", err=True)
        raise typer.Exit(code=1)
    if not validate_model_path(target):
        typer.echo(f"Error: Target path not found: {target}", err=True)
        raise typer.Exit(code=1)

    output_path = Path(output)
    if output_path.exists():
        typer.echo(f"Warning: Output file exists and will be overwritten: {output}")

    # Infer names if not provided
    if source_name is None:
        source_name = Path(source).name
    if target_name is None:
        target_name = Path(target).name

    typer.echo(f"BRIDGE GENERATION: {source_name} → {target_name}")
    typer.echo(f"  Source: {source}")
    typer.echo(f"  Target: {target}")
    typer.echo(f"  Samples: {n_samples}")

    with prevent_sleep():
        model_loader = MLXModelLoader()
        generator = BridgeGenerator(backend)

        # Load embeddings from models
        typer.echo("  Loading source embeddings...")
        source_weights = model_loader.load_weights(source)
        source_embed_key = _find_embedding_key(source_weights)
        if source_embed_key is None:
            typer.echo("Error: Could not find embedding layer in source model", err=True)
            raise typer.Exit(code=1)
        source_embed = source_weights[source_embed_key]
        source_embed = backend.array(source_embed)

        typer.echo("  Loading target embeddings...")
        target_weights = model_loader.load_weights(target)
        target_embed_key = _find_embedding_key(target_weights)
        if target_embed_key is None:
            typer.echo("Error: Could not find embedding layer in target model", err=True)
            raise typer.Exit(code=1)
        target_embed = target_weights[target_embed_key]
        target_embed = backend.array(target_embed)
        backend.eval(source_embed, target_embed)

        typer.echo(f"  Source dim: {source_embed.shape[-1]}, Target dim: {target_embed.shape[-1]}")

        # Sample activations using atlas semantic probes
        # Atlas probes span the manifold systematically, improving overlap diagnostics
        typer.echo("  Loading atlas probes (semantic concepts)...")
        source_activations, target_activations = _sample_atlas_probes(
            backend,
            source_embed,
            target_embed,
            source,
            target,
            n_samples,
            probe_sources,
        )
        backend.eval(source_activations, target_activations)
        typer.echo(f"  Probe samples: {source_activations.shape[0]}")

        # Generate bridge
        typer.echo("  Computing bridge transform (GramAlign)...")
        result = generator.generate(
            source_activations,
            target_activations,
            source_name=source_name,
            target_name=target_name,
        )

        # Save bridge
        typer.echo(f"  Saving bridge to {output}...")
        from modelcypher.cli.composition import get_bridge_service

        bridge_service = get_bridge_service()
        bridge_service.save(result, output_path)

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("BRIDGE GENERATION COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Output: {output}")
    typer.echo(f"  Source dim: {result.source_dim}")
    typer.echo(f"  Target dim: {result.target_dim}")
    typer.echo(f"  CKA achieved: {result.cka_achieved:.6f}")
    typer.echo(f"  Raw CKA: {result.raw_cka:.6f}")
    typer.echo(f"  Scale ratio: {result.scale_ratio:.4f}")
    typer.echo("=" * 60)


@app.command("apply-bridge")
def apply_bridge(
    ctx: typer.Context,
    bridge_path: str = typer.Argument(..., help="Path to bridge file (safetensors)"),
    input_path: str = typer.Argument(..., help="Path to input embeddings (npy or safetensors)"),
    output: str = typer.Option(..., "--output", "-o", help="Output path for transformed embeddings"),
    inverse: bool = typer.Option(False, "--inverse", "-i", help="Apply inverse transform (target → source)"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Apply scale normalization"),
) -> None:
    """Apply a bridge transform to embeddings.

    Transforms embeddings from source space to target space (or vice versa with
    --inverse) using a previously generated bridge.

    Supports input/output in:
    - NumPy (.npy) format
    - Safetensors (.safetensors) format

    Examples:
        mc merge apply-bridge clip_to_lfm2.safetensors image_embeds.npy -o lfm2_embeds.npy
        mc merge apply-bridge bridge.safetensors source.safetensors -o target.safetensors
        mc merge apply-bridge bridge.safetensors target_embeds.npy -o source_embeds.npy --inverse
    """
    import numpy as np
    from safetensors.numpy import load_file as load_safetensors
    from safetensors.numpy import save_file as save_safetensors

    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.cli.composition import get_bridge_service

    context = _context(ctx)
    backend = get_default_backend()

    # Validate bridge path
    bridge_file = Path(bridge_path)
    if not bridge_file.exists():
        typer.echo(f"Error: Bridge file not found: {bridge_path}", err=True)
        raise typer.Exit(code=1)

    # Validate input path
    input_file = Path(input_path)
    if not input_file.exists():
        typer.echo(f"Error: Input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    output_path = Path(output)

    typer.echo(f"APPLY BRIDGE: {'inverse' if inverse else 'forward'} transform")
    typer.echo(f"  Bridge: {bridge_path}")
    typer.echo(f"  Input: {input_path}")
    typer.echo(f"  Output: {output}")

    # Load bridge
    typer.echo("  Loading bridge...")
    bridge_service = get_bridge_service()
    bridge = bridge_service.load(bridge_file)
    typer.echo(f"  Bridge: {bridge.source_name} ({bridge.source_dim}D) → {bridge.target_name} ({bridge.target_dim}D)")

    # Load input embeddings
    typer.echo("  Loading input embeddings...")
    if input_file.suffix == ".npy":
        embeddings_np = np.load(input_file)
        embeddings = backend.array(embeddings_np)
    elif input_file.suffix == ".safetensors":
        data = load_safetensors(input_file)
        # Try common keys
        for key in ["embeddings", "embed", "hidden_states", "data"]:
            if key in data:
                embeddings = backend.array(data[key])
                break
        else:
            # Use first tensor
            first_key = list(data.keys())[0]
            embeddings = backend.array(data[first_key])
            typer.echo(f"  Using tensor key: {first_key}")
    else:
        typer.echo(f"Error: Unsupported input format: {input_file.suffix}", err=True)
        typer.echo("  Supported formats: .npy, .safetensors", err=True)
        raise typer.Exit(code=1)

    backend.eval(embeddings)
    typer.echo(f"  Input shape: {embeddings.shape}")

    # Apply transform
    typer.echo(f"  Applying {'inverse' if inverse else 'forward'} transform...")
    if inverse:
        transformed = bridge.apply_inverse(embeddings, normalize_scale=normalize)
    else:
        transformed = bridge.apply(embeddings, normalize_scale=normalize)
    backend.eval(transformed)

    typer.echo(f"  Output shape: {transformed.shape}")

    # Save output
    typer.echo(f"  Saving to {output}...")
    # Convert to numpy for saving
    transformed_np = np.array(backend.tolist(transformed))

    if output_path.suffix == ".npy":
        np.save(output_path, transformed_np)
    elif output_path.suffix == ".safetensors":
        save_safetensors({"embeddings": transformed_np}, str(output_path))
    else:
        # Default to npy
        np.save(output_path, transformed_np)
        typer.echo(f"  Note: Saved as NumPy format (unknown extension: {output_path.suffix})")

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("APPLY BRIDGE COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Input shape: {embeddings.shape}")
    typer.echo(f"  Output shape: {transformed.shape}")
    typer.echo(f"  Direction: {'inverse' if inverse else 'forward'}")
    typer.echo(f"  Scale normalized: {normalize}")
    typer.echo("=" * 60)


def _find_embedding_key(weights: dict[str, any]) -> str | None:
    """Find the embedding layer key in a model's weight dict."""
    # Common embedding layer names across architectures
    candidates = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embeddings.word_embeddings.weight",
        "embed_tokens.weight",
        "wte.weight",
        "word_embeddings.weight",
    ]
    for key in candidates:
        if key in weights:
            return key
    # Fallback: look for any key with "embed" in it
    for key in weights:
        if "embed" in key.lower() and "weight" in key.lower():
            return key


def _sample_atlas_probes(
    backend: any,
    source_embed: any,
    target_embed: any,
    source_path: str,
    target_path: str,
    n_samples: int,
    probe_sources: str | None = None,
) -> tuple[any, any]:
    """Sample embeddings using atlas semantic probes.

    Loads conceptual probes from the atlas system, tokenizes them using each
    model's tokenizer, and collects the corresponding embeddings.

    Args:
        backend: Backend for array operations
        source_embed: Source embedding table [vocab_size, hidden_dim]
        target_embed: Target embedding table [vocab_size, hidden_dim]
        source_path: Path to source model (for tokenizer)
        target_path: Path to target model (for tokenizer)
        n_samples: Maximum number of probe samples
        probe_sources: Optional comma-separated list of atlas sources to use

    Returns:
        Tuple of (source_activations, target_activations)
    """
    from modelcypher.cli.composition import get_model_loader
    from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory
    from modelcypher.core.use_cases.merge.helpers import load_tokenizer

    # Load atlas probes
    all_probes = UnifiedAtlasInventory.all_probes()

    # Filter by sources if specified
    if probe_sources:
        allowed_sources = {s.strip().lower() for s in probe_sources.split(",")}
        all_probes = [p for p in all_probes if p.source.value.lower() in allowed_sources]

    if not all_probes:
        raise ValueError(f"No probes found for sources: {probe_sources}")

    # Limit to n_samples
    if len(all_probes) > n_samples:
        # Sample evenly across probes
        step = len(all_probes) // n_samples
        all_probes = all_probes[::step][:n_samples]

    # Load tokenizers
    model_loader = get_model_loader()
    source_tokenizer = load_tokenizer(source_path, model_loader)
    target_tokenizer = load_tokenizer(target_path, model_loader)

    if source_tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for source: {source_path}")
    if target_tokenizer is None:
        raise ValueError(f"Failed to load tokenizer for target: {target_path}")

    # Collect embeddings for each probe
    source_embeds_list = []
    target_embeds_list = []
    n_source_vocab = int(source_embed.shape[0])
    n_target_vocab = int(target_embed.shape[0])

    for probe in all_probes:
        # Get probe text (use name as primary, fallback to first support text)
        probe_text = probe.name
        if probe.support_texts:
            probe_text = probe.support_texts[0]

        # Tokenize with each model's tokenizer
        try:
            source_ids = source_tokenizer.encode(probe_text, add_special_tokens=False)
            target_ids = target_tokenizer.encode(probe_text, add_special_tokens=False)

            if not source_ids or not target_ids:
                continue

            # Use first token as representative (most semantic content is there)
            source_id = source_ids[0]
            target_id = target_ids[0]

            # Validate token IDs are within vocab range
            if source_id >= n_source_vocab or target_id >= n_target_vocab:
                continue

            # Get embeddings
            source_idx = backend.array([source_id])
            target_idx = backend.array([target_id])
            source_vec = backend.take(source_embed, source_idx, axis=0)
            target_vec = backend.take(target_embed, target_idx, axis=0)

            source_embeds_list.append(source_vec)
            target_embeds_list.append(target_vec)
        except Exception:
            # Skip probes that fail tokenization
            continue

    if not source_embeds_list:
        raise ValueError("No valid probe embeddings found. Check tokenizers and vocabulary.")

    # Stack into arrays
    source_activations = backend.concatenate(source_embeds_list, axis=0)
    target_activations = backend.concatenate(target_embeds_list, axis=0)

    return source_activations, target_activations
