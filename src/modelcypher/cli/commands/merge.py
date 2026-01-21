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

"""Merge models via null-space knowledge transplant."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.commands.model import prevent_sleep
from modelcypher.cli.composition import get_model_probe_service
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


def _strip_option_prefix(value: str, flags: tuple[str, ...]) -> str:
    cleaned = value.strip()
    for flag in flags:
        if cleaned.startswith(f"{flag}="):
            return cleaned[len(flag) + 1 :].strip()
        if cleaned.startswith(flag):
            remainder = cleaned[len(flag) :].strip()
            return remainder
    return cleaned


def _strip_wrapping_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        return cleaned[1:-1].strip()
    return cleaned


def _resolve_merge_path(
    ctx: typer.Context,
    value: str | None,
    *,
    label: str,
    flags: tuple[str, ...],
    missing_hint: str,
) -> str:
    if value:
        return value

    context = _context(ctx)
    if context.no_prompt:
        error = ErrorDetail(
            code="MC-1101",
            title="Missing merge input",
            detail=f"{label} is required. Provide {missing_hint}.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    response = typer.prompt(f"{label} (paste a path or '{flags[0]} /path')")
    response = _strip_option_prefix(response, flags)
    response = _strip_wrapping_quotes(response)
    if not response:
        error = ErrorDetail(
            code="MC-1102",
            title="Empty merge input",
            detail=f"{label} cannot be empty.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
    return response


def _run_dry_run(
    ctx: typer.Context,
    source: str,
    target: str,
    output_dir: str,
) -> None:
    """Show merge inputs and compatibility without running."""
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
    full_atlas: bool = False,
    dry_run: bool = False,
    auto_profile: bool = False,
    no_profile: bool = False,
) -> None:
    """Run a single-source merge with atlas probes and fixed scale."""
    from modelcypher.cli.composition import get_merge_pipeline_service
    from modelcypher.utils.logging import add_file_logger, remove_file_loggers

    context = _context(ctx)

    # Validate model paths early
    validate_model_path(source, context=context)
    validate_model_path(target, context=context)

    # Dry-run mode: show what would happen without merging
    if dry_run:
        _run_dry_run(ctx, source, target, output_dir)
        return

    # Auto-profile mode: profile models if profiles don't exist
    if auto_profile:
        from modelcypher.cli.composition import get_registry
        from modelcypher.core.domain.profile import GeometricProfileStore
        from modelcypher.core.use_cases.profile_service import ProfileService

        store = GeometricProfileStore()
        registry = get_registry()
        service = ProfileService(
            backend=registry.backend,
            model_loader=registry.model_loader,
            activation_provider=registry.activation_provider,
            store=store,
        )

        for model_path, label in [(source, "source"), (target, "target")]:
            if not store.exists(model_path):
                typer.echo(f"AUTO-PROFILE: Computing profile for {label} model...", err=True)
                result = service.compute_profile(model_path)
                typer.echo(
                    f"AUTO-PROFILE: {label} profile computed ({result.layers_profiled} layers, "
                    f"{result.probes_processed} probes)",
                    err=True,
                )
            else:
                typer.echo(f"AUTO-PROFILE: Using cached profile for {label} model", err=True)

    # Set up automatic file logging for merge operations
    log_path = add_file_logger()
    if log_path:
        typer.echo(f"LOG FILE: {log_path}")
        typer.echo("")

    service = get_merge_pipeline_service()

    try:
        with prevent_sleep():
            # delta_scale=1.0 always - null-space projection handles safety
            # use_profiles=True by default: auto-detect and use profiles when available
            # --no-profile explicitly disables profile usage (forces probe inference)
            result = service.run(
                source_path=source,
                target_path=target,
                output_dir=output_dir,
                probe_mode="atlas_full" if full_atlas else "atlas",
                delta_scale=1.0,
                use_profiles=not no_profile,  # True unless --no-profile specified
            )

        # Build output payload
        payload = {
            "_schema": "mc.merge.pipeline.v1",
            "pipelineId": result.pipeline_id,
            "timestamp": result.timestamp,
            "sourceModel": result.source_model,
            "targetModel": result.target_model,
            "outputDir": result.output_dir,
            "logFile": log_path,
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
            ])

            if log_path:
                lines.extend([
                    "",
                    f"LOG FILE: {log_path}",
                ])

            lines.append("=" * 70)

            write_output("\n".join(lines), context.output_format, context.pretty)
            remove_file_loggers()
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-1100",
            title="Pipeline failed",
            detail=str(e),
            hint=f"Check model paths and merge parameters. Log file: {log_path}" if log_path else "Check model paths and merge parameters",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict())
        raise typer.Exit(code=1)
    finally:
        # Always clean up file loggers
        remove_file_loggers()


@app.callback()
def merge_callback(
    ctx: typer.Context,
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Path to source model (knowledge donor). Prompted if omitted",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Path to target model (receives knowledge). Prompted if omitted",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for merged model. Prompted if omitted",
    ),
    output_file: str | None = typer.Option(None, "--output-file", "-f", help="Save full pipeline result to JSON file"),
    full_atlas: bool = typer.Option(
        False,
        "--full-atlas/--geometry-min",
        help="Use full atlas probes instead of geometry-minimum set",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would happen without actually merging"),
    auto_profile: bool = typer.Option(
        False,
        "--auto-profile/--no-auto-profile",
        help="Automatically profile models if profiles don't exist (profile once, merge many)",
    ),
    no_profile: bool = typer.Option(
        False,
        "--no-profile",
        help="Force probe inference even if profiles exist (skip profile-based merge)",
    ),
) -> None:
    """Merge two models via null-space knowledge transplant.

    By default, uses cached geometric profiles if available (profile once, merge many).
    Use --no-profile to force probe inference.

    Example: mc merge -s ./qwen -t ./smol -o ./merged
    """
    # If a subcommand was invoked (like 'run'), don't do anything here
    if ctx.invoked_subcommand is not None:
        return

    # Resolve required inputs (prompt if missing unless --no-prompt/--ai)
    if source or target or output_dir:
        source = _resolve_merge_path(
            ctx,
            source,
            label="Source model path",
            flags=("-s", "--source"),
            missing_hint="--source/-s",
        )
        target = _resolve_merge_path(
            ctx,
            target,
            label="Target model path",
            flags=("-t", "--target"),
            missing_hint="--target/-t",
        )
        output_dir = _resolve_merge_path(
            ctx,
            output_dir,
            label="Output directory",
            flags=("-o", "--output-dir"),
            missing_hint="--output-dir/-o",
        )

        _run_merge(
            ctx,
            source,
            target,
            output_dir,
            output_file,
            full_atlas=full_atlas,
            dry_run=dry_run,
            auto_profile=auto_profile,
            no_profile=no_profile,
        )
        return
    # else: no options, show help (handled by Typer's no_args_is_help behavior)


@app.command()
def run(
    ctx: typer.Context,
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Path to source model (knowledge donor). Prompted if omitted",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Path to target model (receives knowledge). Prompted if omitted",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for merged model. Prompted if omitted",
    ),
    output_file: str | None = typer.Option(
        None,
        "--output-file",
        "-f",
        help="Save full pipeline result to JSON file",
    ),
    full_atlas: bool = typer.Option(
        False,
        "--full-atlas/--geometry-min",
        help="Use full atlas probes instead of geometry-minimum set",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would happen without actually merging"),
    auto_profile: bool = typer.Option(
        False,
        "--auto-profile/--no-auto-profile",
        help="Automatically profile models if profiles don't exist (profile once, merge many)",
    ),
    no_profile: bool = typer.Option(
        False,
        "--no-profile",
        help="Force probe inference even if profiles exist (skip profile-based merge)",
    ),
) -> None:
    """Merge two models via null-space knowledge transplant.

    By default, uses cached geometric profiles if available (profile once, merge many).
    Use --no-profile to force probe inference.

    Example: mc merge run -s ./qwen -t ./smol -o ./merged
    """
    source = _resolve_merge_path(
        ctx,
        source,
        label="Source model path",
        flags=("-s", "--source"),
        missing_hint="--source/-s",
    )
    target = _resolve_merge_path(
        ctx,
        target,
        label="Target model path",
        flags=("-t", "--target"),
        missing_hint="--target/-t",
    )
    output_dir = _resolve_merge_path(
        ctx,
        output_dir,
        label="Output directory",
        flags=("-o", "--output-dir"),
        missing_hint="--output-dir/-o",
    )

    _run_merge(
        ctx,
        source,
        target,
        output_dir,
        output_file,
        full_atlas=full_atlas,
        dry_run=dry_run,
        auto_profile=auto_profile,
        no_profile=no_profile,
    )


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
    """Merge multiple source models into one target.

    Examples:
        mc merge batch -s ./m1 -s ./m2 -t ./target -o ./out
        mc merge batch -s ./qwen -s ./llama -t ./smol -o ./out --consensus
        mc merge batch -s ./m1 -s ./m2 -t ./target -o ./out --detect-outliers
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger

    backend = get_default_backend()

    # Validate paths
    context = _context(ctx)
    for source in sources:
        validate_model_path(source, context=context)
    validate_model_path(target, context=context)

    # Set up automatic file logging for merge operations
    from modelcypher.utils.logging import add_file_logger, remove_file_loggers

    log_path = add_file_logger()
    if log_path:
        typer.echo(f"LOG FILE: {log_path}")
        typer.echo("")

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
        from modelcypher.core.domain.agents.probe_loader import ProbeLoader
        from modelcypher.core.use_cases.merge.helpers import get_hidden_state

        model_loader = MLXModelLoader()
        all_paths = [target] + list(sources)
        path_names = [Path(path).name for path in all_paths]

        # Load probes for quick Gram comparison
        # Use sqrt(available) probes - balances coverage vs speed
        probe_loader = ProbeLoader()
        available_probes = probe_loader.load_probes()
        n_probes = max(len(all_paths) + 2, int(len(available_probes) ** 0.5))
        n_probes = min(n_probes, 128)  # Cap for pre-merge speed
        probes = available_probes[:n_probes]
        min_valid = len(all_paths) + 1

        def _collect_probe_activations(model: object, tokenizer: object) -> list[object]:
            activations = []
            for probe in probes:
                try:
                    act = get_hidden_state(model, tokenizer, probe.text, backend)
                    if act is not None:
                        activations.append(act)
                except Exception:
                    continue
            return activations

        def _build_gram_matrix(activations: list[object]) -> object:
            X = backend.stack(activations, axis=0)
            backend.eval(X)
            sq_dist = geodesic_squared_distances(X, backend)
            sigma = _shared_rbf_sigma(sq_dist, sq_dist, backend)
            K = _rbf_gram_from_sq_distances(sq_dist, sigma, backend)
            backend.eval(K)
            return K

        # Collect Gram matrices for each model
        gram_matrices = []
        for path in all_paths:
            model, tokenizer = model_loader.load_model(path)

            # Get activations for probes (middle layer)
            activations = _collect_probe_activations(model, tokenizer)

            # Need at least n_models + 1 valid probes for meaningful Gram comparison
            if len(activations) < min_valid:
                typer.echo(f"  Warning: Too few valid probes ({len(activations)} < {min_valid}) for {path}")
                gram_matrices.append(None)
                continue

            # Build geodesic RBF Gram matrix (n_probes × n_probes, dimension-invariant)
            # Uses k-NN graph shortest paths for proper manifold distances
            gram_matrices.append(_build_gram_matrix(activations))

            # Clean up model
            del model
            if hasattr(backend, "clear_cache"):
                backend.clear_cache()

        # Compare Gram matrices using geodesic CKA (overlap diagnostic)
        detector = OutlierDetector(backend)
        valid_grams = [(i, K) for i, K in enumerate(gram_matrices) if K is not None]

        if len(valid_grams) < 2:
            typer.echo("  Error: Need at least 2 valid Gram matrices for comparison.")
            result_detect = None
        else:
            # Compute pairwise CKA distances (1 - CKA = distance)
            pairwise_cka: dict[tuple[int, int], float] = {}
            cka_sums = {idx: 0.0 for idx, _ in valid_grams}
            cka_counts = {idx: 0 for idx, _ in valid_grams}

            for i in range(len(valid_grams)):
                idx_i, K_i = valid_grams[i]
                for j in range(i + 1, len(valid_grams)):
                    idx_j, K_j = valid_grams[j]
                    cka_val = compute_cka_from_grams(K_i, K_j, backend)
                    pairwise_cka[(idx_i, idx_j)] = cka_val
                    cka_sums[idx_i] += cka_val
                    cka_sums[idx_j] += cka_val
                    cka_counts[idx_i] += 1
                    cka_counts[idx_j] += 1

            mean_cka = [
                (idx, cka_sums[idx] / cka_counts[idx] if cka_counts[idx] > 0 else 0.0)
                for idx, _ in valid_grams
            ]
            mean_cka_map = {idx: mean for idx, mean in mean_cka}

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
                typer.echo(f"    {path_names[i]} <-> {path_names[j]}: CKA={cka:.4f}")
            typer.echo("")

            if result_detect.outlier_indices:
                typer.echo("  OUTLIERS DETECTED (low CKA with others):")
                for idx in result_detect.outlier_indices:
                    model_mean_cka = mean_cka_map.get(idx, 0.0)
                    role = "TARGET" if idx == 0 else f"SOURCE-{idx}"
                    typer.echo(f"    [{role}] {path_names[idx]} (mean CKA: {model_mean_cka:.4f})")
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
    if log_path:
        typer.echo(f"  Log file: {log_path}")

    # Clean up file loggers
    remove_file_loggers()


@app.command()
def deviation(
    ctx: typer.Context,
    baseline: str = typer.Option(..., "--baseline", "-b", help="Path to original baseline model"),
    current: str = typer.Option(..., "--current", "-c", help="Path to current (merged) model"),
) -> None:
    """Measure deviation from a baseline model.

    Example: mc merge deviation --baseline ./original --current ./merged2
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.deviation_budget import DeviationTracker

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    validate_model_path(baseline, context=context)
    validate_model_path(current, context=context)

    typer.echo(f"DEVIATION MEASUREMENT: {current} vs baseline {baseline}")

    # Load weights
    model_loader = MLXModelLoader()

    typer.echo("  Loading baseline weights...")
    baseline_weights = model_loader.load_weights(baseline)

    typer.echo("  Loading current weights...")
    current_weights = model_loader.load_weights(current)

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
    """Merge multiple channels via Birkhoff routing.

    Examples:
        mc merge multi-channel -c spatial:/path -c text:/path -t ./lfm2 -o ./merged
        mc merge multi-channel -c spatial:./world -c text:./llm -t ./target -o ./out
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.use_cases.merge.merger import UnifiedGeometricMerger

    context = _context(ctx)
    backend = get_default_backend()

    def _fail_channel(detail: str) -> None:
        error = ErrorDetail(
            code="MC-1201",
            title="Invalid channel format",
            detail=detail,
            hint="Use 'name:path' format for each channel.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Parse channel arguments (format: "name:path")
    channel_paths: dict[str, str] = {}
    for channel_spec in channels:
        if ":" not in channel_spec:
            _fail_channel(
                f"Invalid channel format '{channel_spec}'. Use 'name:path' format."
            )
        name, path = channel_spec.split(":", 1)
        if not name or not path:
            _fail_channel(
                f"Invalid channel format '{channel_spec}'. Both name and path required."
            )
        channel_paths[name] = path

    # Validate paths
    for name, path in channel_paths.items():
        validate_model_path(path, context=context)
    validate_model_path(target, context=context)

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
    """Generate an embedding bridge between two encoders.

    Examples:
        mc merge bridge /path/to/clip /path/to/lfm2 -o clip_to_lfm2.safetensors
        mc merge bridge ./model_a ./model_b -o bridge.safetensors --samples 200
    """
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.bridge.generator import BridgeGenerator

    context = _context(ctx)
    backend = get_default_backend()

    # Validate paths
    validate_model_path(source, context=context)
    validate_model_path(target, context=context)

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

    Examples:
        mc merge apply-bridge bridge.safetensors source.npy -o target.npy
        mc merge apply-bridge bridge.safetensors target.npy -o source.npy --inverse
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
    """Sample embeddings using atlas semantic probes."""
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


@app.command()
def validate(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to merged model to validate"),
    baseline: str | None = typer.Option(
        None, "--baseline", "-b", help="Path to baseline measurements JSON for comparison"
    ),
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Save validation report to JSON file"
    ),
    skip_density: bool = typer.Option(
        False, "--skip-density", help="Skip intrinsic dimension and null space measurements"
    ),
    num_prompts: int | None = typer.Option(
        None, "--num-prompts", "-n", help="Number of test prompts for coherence validation"
    ),
    baseline_model: str | None = typer.Option(
        None, "--baseline-model", help="Baseline model path for coherence comparison"
    ),
) -> None:
    """Validate a merged model for coherence and density.

    Examples:
        mc merge validate /path/to/merged
        mc merge validate /path/to/merged --baseline baselines.json
        mc merge validate /path/to/merged -o results/validation.json
    """
    from datetime import datetime

    from modelcypher.cli.composition import get_inference_engine, get_model_loader, get_registry
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.trajectory_coherence import validate_merge_coherence

    context = _context(ctx)
    backend = get_default_backend()

    # Validate model path
    if not validate_model_path(model, context=context):
        raise typer.Exit(code=1)

    typer.echo("=" * 70)
    typer.echo("POST-MERGE VALIDATION")
    typer.echo("=" * 70)
    typer.echo(f"Model: {model}")
    typer.echo("")

    validation_results: dict[str, any] = {
        "_schema": "mc.merge.validation.v1",
        "timestamp": datetime.utcnow().isoformat(),
        "model_path": model,
    }

    # === PHASE 1: COHERENCE VALIDATION (Required) ===
    typer.echo("PHASE 1: COHERENCE VALIDATION")
    typer.echo("-" * 40)

    # Extended test prompts for thorough validation
    test_prompts = [
        "The most important thing about machine learning is",
        "In mathematics, a function is defined as",
        "The capital of France is Paris, and the capital of Germany is",
        "When you want to write good code, you should",
        "A neural network consists of layers that",
        "The difference between supervised and unsupervised learning is",
        "In physics, energy is conserved, which means",
        "A good explanation should be clear and",
        "The relationship between input and output in a system",
        "When debugging code, the first step is usually to",
    ]
    if num_prompts is not None:
        test_prompts = test_prompts[:num_prompts]

    try:
        inference_engine = get_inference_engine()
        typer.echo(f"  Running {len(test_prompts)} coherence tests...")

        coherence_result = validate_merge_coherence(
            model_path=model,
            inference_engine=inference_engine,
            test_prompts=test_prompts,
            baseline_model_path=baseline_model,
        )

        passed_count = (
            coherence_result.total_count - coherence_result.failed_count
            if coherence_result.failed_count is not None
            else None
        )
        validation_results["coherence"] = {
            "is_coherent": coherence_result.is_coherent,
            "passed_count": passed_count,
            "failed_count": coherence_result.failed_count,
            "total_count": coherence_result.total_count,
            "mean_repetition_score": round(coherence_result.mean_repetition_score, 4),
            "failed_prompts": coherence_result.failed_prompts,
        }

        if coherence_result.is_coherent is True:
            typer.echo(
                f"  [PASS] Coherence: {passed_count}/{coherence_result.total_count} prompts passed"
            )
            typer.echo(f"  Mean repetition score: {coherence_result.mean_repetition_score:.4f}")
        elif coherence_result.is_coherent is False:
            typer.echo(
                f"  [FAIL] Coherence: {coherence_result.failed_count}/{coherence_result.total_count} prompts failed"
            )
            typer.echo(f"  Mean repetition score: {coherence_result.mean_repetition_score:.4f}")
            typer.echo("")
            typer.echo("  FAILED PROMPTS:")
            for i, (prompt, metrics) in enumerate(
                zip(
                    coherence_result.failed_prompts,
                    [m for m in coherence_result.metrics if m.is_degenerate],
                )
            ):
                typer.echo(f"    {i+1}. '{prompt[:40]}...'")
                typer.echo(f"       Reason: {metrics.degenerate_reason}")
        else:
            typer.echo("  [INFO] Coherence metrics computed (no baseline model).")
            typer.echo(f"  Mean repetition score: {coherence_result.mean_repetition_score:.4f}")

    except Exception as e:
        typer.echo(f"  [ERROR] Coherence validation failed: {e}")
        validation_results["coherence"] = {
            "is_coherent": False,
            "error": str(e),
        }

    typer.echo("")

    # === PHASE 2: DENSITY METRICS (Optional) ===
    if not skip_density:
        typer.echo("PHASE 2: DENSITY METRICS")
        typer.echo("-" * 40)

        try:
            registry = get_registry()
            model_loader = get_model_loader()

            # Load model for activation extraction
            typer.echo("  Loading model for density measurement...")
            loaded_model, tokenizer = model_loader.load_model_for_training(model)

            # Collect activations using probes
            typer.echo("  Collecting activations from probe prompts...")
            from modelcypher.core.domain.agents.probe_loader import ProbeLoader

            probe_loader = ProbeLoader()
            all_probes = probe_loader.load_probes()

            # Use subset of probes for density measurement (faster)
            density_probes = all_probes[:min(100, len(all_probes))]

            # Collect hidden state activations
            activation_provider = registry.activation_provider
            probe_texts = [p.text for p in density_probes]

            try:
                activations_list = activation_provider.collect_hidden_activations_batch(
                    loaded_model, tokenizer, probe_texts
                )
            except NotImplementedError:
                # Fall back to sequential
                activations_list = []
                for text in probe_texts:
                    try:
                        act_dict = activation_provider.collect_hidden_activations(
                            loaded_model, tokenizer, text
                        )
                        activations_list.append(act_dict)
                    except Exception:
                        continue

            if activations_list:
                # Compute per-layer intrinsic dimension
                typer.echo("  Computing intrinsic dimension per layer...")
                id_estimator = IntrinsicDimension(backend)

                layer_ids = {}
                # Get layer indices from first activation dict
                if activations_list:
                    layer_indices = sorted(activations_list[0].keys())

                    for layer_idx in layer_indices:
                        # Collect activations for this layer across all probes
                        layer_acts = []
                        for act_dict in activations_list:
                            if layer_idx in act_dict:
                                layer_acts.append(act_dict[layer_idx])

                        if len(layer_acts) >= 10:
                            points = backend.stack(layer_acts, axis=0)
                            backend.eval(points)

                            try:
                                estimate = id_estimator.compute(points)
                                layer_ids[layer_idx] = estimate.intrinsic_dimension
                            except Exception:
                                continue

                if layer_ids:
                    mean_id = sum(layer_ids.values()) / len(layer_ids)
                    max_id = max(layer_ids.values())
                    min_id = min(layer_ids.values())

                    # Estimate null space ratio from variance
                    # Null space = directions with near-zero variance
                    # Simplified: (hidden_dim - mean_ID) / hidden_dim
                    hidden_dim = next(iter(activations_list[0].values())).shape[-1] if activations_list else 0
                    null_space_ratio = (hidden_dim - mean_id) / hidden_dim if hidden_dim > 0 else 0.0

                    validation_results["density"] = {
                        "mean_intrinsic_dimension": round(mean_id, 4),
                        "max_intrinsic_dimension": round(max_id, 4),
                        "min_intrinsic_dimension": round(min_id, 4),
                        "hidden_dimension": hidden_dim,
                        "estimated_null_space_ratio": round(null_space_ratio, 6),
                        "layers_measured": len(layer_ids),
                        "probes_used": len(activations_list),
                        "layer_dimensions": {str(k): round(v, 4) for k, v in layer_ids.items()},
                    }

                    typer.echo(f"  Mean intrinsic dimension: {mean_id:.2f}")
                    typer.echo(f"  Range: [{min_id:.2f}, {max_id:.2f}]")
                    typer.echo(f"  Hidden dimension: {hidden_dim}")
                    typer.echo(f"  Estimated null space ratio: {null_space_ratio:.2%}")
                else:
                    typer.echo("  [WARN] Could not compute intrinsic dimension")
                    validation_results["density"] = {"error": "Failed to compute ID"}
            else:
                typer.echo("  [WARN] No activations collected")
                validation_results["density"] = {"error": "No activations collected"}

            # Clean up
            del loaded_model
            if hasattr(backend, "clear_cache"):
                backend.clear_cache()

        except Exception as e:
            typer.echo(f"  [ERROR] Density measurement failed: {e}")
            validation_results["density"] = {"error": str(e)}

        typer.echo("")

    # === PHASE 3: BASELINE COMPARISON (Optional) ===
    if baseline:
        typer.echo("PHASE 3: BASELINE COMPARISON")
        typer.echo("-" * 40)

        try:
            baseline_path = Path(baseline)
            if baseline_path.exists():
                baseline_data = json.loads(baseline_path.read_text())

                comparison = {}

                # Compare intrinsic dimension if available
                if "density" in validation_results and "mean_intrinsic_dimension" in validation_results["density"]:
                    merged_id = validation_results["density"]["mean_intrinsic_dimension"]

                    if "comparison" in baseline_data and "mean_intrinsic_dimension" in baseline_data["comparison"]:
                        baseline_ids = baseline_data["comparison"]["mean_intrinsic_dimension"]
                        for model_name, baseline_id in baseline_ids.items():
                            delta = merged_id - baseline_id
                            comparison[f"id_delta_vs_{model_name}"] = round(delta, 4)
                            improvement = "IMPROVED" if delta > 0 else "DECREASED"
                            typer.echo(f"  ID vs {model_name}: {delta:+.2f} ({improvement})")

                    if "comparison" in baseline_data and "mean_null_space_ratio" in baseline_data["comparison"]:
                        baseline_nulls = baseline_data["comparison"]["mean_null_space_ratio"]
                        merged_null = validation_results["density"].get("estimated_null_space_ratio", 0)
                        for model_name, baseline_null in baseline_nulls.items():
                            delta = merged_null - baseline_null
                            comparison[f"null_delta_vs_{model_name}"] = round(delta, 6)
                            improvement = "MORE DENSE" if delta < 0 else "LESS DENSE"
                            typer.echo(f"  Null space vs {model_name}: {delta:+.4f} ({improvement})")

                validation_results["baseline_comparison"] = comparison
            else:
                typer.echo(f"  [WARN] Baseline file not found: {baseline}")
                validation_results["baseline_comparison"] = {"error": f"File not found: {baseline}"}

        except Exception as e:
            typer.echo(f"  [ERROR] Baseline comparison failed: {e}")
            validation_results["baseline_comparison"] = {"error": str(e)}

        typer.echo("")

    # === FINAL SUMMARY ===
    typer.echo("=" * 70)
    typer.echo("VALIDATION SUMMARY")
    typer.echo("=" * 70)

    coherence_status = validation_results.get("coherence", {})
    is_coherent = coherence_status.get("is_coherent", False)

    if is_coherent:
        typer.echo("  [PASS] Model generates coherent output")
        validation_results["overall_status"] = "PASS"
    else:
        typer.echo("  [FAIL] Model produces degenerate output")
        validation_results["overall_status"] = "FAIL"

    if "density" in validation_results and "mean_intrinsic_dimension" in validation_results["density"]:
        mean_id = validation_results["density"]["mean_intrinsic_dimension"]
        null_ratio = validation_results["density"].get("estimated_null_space_ratio", 0)
        typer.echo(f"  Intrinsic dimension: {mean_id:.2f}")
        typer.echo(f"  Null space utilization: {(1 - null_ratio) * 100:.1f}%")

    typer.echo("=" * 70)

    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(validation_results, indent=2))
        typer.echo(f"\nResults saved to {output_file}")

    # JSON output
    if context.output_format == "json":
        write_output(validation_results, context.output_format, context.pretty)

    # Exit with appropriate code
    if not is_coherent:
        raise typer.Exit(code=1)
