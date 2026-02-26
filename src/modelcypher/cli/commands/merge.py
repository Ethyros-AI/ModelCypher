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

"""Merge CLI - Geometric model merging.

Null-space constrained transplant: W' = W_target + P_null @ (W_source_aligned - W_target)

Commands:
    mc merge run -s SOURCE -t TARGET -o OUTPUT       # Single merge
    mc merge batch -s SRC1 -s SRC2 -t TARGET -o OUT  # N→1 merge
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

merge_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _validate_path(path: Path, label: str, context: CLIContext) -> None:
    """Validate a model path exists, exit with error if not."""
    if not path.exists():
        error = ErrorDetail(
            code="MC-3001",
            title=f"{label} not found",
            detail=f"Path does not exist: {path}",
            hint="Provide a valid path to a model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)


def _result_to_dict(result) -> dict:
    """Extract serializable metrics from UnifiedMergeResult."""
    return {
        "merge_strategy": result.merge_strategy,
        "output_path": result.output_path,
        "layer_count": result.layer_count,
        "weight_count": result.weight_count,
        "mean_preserved_fraction": result.mean_preserved_fraction,
        "mean_procrustes_error": result.mean_procrustes_error,
        "refusal_preserved": result.refusal_preserved,
        "timestamp": result.timestamp.isoformat(),
        "probe_metrics": result.probe_metrics,
        "density_metrics": result.density_metrics,
        "transplant_metrics": result.transplant_metrics,
    }


@merge_app.command("run")
def merge_run(
    ctx: typer.Context,
    source: str = typer.Option(
        ..., "--source", "-s", help="Path to source model (capabilities to transfer)"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="Path to target model (base architecture)"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output directory for merged model"
    ),
    behavior_jacobian: bool = typer.Option(
        False,
        "--behavior-jacobian",
        help="Use behavior Jacobian null-space projector (per-probe CE gradients) instead of activation-covariance",
    ),
) -> None:
    """Merge source model into target using null-space geometric transplant.

    Transfers knowledge from source into target's unused representation space
    without disrupting target's existing capabilities. The pipeline discovers
    which directions the target doesn't use (null space), aligns source knowledge
    into those directions, and writes the combined model.

    Pipeline stages: PROBE (map manifolds) → DENSITY (find transfer regions) → TRANSPLANT (null-space projection).

    Output fields (when --json):
        merge_strategy: Always "transplant"
        output_path: Path to merged model
        layer_count: Number of layers merged
        weight_count: Total weights in output
        mean_preserved_fraction: Fraction of source knowledge preserved (0-1)
        mean_procrustes_error: Alignment error (lower = better)
        probe_metrics: CKA alignment and layer correspondence details
        density_metrics: Knowledge density analysis
        transplant_metrics: Per-layer transplant details

    Example:
        mc merge run -s /path/to/source -t /path/to/target -o /path/to/output
    """
    context = _context(ctx)
    source_path = Path(source)
    target_path = Path(target)

    _validate_path(source_path, "Source model", context)
    _validate_path(target_path, "Target model", context)

    import sys

    from modelcypher.cli.composition import get_merge_service

    # Create progress reporter for structured stderr events when in AI mode
    # or when stderr is not a TTY (agent consumption).
    from modelcypher.cli.progress import ProgressReporter

    reporter = None
    if context.ai_mode or not sys.stderr.isatty():
        reporter = ProgressReporter()

    merger = get_merge_service()
    merger._progress_reporter = reporter
    result = merger.merge(
        source_path=str(source_path),
        target_path=str(target_path),
        output_dir=output,
        behavior_jacobian=behavior_jacobian,
    )

    write_output(_result_to_dict(result), context.output_format, context.pretty)


@merge_app.command("batch")
def merge_batch(
    ctx: typer.Context,
    source: list[str] = typer.Option(
        ..., "--source", "-s", help="Source model paths (repeatable)"
    ),
    target: str = typer.Option(
        ..., "--target", "-t", help="Path to target model"
    ),
    output: str = typer.Option(
        ..., "--output", "-o", help="Output directory for merged model"
    ),
) -> None:
    """Merge multiple source models into one target (N→1 batch merge).

    Target is loaded and probed once, then reused for all sources.
    Deltas accumulate in the target's null space. Each source contributes
    knowledge to unused directions, progressively filling capacity.

    Output: Same structure as `mc merge run`.

    Example:
        mc merge batch -s /path/src1 -s /path/src2 -t /path/target -o /path/out
    """
    context = _context(ctx)
    target_path = Path(target)

    _validate_path(target_path, "Target model", context)
    for i, src in enumerate(source):
        _validate_path(Path(src), f"Source model {i + 1}", context)

    from modelcypher.cli.composition import get_merge_service

    merger = get_merge_service()
    result = merger.merge_batch(
        source_paths=source,
        target_path=str(target_path),
        output_dir=output,
    )

    write_output(_result_to_dict(result), context.output_format, context.pretty)
