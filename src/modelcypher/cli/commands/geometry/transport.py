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

"""Geometry transport-guided merge CLI commands.

Provides commands for transport-guided model weight merging using
Gromov-Wasserstein optimal transport.

Commands:
    mc geometry transport merge --source <file> --target <file> --plan <file>
    mc geometry transport synthesize --source-act <file> --target-act <file> ...
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_transport_service import (
    GeometryTransportService,
    MergeConfig,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("merge")
def geometry_transport_merge(
    ctx: typer.Context,
    source_file: Path = typer.Option(
        ..., "--source", "-s", help="JSON file with source weights [N x D]"
    ),
    target_file: Path = typer.Option(
        ..., "--target", "-t", help="JSON file with target weights [M x D]"
    ),
    plan_file: Path = typer.Option(
        ..., "--plan", "-p", help="JSON file with transport plan [N x M]"
    ),
    output: Path | None = typer.Option(
        None, "--output-file", "-o", help="Output file for merged weights"
    ),
):
    """
    Merge weights using a transport plan.

    Coupling threshold and blend alpha are derived from the transport
    plan structure - not user-specified.

    Uses the transport plan pi[i,j] to guide weighted averaging:
    W_merged[j,:] = sum_i pi[i,j] * W_source[i,:]
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source = json.loads(Path(source_file).read_text())
    target = json.loads(Path(target_file).read_text())
    plan = json.loads(Path(plan_file).read_text())

    # Threshold and alpha derived from transport plan, not user-specified
    merged = service.synthesize_weights(
        source_weights=source,
        target_weights=target,
        transport_plan=plan,
        coupling_threshold=None,  # Derived from plan distribution
        normalize_rows=True,
        blend_alpha=None,  # Derived from coupling strength
    )

    if merged is None:
        write_error(
            ErrorDetail(
                code="MC-4012",
                message="Failed to merge weights",
                detail="Invalid dimensions or empty input",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(merged))

    payload = {
        "mergedShape": [len(merged), len(merged[0]) if merged else 0],
        "outputFile": str(output) if output else None,
        "nextActions": [
            "mc geometry gromov-wasserstein to compute transport plan",
            "mc model merge for full model merging",
        ],
    }

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED MERGE",
            f"Source: {len(source)} x {len(source[0]) if source else 0}",
            f"Target: {len(target)} x {len(target[0]) if target else 0}",
            f"Merged: {len(merged)} x {len(merged[0]) if merged else 0}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("synthesize")
def geometry_transport_synthesize(
    ctx: typer.Context,
    source_act_file: Path = typer.Option(
        ..., "--source-act", help="JSON file with source activations"
    ),
    target_act_file: Path = typer.Option(
        ..., "--target-act", help="JSON file with target activations"
    ),
    source_weights_file: Path = typer.Option(
        ..., "--source-weights", help="JSON file with source weights"
    ),
    target_weights_file: Path = typer.Option(
        ..., "--target-weights", help="JSON file with target weights"
    ),
    output: Path | None = typer.Option(
        None, "--output-file", "-o", help="Output file for merged weights"
    ),
):
    """
    Compute GW transport plan and synthesize merged weights.

    All parameters (epsilon, threshold, alpha) are derived from the
    activation geometry - not user-specified.
    """
    context = _context(ctx)
    service = GeometryTransportService()

    source_act = json.loads(Path(source_act_file).read_text())
    target_act = json.loads(Path(target_act_file).read_text())
    source_weights = json.loads(Path(source_weights_file).read_text())
    target_weights = json.loads(Path(target_weights_file).read_text())

    # All parameters derived from geometry
    config = MergeConfig(
        coupling_threshold=None,  # Derived from coupling distribution
        blend_alpha=None,  # Derived from coupling strength
        gw_epsilon=None,  # Derived from distance scale
        gw_max_iterations=None,  # Derived from convergence behavior
    )

    result = service.synthesize_with_gw(
        source_activations=source_act,
        target_activations=target_act,
        source_weights=source_weights,
        target_weights=target_weights,
        config=config,
    )

    if result is None:
        write_error(
            ErrorDetail(
                code="MC-4013",
                message="Failed to synthesize with GW",
                detail="Insufficient samples or dimension mismatch",
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if output:
        Path(output).write_text(json.dumps(result.merged_weights))

    payload = service.merge_result_payload(result)
    payload["outputFile"] = str(output) if output else None
    payload["nextActions"] = [
        "mc geometry intrinsic-dimension to analyze merged space",
        "mc model merge for full model merging",
    ]

    if context.output_format == "text":
        lines = [
            "TRANSPORT-GUIDED SYNTHESIS",
            f"GW Distance: {result.gw_distance:.4f}",
            f"Marginal Error: {result.marginal_error:.4f}",
            f"Effective Rank: {result.effective_rank}",
            f"Converged: {'Yes' if result.converged else 'No'} ({result.iterations} iterations)",
            f"Merged Shape: {len(result.merged_weights)} x {len(result.merged_weights[0]) if result.merged_weights else 0}",
        ]
        if output:
            lines.append(f"Output: {output}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
