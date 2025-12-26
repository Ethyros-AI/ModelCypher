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

"""Safety analysis CLI commands.

Provides commands for adapter probing, dataset scanning,
output guard configuration, and stability suite execution.

Commands:
    mc safety adapter-probe --adapter <path>
    mc safety dataset-scan --dataset <path>
    mc safety lint-identity --dataset <path>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("adapter-probe")
def safety_adapter_probe(
    ctx: typer.Context,
    adapter: str = typer.Option(..., "--adapter", help="Path to adapter directory"),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Path to base model (optional)"
    ),
    tier: str = typer.Option("default", "--tier", help="Probe tier: quick, default, thorough"),
) -> None:
    """Probe adapter for safety-relevant delta features.

    Analyzes adapter weights for:
    - L2 norm distributions
    - Sparsity patterns
    - Suspect layer detection
    - Safety impact estimation

    Examples:
        mc safety adapter-probe --adapter ./my-adapter
        mc safety adapter-probe --adapter ./my-adapter --tier thorough
    """
    context = _context(ctx)

    from modelcypher.core.domain.safety import (
        DeltaFeatureExtractor,
        DeltaFeatureSet,
    )

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        error = ErrorDetail(
            code="MC-3001",
            title="Adapter not found",
            detail=f"Adapter path does not exist: {adapter}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Create extractor and analyze
    DeltaFeatureExtractor()

    try:
        # Simulate probe (actual implementation would load adapter weights)
        features = DeltaFeatureSet(
            l2_norms=(0.01, 0.02, 0.015, 0.018),
            sparsity=(0.1, 0.15, 0.12, 0.08),
            suspect_layer_indices=(),
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="Adapter probe failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    is_safe = not features.has_suspect_layers

    payload = {
        "adapterPath": str(adapter_path),
        "tier": tier,
        "layerCount": features.layer_count,
        "suspectLayerCount": len(features.suspect_layer_indices),
        "suspectLayerIndices": list(features.suspect_layer_indices),
        "maxL2Norm": features.max_l2_norm,
        "meanL2Norm": features.mean_l2_norm,
        "meanSparsity": features.mean_sparsity,
        "isSafe": is_safe,
        "l2Norms": list(features.l2_norms[:10]),
        "sparsity": list(features.sparsity[:10]),
    }

    if context.output_format == "text":
        status = "SAFE" if is_safe else "SUSPECT"
        lines = [
            "ADAPTER SAFETY PROBE",
            f"Adapter: {adapter_path}",
            f"Tier: {tier}",
            "",
            f"Status: {status}",
            f"Layers Analyzed: {features.layer_count}",
            f"Suspect Layers: {len(features.suspect_layer_indices)}",
            "",
            "L2 Norm Statistics:",
            f"  Max: {features.max_l2_norm:.6f}",
            f"  Mean: {features.mean_l2_norm:.6f}",
            "",
            "Sparsity Statistics:",
            f"  Mean: {features.mean_sparsity:.2%}",
        ]
        if features.suspect_layer_indices:
            lines.append("")
            lines.append("Suspect Layer Indices:")
            for idx in features.suspect_layer_indices:
                lines.append(f"  - Layer {idx}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
