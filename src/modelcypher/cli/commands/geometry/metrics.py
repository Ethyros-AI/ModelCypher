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

"""Geometry metrics CLI commands.

Provides commands for geometric analysis of model representations
with no user-configurable parameters. Inputs are point clouds only.

Commands:
    mc geometry metrics gromov-wasserstein <source_file> <target_file>
    mc geometry metrics intrinsic-dimension <points_file>
    mc geometry metrics effective-rank <points_file>
    mc geometry metrics topological-fingerprint <points_file>
    mc geometry metrics spectral-signature <points_file>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import (
    validate_file_exists,
    validate_json_file,
    validate_model_path,
)
from modelcypher.cli.commands.geometry.helpers import (
    extract_anchor_activations,
    resolve_model_backbone,
)
from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

app = typer.Typer(no_args_is_help=True)

def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _load_prompts(prompts_file: str, context: CLIContext) -> list[str]:
    prompts_path = validate_file_exists(
        prompts_file,
        description="Prompts file",
        context=context,
    )
    content = prompts_path.read_text(encoding="utf-8")
    try:
        prompt_data = json.loads(content)
        if not isinstance(prompt_data, list) or not all(
            isinstance(prompt, str) for prompt in prompt_data
        ):
            raise typer.BadParameter("Prompts file must contain a JSON array of strings")
        prompts = [prompt.strip() for prompt in prompt_data if prompt.strip()]
    except json.JSONDecodeError:
        prompts = [line.strip() for line in content.splitlines() if line.strip()]

    if not prompts:
        raise typer.BadParameter("Prompts file is empty")

    return prompts


@app.command("gromov-wasserstein")
def geometry_metrics_gromov_wasserstein(
    ctx: typer.Context,
    source_file: str = typer.Argument(
        ..., help="Path to source point cloud (JSON array of arrays)"
    ),
    target_file: str = typer.Argument(
        ..., help="Path to target point cloud (JSON array of arrays)"
    ),
) -> None:
    """Compute Gromov-Wasserstein distance between two point clouds."""
    context = _context(ctx)

    # Validate inputs early for clear error messages
    source_points = validate_json_file(
        source_file, description="Source point cloud", context=context
    )
    target_points = validate_json_file(
        target_file, description="Target point cloud", context=context
    )

    service = GeometryMetricsService()
    result = service.compute_gromov_wasserstein(
        source_points=source_points,
        target_points=target_points,
    )

    payload = service.gromov_wasserstein_payload(result)
    payload["_schema"] = "mc.geometry.gromov_wasserstein.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("intrinsic-dimension")
def geometry_metrics_intrinsic_dimension(
    ctx: typer.Context,
    points_file: str = typer.Argument(
        ..., help="Path to point cloud (JSON array of arrays or activations dict)"
    ),
) -> None:
    """Estimate intrinsic dimension of a point cloud using TwoNN."""
    context = _context(ctx)

    # Validate input early for clear error messages
    raw_points = validate_json_file(
        points_file, description="Point cloud", context=context
    )
    if isinstance(raw_points, dict):
        points = [raw_points[key] for key in sorted(raw_points.keys())]
    else:
        points = raw_points

    service = GeometryMetricsService()
    result = service.estimate_intrinsic_dimension(points=points)

    payload = service.intrinsic_dimension_payload(result)
    payload["_schema"] = "mc.geometry.intrinsic_dimension.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("effective-rank")
def geometry_metrics_effective_rank(
    ctx: typer.Context,
    points_file: str | None = typer.Argument(
        None, help="Path to point cloud (JSON array of arrays or activations dict)"
    ),
    model: str | None = typer.Option(
        None, "--model", help="Path to model directory for activation probing"
    ),
    prompts: str | None = typer.Option(
        None, "--prompts", help="Path to prompts file (JSON array or newline-separated)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer index (defaults to middle layer)"
    ),
) -> None:
    """Compute effective rank (Renyi/Shannon) of a point cloud or activations."""
    context = _context(ctx)

    if points_file and (model or prompts):
        raise typer.BadParameter("Provide either points_file or --model/--prompts, not both.")
    if points_file is None and (model is None or prompts is None):
        raise typer.BadParameter("Provide points_file or both --model and --prompts.")

    service = GeometryMetricsService()
    payload_extra: dict[str, object] = {}

    if points_file:
        raw_points = validate_json_file(
            points_file, description="Point cloud", context=context
        )
        if isinstance(raw_points, dict):
            points = [raw_points[key] for key in sorted(raw_points.keys())]
        else:
            points = raw_points
        result = service.compute_effective_rank(points=points)
    else:
        validate_model_path(model, context=context)
        prompt_list = _load_prompts(prompts, context)

        from modelcypher.adapters.model_loader import load_model_for_training
        from modelcypher.core.domain._backend import get_default_backend

        model_obj, tokenizer = load_model_for_training(model)
        backbone = resolve_model_backbone(model_obj, getattr(model_obj, "model_type", None))
        if backbone is None:
            raise typer.BadParameter("Failed to resolve model backbone.")
        embed_tokens, layers, norm = backbone
        num_layers = len(layers)

        if layer is None:
            layer_idx = num_layers // 2
        else:
            if layer < 0 or layer >= num_layers:
                raise typer.BadParameter(
                    f"Layer {layer} out of range for model with {num_layers} layers."
                )
            layer_idx = layer

        class PromptAnchor:
            def __init__(self, name: str, prompt: str) -> None:
                self.name = name
                self.prompt = prompt

        anchors = [
            PromptAnchor(f"prompt_{idx}", prompt)
            for idx, prompt in enumerate(prompt_list)
        ]

        backend = get_default_backend()
        activations = extract_anchor_activations(
            anchors=anchors,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            target_layer=layer_idx,
            backend=backend,
            prompt_attr="prompt",
            name_attr="name",
        )
        if not activations:
            raise typer.BadParameter("Activation extraction returned no samples.")

        names = list(activations.keys())
        vectors = [activations[name] for name in names]
        matrix = backend.stack(vectors, axis=0)
        backend.eval(matrix)
        result = service.compute_effective_rank(points=matrix)

        payload_extra = {
            "modelPath": model,
            "layer": layer_idx,
            "promptCount": len(prompt_list),
            "activationCount": len(activations),
        }

    payload = service.effective_rank_payload(result)
    payload.update(payload_extra)
    payload["_schema"] = "mc.geometry.effective_rank.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("topological-fingerprint")
def geometry_metrics_topological_fingerprint(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
) -> None:
    """Compute topological fingerprint using persistent homology."""
    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_topological_fingerprint(points=points)

    context = _context(ctx)
    payload = service.topological_fingerprint_payload(result)
    payload["_schema"] = "mc.geometry.topological_fingerprint.v1"
    write_output(payload, context.output_format, context.pretty)


@app.command("spectral-signature")
def geometry_metrics_spectral_signature(
    ctx: typer.Context,
    points_file: str = typer.Argument(..., help="Path to point cloud (JSON array of arrays)"),
) -> None:
    """Compute spectral signature of a point cloud."""
    points = json.loads(Path(points_file).read_text())

    service = GeometryMetricsService()
    result = service.compute_spectral_signature(points=points)

    context = _context(ctx)
    payload = service.spectral_signature_payload(result)
    payload["_schema"] = "mc.geometry.spectral_signature.v1"
    write_output(payload, context.output_format, context.pretty)
