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

"""Geodesic comparison CLI command.

Experimental. Compares geodesic trajectories across categorized prompt sets.

    mc analyze geodesic-compare --model PATH --prompts FILE [--layer N]
"""

from __future__ import annotations

import json

from ._common import (
    ErrorDetail,
    Path,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


@app.command("geodesic-compare")
def geodesic_compare(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", help="Path to model directory"
    ),
    prompts: str = typer.Option(
        ..., "--prompts", help="JSON file with categorized prompts: {category: [prompt, ...]}"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer to analyze (default: last layer)"
    ),
    annotate: bool = typer.Option(
        False, "--annotate/--no-annotate",
        help="Include per-token labels (slower, more output)",
    ),
) -> None:
    """Compare geodesic trajectories across prompt categories.

    Experimental. Runs geodesic analysis on each prompt in a categorized
    JSON file and aggregates metrics per category.

    Prompt file format:
        {"cot_reasoning": ["prompt1", ...], "simple_narrative": ["prompt2", ...]}

    Example:
        mc analyze geodesic-compare --model /path/to/model --prompts data/probes/geodesic_comparison.json
    """
    context = get_context(ctx)
    model_path = Path(model)
    prompts_path = Path(prompts)

    if not model_path.exists():
        error = ErrorDetail(
            code="MC-8001",
            title="Model not found",
            detail=f"Path does not exist: {model_path}",
            hint="Provide a valid path to a model directory",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if not prompts_path.exists():
        error = ErrorDetail(
            code="MC-8002",
            title="Prompts file not found",
            detail=f"Path does not exist: {prompts_path}",
            hint="Provide a JSON file with categorized prompts",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    with open(prompts_path) as f:
        categorized_prompts = json.load(f)

    from modelcypher.cli.composition import (
        get_activation_provider,
        get_backend,
    )
    from modelcypher.adapters.model_loader import ModelLoader
    from modelcypher.core.use_cases.geodesic_trajectory_service import (
        GeodesicTrajectoryService,
    )

    backend = get_backend()
    loader = ModelLoader(backend)
    model_obj, tokenizer_obj = loader.load_model(str(model_path))

    service = GeodesicTrajectoryService(
        backend=backend,
        activation_provider=get_activation_provider(),
    )

    result = service.measure_batch(
        model=model_obj,
        tokenizer=tokenizer_obj,
        categorized_prompts=categorized_prompts,
        model_path=str(model_path),
        target_layer=layer,
        annotate_tokens=annotate,
    )

    write_output(result.to_dict(), context.output_format, context.pretty)
