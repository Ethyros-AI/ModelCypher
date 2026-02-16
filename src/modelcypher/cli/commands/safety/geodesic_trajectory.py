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

"""Geodesic trajectory CLI command.

Experimental. Measures geodesic deviation of activation trajectories.

    mc analyze geodesic-trajectory --model PATH --prompt TEXT [--layer N]
"""

from __future__ import annotations

from ._common import (
    ErrorDetail,
    Path,
    get_context,
    typer,
    write_error,
    write_output,
)

app = typer.Typer(no_args_is_help=True)


@app.command("geodesic-trajectory")
def geodesic_trajectory(
    ctx: typer.Context,
    model: str = typer.Option(
        ..., "--model", help="Path to model directory"
    ),
    prompt: str = typer.Option(
        ..., "--prompt", help="Text to analyze (prompt or prompt+response)"
    ),
    layer: int | None = typer.Option(
        None, "--layer", help="Layer to analyze (default: last layer)"
    ),
) -> None:
    """Measure geodesic deviation of activation trajectories.

    Experimental. Collects per-token hidden states and measures whether
    the trajectory follows geodesics on the activation manifold.

    Output:
        mean_deviation: 0.0 = straight line, >0 = curved path
        path_length_ratio: 1.0 = straight, >1.0 = winding
        intrinsic_dimension: manifold dimensionality of the trajectory

    Example:
        mc analyze geodesic-trajectory --model /path/to/model --prompt "What is 2+2? Let me think step by step."
    """
    context = get_context(ctx)
    model_path = Path(model)

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

    from modelcypher.cli.composition import (
        get_activation_provider,
        get_backend,
        get_model_loader,
    )
    from modelcypher.core.use_cases.geodesic_trajectory_service import (
        GeodesicTrajectoryService,
    )

    loader = get_model_loader()
    model_obj, tokenizer_obj = loader.load_model(str(model_path))

    service = GeodesicTrajectoryService(
        backend=get_backend(),
        activation_provider=get_activation_provider(),
    )

    result = service.measure(
        model=model_obj,
        tokenizer=tokenizer_obj,
        text=prompt,
        target_layer=layer,
    )

    write_output(result.to_dict(), context.output_format, context.pretty)
