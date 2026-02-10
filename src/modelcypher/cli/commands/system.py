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

"""System CLI commands.

Provides commands for:
- System status and health checks
- System probing
- Cache benchmarking
- Merge cache testing

Commands:
    mc system status
    mc system probe <target>
    mc system benchmark cache
    mc system test-cache [model_path]
"""

from __future__ import annotations

import typer

from modelcypher.cli.composition import (
    get_system_cache_service,
    get_system_service,
)
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(no_args_is_help=True)
benchmark_app = typer.Typer(no_args_is_help=True, help="Benchmark system components")
app.add_typer(benchmark_app, name="benchmark")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def system_status(
    ctx: typer.Context,
    require_backend: str | None = typer.Option(None, "--require-backend"),
) -> None:
    """Get system status.

    Examples:
        mc system status
        mc system status --require-backend <backend-key>
    """
    context = _context(ctx)
    service = get_system_service()
    status = service.status()
    if require_backend:
        backends = status.get("backends", [])
        match = next(
            (backend for backend in backends if backend.get("key") == require_backend),
            None,
        )
        if not match or not match.get("available"):
            raise typer.Exit(code=3)
    write_output(status, context.output_format, context.pretty)


@app.command("probe")
def system_probe(ctx: typer.Context, target: str = typer.Argument(...)) -> None:
    """Probe a system target.

    Examples:
        mc system probe backends
        mc system probe memory
        mc system probe <backend-key>
    """
    context = _context(ctx)
    service = get_system_service()
    write_output(service.probe(target), context.output_format, context.pretty)


@benchmark_app.command("cache")
def benchmark_cache(ctx: typer.Context) -> None:
    """Benchmark computation cache performance.

    Runs benchmarks for CKA, geodesic distances, and Fréchet mean
    operations to measure cache speedup.

    Examples:
        mc system benchmark cache
        mc system benchmark cache --output json
    """
    context = _context(ctx)
    service = get_system_cache_service()
    result = service.benchmark_cache()
    write_output(result.to_dict(), context.output_format, context.pretty)


@app.command("test-cache")
def test_cache(
    ctx: typer.Context,
    model_path: str | None = typer.Argument(
        None, help="Path to model for real weight testing"
    ),
    n_pairs: int = typer.Option(5, "--pairs", "-p", help="Number of layer pairs to test"),
) -> None:
    """Test computation cache with real or synthetic model weights.

    If a model path is provided, uses real model weights for testing.
    Otherwise, falls back to synthetic data.

    Examples:
        mc system test-cache
        mc system test-cache /path/to/model
        mc system test-cache /path/to/model --pairs 10
    """
    from modelcypher.core.use_cases.system_cache_service import CacheProbeRequest

    context = _context(ctx)
    service = get_system_cache_service()
    request = CacheProbeRequest(model_path=model_path, n_pairs=n_pairs)
    result = service.test_cache(request)
    write_output(result.to_dict(), context.output_format, context.pretty)
