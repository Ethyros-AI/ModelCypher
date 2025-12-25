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

"""Storage CLI commands."""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("status")
def storage_status(ctx: typer.Context) -> None:
    """Return storage usage breakdown by category."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_storage_service

    service = get_storage_service()
    snapshot = service.compute_snapshot()
    usage = snapshot.usage
    disk = snapshot.disk

    payload = {
        "totalGb": usage.total_gb,
        "modelsGb": usage.models_gb,
        "checkpointsGb": usage.checkpoints_gb,
        "otherGb": usage.other_gb,
        "disk": {
            "totalBytes": disk.total_bytes,
            "freeBytes": disk.free_bytes,
        },
    }

    if context.output_format == "text":
        lines = [
            "STORAGE STATUS",
            f"Total Disk: {usage.total_gb:.2f} GB",
            f"Free Disk: {disk.free_bytes / (1024**3):.2f} GB",
            "",
            "Usage Breakdown:",
            f"  Models: {usage.models_gb:.2f} GB",
            f"  Checkpoints: {usage.checkpoints_gb:.2f} GB",
            f"  Other: {usage.other_gb:.2f} GB",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("usage")
def storage_usage(ctx: typer.Context) -> None:
    """Alias for storage status to match MCP naming."""
    storage_status(ctx)


@app.command("cleanup")
def storage_cleanup(
    ctx: typer.Context,
    targets: list[str] = typer.Option(..., "--target", help="Cleanup targets: caches"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview cleanup without deleting"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
) -> None:
    """Remove old artifacts and return freed space."""
    context = _context(ctx)
    from modelcypher.cli.composition import get_storage_service

    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        targets_str = ", ".join(targets)
        if not typer.confirm(f"Clean up {targets_str}? This cannot be undone."):
            raise typer.Exit(code=1)

    service = get_storage_service()
    before_snapshot = service.compute_snapshot()

    if dry_run:
        payload = {
            "dryRun": True,
            "targets": targets,
            "message": "Dry run - no files deleted",
        }
        write_output(payload, context.output_format, context.pretty)
        return

    try:
        cleared = service.cleanup(targets)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1018",
            title="Storage cleanup failed",
            detail=str(exc),
            hint="Valid targets are: caches",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    after_snapshot = service.compute_snapshot()
    freed_bytes = max(0, after_snapshot.disk.free_bytes - before_snapshot.disk.free_bytes)

    payload = {
        "freedBytes": freed_bytes,
        "freedGb": freed_bytes / (1024**3),
        "categoriesCleaned": cleared,
    }

    if context.output_format == "text":
        lines = [
            "STORAGE CLEANUP",
            f"Categories cleaned: {', '.join(cleared)}",
            f"Space freed: {freed_bytes / (1024**3):.2f} GB",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
