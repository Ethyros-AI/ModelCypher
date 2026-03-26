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

"""ModelCypher CLI - workflow-first commands with expert surfaces.

mc train    - Train LoRA adapters
mc merge    - Geometric model merging
mc infer    - Run inference
mc analyze  - Workflow-first observation and expert metrics
mc model    - Model registry
mc system   - System status
mc adapter  - LoRA adapter analysis
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Respect NO_COLOR convention (https://no-color.org/)
if os.environ.get("NO_COLOR"):
    os.environ.setdefault("TERM", "dumb")

import logging
import sys

logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import typer
from typer.core import TyperGroup

from modelcypher.cli.typer_compat import apply_typer_compat

apply_typer_compat()

from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories

register_default_atlas_inventories()

# The 9 commands
from modelcypher.cli.commands import analyze as analyze_commands
from modelcypher.cli.commands import adapter as adapter_commands
from modelcypher.cli.commands import data as data_commands
from modelcypher.cli.commands import infer as infer_commands
from modelcypher.cli.commands import merge as merge_commands
from modelcypher.cli.commands import model as model_commands
from modelcypher.cli.commands import quantize as quantize_commands
from modelcypher.cli.commands import system as system_commands
from modelcypher.cli.commands import train as train_commands
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.warnings import warn_trust_remote_code
from modelcypher.utils.logging import configure_logging

_GLOBAL_FLAGS_WITH_VALUES = {"--log-level", "--trace-id"}
# --output is special: only hoist when the value is a known format string,
# to avoid stealing --output /path/to/file from subcommands.
_OUTPUT_FORMAT_VALUES = {"json", "yaml", "text"}
_GLOBAL_FLAG_ALIASES = {
    "--ai",
    "--text",
    "-j", "--json",
    "-q", "--quiet",
    "-qq", "--very-quiet",
    "-y", "--yes",
    "--no-prompt",
    "-p", "--pretty",
    "--log-level",
    "--trace-id",
    "--version", "-V",
}


def _hoist_global_flags(args: list[str]) -> list[str]:
    """Allow global flags to appear anywhere in the command."""
    extracted: list[str] = []
    remaining: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            remaining.extend(args[i:])
            break

        if any(arg.startswith(f"{flag}=") for flag in _GLOBAL_FLAGS_WITH_VALUES):
            extracted.append(arg)
            i += 1
            continue

        # --output is only a global flag when its value is a known format
        if arg.startswith("--output="):
            val = arg.split("=", 1)[1]
            if val in _OUTPUT_FORMAT_VALUES:
                extracted.append(arg)
                i += 1
                continue
            # Not a format → leave for subcommand
            remaining.append(arg)
            i += 1
            continue

        if arg == "--output":
            next_val = args[i + 1] if i + 1 < len(args) else ""
            if next_val in _OUTPUT_FORMAT_VALUES:
                extracted.append(arg)
                extracted.append(next_val)
                i += 2
                continue
            # Not a format → leave for subcommand
            remaining.append(arg)
            i += 1
            continue

        if arg in _GLOBAL_FLAGS_WITH_VALUES:
            extracted.append(arg)
            if i + 1 < len(args):
                extracted.append(args[i + 1])
                i += 2
            else:
                i += 1
            continue

        if arg in _GLOBAL_FLAG_ALIASES:
            extracted.append(arg)
            i += 1
            continue

        remaining.append(arg)
        i += 1

    return extracted + remaining


class _GlobalOptionsTyperGroup(TyperGroup):
    def parse_args(self, ctx, args: list[str]) -> list[str]:
        return super().parse_args(ctx, _hoist_global_flags(args))


app = typer.Typer(no_args_is_help=True, add_completion=False, cls=_GlobalOptionsTyperGroup)

# Register commands
app.add_typer(train_commands.train_app, name="train", help="Train LoRA adapters")
app.add_typer(data_commands.data_app, name="data", help="Data preparation and validation")
app.add_typer(merge_commands.merge_app, name="merge", help="Geometric model merging")
app.add_typer(infer_commands.app, name="infer", help="Run inference")
app.add_typer(
    analyze_commands.app,
    name="analyze",
    help="Workflow-first model observation and expert metrics",
)
app.add_typer(model_commands.app, name="model", help="Model operations: info, capacity, quantize (bf16→4bit), moe-profile")
app.add_typer(system_commands.app, name="system", help="System status")
app.add_typer(adapter_commands.app, name="adapter", help="LoRA adapter analysis")
app.add_typer(quantize_commands.quantize_app, name="quantize", help="Post-quantization Tikhonov spectral correction (not for initial quantization; use: mc model quantize)")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version as pkg_version

        try:
            ver = pkg_version("modelcypher")
        except Exception:
            ver = "unknown"
        typer.echo(ver)
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit",
    ),
    ai: bool | None = typer.Option(
        None, "--ai", help="AI mode: force JSON output, suppress prompts/logs"
    ),
    output: str | None = typer.Option(
        None, "--output", help="Output format: json, yaml, text"
    ),
    text_output: bool = typer.Option(False, "--text", help="Shorthand for --output text"),
    json_output: bool = typer.Option(False, "-j", "--json", help="Shorthand for --output json"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info logs"),
    very_quiet: bool = typer.Option(False, "--very-quiet", "-qq", help="Suppress all logs"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm prompts"),
    no_prompt: bool = typer.Option(False, "--no-prompt", help="Fail if confirmation required"),
    pretty: bool = typer.Option(False, "--pretty", "-p", help="Pretty print output"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Trace ID"),
) -> None:
    effective_output = output
    if text_output:
        effective_output = "text"
    elif json_output:
        effective_output = "json"

    ai_mode = resolve_ai_mode(ai)
    output_format = resolve_output_format(ai_mode, effective_output)
    quiet_mode = very_quiet or quiet or ai_mode

    if very_quiet:
        effective_log_level = "error"
    elif quiet:
        effective_log_level = "warn"
    else:
        effective_log_level = log_level
    configure_logging(effective_log_level, quiet=quiet_mode)

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        ctx.obj = CLIContext(
            ai_mode=ai_mode,
            output_format=output_format,
            quiet=quiet,
            very_quiet=very_quiet,
            yes=yes or ai_mode,
            no_prompt=no_prompt or ai_mode,
            pretty=pretty,
            log_level=effective_log_level,
            trace_id=trace_id,
        )
        return

    from modelcypher.backends import detect_default_backend_type, get_backend
    from modelcypher.core.domain._backend import set_default_backend

    set_default_backend(get_backend(detect_default_backend_type()))

    ctx.obj = CLIContext(
        ai_mode=ai_mode,
        output_format=output_format,
        quiet=quiet,
        very_quiet=very_quiet,
        yes=yes or ai_mode,
        no_prompt=no_prompt or ai_mode,
        pretty=pretty,
        log_level=effective_log_level,
        trace_id=trace_id,
    )
    warn_trust_remote_code(ctx.obj)
