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

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from typer.core import TyperGroup

from modelcypher.cli.typer_compat import apply_typer_compat

apply_typer_compat()

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.commands import adapter as adapter_commands
from modelcypher.cli.commands import agent as agent_commands
from modelcypher.cli.commands import agent_eval as agent_eval_commands
from modelcypher.cli.commands import dashboard as dashboard_commands
from modelcypher.cli.commands import ensemble as ensemble_commands
from modelcypher.cli.commands import entropy as entropy_commands
from modelcypher.cli.commands import help_cmd as help_commands
from modelcypher.cli.commands import infer as infer_commands
from modelcypher.cli.commands import model as model_commands
from modelcypher.cli.commands import research as research_commands
from modelcypher.cli.commands import safety as safety_commands
from modelcypher.cli.commands import stability as stability_commands
from modelcypher.cli.commands import storage as storage_commands
from modelcypher.cli.commands import system as system_commands
from modelcypher.cli.commands import thermo as thermo_commands
from modelcypher.cli.commands.geometry import crm as geometry_crm_commands
from modelcypher.cli.commands.geometry import atlas as geometry_atlas_commands
from modelcypher.cli.commands.geometry import emotion as geometry_emotion_commands
from modelcypher.cli.commands.geometry import geom_adapter as geometry_adapter_commands
from modelcypher.cli.commands.geometry import interference as geometry_interference_commands
from modelcypher.cli.commands.geometry import invariant as geometry_invariant_commands
from modelcypher.cli.commands.geometry import manifold as geometry_manifold_commands
from modelcypher.cli.commands.geometry import merge_entropy as geometry_merge_entropy_commands
from modelcypher.cli.commands.geometry import metrics as geometry_metrics_commands
from modelcypher.cli.commands.geometry import moral as geometry_moral_commands
from modelcypher.cli.commands.geometry import path as geometry_path_commands
from modelcypher.cli.commands.geometry import persona as geometry_persona_commands
from modelcypher.cli.commands.geometry import primes as geometry_primes_commands
from modelcypher.cli.commands.geometry import refinement as geometry_refinement_commands
from modelcypher.cli.commands.geometry import refusal as geometry_refusal_commands
from modelcypher.cli.commands.geometry import safety as geometry_safety_commands
from modelcypher.cli.commands.geometry import social as geometry_social_commands
from modelcypher.cli.commands.geometry import sparse as geometry_sparse_commands
from modelcypher.cli.commands.geometry import spatial as geometry_spatial_commands
from modelcypher.cli.commands.geometry import stitch as geometry_stitch_commands
from modelcypher.cli.commands.geometry import temporal as geometry_temporal_commands
from modelcypher.cli.commands.geometry import training as geometry_training_commands
from modelcypher.cli.commands.geometry import transfer as geometry_transfer_cabe_commands
from modelcypher.cli.commands.geometry import transport as geometry_transport_commands
from modelcypher.cli.commands.geometry import waypoint as geometry_waypoint_commands
from modelcypher.cli.context import CLIContext, resolve_ai_mode, resolve_output_format
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json
from modelcypher.utils.logging import configure_logging

_GLOBAL_FLAGS_WITH_VALUES = {"--output", "--log-level", "--trace-id"}
_GLOBAL_FLAG_ALIASES = {
    "--ai",
    "--output",
    "--quiet",
    "--very-quiet",
    "--yes",
    "--no-prompt",
    "--pretty",
    "--log-level",
    "--trace-id",
}


def _hoist_global_flags(args: list[str]) -> list[str]:
    """Allow global flags to appear anywhere in the command.

    Click/Typer only parse group-level options *before* the subcommand token.
    ModelCypher-style usage places flags at the end (e.g. `mc inventory --output json`).

    This pre-parser moves known global flags (and their values) to the front so the
    Typer app callback can consume them, without requiring every subcommand to
    re-declare the same options.
    """

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
geometry_app = typer.Typer(no_args_is_help=True)

# Hidden dev group for diagnostic/internal commands
dev_app = typer.Typer(no_args_is_help=True, hidden=True)

app.add_typer(model_commands.app, name="model")
app.add_typer(system_commands.app, name="system")
app.add_typer(geometry_app, name="geometry")
geometry_app.add_typer(geometry_path_commands.app, name="path")
geometry_app.add_typer(geometry_training_commands.app, name="training")
geometry_app.add_typer(geometry_safety_commands.app, name="safety")
geometry_app.add_typer(geometry_adapter_commands.app, name="adapter")
geometry_app.add_typer(geometry_atlas_commands.app, name="atlas")
geometry_app.add_typer(geometry_primes_commands.app, name="primes")
geometry_app.add_typer(geometry_stitch_commands.app, name="stitch")
geometry_app.add_typer(geometry_crm_commands.app, name="crm")
geometry_app.add_typer(geometry_metrics_commands.app, name="metrics")
geometry_app.add_typer(geometry_sparse_commands.app, name="sparse")
geometry_app.add_typer(geometry_refusal_commands.app, name="refusal")
geometry_app.add_typer(geometry_persona_commands.app, name="persona")
geometry_app.add_typer(geometry_manifold_commands.app, name="manifold")
geometry_app.add_typer(geometry_transport_commands.app, name="transport")
geometry_app.add_typer(geometry_refinement_commands.app, name="refinement")
geometry_app.add_typer(geometry_invariant_commands.app, name="invariant")
geometry_app.add_typer(geometry_emotion_commands.app, name="emotion")
geometry_app.add_typer(geometry_merge_entropy_commands.app, name="merge-entropy")
geometry_app.add_typer(geometry_transfer_cabe_commands.app, name="transfer")
geometry_app.add_typer(geometry_spatial_commands.app, name="spatial")
geometry_app.add_typer(geometry_social_commands.app, name="social")
geometry_app.add_typer(geometry_temporal_commands.app, name="temporal")
geometry_app.add_typer(geometry_moral_commands.app, name="moral")
geometry_app.add_typer(geometry_waypoint_commands.app, name="waypoint")
geometry_app.add_typer(geometry_interference_commands.app, name="interference")
app.add_typer(entropy_commands.app, name="entropy")
app.add_typer(adapter_commands.adapter_app, name="adapter")
app.add_typer(thermo_commands.app, name="thermo")
app.add_typer(safety_commands.app, name="safety")
app.add_typer(agent_commands.app, name="agent")
app.add_typer(stability_commands.app, name="stability")
app.add_typer(dashboard_commands.app, name="dashboard")
app.add_typer(storage_commands.app, name="storage")
app.add_typer(ensemble_commands.app, name="ensemble")
app.add_typer(infer_commands.app, name="infer")
app.add_typer(help_commands.app, name="help")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    ai: bool | None = typer.Option(None, "--ai", help="AI mode: JSON output, no prompts"),
    output: str | None = typer.Option(None, "--output", help="Output format: json, yaml, text"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress info logs"),
    very_quiet: bool = typer.Option(False, "--very-quiet", help="Suppress all logs"),
    yes: bool = typer.Option(False, "--yes", help="Auto-confirm prompts"),
    no_prompt: bool = typer.Option(False, "--no-prompt", help="Fail if confirmation required"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print JSON output"),
    log_level: str = typer.Option(
        "info", "--log-level", help="Log level: trace, debug, info, warn, error"
    ),
    trace_id: str | None = typer.Option(None, "--trace-id", help="Trace ID"),
) -> None:
    ai_mode = resolve_ai_mode(ai)
    output_format = resolve_output_format(ai_mode, output)
    quiet_mode = very_quiet or quiet or ai_mode
    effective_log_level = "error" if very_quiet else log_level
    configure_logging(effective_log_level, quiet=quiet_mode)

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


@app.command("inventory")
def inventory(ctx: typer.Context) -> None:
    context = _context(ctx)
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory

    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)
    service = factory.inventory_service()
    write_output(service.inventory(), context.output_format, context.pretty)


@app.command("explain")
def explain(ctx: typer.Context, command: str = typer.Argument(...)) -> None:
    context = _context(ctx)
    from modelcypher.core.use_cases.help_service import HelpService

    service = HelpService()
    payload = service.explain(command)
    write_output(payload, context.output_format, context.pretty)


@geometry_app.command("validate")
def geometry_validate(
    ctx: typer.Context,
    include_fixtures: bool = typer.Option(False, "--include-fixtures"),
    file: str | None = typer.Option(None, "--file"),
) -> None:
    context = _context(ctx)
    embedder = EmbeddingDefaults.make_default_embedder()
    service = GeometryService(embedder=embedder)
    report = service.validate(include_fixtures=include_fixtures)
    payload = service.validation_payload(
        report,
        include_schema=context.output_format in {"json", "yaml"},
    )
    if file:
        Path(file).write_text(dump_json(payload, pretty=context.pretty), encoding="utf-8")

    if context.output_format == "text":
        status = "PASS" if report.passed else "FAIL"
        lines = [
            f"Geometry validation: {status}",
            f"GW distance (perm): {report.gromov_wasserstein.distance_permutation:.6f}",
            f"GW symmetry delta: {report.gromov_wasserstein.symmetry_delta:.6f}",
            f"Traversal coherence (self): {report.traversal_coherence.self_correlation:.5f}",
            f"Path signature similarity: {report.path_signature.signature_similarity:.5f}",
            f"Frechet distance: {report.path_signature.frechet_distance:.6f}",
        ]
        if report.fixtures is not None:
            lines.append("Fixtures: included")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("infer")
def infer(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model"),
    prompt: str = typer.Option(..., "--prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.95, "--top-p"),
    scan: bool = typer.Option(False, "--scan", help="Run security scan on output"),
) -> None:
    context = _context(ctx)
    engine = LocalInferenceEngine()

    # Use the more capable 'run' method
    from dataclasses import asdict

    result = engine.run(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        security_scan=scan,
    )

    # Convert dataclass to dict for output
    payload = asdict(result)

    # Flatten security info for easier reading if present
    if result.security:
        payload["hasSecurityFlags"] = result.security.has_security_flags
        payload["maxAnomalyScore"] = result.security.max_anomaly_score
        payload["securityAnomalies"] = result.security.anomaly_count

    write_output(payload, context.output_format, context.pretty)


# Agent-eval commands (extracted to commands/agent_eval.py)
app.add_typer(agent_eval_commands.app, name="agent-eval")


# Research commands (all commands in research_commands.app)
app.add_typer(research_commands.app, name="research")
