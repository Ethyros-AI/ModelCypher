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

"""
CLI commands for Domain Geometry Waypoints.

Provides commands for:
- Computing model geometry profiles
- Pre-merge geometry audit
- Post-merge geometry validation
- Domain strength profiles
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(help="Domain geometry waypoint commands for merge guidance")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("profile")
def waypoint_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
    domains: str | None = typer.Option(
        None,
        "--domains",
        help="Comma-separated domains to analyze (spatial,social,temporal,moral). Default: all",
    ),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save profile to file"
    ),
) -> None:
    """
    Compute geometry profile for a model across validated domains.

    Tests all four validated geometric hypotheses:
    - Spatial: 3D world model (Euclidean, gravity, occlusion)
    - Social: Power hierarchies, kinship, formality
    - Temporal: Direction, duration, causality
    - Moral: Valence, agency, scope (Haidt foundations)
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service
    from modelcypher.core.domain.domains import AtlasDomain, resolve_domain

    typer.echo(f"Computing geometry profile for {model_path}...")

    service = get_domain_geometry_waypoint_service()

    # Parse domains
    domain_list = None
    if domains:
        supported = {
            AtlasDomain.SPATIAL,
            AtlasDomain.SOCIAL,
            AtlasDomain.TEMPORAL,
            AtlasDomain.MORAL,
        }
        domain_list = []
        for raw in domains.split(","):
            name = raw.strip()
            if not name:
                continue
            resolved = resolve_domain(name)
            if resolved is None:
                typer.echo(
                    f"Invalid domain: {name}. Valid: spatial, social, temporal, moral",
                    err=True,
                )
                raise typer.Exit(1)
            if resolved not in supported:
                typer.echo(
                    f"Unsupported domain for waypoint analysis: {resolved.value}. "
                    "Valid: spatial, social, temporal, moral",
                    err=True,
                )
                raise typer.Exit(1)
            domain_list.append(resolved)

    try:
        profile = service.compute_profile(model_path, layer, domain_list)
    except Exception as e:
        typer.echo(f"Error computing profile: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.waypoint.profile.v1",
        **profile.to_dict(),
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Profile saved to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 70,
            f"GEOMETRY PROFILE: {Path(model_path).name}",
            "=" * 70,
            "",
            f"Layer Analyzed: {layer if layer != -1 else 'last'}",
            f"Total Anchors: {profile.total_anchors}",
            f"Mean Manifold Score: {profile.mean_manifold_score:.4f}",
            "",
            "-" * 50,
            "Per-Domain Scores:",
        ]
        for domain, score in profile.domain_scores.items():
            lines.append(
                f"  {domain.value.upper():<10} MMS={score.manifold_score:.3f} "
                f"Ortho={score.axis_orthogonality:.2f} "
                f"Grad={score.gradient_consistency:.2f} "
                f"Anchors={score.anchors_probed}"
            )
        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("audit")
def waypoint_audit(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    target_path: str = typer.Argument(..., help="Path to target model"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
    output_file: str | None = typer.Option(None, "--output-file", "-o", help="Save audit to file"),
) -> None:
    """
    Pre-merge geometry audit comparing source and target models.

    Computes geometry deltas and strength ratio variance before merging.
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service

    typer.echo("Auditing geometry alignment...")
    typer.echo(f"  Source: {source_path}")
    typer.echo(f"  Target: {target_path}")

    service = get_domain_geometry_waypoint_service()

    try:
        audit = service.pre_merge_audit(source_path, target_path, layer)
    except Exception as e:
        typer.echo(f"Error during audit: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.waypoint.audit.v1",
        **audit.to_dict(),
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Audit saved to {output_file}")

    if context.output_format == "text":
        strength_profile = service.compute_domain_strength_profile(audit)
        lines = [
            "=" * 70,
            "PRE-MERGE GEOMETRY AUDIT",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Target: {Path(target_path).name}",
            f"Strength Ratio Variance: {audit.strength_ratio_variance:.4f}",
            "",
            "-" * 50,
            "Domain Deltas:",
        ]

        for delta in audit.domain_deltas:
            lines.append(
                f"  {delta.domain.value:<10}: source={delta.source_score:.3f} "
                f"target={delta.target_score:.3f} delta={delta.delta:.3f}"
            )

        lines.append("")
        lines.append("Target Strength Ratio by Domain:")
        for domain, ratio in strength_profile.items():
            lines.append(f"  {domain.value:<10}: ratio={ratio:.2f}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("validate")
def waypoint_validate(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    merged_path: str = typer.Argument(..., help="Path to merged model"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Save validation to file"
    ),
) -> None:
    """
    Post-merge geometry validation.

    Compares source and merged model geometry to detect degradation.
    Run this AFTER merging to verify geometry preservation.
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service

    typer.echo("Validating geometry preservation...")
    typer.echo(f"  Source: {source_path}")
    typer.echo(f"  Merged: {merged_path}")

    service = get_domain_geometry_waypoint_service()

    try:
        validation = service.post_merge_validate(source_path, merged_path, layer)
    except Exception as e:
        typer.echo(f"Error during validation: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.waypoint.validate.v1",
        **validation.to_dict(),
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))
        typer.echo(f"Validation saved to {output_file}")

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "POST-MERGE GEOMETRY VALIDATION",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Merged: {Path(merged_path).name}",
            f"Overall Preservation: {validation.overall_preservation:.1%}",
            "",
            "-" * 50,
            "Preservation by Domain:",
        ]

        for domain, preservation in validation.preservation_by_domain.items():
            lines.append(f"  {domain.value:<10}: {preservation:.1%}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("strength-profile")
def waypoint_strength_profile(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    target_path: str = typer.Argument(..., help="Path to target model"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
) -> None:
    """
    Compute domain strength profile for merging.

    Runs geometry audit and computes per-domain strength ratios derived
    entirely from the geometric relationship between source and target.
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service

    typer.echo("Computing domain strength profile...")

    service = get_domain_geometry_waypoint_service()

    try:
        audit = service.pre_merge_audit(source_path, target_path, layer)
        strength_profile = service.compute_domain_strength_profile(audit)
    except Exception as e:
        typer.echo(f"Error computing strength profile: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.waypoint.strength_profile.v1",
        "sourceModel": source_path,
        "targetModel": target_path,
        "strengthProfile": {d.value: a for d, a in strength_profile.items()},
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "DOMAIN STRENGTH PROFILE",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Target: {Path(target_path).name}",
            "",
            "-" * 50,
            "Target Strength Ratio by Domain:",
        ]
        for domain, ratio in strength_profile.items():
            lines.append(f"  {domain.value:<10}: ratio={ratio:.3f}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
