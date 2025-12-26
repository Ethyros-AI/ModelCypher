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
- Domain-aware alpha recommendations
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
    from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
        GeometryDomain,
    )

    typer.echo(f"Computing geometry profile for {model_path}...")

    service = get_domain_geometry_waypoint_service()

    # Parse domains
    domain_list = None
    if domains:
        domain_list = []
        for d in domains.split(","):
            try:
                domain_list.append(GeometryDomain(d.strip().lower()))
            except ValueError:
                typer.echo(
                    f"Invalid domain: {d}. Valid: spatial, social, temporal, moral", err=True
                )
                raise typer.Exit(1)

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
            f"Strongest Domain: {profile.strongest_domain.value if profile.strongest_domain else 'N/A'}",
            f"Weakest Domain: {profile.weakest_domain.value if profile.weakest_domain else 'N/A'}",
            "",
            "-" * 50,
            "Per-Domain Scores:",
        ]
        for domain, score in profile.domain_scores.items():
            status = "YES" if score.has_manifold else "NO"
            lines.append(
                f"  {domain.value.upper():<10} MMS={score.manifold_score:.3f} "
                f"Ortho={score.axis_orthogonality:.2f} "
                f"Manifold={status}"
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

    Identifies geometry conflict zones and recommends domain-aware alpha values.
    Run this BEFORE merging to understand geometric compatibility.
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service

    typer.echo("Auditing geometry compatibility...")
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
        lines = [
            "=" * 70,
            "PRE-MERGE GEOMETRY AUDIT",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Target: {Path(target_path).name}",
            f"Overall Compatibility: {audit.overall_compatibility:.1%}",
            "",
            "-" * 50,
            f"VERDICT: {audit.audit_verdict}",
            "-" * 50,
            "",
        ]

        if audit.conflict_zones:
            lines.append("Conflict Zones:")
            for zone in audit.conflict_zones:
                lines.append(f"  [{zone.severity.upper()}] {zone.domain.value}")
                lines.append(
                    f"    Source: {zone.source_score:.3f} → Target: {zone.target_score:.3f}"
                )
                lines.append(f"    Δ = {zone.delta:.3f}")
                lines.append(f"    {zone.recommendation}")
            lines.append("")

        lines.append("Recommended Alpha by Domain:")
        for domain, alpha in audit.recommended_alpha_by_domain.items():
            lines.append(f"  {domain.value:<10}: α = {alpha:.2f}")

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
            f"Status: {validation.validation_status.upper()}",
            "",
            "-" * 50,
            "Preservation by Domain:",
        ]

        for domain, preservation in validation.preservation_by_domain.items():
            status = ""
            if domain in validation.degraded_domains:
                status = " [DEGRADED]"
            elif domain in validation.enhanced_domains:
                status = " [ENHANCED]"
            lines.append(f"  {domain.value:<10}: {preservation:.1%}{status}")

        lines.append("")
        lines.append("Recommendations:")
        for rec in validation.recommendations:
            lines.append(f"  • {rec}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("alpha-profile")
def waypoint_alpha_profile(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model"),
    target_path: str = typer.Argument(..., help="Path to target model"),
    base_alpha: float = typer.Option(0.5, "--base-alpha", help="Base alpha value"),
    strength: float = typer.Option(0.5, "--strength", help="Domain adjustment strength (0-1)"),
    layer: int = typer.Option(-1, "--layer", help="Layer to analyze (-1 for last)"),
) -> None:
    """
    Compute domain-aware alpha profile for merging.

    Runs geometry audit and computes recommended alpha values for each domain
    based on geometric compatibility between source and target.
    """
    context = _context(ctx)

    from modelcypher.cli.composition import get_domain_geometry_waypoint_service

    typer.echo("Computing domain-aware alpha profile...")

    service = get_domain_geometry_waypoint_service()

    try:
        audit = service.pre_merge_audit(source_path, target_path, layer)
        alpha_profile = service.compute_domain_alpha_profile(audit, base_alpha, strength)
    except Exception as e:
        typer.echo(f"Error computing alpha profile: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.waypoint.alpha_profile.v1",
        "sourceModel": source_path,
        "targetModel": target_path,
        "baseAlpha": base_alpha,
        "strength": strength,
        "compatibility": audit.overall_compatibility,
        "alphaProfile": {d.value: a for d, a in alpha_profile.items()},
        "verdict": audit.audit_verdict,
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "DOMAIN-AWARE ALPHA PROFILE",
            "=" * 70,
            "",
            f"Source: {Path(source_path).name}",
            f"Target: {Path(target_path).name}",
            f"Base α: {base_alpha:.2f}  Strength: {strength:.2f}",
            f"Compatibility: {audit.overall_compatibility:.1%}",
            "",
            "-" * 50,
            "Recommended Alpha by Domain:",
        ]
        for domain, alpha in alpha_profile.items():
            delta = alpha - base_alpha
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            lines.append(f"  {domain.value:<10}: α = {alpha:.2f}  {direction} ({delta:+.2f})")

        lines.append("")
        lines.append(f"Verdict: {audit.audit_verdict}")
        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
