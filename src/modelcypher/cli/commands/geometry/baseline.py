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
CLI commands for Model Geometry Profiles.

Provides commands for:
- Extracting geometry profiles from models
- Comparing model geometry profiles
- Listing stored profiles
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(help="Model geometry profile extraction and comparison")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("extract")
def profile_extract(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    domains: str | None = typer.Option(
        None,
        "--domains",
        "-d",
        help="Comma-separated domains for domain-specific metrics (spatial,social,temporal,moral)",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", help="Layer to analyze (-1 for sampled layers)"
    ),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", "-k", help="k for k-NN graph in Ollivier-Ricci"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Directory to save profile (default: ~/.modelcypher/profiles)"
    ),
) -> None:
    """
    Extract geometry profile from a model.

    Uses Ollivier-Ricci curvature and intrinsic dimension to create
    a geometry profile. Profiles are used for:

    - Model comparison and alignment assessment
    - Pre/post merge geometry checks
    - Cross-model geometry analysis

    Example:
        mc geometry baseline extract /path/to/Qwen2.5-0.5B
        mc geometry baseline extract /path/to/model --domains spatial,social
    """
    context = _context(ctx)

    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain.geometry.model_profile import (
        ModelProfileExtractor,
        ProfileRepository,
    )

    domain_list = None
    if domains:
        domain_list = [d.strip().lower() for d in domains.split(",")]
        valid_domains = ["spatial", "social", "temporal", "moral"]
        for d in domain_list:
            if d not in valid_domains:
                typer.echo(f"Invalid domain: {d}. Valid: {', '.join(valid_domains)}", err=True)
                raise typer.Exit(1)

    typer.echo(f"Extracting geometry profile from {model_path}...")
    typer.echo(f"  k-neighbors: {k_neighbors}")
    if domain_list:
        typer.echo(f"  Domains: {', '.join(domain_list)}")

    try:
        model_loader = MLXModelLoader()
        extractor = ModelProfileExtractor(model_loader=model_loader)
        profile = extractor.extract_profile(
            model_path=model_path,
            layers=[layer] if layer != -1 else None,
            k_neighbors=k_neighbors,
            domains=domain_list,
        )
    except Exception as e:
        typer.echo(f"Error extracting profile: {e}", err=True)
        raise typer.Exit(1)

    # Save profile
    try:
        repo = ProfileRepository(profile_dir=output_dir)
        saved_path = repo.save_profile(profile)
        typer.echo(f"Profile saved to: {saved_path}")
    except Exception as e:
        typer.echo(f"Error saving profile: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.profile.extract.v1",
        "model_family": profile.model_family,
        "model_path": profile.model_path,
        "global_ollivier_ricci_mean": profile.global_ollivier_ricci_mean,
        "global_ollivier_ricci_std": profile.global_ollivier_ricci_std,
        "global_intrinsic_dimension_mean": profile.global_intrinsic_dimension_mean,
        "layers_analyzed": len(profile.layer_profiles),
        "domain_metrics": profile.domain_metrics,
        "saved_path": str(saved_path),
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "PROFILE EXTRACTED",
            "=" * 70,
            "",
            f"Model: {profile.model_family}",
            f"Path: {profile.model_path}",
            f"Layers analyzed: {len(profile.layer_profiles)}",
            "",
            f"Ollivier-Ricci Mean: {profile.global_ollivier_ricci_mean:.4f}",
            f"Ollivier-Ricci Std: {profile.global_ollivier_ricci_std:.4f}",
            f"Intrinsic Dimension: {profile.global_intrinsic_dimension_mean:.1f}",
            "",
            f"Saved to: {saved_path}",
            "",
        ]
        if profile.domain_metrics:
            lines.append("Domain Metrics:")
            for domain, metrics in profile.domain_metrics.items():
                lines.append(f"  {domain}:")
                for metric, value in metrics.items():
                    lines.append(f"    {metric}: {value:.4f}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def profile_compare(
    ctx: typer.Context,
    model1_path: str = typer.Argument(..., help="Path to first model"),
    model2_path: str = typer.Argument(..., help="Path to second model"),
    domains: str | None = typer.Option(
        None,
        "--domains",
        "-d",
        help="Comma-separated domains to include (spatial,social,temporal,moral)",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", help="Layer to analyze (-1 for sampled layers)"
    ),
) -> None:
    """
    Compare geometry profiles of two models.

    Extracts profiles from both models and computes divergence metrics.
    Useful for:

    - Pre-merge alignment assessment
    - Model family similarity analysis
    - Fine-tuning impact measurement

    Example:
        mc geometry baseline compare /path/to/model1 /path/to/model2
    """
    context = _context(ctx)

    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.domain.geometry.model_profile import ModelProfileExtractor

    domain_list = None
    if domains:
        domain_list = [d.strip().lower() for d in domains.split(",")]

    typer.echo("Comparing model geometry...")
    typer.echo(f"  Model 1: {model1_path}")
    typer.echo(f"  Model 2: {model2_path}")

    try:
        model_loader = MLXModelLoader()
        extractor = ModelProfileExtractor(model_loader=model_loader)

        typer.echo("Extracting profile from model 1...")
        profile1 = extractor.extract_profile(
            model_path=model1_path,
            layers=[layer] if layer != -1 else None,
            domains=domain_list,
        )

        typer.echo("Extracting profile from model 2...")
        profile2 = extractor.extract_profile(
            model_path=model2_path,
            layers=[layer] if layer != -1 else None,
            domains=domain_list,
        )
    except Exception as e:
        typer.echo(f"Error extracting profiles: {e}", err=True)
        raise typer.Exit(1)

    # Compute divergence
    ricci_divergence = abs(profile1.global_ollivier_ricci_mean - profile2.global_ollivier_ricci_mean)
    id_divergence = abs(profile1.global_intrinsic_dimension_mean - profile2.global_intrinsic_dimension_mean)

    # Compute domain metric divergence
    domain_divergence: dict[str, dict[str, float]] = {}
    for domain in set(profile1.domain_metrics.keys()) | set(profile2.domain_metrics.keys()):
        metrics1 = profile1.domain_metrics.get(domain, {})
        metrics2 = profile2.domain_metrics.get(domain, {})
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if common_metrics:
            domain_divergence[domain] = {
                m: abs(metrics1[m] - metrics2[m]) for m in common_metrics
            }

    payload = {
        "_schema": "mc.geometry.profile.compare.v1",
        "model1": {
            "path": model1_path,
            "family": profile1.model_family,
            "ollivier_ricci_mean": profile1.global_ollivier_ricci_mean,
            "intrinsic_dimension": profile1.global_intrinsic_dimension_mean,
        },
        "model2": {
            "path": model2_path,
            "family": profile2.model_family,
            "ollivier_ricci_mean": profile2.global_ollivier_ricci_mean,
            "intrinsic_dimension": profile2.global_intrinsic_dimension_mean,
        },
        "divergence": {
            "ollivier_ricci": ricci_divergence,
            "intrinsic_dimension": id_divergence,
            "domain_metrics": domain_divergence,
        },
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "GEOMETRY COMPARISON",
            "=" * 70,
            "",
            f"Model 1: {profile1.model_family}",
            f"  Ollivier-Ricci: {profile1.global_ollivier_ricci_mean:.4f}",
            f"  Intrinsic Dim:  {profile1.global_intrinsic_dimension_mean:.1f}",
            "",
            f"Model 2: {profile2.model_family}",
            f"  Ollivier-Ricci: {profile2.global_ollivier_ricci_mean:.4f}",
            f"  Intrinsic Dim:  {profile2.global_intrinsic_dimension_mean:.1f}",
            "",
            "-" * 50,
            "DIVERGENCE:",
            f"  Ricci Curvature: {ricci_divergence:.4f}",
            f"  Intrinsic Dimension: {id_divergence:.1f}",
        ]

        if domain_divergence:
            lines.append("")
            lines.append("Domain Metrics:")
            for domain, metrics in domain_divergence.items():
                lines.append(f"  {domain}:")
                for metric, div in metrics.items():
                    lines.append(f"    {metric}: {div:.4f}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("list")
def profile_list(
    ctx: typer.Context,
    family: str | None = typer.Option(
        None,
        "--family",
        "-f",
        help="Filter by model family (qwen, llama, mistral, etc.)",
    ),
) -> None:
    """
    List available geometry profiles.

    Shows all extracted profiles stored in the repository.
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.model_profile import ProfileRepository

    repo = ProfileRepository()
    if family:
        profiles = repo.get_profiles_for_family(family)
    else:
        profiles = repo.get_all_profiles()

    payload = {
        "_schema": "mc.geometry.profile.list.v1",
        "profiles": [
            {
                "model_family": p.model_family,
                "model_path": p.model_path,
                "ollivier_ricci_mean": p.global_ollivier_ricci_mean,
                "intrinsic_dimension": p.global_intrinsic_dimension_mean,
                "computed_at": p.computed_at,
                "layers_analyzed": len(p.layer_profiles),
            }
            for p in profiles
        ],
    }

    if context.output_format == "text":
        if not profiles:
            typer.echo("No profiles found.")
            return

        lines = [
            "=" * 70,
            "AVAILABLE PROFILES",
            "=" * 70,
            "",
            f"{'Family':<10} {'Ricci':<10} {'ID':<8} {'Layers':<8} {'Date'}",
            "-" * 70,
        ]
        for p in profiles:
            lines.append(
                f"{p.model_family:<10} {p.global_ollivier_ricci_mean:+.4f}   "
                f"{p.global_intrinsic_dimension_mean:<8.1f} {len(p.layer_profiles):<8} "
                f"{p.computed_at[:10] if p.computed_at else 'n/a'}"
            )
        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


# Remove the old validate command - validation is now just comparison against profiles
# If validation is needed, use `mc geometry baseline compare`


__all__ = ["app"]
