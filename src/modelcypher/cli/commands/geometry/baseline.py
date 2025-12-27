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
CLI commands for Domain Geometry Baselines.

Provides commands for:
- Extracting geometry baselines from reference models
- Validating models against established baselines
- Comparing model geometry profiles
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output

app = typer.Typer(help="Domain geometry baseline extraction and validation")
logger = logging.getLogger(__name__)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("extract")
def baseline_extract(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
    domain: str = typer.Option(
        "spatial",
        "--domain",
        "-d",
        help="Domain to extract (spatial, social, temporal, moral)",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", help="Layer to analyze (-1 for last)"
    ),
    k_neighbors: int = typer.Option(
        10, "--k-neighbors", "-k", help="k for k-NN graph in Ollivier-Ricci"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Directory to save baseline (default: builtin)"
    ),
) -> None:
    """
    Extract geometry baseline from a reference model.

    Uses Ollivier-Ricci curvature and domain-specific analyzers to create
    an empirical baseline for healthy LLM geometry. Baselines are used for:

    - Validating model health (negative Ricci = healthy, positive = collapsed)
    - Pre/post merge geometry checks
    - Cross-model geometry comparison

    Example:
        mc geometry baseline extract /path/to/Qwen2.5-0.5B --domain spatial
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.domain_geometry_baselines import (
        BaselineRepository,
        DomainGeometryBaselineExtractor,
    )

    valid_domains = ["spatial", "social", "temporal", "moral"]
    if domain.lower() not in valid_domains:
        typer.echo(f"Invalid domain: {domain}. Valid: {', '.join(valid_domains)}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Extracting {domain} geometry baseline from {model_path}...")
    typer.echo(f"  Layer: {layer if layer != -1 else 'last'}")
    typer.echo(f"  k-neighbors: {k_neighbors}")

    try:
        extractor = DomainGeometryBaselineExtractor()
        baseline = extractor.extract_baseline(
            model_path=model_path,
            domain=domain.lower(),
            layers=[layer] if layer != -1 else None,
            k_neighbors=k_neighbors,
        )
    except Exception as e:
        typer.echo(f"Error extracting baseline: {e}", err=True)
        raise typer.Exit(1)

    # Save baseline
    try:
        repo = BaselineRepository(baseline_dir=output_dir)
        saved_path = repo.save_baseline(baseline)
        typer.echo(f"Baseline saved to: {saved_path}")
    except Exception as e:
        typer.echo(f"Error saving baseline: {e}", err=True)
        raise typer.Exit(1)

    payload = {
        "_schema": "mc.geometry.baseline.extract.v1",
        "domain": baseline.domain,
        "model_family": baseline.model_family,
        "model_size": baseline.model_size,
        "ollivier_ricci_mean": baseline.ollivier_ricci_mean,
        "ollivier_ricci_std": baseline.ollivier_ricci_std,
        "manifold_health_distribution": baseline.manifold_health_distribution,
        "intrinsic_dimension_mean": baseline.intrinsic_dimension_mean,
        "domain_metrics": baseline.domain_metrics,
        "saved_path": str(saved_path),
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            f"BASELINE EXTRACTED: {domain.upper()}",
            "=" * 70,
            "",
            f"Model: {baseline.model_family}-{baseline.model_size}",
            f"Ollivier-Ricci Mean: {baseline.ollivier_ricci_mean:.4f}",
            f"Ollivier-Ricci Std: {baseline.ollivier_ricci_std:.4f}",
            f"Intrinsic Dimension: {baseline.intrinsic_dimension_mean:.1f}",
            "",
            "-" * 50,
            "Manifold Health Distribution:",
        ]
        for health, pct in baseline.manifold_health_distribution.items():
            lines.append(f"  {health:<12}: {pct:.1%}")
        lines.append("")
        lines.append(f"Saved to: {saved_path}")
        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("validate")
def baseline_validate(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model to validate"),
    domains: str | None = typer.Option(
        None,
        "--domains",
        "-d",
        help="Comma-separated domains (spatial,social,temporal,moral). Default: all",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", help="Layer to analyze (-1 for last)"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail on any geometry deviation"
    ),
) -> None:
    """
    Validate model geometry against established baselines.

    Compares model's Ollivier-Ricci curvature and domain metrics against
    known-good baselines. Useful for:

    - Post-merge validation (did the merge preserve geometry?)
    - Model health checks (is the model collapsed?)
    - Regression testing after fine-tuning

    Example:
        mc geometry baseline validate /path/to/merged-model --domains spatial,social
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.domain_geometry_validator import (
        DomainGeometryValidator,
    )

    domain_list = None
    if domains:
        domain_list = [d.strip().lower() for d in domains.split(",")]
        valid_domains = ["spatial", "social", "temporal", "moral"]
        for d in domain_list:
            if d not in valid_domains:
                typer.echo(f"Invalid domain: {d}. Valid: {', '.join(valid_domains)}", err=True)
                raise typer.Exit(1)

    typer.echo(f"Validating geometry for {model_path}...")

    try:
        validator = DomainGeometryValidator()
        results = validator.validate_model(
            model_path=model_path,
            domains=domain_list,
            layer=layer,
        )
    except Exception as e:
        typer.echo(f"Error validating model: {e}", err=True)
        raise typer.Exit(1)

    # Check overall pass/fail
    all_passed = all(r.passed for r in results)

    payload = {
        "_schema": "mc.geometry.baseline.validate.v1",
        "model_path": model_path,
        "overall_passed": all_passed,
        "results": [
            {
                "domain": r.domain,
                "passed": r.passed,
                "deviation_scores": r.deviation_scores,
                "warnings": r.warnings,
                "recommendations": r.recommendations,
            }
            for r in results
        ],
    }

    if context.output_format == "text":
        lines = [
            "=" * 70,
            "GEOMETRY VALIDATION RESULTS",
            "=" * 70,
            "",
            f"Model: {Path(model_path).name}",
            f"Overall: {'PASSED' if all_passed else 'FAILED'}",
            "",
        ]
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            lines.append("-" * 50)
            lines.append(f"{result.domain.upper()}: {status}")

            if result.deviation_scores:
                lines.append("  Deviations:")
                for metric, dev in result.deviation_scores.items():
                    flag = "" if dev < 0.2 else " [HIGH]"
                    lines.append(f"    {metric}: {dev:.1%}{flag}")

            if result.warnings:
                lines.append("  Warnings:")
                for warn in result.warnings:
                    lines.append(f"    - {warn}")

            if result.recommendations:
                lines.append("  Recommendations:")
                for rec in result.recommendations:
                    lines.append(f"    - {rec}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)

        if strict and not all_passed:
            raise typer.Exit(1)
        return

    write_output(payload, context.output_format, context.pretty)

    if strict and not all_passed:
        raise typer.Exit(1)


@app.command("compare")
def baseline_compare(
    ctx: typer.Context,
    model1_path: str = typer.Argument(..., help="Path to first model"),
    model2_path: str = typer.Argument(..., help="Path to second model"),
    domain: str = typer.Option(
        "spatial",
        "--domain",
        "-d",
        help="Domain to compare (spatial, social, temporal, moral)",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", help="Layer to analyze (-1 for last)"
    ),
) -> None:
    """
    Compare geometry profiles of two models.

    Extracts baselines from both models and computes divergence metrics.
    Useful for:

    - Pre-merge compatibility assessment
    - Model family similarity analysis
    - Fine-tuning impact measurement

    Example:
        mc geometry baseline compare /path/to/model1 /path/to/model2 --domain spatial
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.domain_geometry_baselines import (
        DomainGeometryBaselineExtractor,
    )

    valid_domains = ["spatial", "social", "temporal", "moral"]
    if domain.lower() not in valid_domains:
        typer.echo(f"Invalid domain: {domain}. Valid: {', '.join(valid_domains)}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Comparing {domain} geometry...")
    typer.echo(f"  Model 1: {model1_path}")
    typer.echo(f"  Model 2: {model2_path}")

    try:
        extractor = DomainGeometryBaselineExtractor()

        typer.echo("Extracting baseline from model 1...")
        baseline1 = extractor.extract_baseline(
            model_path=model1_path,
            domain=domain.lower(),
            layers=[layer] if layer != -1 else None,
        )

        typer.echo("Extracting baseline from model 2...")
        baseline2 = extractor.extract_baseline(
            model_path=model2_path,
            domain=domain.lower(),
            layers=[layer] if layer != -1 else None,
        )
    except Exception as e:
        typer.echo(f"Error extracting baselines: {e}", err=True)
        raise typer.Exit(1)

    # Compute divergence
    ricci_divergence = abs(baseline1.ollivier_ricci_mean - baseline2.ollivier_ricci_mean)
    id_divergence = abs(baseline1.intrinsic_dimension_mean - baseline2.intrinsic_dimension_mean)

    # Compute domain metric divergence
    domain_divergence = {}
    common_metrics = set(baseline1.domain_metrics.keys()) & set(baseline2.domain_metrics.keys())
    for metric in common_metrics:
        v1 = baseline1.domain_metrics[metric]
        v2 = baseline2.domain_metrics[metric]
        domain_divergence[metric] = abs(v1 - v2)

    payload = {
        "_schema": "mc.geometry.baseline.compare.v1",
        "domain": domain,
        "model1": {
            "path": model1_path,
            "family": baseline1.model_family,
            "size": baseline1.model_size,
            "ollivier_ricci_mean": baseline1.ollivier_ricci_mean,
            "intrinsic_dimension": baseline1.intrinsic_dimension_mean,
        },
        "model2": {
            "path": model2_path,
            "family": baseline2.model_family,
            "size": baseline2.model_size,
            "ollivier_ricci_mean": baseline2.ollivier_ricci_mean,
            "intrinsic_dimension": baseline2.intrinsic_dimension_mean,
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
            f"GEOMETRY COMPARISON: {domain.upper()}",
            "=" * 70,
            "",
            f"Model 1: {baseline1.model_family}-{baseline1.model_size}",
            f"  Ollivier-Ricci: {baseline1.ollivier_ricci_mean:.4f}",
            f"  Intrinsic Dim:  {baseline1.intrinsic_dimension_mean:.1f}",
            "",
            f"Model 2: {baseline2.model_family}-{baseline2.model_size}",
            f"  Ollivier-Ricci: {baseline2.ollivier_ricci_mean:.4f}",
            f"  Intrinsic Dim:  {baseline2.intrinsic_dimension_mean:.1f}",
            "",
            "-" * 50,
            "DIVERGENCE:",
            f"  Ricci Curvature: {ricci_divergence:.4f}",
            f"  Intrinsic Dimension: {id_divergence:.1f}",
        ]

        if domain_divergence:
            lines.append("")
            lines.append("Domain Metrics:")
            for metric, div in domain_divergence.items():
                lines.append(f"  {metric}: {div:.4f}")

        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("list")
def baseline_list(
    ctx: typer.Context,
    domain: str | None = typer.Option(
        None,
        "--domain",
        "-d",
        help="Filter by domain (spatial, social, temporal, moral)",
    ),
) -> None:
    """
    List available geometry baselines.

    Shows all extracted baselines stored in the repository.
    """
    context = _context(ctx)

    from modelcypher.core.domain.geometry.domain_geometry_baselines import (
        BaselineRepository,
    )

    repo = BaselineRepository()
    if domain:
        baselines = repo.get_baselines_for_domain(domain)
    else:
        baselines = repo.get_all_baselines()

    payload = {
        "_schema": "mc.geometry.baseline.list.v1",
        "baselines": [
            {
                "domain": b.domain,
                "model_family": b.model_family,
                "model_size": b.model_size,
                "ollivier_ricci_mean": b.ollivier_ricci_mean,
                "extraction_date": b.extraction_date,
            }
            for b in baselines
        ],
    }

    if context.output_format == "text":
        if not baselines:
            typer.echo("No baselines found.")
            return

        lines = [
            "=" * 70,
            "AVAILABLE BASELINES",
            "=" * 70,
            "",
            f"{'Domain':<10} {'Family':<10} {'Size':<8} {'Ricci':<10} {'Date'}",
            "-" * 70,
        ]
        for b in baselines:
            lines.append(
                f"{b.domain:<10} {b.model_family:<10} {b.model_size:<8} "
                f"{b.ollivier_ricci_mean:+.4f}   {b.extraction_date}"
            )
        lines.append("")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


__all__ = ["app"]
