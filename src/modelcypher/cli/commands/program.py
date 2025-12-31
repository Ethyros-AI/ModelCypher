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

"""Program management CLI commands.

Provides commands for:
- Executing multi-donor transplant programs
- Managing program status and checkpoints
- Comparing results across programs

Commands:
    mc program run <config>
    mc program status <program_id>
    mc program list
    mc program compare <programs...>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output

app = typer.Typer(
    name="program",
    help="Manage and execute multi-donor transplant programs.",
    no_args_is_help=True,
)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _get_multi_donor_service():
    """Get MultiDonorMergeService with proper dependency injection."""
    from modelcypher.adapters.mlx_model_loader import MLXModelLoader
    from modelcypher.core.use_cases.multi_donor_merge import MultiDonorMergeService

    model_loader = MLXModelLoader()
    return MultiDonorMergeService(model_loader=model_loader)


@app.command("run")
def program_run(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to program YAML/JSON config file",
            exists=True,
            readable=True,
        ),
    ],
    parallel: bool = typer.Option(
        False,
        "--parallel",
        "-p",
        help="Process base models in parallel",
    ),
    max_workers: int = typer.Option(
        2,
        "--max-workers",
        help="Max parallel workers (requires --parallel)",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Resume from last checkpoint if program exists",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Validate program without execution",
    ),
    base_filter: str | None = typer.Option(
        None,
        "--base",
        "-b",
        help="Only process specific base model (by ID)",
    ),
) -> None:
    """Execute a multi-donor transplant program.

    Sequentially transplants knowledge from multiple donors into base model(s).
    Supports checkpointing for resumability and parallel execution across bases.

    Examples:
        mc program run ./configs/program_a.yaml
        mc program run ./configs/program_a.yaml --parallel
        mc program run ./configs/program_a.yaml --resume
        mc program run ./configs/program_a.yaml --dry-run
        mc program run ./configs/program_a.yaml --base qwen3-8b
    """
    context = _context(ctx)

    try:
        from modelcypher.core.use_cases.multi_donor_merge import TransplantProgram

        # Load program config
        program = TransplantProgram.from_yaml(config_path)

        # Filter to specific base if requested
        if base_filter:
            matching_bases = tuple(b for b in program.bases if b.id == base_filter)
            if not matching_bases:
                write_error(
                    f"Base '{base_filter}' not found in program. "
                    f"Available: {', '.join(b.id for b in program.bases)}",
                    context.output_format,
                )
                raise typer.Exit(1)
            # Create filtered program
            program = TransplantProgram(
                name=program.name,
                description=program.description,
                bases=matching_bases,
                donors=program.donors,
                evaluation=program.evaluation,
                output_dir=program.output_dir,
            )

        service = _get_multi_donor_service()

        if dry_run:
            # Validation only
            write_output(
                {
                    "status": "valid",
                    "program": program.name,
                    "bases": len(program.bases),
                    "donors": len(program.donors),
                    "base_ids": [b.id for b in program.bases],
                    "donor_ids": [d.id for d in program.donors],
                },
                context.output_format,
                context.pretty,
            )
            return

        # Execute program
        result = service.execute_program(
            program=program,
            parallel=parallel,
            max_workers=max_workers,
            dry_run=False,
        )

        # Output result
        write_output(result.to_dict(), context.output_format, context.pretty)

    except FileNotFoundError as e:
        write_error(str(e), context.output_format)
        raise typer.Exit(1) from e
    except ValueError as e:
        write_error(f"Invalid program config: {e}", context.output_format)
        raise typer.Exit(1) from e
    except Exception as e:
        write_error(f"Program execution failed: {e}", context.output_format)
        raise typer.Exit(1) from e


@app.command("status")
def program_status(
    ctx: typer.Context,
    program_id: Annotated[
        str,
        typer.Argument(help="Program ID to check status"),
    ],
) -> None:
    """Get status of a running or completed program.

    Examples:
        mc program status abc123
    """
    context = _context(ctx)

    try:
        service = _get_multi_donor_service()
        status = service.get_program_status(program_id)

        write_output(status.to_dict(), context.output_format, context.pretty)

    except FileNotFoundError:
        write_error(f"Program '{program_id}' not found", context.output_format)
        raise typer.Exit(1)
    except Exception as e:
        write_error(f"Failed to get status: {e}", context.output_format)
        raise typer.Exit(1) from e


@app.command("list")
def program_list(ctx: typer.Context) -> None:
    """List all programs (running, completed, failed).

    Examples:
        mc program list
    """
    context = _context(ctx)

    try:
        service = _get_multi_donor_service()
        programs = service.list_programs()

        output = [
            {
                "program_id": p.program_id,
                "program_name": p.program_name,
                "status": p.status,
                "started_at": p.started_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
            }
            for p in programs
        ]

        write_output(output, context.output_format, context.pretty)

    except Exception as e:
        write_error(f"Failed to list programs: {e}", context.output_format)
        raise typer.Exit(1) from e


@app.command("show")
def program_show(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to program YAML/JSON config file",
            exists=True,
            readable=True,
        ),
    ],
) -> None:
    """Show details of a program configuration.

    Examples:
        mc program show ./configs/program_a.yaml
    """
    context = _context(ctx)

    try:
        from modelcypher.core.use_cases.multi_donor_merge import TransplantProgram

        program = TransplantProgram.from_yaml(config_path)

        output = {
            "name": program.name,
            "description": program.description,
            "bases": [
                {
                    "id": b.id,
                    "source": b.source,
                    "alias": b.effective_alias,
                }
                for b in program.bases
            ],
            "donors": [
                {
                    "id": d.id,
                    "source": d.source,
                    "domains": list(d.domains),
                    "priority": d.priority,
                    "layers": list(d.layers) if d.layers else None,
                }
                for d in program.donors
            ],
            "evaluation": program.evaluation.to_dict(),
            "output_dir": program.output_dir,
        }

        write_output(output, context.output_format, context.pretty)

    except FileNotFoundError as e:
        write_error(str(e), context.output_format)
        raise typer.Exit(1) from e
    except ValueError as e:
        write_error(f"Invalid program config: {e}", context.output_format)
        raise typer.Exit(1) from e


@app.command("compare")
def program_compare(
    ctx: typer.Context,
    programs: Annotated[
        list[str],
        typer.Argument(
            help="Program results to compare (format: LABEL:PATH or just PATH)"
        ),
    ],
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        "-o",
        help="Save comparison to JSON file",
    ),
    output_md: Path | None = typer.Option(
        None,
        "--output-md",
        help="Save comparison to Markdown file",
    ),
) -> None:
    """Compare results from multiple programs.

    Examples:
        mc program compare A:./out-A B:./out-B C:./out-C
        mc program compare ./out-A ./out-B --output-json comparison.json
    """
    context = _context(ctx)

    try:
        # Parse program specifications
        program_specs = []
        for spec in programs:
            if ":" in spec:
                label, path = spec.split(":", 1)
            else:
                label = Path(spec).name
                path = spec

            # Load result from path
            result_path = Path(path) / "result.json"
            if not result_path.exists():
                write_error(f"Result not found: {result_path}", context.output_format)
                raise typer.Exit(1)

            with result_path.open() as f:
                result = json.load(f)

            program_specs.append({"label": label, "path": path, "result": result})

        # Build comparison
        comparison = _build_comparison(program_specs)

        # Output to file if requested
        if output_json:
            with output_json.open("w") as f:
                json.dump(comparison, f, indent=2)
            sys.stderr.write(f"Comparison saved to {output_json}\n")

        if output_md:
            md_content = _format_comparison_markdown(comparison)
            output_md.write_text(md_content)
            sys.stderr.write(f"Comparison saved to {output_md}\n")

        write_output(comparison, context.output_format, context.pretty)

    except Exception as e:
        write_error(f"Comparison failed: {e}", context.output_format)
        raise typer.Exit(1) from e


def _build_comparison(program_specs: list[dict]) -> dict:
    """Build comparison structure from program results."""
    comparison = {
        "_schema": "mc.comparison.programs.v1",
        "programs": {},
        "metrics": {},
        "rankings": {},
    }

    for spec in program_specs:
        label = spec["label"]
        result = spec["result"]

        comparison["programs"][label] = {
            "program_name": result.get("program_name", label),
            "status": result.get("status", "unknown"),
            "total_duration_seconds": result.get("total_duration_seconds", 0),
        }

        # Extract metrics from base results
        base_results = result.get("base_results", [])
        for base in base_results:
            base_key = f"{label}:{base.get('base_alias', 'unknown')}"
            comparison["metrics"][base_key] = {
                "total_cka_improvement": base.get("total_cka_improvement", 0),
                "mean_boundary_preserved": base.get("mean_boundary_preserved", 0),
                "total_donors_applied": base.get("total_donors_applied", 0),
            }

    # Compute rankings
    if comparison["metrics"]:
        # Rank by CKA improvement
        sorted_by_cka = sorted(
            comparison["metrics"].items(),
            key=lambda x: x[1].get("total_cka_improvement", 0),
            reverse=True,
        )
        comparison["rankings"]["by_cka_improvement"] = [k for k, _ in sorted_by_cka]

        # Rank by boundary preservation
        sorted_by_boundary = sorted(
            comparison["metrics"].items(),
            key=lambda x: x[1].get("mean_boundary_preserved", 0),
            reverse=True,
        )
        comparison["rankings"]["by_boundary_preservation"] = [
            k for k, _ in sorted_by_boundary
        ]

    return comparison


def _format_comparison_markdown(comparison: dict) -> str:
    """Format comparison as Markdown."""
    lines = ["# Program Comparison Report", ""]

    # Programs summary
    lines.append("## Programs")
    lines.append("")
    lines.append("| Label | Name | Status | Duration |")
    lines.append("|-------|------|--------|----------|")
    for label, info in comparison.get("programs", {}).items():
        duration = info.get("total_duration_seconds", 0)
        duration_str = f"{duration:.1f}s" if duration < 3600 else f"{duration/3600:.1f}h"
        lines.append(
            f"| {label} | {info.get('program_name', '')} | "
            f"{info.get('status', '')} | {duration_str} |"
        )
    lines.append("")

    # Metrics table
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Program:Base | CKA Improvement | Boundary Preserved | Donors Applied |")
    lines.append("|--------------|-----------------|-------------------|----------------|")
    for key, metrics in comparison.get("metrics", {}).items():
        lines.append(
            f"| {key} | {metrics.get('total_cka_improvement', 0):.4f} | "
            f"{metrics.get('mean_boundary_preserved', 0):.2%} | "
            f"{metrics.get('total_donors_applied', 0)} |"
        )
    lines.append("")

    # Rankings
    if comparison.get("rankings"):
        lines.append("## Rankings")
        lines.append("")
        for metric, ranking in comparison["rankings"].items():
            lines.append(f"### {metric.replace('_', ' ').title()}")
            for i, item in enumerate(ranking, 1):
                lines.append(f"{i}. {item}")
            lines.append("")

    return "\n".join(lines)


@app.command("generate")
def program_generate(
    ctx: typer.Context,
    target_profile: Annotated[
        Path,
        typer.Argument(
            help="Path to target model's density profile JSON",
            exists=True,
            readable=True,
        ),
    ],
    donor_dirs: list[Path] | None = typer.Option(
        None,
        "--donor-dir",
        "-d",
        help="Directory containing donor profile JSONs (repeatable)",
    ),
    donor_profiles: list[Path] | None = typer.Option(
        None,
        "--donor",
        help="Individual donor profile paths (repeatable)",
    ),
    output_yaml: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save generated YAML program",
    ),
    program_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for generated program",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for merged models",
    ),
    min_opportunity: float | None = typer.Option(
        None,
        "--min-opportunity",
        help="Minimum opportunity score to include donor (derived from data if not set)",
    ),
    max_layers: int = typer.Option(
        10,
        "--max-layers",
        help="Maximum layers per donor",
    ),
) -> None:
    """Generate TransplantProgram YAML from density profiles.

    Analyzes target and donor density profiles to automatically select
    optimal donors for each knowledge domain. Outputs a complete
    TransplantProgram ready for execution with `mc program run`.

    The generator:
    1. Loads target model's density profile
    2. Loads all donor profiles from --donor-dir or --donor options
    3. Computes per-domain opportunity scores (donor_density - target_density)
    4. Selects best donor per domain where opportunity is positive
    5. Generates layer assignments (target weak + donor strong intersection)

    Examples:
        mc program generate ./target.json -d ./experts/ -d ./full-profiles/
        mc program generate ./target.json --donor ./math.json --donor ./code.json
        mc program generate ./target.json -d ./donors/ -n "uber-model" -o ./program.yaml
    """
    context = _context(ctx)

    try:
        from modelcypher.core.use_cases.program_generator_service import (
            ProgramGeneratorConfig,
            ProgramGeneratorService,
        )

        # Validate inputs
        has_dirs = donor_dirs is not None and len(donor_dirs) > 0
        has_profiles = donor_profiles is not None and len(donor_profiles) > 0
        if not has_dirs and not has_profiles:
            write_error(
                "Either --donor-dir or at least one --donor must be provided",
                context.output_format,
            )
            raise typer.Exit(1)

        # Validate donor directories exist
        if donor_dirs:
            for dd in donor_dirs:
                if not dd.exists():
                    write_error(
                        f"Donor directory not found: {dd}",
                        context.output_format,
                    )
                    raise typer.Exit(1)

        # Validate individual donor profiles exist
        if donor_profiles:
            for dp in donor_profiles:
                if not dp.exists():
                    write_error(
                        f"Donor profile not found: {dp}",
                        context.output_format,
                    )
                    raise typer.Exit(1)

        # Collect all donor profiles from directories and individual files
        all_donors: list[Path] = []
        if donor_dirs:
            for dd in donor_dirs:
                all_donors.extend(dd.glob("*.json"))
        if donor_profiles:
            all_donors.extend(donor_profiles)

        # Remove target profile from donors if present
        target_resolved = target_profile.resolve()
        all_donors = [d for d in all_donors if d.resolve() != target_resolved]

        if not all_donors:
            write_error(
                "No donor profiles found after filtering",
                context.output_format,
            )
            raise typer.Exit(1)

        # Build config
        config = ProgramGeneratorConfig(
            min_opportunity_threshold=min_opportunity,
            max_layers_per_donor=max_layers,
        )

        service = ProgramGeneratorService()

        # Generate program using all collected donors
        result = service.generate(
            target_profile=target_profile,
            donor_profiles=all_donors,
            config=config,
            program_name=program_name,
            output_dir=output_dir,
        )

        # Save YAML if requested
        if output_yaml:
            output_yaml.parent.mkdir(parents=True, exist_ok=True)
            result.program.to_yaml(output_yaml)
            import sys

            sys.stderr.write(f"Program saved to {output_yaml}\n")

        # Output result
        write_output(result.to_dict(), context.output_format, context.pretty)

    except FileNotFoundError as e:
        write_error(str(e), context.output_format)
        raise typer.Exit(1) from e
    except ValueError as e:
        write_error(f"Invalid input: {e}", context.output_format)
        raise typer.Exit(1) from e
    except Exception as e:
        write_error(f"Generation failed: {e}", context.output_format)
        raise typer.Exit(1) from e
