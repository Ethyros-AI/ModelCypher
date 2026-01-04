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

"""CLI commands for unified ModelProfile operations.

Commands:
    mc profile inspect   - View a profile's contents
    mc profile compare   - Compare two profiles for alignment summary
    mc profile import    - Import from existing profile formats (curvature, etc.)
    mc profile merge     - Merge multiple partial profiles
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.cli.validation import validate_file_exists, validate_model_path

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.model_profile import ModelProfile

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="profile",
    help="Unified model profile operations - the transparent black box.",
    no_args_is_help=True,
)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("inspect")
def inspect_profile(
    ctx: typer.Context,
    profile_path: str = typer.Argument(..., help="Path to model profile JSON"),
    section: str | None = typer.Option(
        None,
        "--section",
        "-s",
        help="Show only this section (identity, geometry, topology, semantic, density)",
    ),
    layer: int | None = typer.Option(
        None, "--layer", "-l", help="Show only this layer index"
    ),
) -> None:
    """Inspect a model profile.

    Examples:
        mc profile inspect profile.json
        mc profile inspect profile.json --section geometry
        mc profile inspect profile.json --layer 5
    """
    from modelcypher.core.domain.geometry.model_profile import (
        ModelProfile,
    )

    context = _context(ctx)
    validate_file_exists(profile_path, description="Profile", context=context)

    try:
        profile = ModelProfile.load(profile_path)
    except Exception as e:
        typer.echo(f"Error loading profile: {e}", err=True)
        raise typer.Exit(1) from e

    # Build output based on requested section
    if section:
        result = _extract_section(profile, section, layer)
    elif layer is not None:
        result = _extract_layer(profile, layer)
    else:
        result = _build_summary(profile)

    write_output(result, context.output_format, context.pretty)


@app.command("compare")
def compare_profiles_cmd(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to source model profile JSON"),
    target_path: str = typer.Argument(..., help="Path to target model profile JSON"),
    output_path: str | None = typer.Option(
        None, "--save", "-o", help="Save comparison result to JSON file"
    ),
    baseline_path: str | None = typer.Option(
        None, "--baseline", "-b", help="Path to family baseline for z-score computation"
    ),
) -> None:
    """Compare two model profiles and show geometric summary.

    The comparison shows:
    - Structural ratios (architecture, dimensions)
    - Geometric diffs (curvature, topology)
    - Layer correspondence and critical layers
    - Baseline-relative z-scores (when --baseline provided)

    Examples:
        mc profile compare source.json target.json
        mc profile compare source.json target.json --save comparison.json
        mc profile compare source.json target.json --baseline qwen-baseline.json
    """
    from modelcypher.core.domain.geometry.curvature_profile import FamilyBaseline
    from modelcypher.core.domain.geometry.model_profile import ModelProfile
    from modelcypher.core.domain.geometry.profile_comparison import compare_profiles

    context = _context(ctx)

    # Validate input files early
    validate_file_exists(source_path, description="Source profile", context=context)
    validate_file_exists(target_path, description="Target profile", context=context)
    if baseline_path:
        validate_file_exists(baseline_path, description="Baseline profile", context=context)

    try:
        source = ModelProfile.load(source_path)
        target = ModelProfile.load(target_path)
    except Exception as e:
        typer.echo(f"Error loading profiles: {e}", err=True)
        raise typer.Exit(1) from e

    # Load baseline if provided
    baseline = None
    if baseline_path:
        try:
            baseline = FamilyBaseline.load(baseline_path)
        except Exception as e:
            typer.echo(f"Warning: Failed to load baseline: {e}", err=True)

    comparison = compare_profiles(source, target, baseline=baseline)
    result = comparison.to_dict()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        typer.echo(f"Saved comparison to {output_path}")

    write_output(result, context.output_format, context.pretty)


@app.command("import")
def import_profile(
    ctx: typer.Context,
    source_path: str = typer.Argument(..., help="Path to existing profile JSON"),
    profile_type: str = typer.Option(
        "curvature",
        "--type",
        "-t",
        help="Profile type: curvature, density, topology",
    ),
    output_path: str = typer.Option(
        ..., "--output", "-o", help="Output path for unified profile"
    ),
    base_path: str | None = typer.Option(
        None,
        "--base",
        "-b",
        help="Base unified profile to merge into (optional)",
    ),
) -> None:
    """Import from existing profile format to unified ModelProfile.

    Supports importing:
    - curvature: CurvatureProfile JSON files
    - (more formats to be added)

    Examples:
        mc profile import curvature.json --type curvature --output unified.json
        mc profile import density.json --type density --base unified.json --output updated.json
    """
    from modelcypher.core.domain.geometry.model_profile import ModelProfile

    context = _context(ctx)

    # Validate input files early
    validate_file_exists(source_path, description="Source profile", context=context)
    if base_path:
        validate_file_exists(base_path, description="Base profile", context=context)

    # Load base profile if provided
    base_profile = None
    if base_path:
        try:
            base_profile = ModelProfile.load(base_path)
        except Exception as e:
            typer.echo(f"Error loading base profile: {e}", err=True)
            raise typer.Exit(1) from e

    # Import based on type
    try:
        if profile_type == "curvature":
            from modelcypher.core.domain.geometry.curvature_profile import (
                CurvatureProfile,
            )

            curvature = CurvatureProfile.load(source_path)
            imported = ModelProfile.from_curvature_profile(curvature)
        else:
            typer.echo(f"Unsupported profile type: {profile_type}", err=True)
            typer.echo("Supported types: curvature", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error importing profile: {e}", err=True)
        raise typer.Exit(1) from e

    # Merge if base provided
    if base_profile:
        result_profile = base_profile.merge_with(imported)
    else:
        result_profile = imported

    # Save
    result_profile.save(output_path)

    result = {
        "status": "success",
        "imported_from": source_path,
        "profile_type": profile_type,
        "output_path": output_path,
        "computed_sections": result_profile.computed_sections,
        "layer_count": len(result_profile.layer_profiles),
    }

    write_output(result, context.output_format, context.pretty)


@app.command("merge")
def merge_profiles(
    ctx: typer.Context,
    profiles: list[str] = typer.Argument(..., help="Paths to profile JSON files"),
    output_path: str = typer.Option(
        ..., "--output", "-o", help="Output path for merged profile"
    ),
) -> None:
    """Merge multiple partial profiles into one.

    Profiles are merged in order, with later profiles filling in
    missing sections from earlier ones.

    Examples:
        mc profile merge geometry.json topology.json semantic.json --output complete.json
    """
    from modelcypher.core.domain.geometry.model_profile import ModelProfile

    context = _context(ctx)

    if not profiles:
        typer.echo("No profiles provided", err=True)
        raise typer.Exit(1)

    # Validate all input files exist before doing any work
    for path in profiles:
        validate_file_exists(path, description="Profile", context=context)

    # Load and merge profiles
    merged = None
    for path in profiles:
        try:
            profile = ModelProfile.load(path)
            if merged is None:
                merged = profile
            else:
                merged = merged.merge_with(profile)
        except Exception as e:
            typer.echo(f"Error loading {path}: {e}", err=True)
            raise typer.Exit(1) from e

    if merged is None:
        typer.echo("No valid profiles to merge", err=True)
        raise typer.Exit(1)

    # Save
    merged.save(output_path)

    result = {
        "status": "success",
        "merged_profiles": profiles,
        "output_path": output_path,
        "computed_sections": merged.computed_sections,
        "layer_count": len(merged.layer_profiles),
    }

    write_output(result, context.output_format, context.pretty)


def _build_summary(profile: "ModelProfile") -> dict[str, Any]:
    """Build summary view of profile."""

    summary: dict[str, Any] = {
        "model_path": profile.model_path,
        "model_family": profile.model_family,
        "architecture": profile.architecture,
        "identity": {
            "parameter_count": profile.parameter_count,
            "hidden_dim": profile.hidden_dim,
            "num_layers": profile.num_layers,
            "num_attention_heads": profile.num_attention_heads,
            "vocab_size": profile.vocab_size,
        },
        "computed_sections": profile.computed_sections,
        "layer_profile_count": len(profile.layer_profiles),
    }

    # Add curvature summary if available
    if profile.global_sectional_mean != 0.0 or profile.global_ollivier_ricci_mean != 0.0:
        summary["curvature"] = {
            "global_sectional_mean": profile.global_sectional_mean,
            "global_sectional_std": profile.global_sectional_std,
            "global_ollivier_ricci_mean": profile.global_ollivier_ricci_mean,
            "global_ollivier_ricci_std": profile.global_ollivier_ricci_std,
            "global_intrinsic_dimension_mean": profile.global_intrinsic_dimension_mean,
        }

    # Add optional sections if present
    if profile.topology_summary:
        summary["topology"] = profile.topology_summary.to_dict()

    if profile.semantic_signature:
        summary["semantic"] = {
            "vector_dim": len(profile.semantic_signature.vector),
            "dominant_primes": profile.semantic_signature.dominant_primes,
        }

    if profile.density_summary:
        summary["density"] = profile.density_summary.to_dict()

    return summary


def _extract_section(
    profile: "ModelProfile", section: str, layer: int | None
) -> dict[str, Any]:
    """Extract a specific section from profile."""
    if section == "identity":
        return {
            "model_path": profile.model_path,
            "model_family": profile.model_family,
            "architecture": profile.architecture,
            "parameter_count": profile.parameter_count,
            "hidden_dim": profile.hidden_dim,
            "num_layers": profile.num_layers,
            "num_attention_heads": profile.num_attention_heads,
            "vocab_size": profile.vocab_size,
        }

    elif section == "geometry":
        if layer is not None:
            return _extract_layer(profile, layer)
        return {
            "global_sectional_mean": profile.global_sectional_mean,
            "global_sectional_std": profile.global_sectional_std,
            "global_ollivier_ricci_mean": profile.global_ollivier_ricci_mean,
            "global_ollivier_ricci_std": profile.global_ollivier_ricci_std,
            "global_intrinsic_dimension_mean": profile.global_intrinsic_dimension_mean,
            "layer_count": len(profile.layer_profiles),
        }

    elif section == "topology":
        if profile.topology_summary:
            return profile.topology_summary.to_dict()
        return {"status": "not_computed"}

    elif section == "semantic":
        if profile.semantic_signature:
            return profile.semantic_signature.to_dict()
        return {"status": "not_computed"}

    elif section == "density":
        if profile.density_summary:
            return profile.density_summary.to_dict()
        return {"status": "not_computed"}

    elif section == "layers":
        return {
            "layer_count": len(profile.layer_profiles),
            "layers": [lp.to_dict() for lp in profile.layer_profiles],
        }

    else:
        return {"error": f"Unknown section: {section}"}


def _extract_layer(profile: "ModelProfile", layer_idx: int) -> dict[str, Any]:
    """Extract a specific layer from profile."""
    for lp in profile.layer_profiles:
        if lp.layer_idx == layer_idx:
            return lp.to_dict()

    return {"error": f"Layer {layer_idx} not found"}


@app.command("generate")
def generate_profile(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to the model directory"),
    output_path: str = typer.Option(
        ..., "--output", "-o", help="Output path for the profile JSON"
    ),
    identity_only: bool = typer.Option(
        False, "--identity-only", help="Only extract identity (fast, no model loading)"
    ),
) -> None:
    """Generate a unified profile from a model.

    By default, extracts identity information from config.json (fast).
    For full geometry profiling, use 'mc geometry research curvature-profile'
    then import with 'mc profile import'.

    Examples:
        mc profile generate /path/to/model -o profile.json --identity-only
        mc profile generate /path/to/model -o profile.json
    """
    from modelcypher.core.domain.geometry.model_profile import ModelProfile
    from modelcypher.core.use_cases.model_profiler_service import ModelProfilerService

    context = _context(ctx)
    validate_model_path(model_path, context=context)
    service = ModelProfilerService()

    # Start with empty profile
    profile = ModelProfile(model_path=model_path)

    # Always try to get identity
    try:
        profile = service.update_identity(profile, model_path)
    except Exception as e:
        logger.warning(f"Failed to extract identity: {e}")

    if identity_only:
        profile.save(output_path)
        result = {
            "status": "success",
            "model_path": model_path,
            "output_path": output_path,
            "computed_sections": profile.computed_sections,
            "identity": {
                "architecture": profile.architecture,
                "num_layers": profile.num_layers,
                "hidden_dim": profile.hidden_dim,
            },
        }
        write_output(result, context.output_format, context.pretty)
        return

    # For full geometry, delegate to curvature profiling and import
    typer.echo(
        "For full geometry profiling, use:\n"
        "  mc geometry research curvature-profile /path/to/model --save curvature.json\n"
        "  mc profile import curvature.json --type curvature -o profile.json",
        err=True,
    )

    # Save what we have
    profile.save(output_path)
    result = {
        "status": "partial",
        "model_path": model_path,
        "output_path": output_path,
        "computed_sections": profile.computed_sections,
        "note": "Use 'mc geometry research curvature-profile' for full geometry",
    }
    write_output(result, context.output_format, context.pretty)


@app.command("update")
def update_profile(
    ctx: typer.Context,
    profile_path: str = typer.Argument(..., help="Path to existing profile JSON"),
    model_path: str | None = typer.Option(
        None, "--model", "-m", help="Model path to extract identity from"
    ),
    output_path: str | None = typer.Option(
        None, "--output", "-o", help="Output path (defaults to overwrite input)"
    ),
) -> None:
    """Update an existing profile with additional information.

    Currently supports:
    - Adding identity information from a model directory

    Examples:
        mc profile update profile.json --model /path/to/model
        mc profile update profile.json --model /path/to/model -o updated.json
    """
    from modelcypher.core.domain.geometry.model_profile import ModelProfile
    from modelcypher.core.use_cases.model_profiler_service import ModelProfilerService

    context = _context(ctx)

    # Load existing profile
    try:
        profile = ModelProfile.load(profile_path)
    except Exception as e:
        typer.echo(f"Error loading profile: {e}", err=True)
        raise typer.Exit(1) from e

    # Update identity if model path provided
    if model_path:
        service = ModelProfilerService()
        try:
            profile = service.update_identity(profile, model_path)
        except Exception as e:
            typer.echo(f"Warning: Failed to extract identity: {e}", err=True)

    # Save
    save_path = output_path or profile_path
    profile.save(save_path)

    result = {
        "status": "success",
        "input_path": profile_path,
        "output_path": save_path,
        "computed_sections": profile.computed_sections,
    }
    write_output(result, context.output_format, context.pretty)
