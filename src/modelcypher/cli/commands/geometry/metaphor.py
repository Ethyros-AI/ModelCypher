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

"""Geometry metaphor CLI commands.

Provides commands for analyzing metaphor convergence across models.

Universal metaphors (e.g., CHANGE IS MOTION, UNDERSTANDING IS GRASPING)
activate at predictable layers across architectures.

Commands:
    mc geometry metaphor anchors
    mc geometry metaphor compare --source <path> --target <path>
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.composition import get_backend
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.warnings import warn_network
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


def _get_metaphor_probes():
    """Get metaphor probes from unified atlas."""
    from modelcypher.core.domain.atlas.unified_atlas import (
        AtlasSource,
        UnifiedAtlasInventory,
    )

    all_probes = UnifiedAtlasInventory.all_probes()
    return [
        p
        for p in all_probes
        if p.source in (AtlasSource.METAPHOR_INVARIANT, AtlasSource.CONCEPTUAL_METAPHOR)
    ]


@app.command("anchors")
def metaphor_anchors(ctx: typer.Context):
    """List all metaphor invariants used for convergence analysis."""
    from modelcypher.core.use_cases.atlas_bootstrap import (
        register_default_atlas_inventories,
    )

    context = _context(ctx)
    register_default_atlas_inventories()

    probes = _get_metaphor_probes()

    # Group by source
    by_source: dict[str, list] = {}
    for p in probes:
        source = p.source.value
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(p)

    if context.output_format == "text":
        lines = [
            "METAPHOR INVARIANTS",
            f"Total probes: {len(probes)}",
            f"Sources: {len(by_source)}",
            "",
        ]
        for source, members in sorted(by_source.items()):
            lines.append(f"  {source} ({len(members)} probes):")
            for probe in members[:5]:  # Show first 5
                lines.append(f"    - {probe.name}")
            if len(members) > 5:
                lines.append(f"    ... and {len(members) - 5} more")
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "sources": {
            source: [{"name": p.name, "support_texts": p.support_texts} for p in members]
            for source, members in by_source.items()
        },
        "total_probes": len(probes),
        "total_sources": len(by_source),
    }
    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def metaphor_compare(
    ctx: typer.Context,
    source: Path = typer.Option(..., "--source", "-s", help="Path to source model"),
    target: Path = typer.Option(..., "--target", "-t", help="Path to target model"),
):
    """
    Compare metaphor convergence between two models.

    Computes layer-wise cosine similarity for each metaphor family,
    revealing how universal conceptual structures emerge at different depths.
    """
    from modelcypher.adapters.hf_hub import HfHubAdapter
    from modelcypher.core.domain.geometry.manifold_stitcher import (
        ManifoldStitcher,
        ProbeSpace,
    )
    from modelcypher.core.domain.geometry.metaphor_convergence_analyzer import (
        MetaphorConvergenceAnalyzer,
    )
    from modelcypher.core.use_cases.atlas_bootstrap import (
        register_default_atlas_inventories,
    )

    context = _context(ctx)
    register_default_atlas_inventories()
    backend = get_backend()

    mode = MetaphorConvergenceAnalyzer.AlignMode.LAYER

    # Load models
    warn_network(context, "Loading models from Hugging Face Hub if not cached.")
    adapter = HfHubAdapter()
    try:
        source_model, source_tokenizer = adapter.load_model_and_tokenizer(str(source))
    except Exception as e:
        write_error(
            ErrorDetail(code="MC-4001", message="Failed to load source model", detail=str(e)),
            context.output_format,
        )
        raise typer.Exit(1)

    try:
        target_model, target_tokenizer = adapter.load_model_and_tokenizer(str(target))
    except Exception as e:
        write_error(
            ErrorDetail(code="MC-4001", message="Failed to load target model", detail=str(e)),
            context.output_format,
        )
        raise typer.Exit(1)

    # Compute fingerprints using ManifoldStitcher
    stitcher = ManifoldStitcher(backend)
    try:
        source_fps = stitcher.compute_fingerprints(
            model=source_model,
            tokenizer=source_tokenizer,
            model_id=str(source),
            probe_space=ProbeSpace.METAPHOR,
        )
        target_fps = stitcher.compute_fingerprints(
            model=target_model,
            tokenizer=target_tokenizer,
            model_id=str(target),
            probe_space=ProbeSpace.METAPHOR,
        )
    except Exception as e:
        write_error(
            ErrorDetail(
                code="MC-4005",
                message="Failed to compute fingerprints",
                detail=str(e),
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    # Run metaphor convergence analysis
    try:
        report = MetaphorConvergenceAnalyzer.analyze(source_fps, target_fps, mode)
    except Exception as e:
        write_error(
            ErrorDetail(
                code="MC-4006",
                message="Metaphor analysis failed",
                detail=str(e),
            ),
            context.output_format,
        )
        raise typer.Exit(1)

    if context.output_format == "text":
        lines = [
            "METAPHOR CONVERGENCE ANALYSIS",
            f"Source: {report.models.model_a}",
            f"Target: {report.models.model_b}",
            f"Align mode: {report.align_mode.value}",
            f"Layers analyzed: {report.layer_count}",
            "",
            "MEAN COSINE BY FAMILY:",
        ]
        for family, mean_cos in sorted(report.summary.mean_cosine_by_family.items()):
            if mean_cos is not None:
                lines.append(f"  {family}: {mean_cos:.4f}")
            else:
                lines.append(f"  {family}: N/A")

        lines.append("")
        lines.append("MEAN COSINE BY LAYER:")
        for layer_label, mean_cos in sorted(
            report.summary.mean_cosine_by_layer.items(),
            key=lambda x: float(x[0]) if x[0].replace(".", "").isdigit() else 0,
        ):
            lines.append(f"  {layer_label}: {mean_cos:.4f}")

        lines.append("")
        lines.append(
            f"DIMENSION ALIGNMENT: {report.dimension_alignment.total_aligned_dimensions} aligned dimensions"
        )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    payload = {
        "models": {
            "source": report.models.model_a,
            "target": report.models.model_b,
        },
        "align_mode": report.align_mode.value,
        "layer_count": report.layer_count,
        "summary": {
            "mean_cosine_by_family": report.summary.mean_cosine_by_family,
            "mean_cosine_by_layer": report.summary.mean_cosine_by_layer,
        },
        "dimension_alignment": {
            "mode": report.dimension_alignment.mode.value,
            "total_aligned": report.dimension_alignment.total_aligned_dimensions,
        },
        "heatmap": {
            "families": report.heatmap.families,
            "layers": report.heatmap.layers,
            "values": report.heatmap.values,
        },
    }
    write_output(payload, context.output_format, context.pretty)
