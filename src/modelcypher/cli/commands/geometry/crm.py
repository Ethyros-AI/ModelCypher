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

"""Geometry CRM (Concept Response Matrix) CLI commands.

Provides commands for:
- Building concept response matrices
- Comparing CRMs between models
- Listing sequence invariant probes

Commands:
    mc geometry crm build --model <path> --output-path <path>
    mc geometry crm compare --source <path> --target <path>
    mc geometry crm delta-mask --source <path> --target <path>
    mc geometry crm sequence-inventory
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.composition import get_inference_engine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.agents import (
    AtlasSource,
    UnifiedAtlasInventory,
)
from modelcypher.core.use_cases.concept_response_matrix_service import (
    ConceptResponseMatrixService,
)
from modelcypher.utils.errors import ErrorDetail
from modelcypher.utils.json import dump_json

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("build")
def geometry_crm_build(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model", help="Path to model directory"),
    output_path: str = typer.Option(..., "--output-path", help="Output CRM JSON path"),
    adapter: str | None = typer.Option(None, "--adapter", help="Optional adapter directory"),
) -> None:
    """Build a concept response matrix (CRM) for a model.

    Examples:
        mc geometry crm build --model ./model --output-path ./crm.json
    """
    context = _context(ctx)
    service = ConceptResponseMatrixService(engine=get_inference_engine())

    try:
        summary = service.build(
            model_path=model_path,
            output_path=output_path,
            adapter=adapter,
        )
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1018",
            title="CRM build failed",
            detail=str(exc),
            hint="Ensure the model directory contains config.json and weights.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": summary.model_path,
        "outputPath": summary.output_path,
        "layerCount": summary.layer_count,
        "hiddenDim": summary.hidden_dim,
        "anchorCount": summary.anchor_count,
        "primeCount": summary.prime_count,
        "gateCount": summary.gate_count,
        "sequenceInvariantCount": summary.sequence_invariant_count,
        "emotionCount": summary.emotion_count,
        "primeNumberCount": summary.prime_number_count,
    }

    if context.output_format == "text":
        lines = [
            "CONCEPT RESPONSE MATRIX",
            f"Model: {summary.model_path}",
            f"Output: {summary.output_path}",
            f"Layers: {summary.layer_count}",
            f"Hidden Dim: {summary.hidden_dim}",
            f"Probes: {summary.anchor_count}",
            f"  semantic_prime: {summary.prime_count}",
            f"  computational_gate: {summary.gate_count}",
            f"  sequence_invariant: {summary.sequence_invariant_count}",
            f"  emotion_concept: {summary.emotion_count}",
            f"  prime_number: {summary.prime_number_count}",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def geometry_crm_compare(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source CRM JSON path"),
    target: str = typer.Option(..., "--target", help="Target CRM JSON path"),
) -> None:
    """Compare two CRMs and compute layer correspondence via CKA.

    Examples:
        mc geometry crm compare --source ./crm1.json --target ./crm2.json
    """
    context = _context(ctx)
    service = ConceptResponseMatrixService()

    try:
        summary = service.compare(source, target)
    except (ValueError, OSError) as exc:
        error = ErrorDetail(
            code="MC-1019",
            title="CRM comparison failed",
            detail=str(exc),
            hint="Ensure both CRM paths exist and are valid JSON exports.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "sourcePath": summary.source_path,
        "targetPath": summary.target_path,
        "commonAnchorCount": summary.common_anchor_count,
        "alignmentPrecision": summary.alignment_precision,
        "meanLayerPrecision": summary.mean_cka,
        "aligned": summary.aligned,
        "layerCorrespondence": summary.layer_correspondence,
    }
    if summary.cka_matrix is not None:
        payload["ckaMatrix"] = summary.cka_matrix

    if context.output_format == "text":
        # Gram-space kernel alignment is exact by construction.
        # We report numerical precision of that computation.
        lines = [
            "CRM COMPARISON",
            f"Source: {summary.source_path}",
            f"Target: {summary.target_path}",
            f"Common Anchors: {summary.common_anchor_count}",
            "",
            "Alignment (Gram-space kernel precision):",
            f"  Numerical Precision: {summary.alignment_precision:.6f}",
            "",
            "Layer Mapping:",
            f"  Mean Precision: {summary.mean_cka:.4f}",
            f"  Perfect: {summary.aligned}",
        ]
        if summary.layer_correspondence:
            lines.append("")
            lines.append("Layer Correspondence (top 10):")
            for match in summary.layer_correspondence[:10]:
                lines.append(
                    f"  {match['sourceLayer']} -> {match['targetLayer']} (precision {match['cka']:.4f})"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("delta-mask")
def geometry_crm_delta_mask(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source CRM JSON path"),
    target: str = typer.Option(..., "--target", help="Target CRM JSON path"),
    output_path: str | None = typer.Option(
        None, "--output-path", help="Write delta mask JSON to file"
    ),
) -> None:
    """Build a knowledge delta mask from two CRM files.

    Identifies layers where the source activation density exceeds the target
    while the target appears sparse, using distribution-derived thresholds.
    """
    context = _context(ctx)
    service = ConceptResponseMatrixService()

    try:
        summary = service.knowledge_delta_mask(source, target)
    except (ValueError, OSError) as exc:
        error = ErrorDetail(
            code="MC-1052",
            title="CRM delta mask failed",
            detail=str(exc),
            hint="Ensure both CRM paths exist and share common anchors.",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "_schema": "mc.geometry.crm.delta_mask.v1",
        "sourcePath": summary.source_path,
        "targetPath": summary.target_path,
        "commonAnchorCount": summary.common_anchor_count,
        "layerCount": summary.layer_count,
        "thresholds": {
            "targetSparseThreshold": summary.target_sparse_threshold,
            "sourceDenseThreshold": summary.source_dense_threshold,
            "densityRatioThreshold": summary.density_ratio_threshold,
        },
        "graftLayers": summary.graft_layers,
        "graftMaskByLayer": {str(k): v for k, v in summary.graft_mask_by_layer.items()},
        "skippedLayers": summary.skipped_layers,
        "layerMetrics": [
            {
                "layer": entry.layer,
                "anchorCount": entry.anchor_count,
                "coverage": entry.coverage,
                "sourceMeanNorm": entry.source_mean_norm,
                "targetMeanNorm": entry.target_mean_norm,
                "sourceStdNorm": entry.source_std_norm,
                "targetStdNorm": entry.target_std_norm,
                "deltaMeanNorm": entry.delta_mean_norm,
                "densityRatio": entry.density_ratio,
                "graftable": entry.graftable,
            }
            for entry in summary.layer_summaries
        ],
    }

    if output_path:
        Path(output_path).write_text(dump_json(payload))
        payload["outputPath"] = output_path

    if context.output_format == "text":
        lines = [
            "CRM DELTA MASK",
            f"Source: {summary.source_path}",
            f"Target: {summary.target_path}",
            f"Common Anchors: {summary.common_anchor_count}",
            f"Layers: {summary.layer_count}",
            "",
            "Thresholds:",
            f"  Target sparse <= {summary.target_sparse_threshold:.6f}",
            f"  Source dense >= {summary.source_dense_threshold:.6f}",
            f"  Density ratio >= {summary.density_ratio_threshold:.6f}",
            "",
            f"Graft Layers: {summary.graft_layers}",
        ]
        if output_path:
            lines.append(f"Output: {output_path}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("probe-inventory")
def geometry_crm_probe_inventory(
    ctx: typer.Context,
    source: str | None = typer.Option(
        None, "--source", help="Filter by source (e.g., sequence_invariant, semantic_prime)"
    ),
) -> None:
    """List available probes for CRM anchoring.

    Examples:
        mc geometry crm probe-inventory
        mc geometry crm probe-inventory --source sequence_invariant
    """
    context = _context(ctx)

    # Get all probes or filter by source
    if source:
        try:
            atlas_source = AtlasSource(source)
            probes = UnifiedAtlasInventory.probes_by_source({atlas_source})
        except ValueError:
            from modelcypher.utils.errors import ErrorDetail
            error = ErrorDetail(
                code="MC-1053",
                title="Invalid source",
                detail=f"Unknown source: {source}",
                hint=f"Valid sources: {', '.join(s.value for s in AtlasSource)}",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)
    else:
        probes = UnifiedAtlasInventory.all_probes()

    # Count by source
    counts = UnifiedAtlasInventory.probe_count()

    probe_list = [
        {
            "id": probe.id,
            "source": probe.source.value,
            "domain": probe.domain.value,
            "name": probe.name,
            "description": probe.description,
            "weight": probe.cross_domain_weight,
        }
        for probe in probes
    ]

    payload = {
        "totalProbes": len(probes),
        "sourceCounts": {src.value: count for src, count in counts.items()},
        "probes": probe_list,
    }

    if context.output_format == "text":
        lines = [
            "PROBE INVENTORY",
            f"Total Probes: {len(probes)}",
            "",
            "Probes by Source:",
        ]
        for src, count in sorted(counts.items(), key=lambda x: x[0].value):
            lines.append(f"  {src.value}: {count}")
        lines.append("")
        lines.append("Probes (first 20):")
        for probe in probes[:20]:
            lines.append(f"  [{probe.source.value}] {probe.id}: {probe.name}")
        if len(probes) > 20:
            lines.append(f"  ... and {len(probes) - 20} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
