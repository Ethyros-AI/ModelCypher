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

"""Geometry invariant layer mapping CLI commands.

Provides commands for invariant-based layer mapping between models.

See UnifiedAtlasInventory for the full list of atlases and probe counts.
Use `mc geometry invariant atlas-inventory` to view available probes.

Commands:
    mc geometry invariant map-layers --source <path> --target <path>
    mc geometry invariant collapse-risk --model <path>
    mc geometry invariant atlas-inventory
"""

from __future__ import annotations

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.cli.composition import get_invariant_mapping_service
from modelcypher.core.domain.agents.unified_atlas import (
    AtlasSource,
    UnifiedAtlasInventory,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("map-layers")
def geometry_invariant_map_layers(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Path to source model"),
    target: str = typer.Option(..., "--target", help="Path to target model"),
) -> None:
    """Map layers between models using the unified atlas.

    Uses probes from all atlases with cross-domain triangulation
    scoring to find corresponding layers between models.

    Example:
        mc geometry invariant map-layers --source ./model-a --target ./model-b
    """
    context = _context(ctx)
    service = get_invariant_mapping_service()

    try:
        result = service.map_layers(source, target)
        payload = service.result_payload(result)

        if context.output_format == "text":
            summary = result.report.summary
            lines = [
                "INVARIANT LAYER MAPPING",
                f"Source: {result.report.source_model}",
                f"Target: {result.report.target_model}",
                f"Invariants Used: {result.report.invariant_count}",
                "",
                "Results:",
                f"  Mapped Layers: {summary.mapped_layers}",
                f"  Mean Similarity: {summary.mean_similarity:.3f}",
                "",
                f"  Source Collapsed: {summary.source_collapsed_layers}",
                f"  Target Collapsed: {summary.target_collapsed_layers}",
            ]
            lines.extend(
                [
                    "",
                    "Triangulation:",
                    f"  Mean Multiplier: {summary.mean_triangulation_multiplier:.2f}",
                ]
            )

            # Show multi-atlas metrics
            if summary.total_probes_used > 70:
                lines.extend(
                    [
                        "",
                        "Multi-Atlas Coverage:",
                        f"  Total Probes: {summary.total_probes_used}",
                        f"  Atlas Sources: {summary.atlas_sources_detected}",
                        f"  Domains: {summary.atlas_domains_detected}",
                    ]
                )

            if result.report.mappings:
                lines.append("")
                lines.append("Layer Mappings (first 10):")
                for m in result.report.mappings[:10]:
                    lines.append(
                        f"  L{m.source_layer} -> L{m.target_layer}: "
                        f"sim={m.similarity:.3f}"
                    )
                if len(result.report.mappings) > 10:
                    lines.append(f"  ... and {len(result.report.mappings) - 10} more")

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("collapse-risk")
def geometry_invariant_collapse_risk(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Path to model"),
) -> None:
    """Analyze layer collapse risk for a model.

    Identifies layers where invariant activation is too sparse for
    reliable layer correspondence in merge operations.

    Example:
        mc geometry invariant collapse-risk --model ./qwen2.5-7b
    """
    context = _context(ctx)
    service = get_invariant_mapping_service()

    try:
        result = service.analyze_collapse_risk(model)
        payload = service.collapse_risk_payload(result)

        if context.output_format == "text":
            lines = [
                "COLLAPSE RISK ANALYSIS",
                f"Model: {result.model_path}",
                "",
                f"Layer Count: {result.layer_count}",
                f"Collapsed Layers: {result.collapsed_layers}",
                f"Collapse Ratio: {result.collapse_ratio * 100:.1f}%",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc


@app.command("atlas-inventory")
def geometry_invariant_atlas_inventory(
    ctx: typer.Context,
) -> None:
    """Show inventory of available probes across all atlases.

    Displays all probes available for unified atlas layer mapping,
    grouped by atlas source and domain.

    Example:
        mc geometry invariant atlas-inventory
    """
    context = _context(ctx)

    # Get probe counts by source
    counts = UnifiedAtlasInventory.probe_count()
    total = UnifiedAtlasInventory.total_probe_count()

    if context.output_format == "text":
        lines = [
            "MULTI-ATLAS PROBE INVENTORY",
            "",
            f"Total Probes: {total}",
        ]

        lines.extend(
            [
                "",
                "Atlas Sources:",
                f"  Sequence Invariants: {counts.get(AtlasSource.SEQUENCE_INVARIANT, 0):3d} probes  (mathematical, logical)",
                f"  Semantic Primes:     {counts.get(AtlasSource.SEMANTIC_PRIME, 0):3d} probes  (linguistic, mental, relational)",
                f"  Computational Gates: {counts.get(AtlasSource.COMPUTATIONAL_GATE, 0):3d} probes  (computational, structural)",
                f"  Emotion Concepts:    {counts.get(AtlasSource.EMOTION_CONCEPT, 0):3d} probes  (affective, relational)",
                f"  Temporal Concepts:   {counts.get(AtlasSource.TEMPORAL_CONCEPT, 0):3d} probes  (temporal, logical)",
                f"  Social Concepts:     {counts.get(AtlasSource.SOCIAL_CONCEPT, 0):3d} probes  (relational, linguistic)",
                f"  Moral Concepts:      {counts.get(AtlasSource.MORAL_CONCEPT, 0):3d} probes  (moral, relational)",
                f"  Compositional:       {counts.get(AtlasSource.COMPOSITIONAL, 0):3d} probes  (linguistic, mental)",
                f"  Philosophical:       {counts.get(AtlasSource.PHILOSOPHICAL_CONCEPT, 0):3d} probes  (philosophical, logical)",
                f"  Conceptual Genealogy: {counts.get(AtlasSource.CONCEPTUAL_GENEALOGY, 0):3d} probes  (philosophical, moral, relational)",
                "",
                "Triangulation Domains:",
                "  mathematical, logical, linguistic, mental, computational,",
                "  structural, affective, relational, temporal, spatial, moral, philosophical",
                "",
                "Usage:",
                "  Full multi-atlas:    mc geometry invariant map-layers ...",
            ]
        )

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    # JSON output
    payload = {
        "_schema": "mc.geometry.atlas.inventory.v1",
        "totalProbes": total,
        "filteredCount": total,
        "sources": {
            "sequenceInvariant": {
                "count": counts.get(AtlasSource.SEQUENCE_INVARIANT, 0),
                "description": "Mathematical sequences and logical invariants",
                "domains": ["mathematical", "logical"],
            },
            "semanticPrime": {
                "count": counts.get(AtlasSource.SEMANTIC_PRIME, 0),
                "description": "NSM semantic primitives for cross-linguistic concepts",
                "domains": ["linguistic", "mental", "relational", "temporal", "spatial"],
            },
            "computationalGate": {
                "count": counts.get(AtlasSource.COMPUTATIONAL_GATE, 0),
                "description": "Programming primitives and computational patterns",
                "domains": ["computational", "structural", "logical"],
            },
            "emotionConcept": {
                "count": counts.get(AtlasSource.EMOTION_CONCEPT, 0),
                "description": "Plutchik emotion wheel with VAD coordinates",
                "domains": ["affective", "relational", "mental"],
            },
            "temporalConcept": {
                "count": counts.get(AtlasSource.TEMPORAL_CONCEPT, 0),
                "description": "Temporal anchors for direction, duration, causality",
                "domains": ["temporal", "logical"],
            },
            "socialConcept": {
                "count": counts.get(AtlasSource.SOCIAL_CONCEPT, 0),
                "description": "Social structure probes (power, kinship, formality)",
                "domains": ["relational", "linguistic"],
            },
            "moralConcept": {
                "count": counts.get(AtlasSource.MORAL_CONCEPT, 0),
                "description": "Moral foundations and ethical valence",
                "domains": ["moral", "relational"],
            },
            "compositional": {
                "count": counts.get(AtlasSource.COMPOSITIONAL, 0),
                "description": "Semantic prime compositions (multi-prime probes)",
                "domains": ["linguistic", "mental"],
            },
            "philosophicalConcept": {
                "count": counts.get(AtlasSource.PHILOSOPHICAL_CONCEPT, 0),
                "description": "Fundamental categories of thought",
                "domains": ["philosophical", "logical"],
            },
            "conceptualGenealogy": {
                "count": counts.get(AtlasSource.CONCEPTUAL_GENEALOGY, 0),
                "description": "Etymology and lineage probes for concept drift",
                "domains": ["philosophical", "moral", "relational", "mathematical", "affective"],
            },
        },
        "domains": [
            "mathematical",
            "logical",
            "linguistic",
            "mental",
            "computational",
            "structural",
            "affective",
            "relational",
            "temporal",
            "spatial",
            "moral",
            "philosophical",
        ],
    }

    write_output(payload, context.output_format, context.pretty)
