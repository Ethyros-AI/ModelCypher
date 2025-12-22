"""Geometry invariant layer mapping CLI commands.

Provides commands for invariant-based layer mapping between models using
multi-atlas triangulation scoring across 237 probes.

Atlases:
- Sequence Invariants: 68 probes (mathematical/logical)
- Semantic Primes: 65 probes (linguistic/mental)
- Computational Gates: 72 probes (computational/structural)
- Emotion Concepts: 32 probes (affective/relational)

Commands:
    mc geometry invariant map-layers --source <path> --target <path>
    mc geometry invariant map-layers --source <path> --target <path> --scope multiAtlas
    mc geometry invariant collapse-risk --model <path>
"""

from __future__ import annotations

from typing import Optional

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.use_cases.invariant_layer_mapping_service import (
    CollapseRiskConfig,
    InvariantLayerMappingService,
    LayerMappingConfig,
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
    families: Optional[str] = typer.Option(
        None,
        "--families",
        help="Comma-separated sequence families: fibonacci, lucas, tribonacci, primes, catalan, ramanujan, logic, ordering, arithmetic, causality",
    ),
    scope: str = typer.Option(
        "sequenceInvariants",
        "--scope",
        help="Invariant scope: invariants, logicOnly, sequenceInvariants, multiAtlas (237 probes)",
    ),
    atlas_sources: Optional[str] = typer.Option(
        None,
        "--atlas-sources",
        help="Comma-separated atlas sources for multiAtlas scope: sequence, semantic, gate, emotion (default: all)",
    ),
    atlas_domains: Optional[str] = typer.Option(
        None,
        "--atlas-domains",
        help="Comma-separated domains: mathematical, logical, linguistic, mental, computational, structural, affective, relational, temporal, spatial",
    ),
    triangulation: bool = typer.Option(
        True,
        "--triangulation/--no-triangulation",
        help="Enable cross-domain triangulation scoring",
    ),
    collapse_threshold: float = typer.Option(
        0.35,
        "--collapse-threshold",
        help="Threshold for collapse detection (0.0-1.0)",
    ),
    sample_layers: int = typer.Option(
        12,
        "--sample-layers",
        help="Number of sample layers",
    ),
) -> None:
    """Map layers between models using multi-atlas triangulation.

    Uses up to 237 probes across 4 atlases with cross-domain
    triangulation scoring to find corresponding layers between models.

    Scopes:
        invariants        - Default sequence families
        logicOnly         - Logic family only
        sequenceInvariants - All 68 sequence invariants
        multiAtlas        - All 237 probes from all 4 atlases

    Example:
        mc geometry invariant map-layers --source ./model-a --target ./model-b
        mc geometry invariant map-layers --source ./qwen --target ./llama --scope multiAtlas
        mc geometry invariant map-layers --source ./model-a --target ./model-b --atlas-sources sequence,semantic
    """
    context = _context(ctx)
    service = InvariantLayerMappingService()

    family_list = None
    if families:
        family_list = [f.strip() for f in families.split(",")]

    atlas_source_list = None
    if atlas_sources:
        atlas_source_list = [s.strip() for s in atlas_sources.split(",")]

    atlas_domain_list = None
    if atlas_domains:
        atlas_domain_list = [d.strip() for d in atlas_domains.split(",")]

    config = LayerMappingConfig(
        source_model_path=source,
        target_model_path=target,
        invariant_scope=scope,
        families=family_list,
        use_triangulation=triangulation,
        collapse_threshold=collapse_threshold,
        sample_layer_count=sample_layers,
        atlas_sources=atlas_source_list,
        atlas_domains=atlas_domain_list,
    )

    try:
        result = service.map_layers(config)
        payload = InvariantLayerMappingService.result_payload(result)

        if context.output_format == "text":
            summary = result.report.summary
            lines = [
                "INVARIANT LAYER MAPPING",
                f"Source: {result.report.source_model}",
                f"Target: {result.report.target_model}",
                f"Invariant Scope: {result.report.config.invariant_scope.value}",
                f"Invariants Used: {result.report.invariant_count}",
                "",
                "Results:",
                f"  Mapped Layers: {summary.mapped_layers}",
                f"  Skipped Layers: {summary.skipped_layers}",
                f"  Mean Similarity: {summary.mean_similarity:.3f}",
                f"  Alignment Quality: {summary.alignment_quality:.3f}",
                "",
                f"  Source Collapsed: {summary.source_collapsed_layers}",
                f"  Target Collapsed: {summary.target_collapsed_layers}",
            ]

            if summary.triangulation_quality != "none":
                lines.extend([
                    "",
                    "Triangulation:",
                    f"  Quality: {summary.triangulation_quality}",
                    f"  Mean Multiplier: {summary.mean_triangulation_multiplier:.2f}",
                ])

            # Show multi-atlas metrics if using multiAtlas scope
            if summary.total_probes_used > 68:
                lines.extend([
                    "",
                    "Multi-Atlas Coverage:",
                    f"  Total Probes: {summary.total_probes_used}",
                    f"  Atlas Sources: {summary.atlas_sources_detected}",
                    f"  Domains: {summary.atlas_domains_detected}",
                ])

            lines.extend([
                "",
                f"Interpretation: {result.interpretation}",
                f"Recommended Action: {result.recommended_action}",
            ])

            if result.report.mappings:
                lines.append("")
                lines.append("Layer Mappings (first 10):")
                for m in result.report.mappings[:10]:
                    skip_marker = " [skipped]" if m.is_skipped else ""
                    lines.append(
                        f"  L{m.source_layer} -> L{m.target_layer}: "
                        f"sim={m.similarity:.3f} conf={m.confidence.value}{skip_marker}"
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
    families: Optional[str] = typer.Option(
        None,
        "--families",
        help="Comma-separated list of families (default: all)",
    ),
    threshold: float = typer.Option(
        0.35,
        "--threshold",
        help="Collapse detection threshold (0.0-1.0)",
    ),
    sample_layers: int = typer.Option(
        12,
        "--sample-layers",
        help="Number of sample layers",
    ),
) -> None:
    """Analyze layer collapse risk for a model.

    Identifies layers where invariant activation is too sparse for
    reliable layer correspondence in merge operations.

    Example:
        mc geometry invariant collapse-risk --model ./qwen2.5-7b
        mc geometry invariant collapse-risk --model ./model --threshold 0.25
    """
    context = _context(ctx)
    service = InvariantLayerMappingService()

    family_list = None
    if families:
        family_list = [f.strip() for f in families.split(",")]

    config = CollapseRiskConfig(
        model_path=model,
        families=family_list,
        collapse_threshold=threshold,
        sample_layer_count=sample_layers,
    )

    try:
        result = service.analyze_collapse_risk(config)
        payload = InvariantLayerMappingService.collapse_risk_payload(result)

        if context.output_format == "text":
            risk_emoji = {
                "low": "[OK]",
                "medium": "[WARN]",
                "high": "[HIGH]",
                "critical": "[CRIT]",
            }.get(result.risk_level, "[?]")

            lines = [
                "COLLAPSE RISK ANALYSIS",
                f"Model: {result.model_path}",
                "",
                f"Risk Level: {risk_emoji} {result.risk_level.upper()}",
                f"Layer Count: {result.layer_count}",
                f"Collapsed Layers: {result.collapsed_layers}",
                f"Collapse Ratio: {result.collapse_ratio * 100:.1f}%",
                "",
                f"Interpretation: {result.interpretation}",
                f"Recommended Action: {result.recommended_action}",
            ]
            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as exc:
        error = ErrorDetail.from_exception(exc)
        write_error(error.message, context.output_format)
        raise typer.Exit(1) from exc
