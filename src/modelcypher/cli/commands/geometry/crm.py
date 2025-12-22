"""Geometry CRM (Concept Response Matrix) CLI commands.

Provides commands for:
- Building concept response matrices
- Comparing CRMs between models
- Listing sequence invariant probes

Commands:
    mc geometry crm build --model <path> --output-path <path>
    mc geometry crm compare --source <path> --target <path>
    mc geometry crm sequence-inventory
"""

from __future__ import annotations

from typing import Optional

import typer

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.core.domain.agents.sequence_invariant_atlas import (
    SequenceFamily,
    SequenceInvariantInventory,
)
from modelcypher.core.use_cases.concept_response_matrix_service import (
    CRMBuildConfig,
    ConceptResponseMatrixService,
)
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)

_CRM_DEFAULTS = CRMBuildConfig()


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("build")
def geometry_crm_build(
    ctx: typer.Context,
    model_path: str = typer.Option(..., "--model", help="Path to model directory"),
    output_path: str = typer.Option(..., "--output-path", help="Output CRM JSON path"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Optional adapter directory"),
    include_primes: bool = typer.Option(
        True,
        "--include-primes/--no-include-primes",
        help="Include semantic prime anchors",
    ),
    include_gates: bool = typer.Option(
        True,
        "--include-gates/--no-include-gates",
        help="Include computational gate anchors",
    ),
    include_polyglot: bool = typer.Option(
        True,
        "--include-polyglot/--no-include-polyglot",
        help="Include multilingual prime variants",
    ),
    include_sequence_invariants: bool = typer.Option(
        True,
        "--include-sequence-invariants/--no-include-sequence-invariants",
        help="Include sequence invariant anchors (fibonacci, logic, causality, etc.)",
    ),
    sequence_families: Optional[str] = typer.Option(
        None,
        "--sequence-families",
        help="Comma-separated sequence families: fibonacci,lucas,tribonacci,primes,catalan,ramanujan,logic,ordering,arithmetic,causality",
    ),
    max_prompts_per_anchor: int = typer.Option(
        _CRM_DEFAULTS.max_prompts_per_anchor,
        "--max-prompts-per-anchor",
        help="Max prompts per anchor",
    ),
    max_polyglot_texts_per_language: int = typer.Option(
        _CRM_DEFAULTS.max_polyglot_texts_per_language,
        "--max-polyglot-texts-per-language",
        help="Max polyglot texts per language",
    ),
    anchor_prefixes: Optional[str] = typer.Option(
        None,
        "--anchor-prefixes",
        help="Comma-separated anchor prefixes (prime, gate)",
    ),
    max_anchors: Optional[int] = typer.Option(
        None,
        "--max-anchors",
        help="Limit number of anchors for quick runs",
    ),
) -> None:
    """Build a concept response matrix (CRM) for a model.

    Examples:
        mc geometry crm build --model ./model --output-path ./crm.json
        mc geometry crm build --model ./model --output-path ./crm.json --max-anchors 10
    """
    context = _context(ctx)
    service = ConceptResponseMatrixService(engine=LocalInferenceEngine())

    prefixes = None
    if anchor_prefixes:
        prefixes = [value.strip() for value in anchor_prefixes.split(",") if value.strip()]

    parsed_families: frozenset[SequenceFamily] | None = None
    if sequence_families:
        family_list = [val.strip().lower() for val in sequence_families.split(",") if val.strip()]
        family_set: set[SequenceFamily] = set()
        for name in family_list:
            try:
                family_set.add(SequenceFamily(name))
            except ValueError:
                pass  # Ignore invalid family names
        if family_set:
            parsed_families = frozenset(family_set)

    config = CRMBuildConfig(
        include_primes=include_primes,
        include_gates=include_gates,
        include_polyglot=include_polyglot,
        include_sequence_invariants=include_sequence_invariants,
        sequence_families=parsed_families,
        max_prompts_per_anchor=max_prompts_per_anchor,
        max_polyglot_texts_per_language=max_polyglot_texts_per_language,
        anchor_prefixes=prefixes,
        max_anchors=max_anchors,
    )

    try:
        summary = service.build(
            model_path=model_path,
            output_path=output_path,
            config=config,
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
    }

    if context.output_format == "text":
        lines = [
            "CONCEPT RESPONSE MATRIX",
            f"Model: {summary.model_path}",
            f"Output: {summary.output_path}",
            f"Layers: {summary.layer_count}",
            f"Hidden Dim: {summary.hidden_dim}",
            f"Anchors: {summary.anchor_count} (primes {summary.prime_count}, gates {summary.gate_count}, seq {summary.sequence_invariant_count})",
        ]
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("compare")
def geometry_crm_compare(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source CRM JSON path"),
    target: str = typer.Option(..., "--target", help="Target CRM JSON path"),
    include_matrix: bool = typer.Option(False, "--include-matrix", help="Include full CKA matrix"),
) -> None:
    """Compare two CRMs and compute layer correspondence via CKA.

    Examples:
        mc geometry crm compare --source ./crm1.json --target ./crm2.json
        mc geometry crm compare --source ./crm1.json --target ./crm2.json --include-matrix
    """
    context = _context(ctx)
    service = ConceptResponseMatrixService()

    try:
        summary = service.compare(source, target, include_matrix=include_matrix)
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
        "overallAlignment": summary.overall_alignment,
        "layerCorrespondence": summary.layer_correspondence,
    }
    if summary.cka_matrix is not None:
        payload["ckaMatrix"] = summary.cka_matrix

    if context.output_format == "text":
        lines = [
            "CRM COMPARISON",
            f"Source: {summary.source_path}",
            f"Target: {summary.target_path}",
            f"Common Anchors: {summary.common_anchor_count}",
            f"Overall Alignment: {summary.overall_alignment:.4f}",
        ]
        if summary.layer_correspondence:
            lines.append("")
            lines.append("Layer Correspondence (top 10):")
            for match in summary.layer_correspondence[:10]:
                lines.append(
                    f"  {match['sourceLayer']} -> {match['targetLayer']} (CKA {match['cka']:.4f})"
                )
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("sequence-inventory")
def geometry_crm_sequence_inventory(
    ctx: typer.Context,
    family: Optional[str] = typer.Option(
        None,
        "--family",
        help="Filter by family: fibonacci, lucas, tribonacci, primes, catalan, ramanujan, logic, ordering, arithmetic, causality",
    ),
) -> None:
    """List available sequence invariant probes for CRM anchoring.

    Examples:
        mc geometry crm sequence-inventory
        mc geometry crm sequence-inventory --family fibonacci
    """
    context = _context(ctx)

    family_filter: set[SequenceFamily] | None = None
    if family:
        try:
            family_filter = {SequenceFamily(family.strip().lower())}
        except ValueError:
            error = ErrorDetail(
                code="MC-1050",
                title="Invalid sequence family",
                detail=f"Unknown family '{family}'",
                hint="Valid families: fibonacci, lucas, tribonacci, primes, catalan, ramanujan, logic, ordering, arithmetic, causality",
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    probes = SequenceInvariantInventory.probes_for_families(family_filter)
    counts = SequenceInvariantInventory.probe_count_by_family()

    probe_list = [
        {
            "id": probe.id,
            "family": probe.family.value,
            "domain": probe.domain.value,
            "name": probe.name,
            "description": probe.description,
            "weight": probe.cross_domain_weight,
        }
        for probe in probes
    ]

    payload = {
        "totalProbes": len(probes),
        "familyCounts": {fam.value: count for fam, count in counts.items()},
        "probes": probe_list,
    }

    if context.output_format == "text":
        lines = [
            "SEQUENCE INVARIANT INVENTORY",
            f"Total Probes: {len(probes)}",
            "",
            "Probes by Family:",
        ]
        for fam, count in sorted(counts.items(), key=lambda x: x[0].value):
            lines.append(f"  {fam.value}: {count}")
        lines.append("")
        lines.append("Probes (first 20):")
        for probe in probes[:20]:
            lines.append(f"  [{probe.family.value}] {probe.id}: {probe.name}")
        if len(probes) > 20:
            lines.append(f"  ... and {len(probes) - 20} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
