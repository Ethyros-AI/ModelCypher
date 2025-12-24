"""Research CLI commands.

Provides commands for research experiments and analysis,
including jailbreak entropy taxonomy.

Commands:
    mc research taxonomy run --signatures <file> --model <id>
    mc research taxonomy cluster --signatures <file> --k <clusters>
"""

from __future__ import annotations

import json
from pathlib import Path


import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)
taxonomy_app = typer.Typer(no_args_is_help=True)
app.add_typer(taxonomy_app, name="taxonomy")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@taxonomy_app.command("run")
def taxonomy_run(
    ctx: typer.Context,
    signatures_file: str = typer.Argument(..., help="Path to JSON file with entropy signatures"),
    model_id: str = typer.Option("unknown", "--model", help="Model identifier"),
    k: int = typer.Option(5, "--k", help="Number of clusters"),
    test_split: float = typer.Option(0.2, "--test-split", help="Fraction for test set"),
) -> None:
    """Run full C1 jailbreak entropy taxonomy experiment.

    The signatures file should contain an array of objects with:
    - trajectory: array of entropy values
    - attack_category: category label
    - is_harmful: boolean
    - prompt_prefix: prompt text (truncated)

    Examples:
        mc research taxonomy run ./signatures.json --model llama3 --k 5
    """
    context = _context(ctx)
    from modelcypher.core.domain.research import (
        JailbreakEntropyTaxonomy,
        EntropySignature,
    )

    # Load signatures
    try:
        with open(signatures_file) as f:
            data = json.load(f)

        signatures = [
            EntropySignature(
                trajectory=sig["trajectory"],
                attack_category=sig["attack_category"],
                is_harmful=sig["is_harmful"],
                prompt_prefix=sig.get("prompt_prefix", ""),
            )
            for sig in data
        ]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        from modelcypher.cli.output import write_error
        error = ErrorDetail(
            code="MC-2001",
            title="Failed to load signatures",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    if len(signatures) < k:
        from modelcypher.cli.output import write_error
        error = ErrorDetail(
            code="MC-2002",
            title="Insufficient signatures",
            detail=f"Need at least {k} signatures for {k} clusters, got {len(signatures)}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    taxonomy = JailbreakEntropyTaxonomy()
    report = taxonomy.run_experiment(
        signatures=signatures,
        model_id=model_id,
        k=k,
        test_split=test_split,
    )

    payload = {
        "experimentId": str(report.experiment_id),
        "modelId": report.model_id,
        "testAccuracy": report.test_accuracy,
        "successMetricAchieved": report.success_metric_achieved,
        "signatureCount": len(report.signatures),
        "clusterCount": len(report.clusters),
        "categoryLabels": report.category_labels,
        "categoryPrecision": report.category_precision,
        "categoryRecall": report.category_recall,
        "timestamp": report.timestamp.isoformat(),
        "notes": report.notes,
    }

    if context.output_format == "text":
        lines = [
            "C1: JAILBREAK ENTROPY TAXONOMY",
            "",
            f"Experiment ID: {report.experiment_id}",
            f"Model: {report.model_id}",
            f"Timestamp: {report.timestamp}",
            "",
            "Results:",
            f"  Test Accuracy: {report.test_accuracy * 100:.1f}%",
            f"  Success Metric (>70%): {'YES' if report.success_metric_achieved else 'NO'}",
            f"  Signatures: {len(report.signatures)}",
            f"  Clusters: {len(report.clusters)}",
            "",
            "Per-Category Metrics:",
        ]
        for cat in report.category_labels:
            p = report.category_precision.get(cat, 0)
            r = report.category_recall.get(cat, 0)
            lines.append(f"  {cat}: P={p * 100:.1f}%, R={r * 100:.1f}%")

        if report.notes:
            lines.append("")
            lines.append(f"Notes: {report.notes}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@taxonomy_app.command("cluster")
def taxonomy_cluster(
    ctx: typer.Context,
    signatures_file: str = typer.Argument(..., help="Path to JSON file with entropy signatures"),
    k: int = typer.Option(5, "--k", help="Number of clusters"),
) -> None:
    """Cluster entropy signatures into taxonomy.

    Examples:
        mc research taxonomy cluster ./signatures.json --k 5
    """
    context = _context(ctx)
    from modelcypher.core.domain.research import (
        JailbreakEntropyTaxonomy,
        EntropySignature,
    )

    # Load signatures
    try:
        with open(signatures_file) as f:
            data = json.load(f)

        signatures = [
            EntropySignature(
                trajectory=sig["trajectory"],
                attack_category=sig["attack_category"],
                is_harmful=sig["is_harmful"],
                prompt_prefix=sig.get("prompt_prefix", ""),
            )
            for sig in data
        ]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        from modelcypher.cli.output import write_error
        error = ErrorDetail(
            code="MC-2001",
            title="Failed to load signatures",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    taxonomy = JailbreakEntropyTaxonomy()
    clusters = taxonomy.cluster(signatures=signatures, k=k)

    payload = {
        "signatureCount": len(signatures),
        "clusterCount": len(clusters),
        "clusters": [
            {
                "clusterId": c.cluster_id,
                "dominantCategory": c.dominant_category,
                "memberCount": len(c.member_indices),
                "categoryDistribution": c.category_distribution,
            }
            for c in clusters
        ],
    }

    if context.output_format == "text":
        lines = [
            "ENTROPY SIGNATURE CLUSTERS",
            "",
            f"Signatures: {len(signatures)}",
            f"Clusters: {len(clusters)}",
            "",
        ]
        for c in sorted(clusters, key=lambda x: x.cluster_id):
            lines.append(f"Cluster {c.cluster_id}: {c.dominant_category}")
            lines.append(f"  Members: {len(c.member_indices)}")
            lines.append("  Categories:")
            for cat, count in sorted(c.category_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"    {cat}: {count}")
            lines.append("")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@taxonomy_app.command("report")
def taxonomy_report(
    ctx: typer.Context,
    signatures_file: str = typer.Argument(..., help="Path to JSON file with entropy signatures"),
    model_id: str = typer.Option("unknown", "--model", help="Model identifier"),
    k: int = typer.Option(5, "--k", help="Number of clusters"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output markdown file"),
) -> None:
    """Generate markdown report for taxonomy experiment.

    Examples:
        mc research taxonomy report ./signatures.json --model llama3 -o report.md
    """
    context = _context(ctx)
    from modelcypher.core.domain.research import (
        JailbreakEntropyTaxonomy,
        EntropySignature,
    )

    # Load signatures
    try:
        with open(signatures_file) as f:
            data = json.load(f)

        signatures = [
            EntropySignature(
                trajectory=sig["trajectory"],
                attack_category=sig["attack_category"],
                is_harmful=sig["is_harmful"],
                prompt_prefix=sig.get("prompt_prefix", ""),
            )
            for sig in data
        ]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        from modelcypher.cli.output import write_error
        error = ErrorDetail(
            code="MC-2001",
            title="Failed to load signatures",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    taxonomy = JailbreakEntropyTaxonomy()
    report = taxonomy.run_experiment(
        signatures=signatures,
        model_id=model_id,
        k=k,
    )

    markdown = report.generate_markdown_report()

    if output_file:
        Path(output_file).write_text(markdown)
        write_output(
            {"status": "success", "outputFile": output_file},
            context.output_format,
            context.pretty,
        )
    else:
        write_output(markdown, "text", context.pretty)
