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
        EntropySignature,
        JailbreakEntropyTaxonomy,
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
        EntropySignature,
        JailbreakEntropyTaxonomy,
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
    output_file: str | None = typer.Option(
        None, "--output-file", "-o", help="Output markdown file"
    ),
) -> None:
    """Generate markdown report for taxonomy experiment.

    Examples:
        mc research taxonomy report ./signatures.json --model llama3 -o report.md
    """
    context = _context(ctx)
    from modelcypher.core.domain.research import (
        EntropySignature,
        JailbreakEntropyTaxonomy,
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


@app.command("sparse-region")
def research_sparse_region(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Analyze sparse activation regions in a model."""
    context = _context(ctx)
    from modelcypher.cli.output import write_error
    from modelcypher.core.use_cases.research_service import ResearchService

    service = ResearchService()

    try:
        result = service.sparse_region(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1021",
            title="Sparse region analysis failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": result.model_path,
        "totalSparsity": result.total_sparsity,
        "layerCount": result.layer_count,
        "regions": [
            {
                "layerName": r.layer_name,
                "startIndex": r.start_index,
                "endIndex": r.end_index,
                "sparsityRatio": r.sparsity_ratio,
                "activationPattern": r.activation_pattern,
            }
            for r in result.regions
        ],
    }

    if context.output_format == "text":
        lines = [
            "SPARSE REGION ANALYSIS",
            f"Model: {result.model_path}",
            f"Total Sparsity: {result.total_sparsity:.1%}",
            f"Layers Analyzed: {result.layer_count}",
            "",
            "Regions:",
        ]
        for r in result.regions[:10]:
            lines.append(f"  {r.layer_name}")
            lines.append(f"    Sparsity: {r.sparsity_ratio:.1%}")
            lines.append(f"    Pattern: {r.activation_pattern}")
        if len(result.regions) > 10:
            lines.append(f"  ... and {len(result.regions) - 10} more regions")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("multimodal-merge")
def research_multimodal_merge(
    ctx: typer.Context,
    target_model: str = typer.Argument(..., help="Path to target LLM model"),
    concepts_file: str | None = typer.Option(
        None, "--concepts", "-c", help="JSON file with concept list"
    ),
    include_clip: bool = typer.Option(True, "--clip/--no-clip", help="Include CLIP vision knowledge"),
    include_whisper: bool = typer.Option(True, "--whisper/--no-whisper", help="Include Whisper audio knowledge"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output JSON file for results"),
) -> None:
    """Merge multi-modal knowledge (CLIP, Whisper) into an LLM.

    Takes knowledge from vision (CLIP) and audio (Whisper) models and
    projects it into the target LLM's null space. The LLM gains multi-modal
    understanding without losing existing capabilities.

    Examples:
        mc research multimodal-merge /path/to/llm
        mc research multimodal-merge /path/to/llm --concepts ./concepts.json
        mc research multimodal-merge /path/to/llm --no-whisper -o results.json
    """
    context = _context(ctx)
    from modelcypher.cli.output import write_error
    from modelcypher.core.use_cases.multimodal_merge_service import MultiModalMergeService

    # Default concepts if not provided
    default_concepts = [
        "a red ball",
        "a blue sky",
        "a green tree",
        "running fast",
        "walking slowly",
        "loud noise",
        "quiet whisper",
        "music playing",
        "happiness",
        "sadness",
    ]

    concepts = default_concepts
    if concepts_file:
        try:
            with open(concepts_file) as f:
                concepts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            error = ErrorDetail(
                code="MC-3001",
                title="Failed to load concepts file",
                detail=str(exc),
                trace_id=context.trace_id,
            )
            write_error(error.as_dict(), context.output_format, context.pretty)
            raise typer.Exit(code=1)

    service = MultiModalMergeService()

    try:
        result = service.merge(
            target_model=target_model,
            concepts=concepts,
            include_clip=include_clip,
            include_whisper=include_whisper,
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="Multi-modal merge failed",
            detail=str(exc),
            hint="Ensure target model path is valid and dependencies are installed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "_schema": "mc.research.multimodal_merge.v1",
        "targetModel": result.target_model,
        "sourceModels": list(result.source_models),
        "conceptCount": len(result.concepts),
        "ckaPreservation": result.cka_preservation,
        "alignmentResults": [
            {
                "modality": r.source_modality.value,
                "ckaBefore": r.cka_before,
                "ckaAfter": r.cka_after,
                "transformShape": list(r.transform_shape),
            }
            for r in result.alignment_results
        ],
        "mergeResults": [
            {
                "modality": r.source_modality.value,
                "preservedFraction": r.preserved_fraction,
                "projectionLoss": r.projection_loss,
            }
            for r in result.merge_results
        ],
    }

    if output_file:
        Path(output_file).write_text(json.dumps(payload, indent=2))

    if context.output_format == "text":
        lines = [
            "MULTI-MODAL MERGE RESULTS",
            "",
            f"Target: {result.target_model}",
            f"Concepts: {len(result.concepts)}",
            "",
            "Alignment Results:",
        ]
        for r in result.alignment_results:
            lines.append(f"  {r.source_modality.value}: CKA {r.cka_before:.4f} → {r.cka_after:.4f}")

        lines.append("")
        lines.append("Merge Results:")
        for r in result.merge_results:
            lines.append(f"  {r.source_modality.value}: preserved {r.preserved_fraction:.4f}")

        lines.append("")
        lines.append(f"Geometry Preservation: CKA = {result.cka_preservation:.4f}")

        if result.cka_preservation >= 0.95:
            lines.append("  ✓ Original geometry fully preserved!")
        elif result.cka_preservation >= 0.90:
            lines.append("  ✓ Original geometry mostly preserved")
        else:
            lines.append("  ⚠ Some geometry drift detected")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("multimodal-offramp")
def research_multimodal_offramp(
    ctx: typer.Context,
    target_model: str = typer.Argument(..., help="Path to target LLM model"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Output directory for offramp weights"),
    include_clip: bool = typer.Option(True, "--clip/--no-clip", help="Include CLIP vision offramp"),
    include_whisper: bool = typer.Option(True, "--whisper/--no-whisper", help="Include Whisper audio offramp"),
) -> None:
    """Create multi-modal offramp projections for inference-time knowledge access.

    This command creates bidirectional projection matrices ("offramps") that allow
    the LLM to access multi-modal knowledge (vision, audio) during inference.

    Offramps enable:
    - Forward projection: LLM hidden states → modality-aligned space
    - Inverse projection: Modality embeddings → LLM token space

    Based on DeepSeek mHC (arXiv:2512.24880) and null-space projection theory.
    See docs/research/mhc_null_space_connection.md for theoretical foundation.

    Examples:
        mc research multimodal-offramp /path/to/llm
        mc research multimodal-offramp /path/to/llm -o ./offramps
        mc research multimodal-offramp /path/to/llm --no-whisper
    """
    context = _context(ctx)
    from modelcypher.cli.output import write_error
    from modelcypher.core.domain.multimodal import MultiModalChannelAdapter

    adapter = MultiModalChannelAdapter()

    try:
        result = adapter.create_offramps(
            target_model=target_model,
            include_clip=include_clip,
            include_whisper=include_whisper,
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3003",
            title="Offramp creation failed",
            detail=str(exc),
            hint="Ensure target model path is valid and CLIP/Whisper dependencies installed",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Save offramp weights if output directory specified
    if output_dir:
        import mlx.core as mx
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each offramp as a .safetensors file
        for offramp in result.offramps:
            modality_name = offramp.modality.value
            weights = {
                "projection_matrix": offramp.projection_matrix,
                "inverse_projection": offramp.inverse_projection,
            }
            # Save using MLX's native format
            mx.save_safetensors(str(output_path / f"{modality_name}_offramp.safetensors"), weights)

        # Save metadata
        metadata = {
            "target_model": result.target_model,
            "cka_preservation": result.cka_preservation,
            "concepts_used": list(result.concepts_used),
            "offramps": [
                {
                    "modality": o.modality.value,
                    "source_model": o.source_model,
                    "cka_alignment": o.cka_alignment,
                }
                for o in result.offramps
            ],
        }
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    payload = {
        "_schema": "mc.research.multimodal_offramp.v1",
        "targetModel": result.target_model,
        "ckaPreservation": result.cka_preservation,
        "conceptCount": len(result.concepts_used),
        "offramps": [
            {
                "modality": o.modality.value,
                "sourceModel": o.source_model,
                "ckaAlignment": o.cka_alignment,
            }
            for o in result.offramps
        ],
    }

    if context.output_format == "text":
        lines = [
            "MULTI-MODAL OFFRAMP CREATION",
            "",
            f"Target: {result.target_model}",
            f"CKA Preservation: {result.cka_preservation:.4f}",
            f"Concepts: {len(result.concepts_used)}",
            "",
            "Offramps Created:",
        ]
        for o in result.offramps:
            lines.append(f"  {o.modality.value}:")
            lines.append(f"    Source: {o.source_model}")
            lines.append(f"    CKA: {o.cka_alignment:.4f}")

        if output_dir:
            lines.append("")
            lines.append(f"Weights saved to: {output_dir}")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("afm")
def research_afm(
    ctx: typer.Context,
    model_path: str = typer.Argument(..., help="Path to model directory"),
) -> None:
    """Run activation function mapping analysis."""
    context = _context(ctx)
    from modelcypher.cli.output import write_error
    from modelcypher.core.use_cases.research_service import ResearchService

    service = ResearchService()

    try:
        result = service.afm(model_path)
    except ValueError as exc:
        error = ErrorDetail(
            code="MC-1022",
            title="AFM analysis failed",
            detail=str(exc),
            hint="Ensure the path points to a valid model directory with config.json",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "modelPath": result.model_path,
        "dominantPatterns": result.dominant_patterns,
        "layerSummaries": [
            {
                "layerName": s.layer_name,
                "dominantPattern": s.dominant_pattern,
                "meanActivation": s.mean_activation,
                "maxActivation": s.max_activation,
            }
            for s in result.layer_summaries
        ],
        "activationMaps": {
            k: v[:5]
            for k, v in result.activation_maps.items()
        },
    }

    if context.output_format == "text":
        lines = [
            "ACTIVATION FUNCTION MAPPING",
            f"Model: {result.model_path}",
            f"Dominant Patterns: {', '.join(result.dominant_patterns)}",
            "",
            "Layer Summaries:",
        ]
        for s in result.layer_summaries[:10]:
            lines.append(f"  {s.layer_name}")
            lines.append(f"    Pattern: {s.dominant_pattern}")
            lines.append(f"    Mean Activation: {s.mean_activation:.4f}")
            lines.append(f"    Max Activation: {s.max_activation:.4f}")
        if len(result.layer_summaries) > 10:
            lines.append(f"  ... and {len(result.layer_summaries) - 10} more layers")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
