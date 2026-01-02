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

"""Merge pipeline CLI commands.

Provides commands for the end-to-end merge pipeline:
    mc merge pipeline --source A --target B --output-dir OUT --transplant-domains mathematical,logical

Pipeline stages:
1. Pre-merge analysis: Interference prediction
2. Execute merge: Unified geometric merge
3. Post-merge validation: Extract geometry metrics
4. Verification: Compare predictions to actuals
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from modelcypher.cli.commands.model import prevent_sleep
from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(help="Merge pipeline commands")


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("pipeline")
def pipeline(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", "-s", help="Path to source model"),
    target: str = typer.Option(..., "--target", "-t", help="Path to target model"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for merged model"),
    transplant_domains: str = typer.Option(
        ...,
        "--transplant-domains",
        "-d",
        help="Comma-separated domains for transplant (e.g., mathematical,logical)",
    ),
    skip_pre_analysis: bool = typer.Option(
        False,
        "--skip-pre-analysis",
        help="Skip pre-merge interference analysis",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Enable/disable prediction verification",
    ),
    registry_path: str | None = typer.Option(
        None,
        "--registry-path",
        help="Path to store prediction/verification registry",
    ),
    output_file: str | None = typer.Option(
        None,
        "--output-file",
        "-f",
        help="Save full pipeline result to JSON file",
    ),
) -> None:
    """Run the complete merge pipeline.

    Executes all stages:
    1. Pre-merge: Interference analysis
    2. Merge: Null-space constrained transplant
    3. Post-merge: Extract geometry metrics
    4. Verify: Compare predictions to actuals

    Examples:
        mc merge pipeline --source ./instruct --target ./coder --output-dir ./merged \\
            --transplant-domains mathematical,logical

        mc merge pipeline -s /path/a -t /path/b -o /out -d spatial,social \\
            --skip-pre-analysis --output-file result.json
    """
    from modelcypher.core.use_cases.merge import MergePipelineService

    context = _context(ctx)

    # Parse domains
    domain_list = [d.strip() for d in transplant_domains.split(",") if d.strip()]
    if not domain_list:
        raise typer.BadParameter(
            "transplant-domains must specify at least one domain (e.g., mathematical,logical)"
        )

    service = MergePipelineService(verification_registry_path=registry_path)

    try:
        with prevent_sleep():
            result = service.run(
                source_path=source,
                target_path=target,
                output_dir=output_dir,
                transplant_domains=domain_list,
                skip_pre_analysis=skip_pre_analysis,
                verify_predictions=verify,
            )

        # Build output payload
        payload = {
            "_schema": "mc.merge.pipeline.v1",
            "pipelineId": result.pipeline_id,
            "timestamp": result.timestamp,
            "sourceModel": result.source_model,
            "targetModel": result.target_model,
            "outputDir": result.output_dir,
            "preMerge": {
                "domainsAnalyzed": result.pre_merge.domains_analyzed,
                "meanOverlap": result.pre_merge.mean_overlap,
                "meanAlignment": result.pre_merge.mean_alignment,
                "transformationCounts": result.pre_merge.transformation_counts,
                "totalTransformationsNeeded": result.pre_merge.total_transformations_needed,
            },
            "mergeResult": {
                "layerCount": result.merge_result.get("layer_count"),
                "weightCount": result.merge_result.get("weight_count"),
                "meanConfidence": result.merge_result.get("mean_confidence"),
            },
            "postMerge": {
                "meanConfidence": result.post_merge.mean_confidence,
                "layersTransplanted": result.post_merge.layers_transplanted,
                "weightsTransplanted": result.post_merge.weights_transplanted,
                "meanPreservedFraction": result.post_merge.mean_preserved_fraction,
                "meanCkaAfter": result.post_merge.mean_cka_after,
            },
            "verification": result.verification,
            "timing": {
                "preMergeDurationS": round(result.pre_merge_duration_s, 2),
                "mergeDurationS": round(result.merge_duration_s, 2),
                "validationDurationS": round(result.validation_duration_s, 2),
            },
        }

        # Save full result if requested
        if output_file:
            full_result = {
                "_schema": "mc.merge.pipeline.full.v1",
                "pipelineId": result.pipeline_id,
                "timestamp": result.timestamp,
                "sourceModel": result.source_model,
                "targetModel": result.target_model,
                "outputDir": result.output_dir,
                "preMerge": asdict(result.pre_merge),
                "mergeResult": result.merge_result,
                "postMerge": asdict(result.post_merge),
                "verification": result.verification,
                "timing": {
                    "preMergeDurationS": result.pre_merge_duration_s,
                    "mergeDurationS": result.merge_duration_s,
                    "validationDurationS": result.validation_duration_s,
                },
            }
            Path(output_file).write_text(json.dumps(full_result, indent=2, default=str))
            typer.echo(f"Full result saved to {output_file}")

        # Text output
        if context.output_format == "text":
            lines = [
                "=" * 70,
                "MERGE PIPELINE RESULT",
                "=" * 70,
                f"Pipeline ID: {result.pipeline_id}",
                f"Source: {result.source_model}",
                f"Target: {result.target_model}",
                f"Output: {result.output_dir}",
                "",
                "PRE-MERGE ANALYSIS",
                f"  Domains: {', '.join(result.pre_merge.domains_analyzed)}",
                f"  Mean Overlap: {result.pre_merge.mean_overlap:.4f}",
                f"  Mean Alignment: {result.pre_merge.mean_alignment:.4f}",
                f"  Transformations Needed: {result.pre_merge.total_transformations_needed}",
                "",
                "MERGE RESULT",
                f"  Layers: {result.merge_result.get('layer_count')}",
                f"  Weights: {result.merge_result.get('weight_count')}",
                f"  Mean Confidence: {result.merge_result.get('mean_confidence', 0):.4f}",
                # Safety Verdict removed - raw measurements only
                "",
                "POST-MERGE VALIDATION",
                f"  Mean Preserved Fraction: {result.post_merge.mean_preserved_fraction:.4f}",
                f"  Mean CKA After: {result.post_merge.mean_cka_after:.4f}",
                f"  Layers Transplanted: {result.post_merge.layers_transplanted}",
                f"  Weights Transplanted: {result.post_merge.weights_transplanted}",
            ]

            if result.verification:
                lines.extend([
                    "",
                    "VERIFICATION",
                    f"  Merge ID: {result.verification.get('merge_id')}",
                    f"  Mean Absolute Error: {result.verification.get('mean_absolute_error', 0):.4f}",
                    f"  Overlap Delta: {result.verification.get('overlap_delta', 0):.4f}",
                    f"  Alignment Delta: {result.verification.get('alignment_delta', 0):.4f}",
                ])

            lines.extend([
                "",
                "TIMING",
                f"  Pre-merge: {result.pre_merge_duration_s:.2f}s",
                f"  Merge: {result.merge_duration_s:.2f}s",
                f"  Validation: {result.validation_duration_s:.2f}s",
                "=" * 70,
            ])

            write_output("\n".join(lines), context.output_format, context.pretty)
            return

        write_output(payload, context.output_format, context.pretty)

    except Exception as e:
        error = ErrorDetail(
            code="MC-1100",
            title="Pipeline failed",
            detail=str(e),
            hint="Check model paths and merge parameters",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)
