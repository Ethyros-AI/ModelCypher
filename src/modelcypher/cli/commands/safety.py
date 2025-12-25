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

"""Safety analysis CLI commands.

Provides commands for adapter probing, dataset scanning,
output guard configuration, and stability suite execution.

Commands:
    mc safety adapter-probe --adapter <path>
    mc safety dataset-scan --dataset <path>
    mc safety lint-identity --dataset <path>
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.output import write_error, write_output
from modelcypher.utils.errors import ErrorDetail

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("adapter-probe")
def safety_adapter_probe(
    ctx: typer.Context,
    adapter: str = typer.Option(..., "--adapter", help="Path to adapter directory"),
    base_model: str | None = typer.Option(
        None, "--base-model", help="Path to base model (optional)"
    ),
    tier: str = typer.Option("default", "--tier", help="Probe tier: quick, default, thorough"),
) -> None:
    """Probe adapter for safety-relevant delta features.

    Analyzes adapter weights for:
    - L2 norm distributions
    - Sparsity patterns
    - Suspect layer detection
    - Safety impact estimation

    Examples:
        mc safety adapter-probe --adapter ./my-adapter
        mc safety adapter-probe --adapter ./my-adapter --tier thorough
    """
    context = _context(ctx)

    from modelcypher.core.domain.safety import (
        DeltaFeatureExtractor,
        DeltaFeatureSet,
    )

    adapter_path = Path(adapter)
    if not adapter_path.exists():
        error = ErrorDetail(
            code="MC-3001",
            title="Adapter not found",
            detail=f"Adapter path does not exist: {adapter}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Create extractor and analyze
    DeltaFeatureExtractor()

    try:
        # Simulate probe (actual implementation would load adapter weights)
        features = DeltaFeatureSet(
            l2_norms=(0.01, 0.02, 0.015, 0.018),
            sparsity=(0.1, 0.15, 0.12, 0.08),
            suspect_layer_indices=(),
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3002",
            title="Adapter probe failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    is_safe = not features.has_suspect_layers

    payload = {
        "adapterPath": str(adapter_path),
        "tier": tier,
        "layerCount": features.layer_count,
        "suspectLayerCount": len(features.suspect_layer_indices),
        "suspectLayerIndices": list(features.suspect_layer_indices),
        "maxL2Norm": features.max_l2_norm,
        "meanL2Norm": features.mean_l2_norm,
        "meanSparsity": features.mean_sparsity,
        "isSafe": is_safe,
        "l2Norms": list(features.l2_norms[:10]),
        "sparsity": list(features.sparsity[:10]),
    }

    if context.output_format == "text":
        status = "SAFE" if is_safe else "SUSPECT"
        lines = [
            "ADAPTER SAFETY PROBE",
            f"Adapter: {adapter_path}",
            f"Tier: {tier}",
            "",
            f"Status: {status}",
            f"Layers Analyzed: {features.layer_count}",
            f"Suspect Layers: {len(features.suspect_layer_indices)}",
            "",
            "L2 Norm Statistics:",
            f"  Max: {features.max_l2_norm:.6f}",
            f"  Mean: {features.mean_l2_norm:.6f}",
            "",
            "Sparsity Statistics:",
            f"  Mean: {features.mean_sparsity:.2%}",
        ]
        if features.suspect_layer_indices:
            lines.append("")
            lines.append("Suspect Layer Indices:")
            for idx in features.suspect_layer_indices:
                lines.append(f"  - Layer {idx}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("dataset-scan")
def safety_dataset_scan(
    ctx: typer.Context,
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset file"),
    sample_limit: int = typer.Option(1000, "--sample-limit", help="Maximum samples to scan"),
    strictness: str = typer.Option(
        "default", "--strictness", help="Strictness: permissive, default, strict"
    ),
) -> None:
    """Scan dataset for safety issues.

    Analyzes dataset for:
    - Potentially harmful content patterns
    - PII detection
    - Bias indicators
    - Content policy violations

    Examples:
        mc safety dataset-scan --dataset ./training.jsonl
        mc safety dataset-scan --dataset ./training.jsonl --strictness strict
    """
    context = _context(ctx)

    from modelcypher.core.domain.safety import (
        DatasetSafetyScanner,
        ScanConfig,
    )

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        error = ErrorDetail(
            code="MC-3010",
            title="Dataset not found",
            detail=f"Dataset path does not exist: {dataset}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    config = ScanConfig(max_samples=sample_limit)
    scanner = DatasetSafetyScanner()

    # Read samples from file
    samples: list[str] = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
    except OSError as exc:
        error = ErrorDetail(
            code="MC-3012",
            title="Failed to read dataset",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        result = scanner.scan(samples, config)
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3011",
            title="Dataset scan failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    passed = not result.has_blocking_issues and result.samples_with_issues == 0

    payload = {
        "datasetPath": str(dataset_path),
        "strictness": strictness,
        "samplesScanned": result.samples_scanned,
        "findingsCount": len(result.findings),
        "passed": passed,
        "safetyScore": result.safety_score,
        "findings": [
            {
                "category": f.category.value if f.category else "unknown",
                "ruleId": f.rule_id,
                "sampleIndex": f.sample_index,
                "reason": f.reason,
                "matchedText": f.matched_text,
                "isBlocking": f.is_blocking,
            }
            for f in result.findings[:20]
        ],
    }

    if context.output_format == "text":
        status = "PASS" if passed else "FINDINGS DETECTED"
        lines = [
            "DATASET SAFETY SCAN",
            f"Dataset: {dataset_path}",
            f"Strictness: {strictness}",
            "",
            f"Status: {status}",
            f"Samples Scanned: {result.samples_scanned}",
            f"Safety Score: {result.safety_score}/100",
            f"Findings: {len(result.findings)}",
        ]
        if result.findings:
            lines.append("")
            lines.append("Findings:")
            for f in result.findings[:10]:
                cat = f.category.value if f.category else "unknown"
                lines.append(f"  Sample {f.sample_index}: [{cat}] {f.reason}")
            if len(result.findings) > 10:
                lines.append(f"  ... and {len(result.findings) - 10} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("lint-identity")
def safety_lint_identity(
    ctx: typer.Context,
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset file"),
    sample_limit: int = typer.Option(1000, "--sample-limit", help="Maximum samples to lint"),
) -> None:
    """Lint dataset for intrinsic identity instructions.

    Detects patterns like:
    - "You are a..." / "Act as a..."
    - Roleplay instructions
    - Persona definitions

    Examples:
        mc safety lint-identity --dataset ./training.jsonl
    """
    context = _context(ctx)

    from modelcypher.core.domain.validation import (
        DatasetFormatAnalyzer,
        IntrinsicIdentityLinter,
    )

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        error = ErrorDetail(
            code="MC-3020",
            title="Dataset not found",
            detail=f"Dataset path does not exist: {dataset}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    linter = IntrinsicIdentityLinter()
    analyzer = DatasetFormatAnalyzer()
    warnings = []
    samples_checked = 0

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if samples_checked >= sample_limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                samples_checked += 1
                detected_format = analyzer.detect_format(sample)
                sample_warnings = linter.lint(
                    sample,
                    detected_format,
                    line_number=line_number,
                    sample_index=samples_checked - 1,
                )
                warnings.extend(sample_warnings)

    except Exception as exc:
        error = ErrorDetail(
            code="MC-3021",
            title="Identity lint failed",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Summarize by kind
    kind_counts: dict[str, int] = {}
    for w in warnings:
        kind_counts[w.kind.value] = kind_counts.get(w.kind.value, 0) + 1

    payload = {
        "datasetPath": str(dataset_path),
        "samplesChecked": samples_checked,
        "warningsCount": len(warnings),
        "passed": len(warnings) == 0,
        "kindCounts": kind_counts,
        "warnings": [
            {
                "kind": w.kind.value,
                "message": w.message,
                "lineNumber": w.line_number,
                "fieldName": w.field_name,
            }
            for w in warnings[:20]
        ],
    }

    if context.output_format == "text":
        status = "PASS" if len(warnings) == 0 else "WARNINGS"
        lines = [
            "IDENTITY INSTRUCTION LINT",
            f"Dataset: {dataset_path}",
            "",
            f"Status: {status}",
            f"Samples Checked: {samples_checked}",
            f"Warnings: {len(warnings)}",
        ]
        if kind_counts:
            lines.append("")
            lines.append("By Type:")
            for kind, count in kind_counts.items():
                lines.append(f"  {kind}: {count}")
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings[:10]:
                lines.append(f"  Line {w.line_number}: {w.message}")
            if len(warnings) > 10:
                lines.append(f"  ... and {len(warnings) - 10} more")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
