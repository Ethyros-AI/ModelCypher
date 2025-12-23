"""Dataset management CLI commands.

Provides commands for:
- Dataset validation, preprocessing, conversion
- Row-level operations (preview, get, update, add, delete)
- Dataset listing and deletion
- ASIF sparse image packaging
- Format analysis, chunking, and templating

Commands:
    mc dataset list
    mc dataset validate <path>
    mc dataset preview <path>
    mc dataset get-row <path> --line <n>
    mc dataset format-analyze <path>
    mc dataset chunk --file <path> --size <tokens>
    mc dataset template --model <family> --format <format>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from modelcypher.adapters.asif_packager import ASIFPackager
from modelcypher.cli.context import CLIContext
from modelcypher.cli.dataset_fields import parse_fields, parse_format, preview_line, pretty_fields
from modelcypher.cli.output import write_output
from modelcypher.cli.presenters import (
    dataset_convert_payload,
    dataset_edit_payload,
    dataset_payload,
    dataset_preview_payload,
    dataset_row_payload,
)
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.utils.limits import MAX_FIELD_BYTES, MAX_PREVIEW_LINES, MAX_RAW_BYTES

app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@app.command("validate")
def dataset_validate(ctx: typer.Context, path: str = typer.Argument(...)) -> None:
    """Validate a dataset file.

    Examples:
        mc dataset validate ./data.jsonl
    """
    context = _context(ctx)
    service = DatasetService()
    write_output(service.validate_dataset(path), context.output_format, context.pretty)


@app.command("preprocess")
def dataset_preprocess(
    ctx: typer.Context,
    input_path: str = typer.Argument(...),
    output_path: str = typer.Option(..., "--output-path", "-o", "--dataset-output", "--processed-output"),
    tokenizer: str = typer.Option(..., "--tokenizer"),
) -> None:
    """Preprocess a dataset for training.

    Examples:
        mc dataset preprocess ./raw.jsonl --output-path ./processed.jsonl --tokenizer gpt2
    """
    context = _context(ctx)
    service = DatasetService()
    result = service.preprocess_dataset(input_path, output_path, tokenizer)
    write_output(result, context.output_format, context.pretty)


@app.command("convert")
def dataset_convert(
    ctx: typer.Context,
    input_path: str = typer.Argument(...),
    to_format: str = typer.Option(..., "--to"),
    output_path: str = typer.Option(..., "--output-path", "-o"),
) -> None:
    """Convert dataset to a different format.

    Examples:
        mc dataset convert ./data.jsonl --to parquet --output-path ./data.parquet
    """
    context = _context(ctx)
    service = DatasetEditorService()
    target_format = parse_format(to_format)
    result = service.convert_dataset(input_path, target_format, output_path)
    payload = dataset_convert_payload(result)
    if context.output_format == "text":
        lines: list[str] = [
            "DATASET CONVERT",
            f"Source: {payload['sourcePath']}",
            f"Output: {payload['outputPath']}",
            f"Target format: {payload['targetFormat']}",
            f"Lines written: {payload['lineCount']}",
        ]
        if payload["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in payload["warnings"])
        write_output("\n".join(lines), context.output_format, context.pretty)
        return
    write_output(payload, context.output_format, context.pretty)


@app.command("preview")
def dataset_preview(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    lines: int = typer.Option(5, "--lines"),
    format: str = typer.Option("json", "--format"),
) -> None:
    """Preview rows from a dataset.

    Examples:
        mc dataset preview ./data.jsonl
        mc dataset preview ./data.jsonl --lines 10 --format table
    """
    context = _context(ctx)
    service = DatasetEditorService()
    requested = max(1, lines)
    limit = min(requested, MAX_PREVIEW_LINES)
    warnings: list[str] = []
    if requested > limit:
        warnings.append(f"Preview capped at {limit} lines (requested {requested}).")
    preview = service.preview(path, limit)

    if context.output_format == "text":
        if warnings:
            sys.stdout.write(f"Warning: {warnings[0]}\n")
        mode = format.lower()
        if mode == "table":
            rows = [
                f"{row.line_number}\t[{row.format.value}]\t{preview_line(row.raw)}"
                for row in preview.rows
            ]
            write_output("\n".join(rows), context.output_format, context.pretty)
            return
        rows = []
        for row in preview.rows:
            message = "none" if not row.validation_messages else "; ".join(row.validation_messages)
            rows.append(
                "\n".join(
                    [
                        f"Line {row.line_number} [{row.format.value}]",
                        pretty_fields(row.fields),
                        f"Validation: {message}",
                    ]
                )
            )
        write_output("\n\n".join(rows), context.output_format, context.pretty)
        return

    write_output(dataset_preview_payload(preview, warnings=warnings), context.output_format, context.pretty)


@app.command("get-row")
def dataset_get_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
) -> None:
    """Get a specific row from a dataset.

    Examples:
        mc dataset get-row ./data.jsonl --line 5
    """
    context = _context(ctx)
    service = DatasetEditorService()
    row = service.get_row(path, line)

    if context.output_format == "text":
        lines: list[str] = []
        lines.append(f"Line {row.line_number} [{row.format.value}]")
        lines.append(pretty_fields(row.fields))
        if row.raw_truncated:
            lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
        if row.fields_truncated:
            joined = ", ".join(row.fields_truncated)
            lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
        if row.validation_messages:
            lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_row_payload(row), context.output_format, context.pretty)


@app.command("update-row")
def dataset_update_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
    content: str = typer.Option(..., "--content"),
) -> None:
    """Update a row in a dataset.

    Examples:
        mc dataset update-row ./data.jsonl --line 5 --content '{"text": "new content"}'
    """
    context = _context(ctx)
    service = DatasetEditorService()
    fields = parse_fields(content, "--content")
    result = service.update_row(path, line, fields)

    if context.output_format == "text":
        lines: list[str] = [f"Updated line {line}"]
        if result.row:
            row = result.row
            lines.append(pretty_fields(row.fields))
            if row.raw_truncated:
                lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
            if row.fields_truncated:
                joined = ", ".join(row.fields_truncated)
                lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
            if row.validation_messages:
                lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@app.command("add-row")
def dataset_add_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    format: str = typer.Option(..., "--format"),
    fields: str = typer.Option(..., "--fields"),
) -> None:
    """Add a new row to a dataset.

    Examples:
        mc dataset add-row ./data.jsonl --format jsonl --fields '{"text": "hello"}'
    """
    context = _context(ctx)
    service = DatasetEditorService()
    format_enum = parse_format(format)
    parsed_fields = parse_fields(fields, "--fields")
    result = service.add_row(path, format_enum, parsed_fields)

    if context.output_format == "text":
        line_label = result.line_number or 0
        lines: list[str] = [f"Added line {line_label} [{format_enum.value}]"]
        if result.row:
            row = result.row
            lines.append(pretty_fields(row.fields))
            if row.raw_truncated:
                lines.append(f"Raw truncated to {MAX_RAW_BYTES} bytes (original {row.raw_full_bytes})")
            if row.fields_truncated:
                joined = ", ".join(row.fields_truncated)
                lines.append(f"Fields truncated: {joined} (limit {MAX_FIELD_BYTES} bytes per field)")
            if row.validation_messages:
                lines.append(f"Validation: {'; '.join(row.validation_messages)}")
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@app.command("delete-row")
def dataset_delete_row(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    line: int = typer.Option(..., "--line"),
) -> None:
    """Delete a row from a dataset.

    Examples:
        mc dataset delete-row ./data.jsonl --line 5
    """
    context = _context(ctx)
    service = DatasetEditorService()
    result = service.delete_row(path, line)

    if context.output_format == "text":
        lines: list[str] = [f"Deleted line {line}"]
        if result.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in result.warnings)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(dataset_edit_payload(result), context.output_format, context.pretty)


@app.command("list")
def dataset_list(ctx: typer.Context) -> None:
    """List all datasets.

    Examples:
        mc dataset list
    """
    context = _context(ctx)
    service = DatasetService()
    datasets = [dataset_payload(dataset) for dataset in service.list_datasets()]
    write_output(datasets, context.output_format, context.pretty)


@app.command("delete")
def dataset_delete(
    ctx: typer.Context,
    path: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Delete a dataset.

    Examples:
        mc dataset delete ./data.jsonl
        mc dataset delete ./data.jsonl --force
    """
    context = _context(ctx)
    if not force and not context.yes:
        if context.no_prompt:
            raise typer.Exit(code=2)
        if not typer.confirm(f"Delete dataset {path}?"):
            raise typer.Exit(code=1)
    service = DatasetService()
    service.delete_dataset(path)
    write_output({"deleted": path}, context.output_format, context.pretty)


@app.command("pack-asif")
def dataset_pack_asif(
    ctx: typer.Context,
    source: str = typer.Argument(...),
    destination: str = typer.Option(..., "--destination"),
    headroom_percent: int = typer.Option(15, "--headroom-percent"),
    minimum_free_gib: int = typer.Option(2, "--minimum-free-gib"),
    filesystem: str = typer.Option("apfs", "--filesystem"),
    volume_name: str = typer.Option("DATASET", "--volume-name"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Package dataset into APFS sparse image.

    Examples:
        mc dataset pack-asif ./data --destination ./data.sparseimage
    """
    context = _context(ctx)
    packager = ASIFPackager()
    result = packager.pack(
        source=source,
        destination=destination,
        headroom_percent=headroom_percent,
        minimum_free_gib=minimum_free_gib,
        filesystem=filesystem,
        volume_name=volume_name,
        overwrite=overwrite,
    )
    write_output(result, context.output_format, context.pretty)


@app.command("quality")
def dataset_quality(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the dataset file"),
) -> None:
    """Calculate quality score for a dataset.

    Analyzes the dataset and returns a quality score (0-100) based on:
    - Number of samples
    - Error count
    - Warning count
    - Average sample length

    Examples:
        mc dataset quality ./data.jsonl
    """
    from pathlib import Path

    context = _context(ctx)
    from modelcypher.core.domain.validation import DatasetQualityScorer

    # First validate to get error/warning counts
    service = DatasetService()
    validation = service.validate_dataset(path)

    # Count samples and compute average length
    sample_count = 0
    total_length = 0
    error_count = len(validation.get("errors", []))
    warning_count = len(validation.get("warnings", []))

    dataset_path = Path(path)
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                trimmed = line.strip()
                if trimmed:
                    sample_count += 1
                    total_length += len(trimmed)

    avg_length = total_length // sample_count if sample_count > 0 else 0

    scorer = DatasetQualityScorer.default()
    score = scorer.calculate_score(
        sample_count=sample_count,
        error_count=error_count,
        warning_count=warning_count,
        avg_length=avg_length,
    )

    payload = {
        "score": score.score,
        "range": score.range.value,
        "rangeDisplayName": score.range.display_name,
        "description": score.range.description,
        "sampleCount": score.sample_count,
        "errorCount": score.error_count,
        "warningCount": score.warning_count,
        "avgLength": score.avg_length,
        "breakdown": score.breakdown,
        "isProductionReady": score.is_production_ready,
    }

    if context.output_format == "text":
        lines = [
            "DATASET QUALITY",
            f"Path: {path}",
            "",
            f"Score: {score.score}/100 ({score.range.display_name})",
            f"Status: {score.range.description}",
            "",
            "Metrics:",
            f"  Samples: {score.sample_count}",
            f"  Errors: {score.error_count}",
            f"  Warnings: {score.warning_count}",
            f"  Avg Length: {score.avg_length} chars",
            "",
            "Score Breakdown:",
        ]
        for key, value in score.breakdown.items():
            sign = "+" if value > 0 else ""
            lines.append(f"  {key}: {sign}{value}")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("auto-fix")
def dataset_auto_fix(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the JSONL file to fix"),
) -> None:
    """Automatically fix common issues in a JSONL dataset.

    Converts various formats to MLX-compatible {"text": "..."} format:
    - Chat messages format
    - Instruction/output pairs
    - Prompt/completion pairs
    - Markdown headers
    - Plain text

    Creates a timestamped backup before making changes.

    Examples:
        mc dataset auto-fix ./data.jsonl
    """
    from pathlib import Path

    context = _context(ctx)
    from modelcypher.core.domain.validation import AutoFixEngine

    engine = AutoFixEngine()
    result = engine.auto_fix(Path(path))

    payload = {
        "fixedCount": result.fixed_count,
        "unfixableCount": result.unfixable_count,
        "isFullyFixed": result.is_fully_fixed,
        "backupPath": str(result.backup_path) if result.backup_path else None,
        "fixes": [
            {
                "lineNumber": fix.line_number,
                "type": fix.type.value,
                "description": fix.description,
            }
            for fix in result.fixes[:20]  # Limit to first 20 for display
        ],
        "unfixableLines": [
            {
                "lineNumber": line.line_number,
                "contentPreview": line.content[:50] + "..." if len(line.content) > 50 else line.content,
            }
            for line in result.unfixable_lines[:10]  # Limit to first 10
        ],
    }

    if context.output_format == "text":
        lines = [
            "DATASET AUTO-FIX",
            f"Path: {path}",
            "",
            result.summary,
            "",
        ]
        if result.backup_path:
            lines.append(f"Backup: {result.backup_path}")
            lines.append("")

        if result.fixes:
            lines.append(f"Fixes Applied ({len(result.fixes)} total):")
            for fix in result.fixes[:10]:
                lines.append(f"  Line {fix.line_number}: {fix.description}")
            if len(result.fixes) > 10:
                lines.append(f"  ... and {len(result.fixes) - 10} more")
            lines.append("")

        if result.unfixable_lines:
            lines.append(f"Unfixable Lines ({len(result.unfixable_lines)} total):")
            for line in result.unfixable_lines[:5]:
                preview = line.content[:40] + "..." if len(line.content) > 40 else line.content
                lines.append(f"  Line {line.line_number}: {preview}")
            if len(result.unfixable_lines) > 5:
                lines.append(f"  ... and {len(result.unfixable_lines) - 5} more need manual review")

        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
