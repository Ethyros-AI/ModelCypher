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


@app.command("format-analyze")
def dataset_format_analyze(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to dataset file"),
    sample_limit: int = typer.Option(100, "--sample-limit", help="Maximum samples to analyze"),
) -> None:
    """Analyze dataset format and structure.

    Detects the format (text, chat, instruction, completion, tools) and
    provides statistics on the dataset structure.

    Examples:
        mc dataset format-analyze ./data.jsonl
        mc dataset format-analyze ./data.jsonl --sample-limit 500
    """
    from modelcypher.cli.output import write_error
    from modelcypher.core.domain.validation import DatasetFormatAnalyzer
    from modelcypher.utils.errors import ErrorDetail

    context = _context(ctx)

    dataset_path = Path(path)
    if not dataset_path.exists():
        error = ErrorDetail(
            code="MC-1070",
            title="Dataset not found",
            detail=f"Dataset path does not exist: {path}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    analyzer = DatasetFormatAnalyzer()

    # Analyze samples
    format_counts: dict[str, int] = {}
    total_samples = 0
    field_frequency: dict[str, int] = {}

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if total_samples >= sample_limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(sample, dict):
                    continue

                total_samples += 1
                detected = analyzer.detect_format(sample)
                format_counts[detected.value] = format_counts.get(detected.value, 0) + 1

                # Track field frequency
                for key in sample.keys():
                    field_frequency[key] = field_frequency.get(key, 0) + 1

    except OSError as exc:
        error = ErrorDetail(
            code="MC-1071",
            title="Failed to read dataset",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Determine primary format
    primary_format = max(format_counts, key=format_counts.get) if format_counts else "unknown"

    payload = {
        "path": str(dataset_path),
        "samplesAnalyzed": total_samples,
        "primaryFormat": primary_format,
        "formatDistribution": format_counts,
        "fieldFrequency": dict(sorted(field_frequency.items(), key=lambda x: -x[1])[:20]),
        "isHomogeneous": len(format_counts) == 1,
    }

    if context.output_format == "text":
        lines = [
            "DATASET FORMAT ANALYSIS",
            f"Path: {dataset_path}",
            f"Samples Analyzed: {total_samples}",
            "",
            f"Primary Format: {primary_format}",
            f"Homogeneous: {'YES' if len(format_counts) == 1 else 'NO'}",
            "",
            "Format Distribution:",
        ]
        for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
            pct = count / total_samples * 100 if total_samples > 0 else 0
            lines.append(f"  {fmt}: {count} ({pct:.1f}%)")
        lines.append("")
        lines.append("Top Fields:")
        for field, count in list(field_frequency.items())[:10]:
            pct = count / total_samples * 100 if total_samples > 0 else 0
            lines.append(f"  {field}: {count} ({pct:.1f}%)")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("chunk")
def dataset_chunk(
    ctx: typer.Context,
    file: str = typer.Option(..., "--file", help="Path to input file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output file"),
    size: int = typer.Option(512, "--size", help="Target chunk size in tokens"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap between chunks in tokens"),
) -> None:
    """Chunk a text file into smaller segments.

    Splits documents into chunks for RAG or fine-tuning.
    Respects paragraph and sentence boundaries.

    Examples:
        mc dataset chunk --file ./document.txt --output ./chunks.jsonl --size 512
        mc dataset chunk --file ./long.txt -o ./chunked.jsonl --size 1024 --overlap 100
    """
    from modelcypher.cli.output import write_error
    from modelcypher.core.domain.dataset import DocumentChunker, TextChunk
    from modelcypher.utils.errors import ErrorDetail

    context = _context(ctx)

    file_path = Path(file)
    if not file_path.exists():
        error = ErrorDetail(
            code="MC-1072",
            title="File not found",
            detail=f"Input file does not exist: {file}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        error = ErrorDetail(
            code="MC-1073",
            title="Failed to read file",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    chunker = DocumentChunker()

    chunks = chunker.chunk(content, target_tokens=size)

    # Write output
    output_path = Path(output)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                line = json.dumps({"text": chunk.text}, ensure_ascii=False)
                f.write(line + "\n")
    except OSError as exc:
        error = ErrorDetail(
            code="MC-1074",
            title="Failed to write output",
            detail=str(exc),
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    payload = {
        "inputPath": str(file_path),
        "outputPath": str(output_path),
        "inputCharacters": len(content),
        "chunkCount": len(chunks),
        "targetSize": size,
        "overlap": overlap,
        "averageChunkSize": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
    }

    if context.output_format == "text":
        lines = [
            "DOCUMENT CHUNKING",
            f"Input: {file_path}",
            f"Output: {output_path}",
            "",
            f"Input Size: {len(content):,} characters",
            f"Chunks Created: {len(chunks)}",
            f"Target Size: {size} tokens",
            f"Overlap: {overlap} tokens",
        ]
        if chunks:
            avg_size = sum(len(c.text) for c in chunks) // len(chunks)
            lines.append(f"Average Chunk Size: {avg_size:,} characters")
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)


@app.command("template")
def dataset_template(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", help="Model family (llama3, qwen, gemma, mistral, etc.)"),
    format: str = typer.Option("chat", "--format", help="Output format: chat, instruction"),
    show_example: bool = typer.Option(True, "--show-example/--no-example", help="Show example output"),
) -> None:
    """Show chat template for a model family.

    Displays the chat template format used by different model families
    for converting between chat messages and text.

    Examples:
        mc dataset template --model llama3
        mc dataset template --model qwen --format instruction
        mc dataset template --model gemma --no-example
    """
    from modelcypher.core.domain.dataset import ChatMessage, ChatTemplate

    context = _context(ctx)

    # Map model name to template
    model_lower = model.lower()
    template_map = {
        "llama3": ChatTemplate.LLAMA3,
        "llama2": ChatTemplate.LLAMA2,
        "llama": ChatTemplate.LLAMA3,
        "qwen": ChatTemplate.QWEN2,
        "qwen2": ChatTemplate.QWEN2,
        "qwen3": ChatTemplate.QWEN3,
        "gemma": ChatTemplate.GEMMA2,
        "gemma2": ChatTemplate.GEMMA2,
        "gemma3": ChatTemplate.GEMMA3,
        "mistral": ChatTemplate.MISTRAL,
        "mixtral": ChatTemplate.MISTRAL,
        "phi": ChatTemplate.PHI3,
        "phi3": ChatTemplate.PHI3,
        "phi4": ChatTemplate.PHI4,
        "cohere": ChatTemplate.COMMAND_R,
        "command_r": ChatTemplate.COMMAND_R,
        "deepseek": ChatTemplate.DEEPSEEK,
        "granite": ChatTemplate.GRANITE,
        "zephyr": ChatTemplate.ZEPHYR,
        "vicuna": ChatTemplate.VICUNA,
        "alpaca": ChatTemplate.ALPACA,
        "chatml": ChatTemplate.CHATML,
    }

    template = template_map.get(model_lower)
    if template is None:
        from modelcypher.cli.output import write_error
        from modelcypher.utils.errors import ErrorDetail

        error = ErrorDetail(
            code="MC-1075",
            title="Unknown model family",
            detail=f"Model family '{model}' not recognized",
            hint=f"Valid families: {', '.join(sorted(template_map.keys()))}",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty)
        raise typer.Exit(code=1)

    # Build example
    example_messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is 2+2?"),
        ChatMessage(role="assistant", content="2+2 equals 4."),
    ]

    if format.lower() == "instruction":
        example_output = template.format_instruction(
            instruction="What is 2+2?",
            output="2+2 equals 4.",
        )
    else:
        example_output = template.format_messages(example_messages)

    payload = {
        "model": model,
        "templateName": template.value,
        "format": format,
        "displayName": template.display_name,
        "description": template.description,
        "example": example_output if show_example else None,
    }

    if context.output_format == "text":
        lines = [
            "CHAT TEMPLATE",
            f"Model Family: {model}",
            f"Template: {template.value}",
            f"Display Name: {template.display_name}",
            f"Format: {format}",
        ]
        if show_example:
            lines.append("")
            lines.append("Example Output:")
            lines.append("-" * 40)
            lines.append(example_output)
            lines.append("-" * 40)
        write_output("\n".join(lines), context.output_format, context.pretty)
        return

    write_output(payload, context.output_format, context.pretty)
