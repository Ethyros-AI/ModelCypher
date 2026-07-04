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

"""Data preparation CLI.

mc data prepare: Convert data from any common format to validated training JSONL.
"""

from __future__ import annotations

from pathlib import Path

import typer

from modelcypher.cli.context import CLIContext
from modelcypher.cli.exit_codes import EXIT_RUNTIME
from modelcypher.cli.output import write_agent_output, write_error
from modelcypher.utils.errors import ErrorDetail

data_app = typer.Typer(no_args_is_help=True)


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.obj


@data_app.callback()
def data() -> None:
    """Data preparation and validation."""


@data_app.command("prepare")
def data_prepare(
    ctx: typer.Context,
    source: str = typer.Argument(
        ..., help="Path to file or HuggingFace dataset name (e.g., 'gsm8k')"
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL path (auto-derived from source if omitted)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model path for tokenizer stats and suggested training command",
    ),
    text_column: str = typer.Option(
        None,
        "--text-column",
        help="Column name for text content in CSV files",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split for HuggingFace datasets",
    ),
) -> None:
    """Prepare data for training.

    Converts data from JSONL, CSV, HuggingFace datasets, conversation JSON,
    or plain text into validated canonical JSONL format. Reports statistics
    and suggests the next training command.

    Output fields (when --json):
        format_detected: Detected source format
        n_samples: Number of valid samples
        n_removed: Number of samples removed (duplicates, invalid, empty)
        char_length_stats: Character length distribution (min, max, mean, median, p95)
        warnings: Any issues found during preparation
        output_path: Path to the prepared JSONL file
        suggested_command: Suggested mc train run command

    Examples:
        mc data prepare data/training/my_data.jsonl
        mc data prepare gsm8k --split train --model /path/to/model
        mc data prepare data.csv --text-column content -o prepared.jsonl
        mc data prepare conversations.json -m /path/to/model
    """
    context = _context(ctx)

    from modelcypher.core.use_cases.data_preparation_service import (
        DataPreparationService,
    )

    service = DataPreparationService()

    output_path = Path(output) if output else None
    model_path = Path(model) if model else None

    try:
        result = service.prepare(
            source=source,
            output=output_path,
            model_path=model_path,
            text_column=text_column,
            split=split,
        )
    except Exception as exc:
        error = ErrorDetail(
            code="MC-3001",
            title="Data preparation failed",
            detail=str(exc),
            hint="Check the source path/name and format",
            trace_id=context.trace_id,
        )
        write_error(error.as_dict(), context.output_format, context.pretty, exit_code=EXIT_RUNTIME)
        raise typer.Exit(code=EXIT_RUNTIME)

    envelope = service.make_envelope(
        result,
        model_path=str(model_path) if model_path else None,
        data_path=source,
    )

    if context.ai_mode or context.output_format != "text":
        write_agent_output(envelope, context.output_format, context.pretty)
    else:
        # Text mode: concise summary
        stats = result.statistics
        lines = [
            f"Prepared {stats.n_samples} samples ({stats.format_detected})",
            f"Output: {stats.output_path}",
        ]
        if stats.char_length_stats:
            cs = stats.char_length_stats
            lines.append(
                f"Lengths: {cs.min}-{cs.max} chars (mean={cs.mean:.0f}, p95={cs.p95:.0f})"
            )
        for w in stats.warnings:
            lines.append(f"  Warning: {w}")
        if result.suggested_command:
            lines.append(f"\nNext: {result.suggested_command}")

        write_agent_output(
            envelope, context.output_format, context.pretty,
            text_result="\n".join(lines),
        )
