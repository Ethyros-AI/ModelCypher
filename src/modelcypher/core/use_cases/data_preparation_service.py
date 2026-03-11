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

"""Data preparation service: ingest, convert, validate, report.

Orchestrates conversion from multiple source formats (JSONL, CSV, HF datasets,
conversation JSON, plain text) into canonical training JSONL.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelcypher.core.domain.agent_protocol import (
    AgentDiagnostics,
    AgentEnvelope,
    AgentRecommendation,
    make_metadata,
)
from modelcypher.core.domain.data_preparation import (
    DatasetStatistics,
    compute_length_stats,
    detect_source_format,
    parse_conversation_json,
    parse_csv_to_samples,
    validate_jsonl_lines,
)

logger = logging.getLogger(__name__)


@dataclass
class DataPrepareResult:
    """Result of data preparation."""

    statistics: DatasetStatistics
    suggested_command: str | None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = self.statistics.to_dict()
        if self.suggested_command:
            d["suggested_command"] = self.suggested_command
        return d


class DataPreparationService:
    """Orchestrate data preparation from multiple formats to canonical JSONL."""

    def prepare(
        self,
        source: str,
        output: Path | None = None,
        model_path: Path | None = None,
        text_column: str | None = None,
        split: str | None = None,
    ) -> DataPrepareResult:
        """Auto-detect format, convert to JSONL, validate, report statistics.

        Args:
            source: Path to file or HuggingFace dataset name.
            output: Output JSONL path. Auto-derived if None.
            model_path: Optional model path for suggested training command.
            text_column: Column name for text in CSV files.
            split: Dataset split for HF datasets (default: "train").
        """
        fmt = detect_source_format(source)
        logger.info("Detected format: %s for source: %s", fmt, source)

        if fmt == "huggingface":
            samples, warnings = self._load_huggingface(source, split or "train")
        elif fmt == "jsonl":
            samples, warnings = self._load_jsonl(Path(source))
        elif fmt == "csv":
            samples, warnings = self._load_csv(Path(source), text_column)
        elif fmt == "conversation_json":
            samples, warnings = self._load_conversation_json(Path(source))
        elif fmt == "parquet":
            samples, warnings = self._load_parquet(Path(source), text_column)
        else:
            # txt, md, code — delegate to DocService-style chunking
            samples, warnings = self._load_text(Path(source))

        # Compute output path
        if output is None:
            if fmt == "huggingface":
                safe_name = source.replace("/", "_")
                output = Path(f"data/training/{safe_name}_{split or 'train'}.jsonl")
            else:
                src_path = Path(source)
                output = src_path.with_suffix(".prepared.jsonl")

        # Write output
        n_total = len(samples)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Compute statistics
        char_lengths = []
        for s in samples:
            text = s.get("text", "")
            if not text and "messages" in s:
                text = " ".join(m.get("content", "") for m in s["messages"])
            char_lengths.append(len(text))

        char_stats = compute_length_stats(char_lengths)

        # Build suggested command
        suggested_cmd = None
        if model_path:
            suggested_cmd = (
                f"mc train run -m {model_path} -d {output}"
            )

        statistics = DatasetStatistics(
            format_detected=fmt,
            n_samples=n_total,
            n_valid=n_total,
            n_removed=0,
            char_length_stats=char_stats,
            token_length_stats=None,  # Would need tokenizer
            warnings=warnings,
            output_path=str(output),
        )

        return DataPrepareResult(
            statistics=statistics,
            suggested_command=suggested_cmd,
        )

    def make_envelope(
        self,
        result: DataPrepareResult,
        model_path: str | None = None,
    ) -> AgentEnvelope:
        """Wrap a DataPrepareResult in an AgentEnvelope."""
        stats = result.statistics
        recs: list[AgentRecommendation] = []

        if result.suggested_command:
            recs.append(
                AgentRecommendation(
                    action="train",
                    reason="Data is ready for training",
                    command=result.suggested_command,
                )
            )

        if stats.n_samples == 0:
            summary = "No valid samples found. Check source format and content."
            status = "failure"
        elif stats.warnings:
            summary = (
                f"Prepared {stats.n_samples} samples from {stats.format_detected} source. "
                f"{len(stats.warnings)} warning(s)."
            )
            status = "partial"
        else:
            summary = (
                f"Prepared {stats.n_samples} samples from {stats.format_detected} source."
            )
            status = "success"

        observations: list[str] = [
            f"Format: {stats.format_detected}",
            f"Samples: {stats.n_valid} valid, {stats.n_removed} removed",
            f"Output: {stats.output_path}",
        ]
        if stats.char_length_stats:
            cs = stats.char_length_stats
            observations.append(
                f"Character lengths: min={cs.min}, max={cs.max}, "
                f"mean={cs.mean:.0f}, median={cs.median:.0f}, p95={cs.p95:.0f}"
            )
        observations.extend(stats.warnings)

        return AgentEnvelope(
            command="mc data prepare",
            status=status,
            result=result.to_dict(),
            diagnostics=AgentDiagnostics(
                summary=summary,
                observations=observations,
                recommendations=recs,
            ),
            metadata=make_metadata(model=model_path),
        )

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _load_jsonl(self, path: Path) -> tuple[list[dict], list[str]]:
        """Load and validate an existing JSONL file."""
        lines = path.read_text(encoding="utf-8").splitlines()
        return validate_jsonl_lines(lines)

    def _load_csv(
        self, path: Path, text_column: str | None,
    ) -> tuple[list[dict], list[str]]:
        """Load CSV and convert to training samples."""
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return parse_csv_to_samples(rows, text_column)

    def _load_conversation_json(
        self, path: Path,
    ) -> tuple[list[dict], list[str]]:
        """Load conversation JSON and convert to training samples."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return parse_conversation_json(raw)

    def _load_text(self, path: Path) -> tuple[list[dict], list[str]]:
        """Load plain text file as training samples (one per paragraph)."""
        content = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            return [], ["File contains no text"]
        samples = [{"text": p} for p in paragraphs]
        return samples, []

    def _load_parquet(
        self, path: Path, text_column: str | None,
    ) -> tuple[list[dict], list[str]]:
        """Load Parquet file and convert to training samples."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return [], ["pyarrow not installed — cannot read Parquet files"]

        table = pq.read_table(path)
        df_dict = table.to_pydict()
        headers = list(df_dict.keys())

        # Resolve text column
        col = text_column
        if col and col not in headers:
            return [], [f"Column '{col}' not found. Available: {', '.join(headers)}"]
        if not col:
            for c in ["text", "content", "question", "input", "prompt"]:
                if c in headers:
                    col = c
                    break
        if not col:
            col = headers[0]

        samples: list[dict] = []
        warnings: list[str] = []
        values = df_dict[col]
        n_empty = 0
        for v in values:
            text = str(v).strip() if v is not None else ""
            if not text:
                n_empty += 1
                continue
            samples.append({"text": text})
        if n_empty:
            warnings.append(f"{n_empty} rows with empty text skipped")
        return samples, warnings

    def _load_huggingface(
        self, dataset_name: str, split: str,
    ) -> tuple[list[dict], list[str]]:
        """Load a HuggingFace dataset and convert to training samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            return [], [
                "The 'datasets' library is required for HuggingFace datasets. "
                "Install with: pip install datasets"
            ]

        warnings: list[str] = []
        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            return [], [f"Failed to load HuggingFace dataset '{dataset_name}': {e}"]

        # Auto-detect text field
        columns = ds.column_names
        text_col = None
        for c in ["text", "content", "question", "input", "prompt", "instruction"]:
            if c in columns:
                text_col = c
                break

        samples: list[dict] = []

        if text_col:
            for row in ds:
                text = str(row[text_col]).strip()
                if text:
                    samples.append({"text": text})
        elif "messages" in columns:
            for row in ds:
                messages = row["messages"]
                if isinstance(messages, list) and messages:
                    samples.append({"messages": messages})
        else:
            # Fall back: concatenate all string fields
            warnings.append(
                f"No standard text column found. Available: {', '.join(columns)}. "
                "Concatenating all string fields."
            )
            for row in ds:
                parts = []
                for col in columns:
                    val = row[col]
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                if parts:
                    samples.append({"text": "\n".join(parts)})

        if not samples:
            warnings.append("No valid samples extracted from dataset")

        return samples, warnings
