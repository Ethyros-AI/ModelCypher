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

"""Dataset MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from modelcypher.mcp.security import ConfirmationError, create_confirmation_response

from .common import (
    DESTRUCTIVE_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    parse_dataset_format,
    require_existing_path,
    row_payload,
)

if TYPE_CHECKING:
    pass


def register_dataset_tools(ctx: ServiceContext) -> None:
    """Register dataset-related MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    if "mc_dataset_validate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_validate(
            path: str,
            format: str | None = None,
            quick: bool = False,
        ) -> dict:
            """Validate a dataset file for training readiness."""
            dataset_path = require_existing_path(path)
            target_format = parse_dataset_format(format) if format else None
            result = ctx.dataset_service.validate(
                dataset_path, target_format=target_format, quick=quick
            )
            return {
                "_schema": "mc.dataset.validate.v1",
                "path": result.path,
                "valid": result.valid,
                "format": result.format,
                "stats": result.stats,
                "errors": result.errors,
                "warnings": result.warnings,
                "nextActions": (
                    ["mc_train_start to begin training"]
                    if result.valid
                    else ["Fix validation errors", "mc_dataset_validate to recheck"]
                ),
            }

    if "mc_dataset_get_row" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_get_row(path: str, lineNumber: int) -> dict:
            """Get a specific row from a dataset."""
            dataset_path = require_existing_path(path)
            row = ctx.dataset_editor_service.get_row(dataset_path, lineNumber)
            return row_payload(row)

    if "mc_dataset_update_row" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_update_row(path: str, lineNumber: int, newContent: str) -> dict:
            """Update a specific row in a dataset."""
            dataset_path = require_existing_path(path)
            row = ctx.dataset_editor_service.update_row(dataset_path, lineNumber, newContent)
            payload = row_payload(row)
            payload["nextActions"] = ["mc_dataset_validate to verify changes"]
            return payload

    if "mc_dataset_add_row" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_add_row(path: str, content: str) -> dict:
            """Add a new row to a dataset."""
            dataset_path = require_existing_path(path)
            row = ctx.dataset_editor_service.add_row(dataset_path, content)
            payload = row_payload(row)
            payload["nextActions"] = ["mc_dataset_validate to verify changes"]
            return payload

    if "mc_dataset_delete_row" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_dataset_delete_row(path: str, lineNumber: int) -> dict:
            """Delete a specific row from a dataset."""
            dataset_path = require_existing_path(path)
            ctx.dataset_editor_service.delete_row(dataset_path, lineNumber)
            return {
                "_schema": "mc.dataset.delete_row.v1",
                "path": dataset_path,
                "lineNumber": lineNumber,
                "deleted": True,
                "nextActions": ["mc_dataset_validate to verify changes"],
            }

    if "mc_dataset_convert" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_convert(path: str, outputPath: str, targetFormat: str) -> dict:
            """Convert a dataset to a different format."""
            dataset_path = require_existing_path(path)
            target = parse_dataset_format(targetFormat)
            result = ctx.dataset_service.convert(dataset_path, outputPath, target)
            return {
                "_schema": "mc.dataset.convert.v1",
                "inputPath": result.input_path,
                "outputPath": result.output_path,
                "inputFormat": result.input_format,
                "targetFormat": result.target_format,
                "rowsConverted": result.rows_converted,
                "nextActions": [f"mc_dataset_validate with path={result.output_path}"],
            }

    if "mc_dataset_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_list() -> dict:
            """List all registered datasets."""
            datasets = ctx.dataset_service.list_datasets()
            return {
                "_schema": "mc.dataset.list.v1",
                "datasets": [
                    {
                        "path": d.path,
                        "name": d.name,
                        "format": d.format,
                        "rows": d.rows,
                        "sizeBytes": d.size_bytes,
                        "lastValidated": d.last_validated,
                    }
                    for d in datasets
                ],
                "count": len(datasets),
                "nextActions": ["mc_dataset_validate to validate a dataset"],
            }

    if "mc_dataset_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_dataset_delete(path: str, confirmationToken: str | None = None) -> dict:
            """Delete a dataset file."""
            dataset_path = require_existing_path(path)
            try:
                ctx.confirmation_manager.require_confirmation(
                    operation="dataset_delete",
                    tool_name="mc_dataset_delete",
                    parameters={"path": path},
                    description=f"Delete dataset: {path}",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete dataset: {path}",
                    timeout_seconds=ctx.security_config.confirmation_timeout_seconds,
                )
            ctx.dataset_service.delete(dataset_path)
            return {
                "_schema": "mc.dataset.delete.v1",
                "path": dataset_path,
                "deleted": True,
                "nextActions": ["mc_dataset_list to see remaining datasets"],
            }

    if "mc_dataset_preprocess" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_preprocess(path: str, outputPath: str | None = None) -> dict:
            """Preprocess a dataset for training."""
            dataset_path = require_existing_path(path)
            result = ctx.dataset_service.preprocess(dataset_path, output_path=outputPath)
            return {
                "_schema": "mc.dataset.preprocess.v1",
                "inputPath": result.input_path,
                "outputPath": result.output_path,
                "rowsProcessed": result.rows_processed,
                "transformationsApplied": result.transformations_applied,
                "nextActions": [f"mc_dataset_validate with path={result.output_path}"],
            }

    # Phase 2: New dataset tools
    if "mc_dataset_format_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_format_analyze(
            path: str,
            sampleLimit: int = 100,
        ) -> dict:
            """Analyze dataset format distribution and field frequency."""
            from modelcypher.core.domain.validation import DatasetFormatAnalyzer

            dataset_path = Path(path).expanduser().resolve()
            if not dataset_path.exists():
                raise ValueError(f"Dataset not found: {dataset_path}")
            analyzer = DatasetFormatAnalyzer()
            format_counts: dict[str, int] = {}
            field_counts: dict[str, int] = {}
            samples_analyzed = 0
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if samples_analyzed >= sampleLimit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    samples_analyzed += 1
                    detected = analyzer.detect_format(sample)
                    format_name = detected.value if hasattr(detected, "value") else str(detected)
                    format_counts[format_name] = format_counts.get(format_name, 0) + 1
                    if isinstance(sample, dict):
                        for key in sample.keys():
                            field_counts[key] = field_counts.get(key, 0) + 1
            dominant = (
                max(format_counts.items(), key=lambda x: x[1])[0] if format_counts else "unknown"
            )
            return {
                "_schema": "mc.dataset.format_analyze.v1",
                "path": str(dataset_path),
                "samplesAnalyzed": samples_analyzed,
                "formatDistribution": format_counts,
                "fieldFrequency": field_counts,
                "dominantFormat": dominant,
                "nextActions": [
                    "mc_dataset_validate for full validation",
                    "mc_dataset_convert to convert format",
                ],
            }

    if "mc_dataset_chunk" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_chunk(
            inputPath: str,
            outputPath: str,
            chunkSize: int = 512,
            overlap: int = 50,
        ) -> dict:
            """Chunk documents into smaller pieces for training."""
            from modelcypher.core.domain.dataset import DocumentChunker

            input_path = Path(inputPath).expanduser().resolve()
            output_path = Path(outputPath).expanduser().resolve()
            if not input_path.exists():
                raise ValueError(f"Input file not found: {input_path}")
            chunker = DocumentChunker(max_chunk_size=chunkSize, overlap=overlap)
            text = input_path.read_text(encoding="utf-8")
            chunks = chunker.chunk(text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps({"text": chunk}) + "\n")
            return {
                "_schema": "mc.dataset.chunk.v1",
                "inputPath": str(input_path),
                "outputPath": str(output_path),
                "chunkSize": chunkSize,
                "overlap": overlap,
                "chunksCreated": len(chunks),
                "nextActions": [
                    f"mc_dataset_validate with path={output_path}",
                    "mc_dataset_format_analyze to verify output",
                ],
            }

    if "mc_dataset_template" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_template(
            modelFamily: str,
            showExample: bool = True,
        ) -> dict:
            """Get chat template information for a model family."""
            from modelcypher.core.domain.dataset import ChatMessage, ChatTemplate

            family_lower = modelFamily.lower().replace("-", "_").replace(" ", "_")
            template_map = {
                "llama3": ChatTemplate.LLAMA3,
                "llama_3": ChatTemplate.LLAMA3,
                "qwen": ChatTemplate.QWEN,
                "qwen2": ChatTemplate.QWEN,
                "gemma": ChatTemplate.GEMMA,
                "gemma2": ChatTemplate.GEMMA,
                "mistral": ChatTemplate.MISTRAL,
                "phi": ChatTemplate.PHI,
                "phi3": ChatTemplate.PHI,
                "cohere": ChatTemplate.COHERE,
                "command": ChatTemplate.COHERE,
                "deepseek": ChatTemplate.DEEPSEEK,
                "granite": ChatTemplate.GRANITE,
                "starcoder": ChatTemplate.STARCODER,
                "codellama": ChatTemplate.CODELLAMA,
                "falcon": ChatTemplate.FALCON,
                "yi": ChatTemplate.YI,
                "internlm": ChatTemplate.INTERNLM,
                "baichuan": ChatTemplate.BAICHUAN,
                "chatglm": ChatTemplate.CHATGLM,
                "olmo": ChatTemplate.OLMO,
                "zephyr": ChatTemplate.ZEPHYR,
                "openchat": ChatTemplate.OPENCHAT,
                "solar": ChatTemplate.SOLAR,
                "vicuna": ChatTemplate.VICUNA,
            }
            template = template_map.get(family_lower)
            if template is None:
                available = sorted(template_map.keys())
                raise ValueError(f"Unknown model family: {modelFamily}. Available: {available}")
            example = None
            if showExample:
                messages = [
                    ChatMessage(role="system", content="You are a helpful assistant."),
                    ChatMessage(role="user", content="Hello!"),
                    ChatMessage(role="assistant", content="Hi there!"),
                ]
                example = template.apply(messages)
            return {
                "_schema": "mc.dataset.template.v1",
                "modelFamily": modelFamily,
                "templateName": template.name,
                "bosToken": template.bos_token,
                "eosToken": template.eos_token,
                "example": example,
                "supportedFamilies": sorted(template_map.keys()),
                "nextActions": [
                    "mc_dataset_convert to apply template",
                    "mc_dataset_format_analyze to check format",
                ],
            }
