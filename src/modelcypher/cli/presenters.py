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

from __future__ import annotations

from datetime import datetime
from typing import Any

from modelcypher.core.domain.model_search import ModelSearchPage, ModelSearchResult
from modelcypher.core.domain.models import CompareCheckpointResult, CompareSession, DatasetInfo, EvaluationResult, ModelInfo
from modelcypher.core.use_cases.doc_service import DocConvertResult
from modelcypher.core.use_cases.dataset_editor_service import (
    DatasetConversionResult,
    DatasetEditResult,
    DatasetPreviewResult,
    DatasetRowSnapshot,
)


def model_payload(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "alias": model.alias,
        "architecture": model.architecture,
        "format": model.format,
        "path": model.path,
        "sizeBytes": model.size_bytes,
        "parameterCount": model.parameter_count,
        "isDefaultChat": model.is_default_chat,
        "createdAt": _format_timestamp(model.created_at),
    }


def dataset_payload(dataset: DatasetInfo) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "path": dataset.path,
        "sizeBytes": dataset.size_bytes,
        "exampleCount": dataset.example_count,
        "createdAt": _format_timestamp(dataset.created_at),
    }


def dataset_row_payload(row: DatasetRowSnapshot) -> dict[str, Any]:
    return {
        "_schema": "mc.dataset.row.v1",
        "lineNumber": row.line_number,
        "raw": row.raw,
        "format": row.format.value,
        "fields": row.fields,
        "validationMessages": row.validation_messages,
        "rawTruncated": row.raw_truncated,
        "rawFullBytes": row.raw_full_bytes,
        "fieldsTruncated": row.fields_truncated,
    }


def dataset_preview_payload(preview: DatasetPreviewResult, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "_schema": "mc.dataset.preview.v1",
        "path": preview.path,
        "rowCount": len(preview.rows),
        "rows": [dataset_row_payload(row) for row in preview.rows],
        "warnings": warnings or [],
    }


def dataset_edit_payload(result: DatasetEditResult) -> dict[str, Any]:
    return {
        "_schema": "mc.dataset.edit.v1",
        "status": result.status,
        "lineNumber": result.line_number,
        "row": dataset_row_payload(result.row) if result.row else None,
        "warnings": result.warnings,
    }


def dataset_convert_payload(result: DatasetConversionResult) -> dict[str, Any]:
    return {
        "_schema": "mc.dataset.convert.v1",
        "sourcePath": result.source_path,
        "outputPath": result.output_path,
        "targetFormat": result.target_format.value,
        "lineCount": result.line_count,
        "warnings": result.warnings,
    }


def evaluation_list_payload(results: list[EvaluationResult]) -> dict[str, Any]:
    return {
        "evaluations": [
            {
                "id": result.id,
                "modelName": result.model_name,
                "datasetName": result.dataset_name,
                "averageLoss": result.average_loss,
                "perplexity": result.perplexity,
                "timestamp": _format_timestamp(result.timestamp),
            }
            for result in results
        ]
    }


def evaluation_detail_payload(result: EvaluationResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "modelPath": result.model_path,
        "modelName": result.model_name,
        "datasetPath": result.dataset_path,
        "datasetName": result.dataset_name,
        "averageLoss": result.average_loss,
        "perplexity": result.perplexity,
        "sampleCount": result.sample_count,
        "timestamp": _format_timestamp(result.timestamp),
        "config": result.config,
        "sampleResults": result.sample_results,
    }


def compare_list_payload(sessions: list[CompareSession]) -> dict[str, Any]:
    return {
        "sessions": [
            {
                "id": session.id,
                "createdAt": _format_timestamp(session.created_at),
                "checkpointCount": len(session.checkpoints),
                "promptPreview": _preview_prompt(session.prompt),
                "status": session.config.get("status", "unknown"),
                "notes": session.notes,
                "tags": session.tags or [],
            }
            for session in sessions
        ]
    }


def compare_detail_payload(session: CompareSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "createdAt": _format_timestamp(session.created_at),
        "prompt": session.prompt,
        "config": session.config,
        "checkpoints": [compare_checkpoint_payload(checkpoint) for checkpoint in session.checkpoints],
        "notes": session.notes,
        "tags": session.tags or [],
    }


def compare_checkpoint_payload(checkpoint: CompareCheckpointResult) -> dict[str, Any]:
    return {
        "checkpointPath": checkpoint.checkpoint_path,
        "modelName": checkpoint.model_name,
        "baseModelName": checkpoint.base_model_name,
        "response": checkpoint.response,
        "status": checkpoint.status,
        "metrics": checkpoint.metrics,
    }


def doc_convert_payload(result: DocConvertResult) -> dict[str, Any]:
    return {
        "jobId": result.job_id,
        "datasetName": result.dataset_name,
        "generator": result.generator,
        "createdAt": result.created_at,
        "durationSeconds": result.duration_seconds,
        "filesProcessed": result.files_processed,
        "sampleCount": result.sample_count,
        "totalTokens": result.total_tokens,
        "totalCharacters": result.total_characters,
        "detectedFormat": result.detected_format,
        "outputFormat": result.output_format,
        "qualityScore": result.quality_score,
        "validationStatus": result.validation_status,
        "validationErrors": result.validation_errors,
        "warnings": result.warnings,
        "sourceFiles": result.source_files,
        "failedFiles": result.failed_files,
    }


def model_search_payload(page: ModelSearchPage) -> dict[str, Any]:
    return {
        "count": len(page.models),
        "hasMore": page.has_more,
        "nextCursor": page.next_cursor,
        "models": [model_search_result_payload(model) for model in page.models],
    }


def model_search_result_payload(result: ModelSearchResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "downloads": result.downloads,
        "likes": result.likes,
        "author": result.author,
        "pipelineTag": result.pipeline_tag,
        "tags": result.tags,
        "isGated": result.is_gated,
        "isPrivate": result.is_private,
        "isRecommended": result.is_recommended,
        "estimatedSizeGB": result.estimated_size_gb,
        "memoryFitStatus": result.memory_fit_status.value if result.memory_fit_status else None,
    }


def _preview_prompt(prompt: str, limit: int = 100) -> str:
    return prompt[:limit]


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + "Z"
