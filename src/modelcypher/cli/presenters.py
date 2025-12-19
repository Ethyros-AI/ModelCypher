from __future__ import annotations

from datetime import datetime
from typing import Any

from modelcypher.core.domain.models import CompareCheckpointResult, CompareSession, DatasetInfo, EvaluationResult, ModelInfo
from modelcypher.core.use_cases.doc_service import DocConvertResult


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


def _preview_prompt(prompt: str, limit: int = 100) -> str:
    return prompt[:limit]


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + "Z"
