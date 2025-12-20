from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.geometry_adapter_service import GeometryAdapterService
from modelcypher.core.use_cases.geometry_safety_service import GeometrySafetyService
from modelcypher.core.use_cases.geometry_training_service import GeometryTrainingService
from modelcypher.core.use_cases.inventory_service import InventoryService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.model_search_service import ModelSearchService
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
from modelcypher.core.use_cases.model_service import ModelService
from modelcypher.core.use_cases.settings_service import SettingsService
from modelcypher.core.use_cases.system_service import SystemService
from modelcypher.core.use_cases.training_service import TrainingService
from modelcypher.core.domain.dataset_validation import DatasetContentFormat
from modelcypher.core.domain.model_search import (
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.core.domain.training import TrainingConfig
from modelcypher.utils.json import dump_json


IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


@dataclass
class _IdempotencyEntry:
    value: str
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


TOOL_PROFILES = {
    "full": {
        "mc_inventory",
        "mc_settings_snapshot",
        "mc_train_start",
        "mc_job_status",
        "mc_job_list",
        "mc_job_detail",
        "mc_job_cancel",
        "mc_job_pause",
        "mc_job_resume",
        "mc_system_status",
        "mc_validate_train",
        "mc_estimate_train",
        "mc_dataset_validate",
        "mc_dataset_get_row",
        "mc_dataset_update_row",
        "mc_dataset_add_row",
        "mc_dataset_delete_row",
        "mc_dataset_convert",
        "mc_model_fetch",
        "mc_model_list",
        "mc_model_search",
        "mc_model_probe",
        "mc_model_validate_merge",
        "mc_model_analyze_alignment",
        "mc_checkpoint_export",
        "mc_geometry_training_status",
        "mc_geometry_training_history",
        "mc_geometry_validate",
        "mc_safety_circuit_breaker",
        "mc_safety_persona_drift",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
        "mc_infer",
    },
    "training": {
        "mc_inventory",
        "mc_settings_snapshot",
        "mc_train_start",
        "mc_job_status",
        "mc_job_list",
        "mc_job_detail",
        "mc_job_cancel",
        "mc_job_pause",
        "mc_job_resume",
        "mc_system_status",
        "mc_validate_train",
        "mc_estimate_train",
        "mc_dataset_validate",
        "mc_dataset_get_row",
        "mc_dataset_update_row",
        "mc_dataset_add_row",
        "mc_dataset_delete_row",
        "mc_dataset_convert",
        "mc_model_fetch",
        "mc_model_list",
        "mc_model_search",
        "mc_checkpoint_export",
        "mc_geometry_training_status",
        "mc_geometry_training_history",
        "mc_geometry_validate",
        "mc_safety_circuit_breaker",
        "mc_safety_persona_drift",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
    },
    "inference": {
        "mc_inventory",
        "mc_settings_snapshot",
        "mc_model_list",
        "mc_infer",
        "mc_system_status",
    },
    "monitoring": {
        "mc_inventory",
        "mc_settings_snapshot",
        "mc_job_status",
        "mc_job_list",
        "mc_job_detail",
        "mc_system_status",
        "mc_geometry_training_status",
        "mc_geometry_training_history",
        "mc_geometry_validate",
        "mc_safety_circuit_breaker",
        "mc_safety_persona_drift",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
    },
}


def _parse_dataset_format(value: str) -> DatasetContentFormat:
    key = value.lower()
    if key == "text":
        return DatasetContentFormat.text
    if key == "chat":
        return DatasetContentFormat.chat
    if key == "completion":
        return DatasetContentFormat.completion
    if key == "tools":
        return DatasetContentFormat.tools
    if key == "instruction":
        return DatasetContentFormat.instruction
    raise ValueError("Unsupported format. Use text, chat, completion, tools, or instruction.")


def _map_job_status(status: str) -> str:
    if status == "pending":
        return "queued"
    if status == "cancelled":
        return "canceled"
    return status


READ_ONLY_ANNOTATIONS = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
MUTATING_ANNOTATIONS = {"readOnlyHint": False, "idempotentHint": False, "openWorldHint": False}
IDEMPOTENT_MUTATING_ANNOTATIONS = {"readOnlyHint": False, "idempotentHint": True, "openWorldHint": False}
DESTRUCTIVE_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": True,
    "openWorldHint": False,
}
NETWORK_ANNOTATIONS = {"readOnlyHint": False, "idempotentHint": True, "openWorldHint": True}


def build_server() -> FastMCP:
    profile = os.environ.get("MC_MCP_PROFILE", "full")
    tool_set = TOOL_PROFILES.get(profile, TOOL_PROFILES["full"])

    mcp = FastMCP("ModelCypher", json_response=True)
    inventory_service = InventoryService()
    training_service = TrainingService()
    job_service = JobService()
    model_service = ModelService()
    model_search_service = ModelSearchService()
    model_probe_service = ModelProbeService()
    dataset_service = DatasetService()
    dataset_editor_service = DatasetEditorService()
    system_service = SystemService()
    settings_service = SettingsService()
    checkpoint_service = CheckpointService()
    inference_engine = LocalInferenceEngine()
    geometry_service = GeometryService()
    geometry_training_service = GeometryTrainingService()
    geometry_safety_service = GeometrySafetyService(geometry_training_service)
    geometry_adapter_service = GeometryAdapterService()

    idempotency_cache: dict[str, _IdempotencyEntry] = {}

    def _namespaced_key(operation: str, key: str) -> str:
        return f"{operation}:{key}"

    def _get_idempotency(operation: str, key: str) -> str | None:
        entry = idempotency_cache.get(_namespaced_key(operation, key))
        if entry is None:
            return None
        if entry.is_expired():
            idempotency_cache.pop(_namespaced_key(operation, key), None)
            return None
        return entry.value

    def _set_idempotency(operation: str, key: str, value: str) -> None:
        idempotency_cache[_namespaced_key(operation, key)] = _IdempotencyEntry(
            value=value,
            expires_at=time.time() + IDEMPOTENCY_TTL_SECONDS,
        )
        if len(idempotency_cache) % 100 == 0:
            expired = [cache_key for cache_key, entry in idempotency_cache.items() if entry.is_expired()]
            for cache_key in expired:
                idempotency_cache.pop(cache_key, None)

    def _require_existing_path(path: str) -> str:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        return str(resolved)

    def _require_existing_directory(path: str) -> str:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Directory does not exist: {resolved}")
        return str(resolved)

    def _row_payload(row) -> dict:
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

    def _system_status_payload() -> dict:
        readiness = system_service.readiness()
        readiness_score = readiness.get("readinessScore", 0)
        if readiness_score >= 80:
            next_actions = ["mc_train_start to begin training"]
        elif readiness_score >= 60:
            next_actions = ["Address blockers first", "mc_system_status to recheck"]
        else:
            next_actions = ["Fix critical blockers", "mc_model_list to verify models"]
        return {
            "_schema": "mc.system.status.v1",
            "machineName": readiness.get("machineName", ""),
            "unifiedMemoryGB": readiness.get("unifiedMemoryGB", 0),
            "mlxVersion": readiness.get("mlxVersion"),
            "readinessScore": readiness_score,
            "scoreBreakdown": readiness.get("scoreBreakdown", {}),
            "blockers": readiness.get("blockers", []),
            "nextActions": next_actions,
        }

    if "mc_inventory" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_inventory() -> dict:
            inventory = inventory_service.inventory()
            jobs = []
            for job in inventory.get("jobs", []):
                if not isinstance(job, dict):
                    continue
                jobs.append(
                    {
                        "jobId": job.get("jobId"),
                        "status": _map_job_status(job.get("status", "")),
                        "progress": job.get("progress", 0.0),
                        "modelId": job.get("modelId"),
                        "datasetPath": job.get("datasetPath"),
                    }
                )
            return {
                "models": inventory.get("models", []),
                "datasets": inventory.get("datasets", []),
                "checkpoints": inventory.get("checkpoints", []),
                "jobs": jobs,
                "workspace": inventory.get("workspace", {}),
                "mlxVersion": inventory.get("mlxVersion"),
                "policies": inventory.get("policies", {}),
            }

    if "mc_train_start" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_train_start(
            model: str,
            dataset: str,
            epochs: int = 3,
            learningRate: float = 1e-5,
            batchSize: int | None = None,
            sequenceLength: int = 2048,
            loraRank: int | None = None,
            loraAlpha: float | None = None,
            captureFisher: bool = False,
            fisherBatches: int = 32,
            fisherOutput: str = "fisher.safetensors",
            fisherSeed: int | None = None,
            idempotencyKey: str | None = None,
            autoEval: bool = False,
            evalDataset: str | None = None,
            evalMetrics: list[str] | None = None,
            evalBatchSize: int = 4,
            evalMaxSamples: int | None = None,
            evalWait: bool = True,
        ) -> dict:
            if epochs <= 0:
                raise ValueError("epochs must be a positive integer")
            if learningRate <= 0:
                raise ValueError("learningRate must be positive")
            if batchSize is not None and batchSize <= 0:
                raise ValueError("batchSize must be a positive integer")
            if sequenceLength <= 0:
                raise ValueError("sequenceLength must be a positive integer")
            if loraRank is not None and loraRank <= 0:
                raise ValueError("loraRank must be a positive integer")
            if loraAlpha is not None and loraAlpha <= 0:
                raise ValueError("loraAlpha must be positive")
            dataset_path = _require_existing_path(dataset)
            if idempotencyKey:
                previous = _get_idempotency("train_start", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.train.start.v1",
                        "jobId": None,
                        "status": "duplicate",
                        "batchSize": None,
                        "wasExecuted": False,
                        "previousJobId": previous,
                        "message": "Job already started with this idempotency key",
                        "autoEval": None,
                        "nextActions": [f"mc_job_status with jobId={previous}"],
                    }

            lora = None
            if loraRank is not None and loraAlpha is not None:
                from modelcypher.core.domain.training import LoRAConfig

                lora = LoRAConfig(rank=loraRank, alpha=loraAlpha, dropout=0.0, targets=["q_proj", "v_proj"])

            batch_size = batchSize if batchSize is not None else 1
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                learning_rate=learningRate,
                batch_size=batch_size,
                epochs=epochs,
                sequence_length=sequenceLength,
                lora=lora,
            )
            if batchSize is None:
                preflight = training_service.preflight(config)
                predicted = preflight.get("predictedBatchSize", 0)
                if predicted > 0:
                    batch_size = predicted
                    config = TrainingConfig(
                        model_id=model,
                        dataset_path=dataset_path,
                        learning_rate=learningRate,
                        batch_size=batch_size,
                        epochs=epochs,
                        sequence_length=sequenceLength,
                        lora=lora,
                    )
            result, _ = training_service.start(config, stream=False)
            job_id = result["jobId"]
            if idempotencyKey:
                _set_idempotency("train_start", idempotencyKey, job_id)

            auto_eval_payload = None
            if autoEval and evalDataset:
                auto_eval_payload = {
                    "enabled": True,
                    "evalDataset": evalDataset,
                    "metrics": evalMetrics or [],
                    "batchSize": evalBatchSize,
                    "maxSamples": evalMaxSamples,
                    "waitForCompletion": evalWait,
                }

            next_actions = [f"mc_job_status with jobId={job_id}", "mc_job_list to see all jobs"]
            if auto_eval_payload is not None:
                next_actions.append("mc_eval_run after training completes (auto-eval configured)")

            return {
                "_schema": "mc.train.start.v1",
                "jobId": job_id,
                "status": "started",
                "batchSize": batch_size,
                "wasExecuted": True,
                "previousJobId": None,
                "message": "Training started with auto-evaluation enabled" if auto_eval_payload else None,
                "autoEval": auto_eval_payload,
                "nextActions": next_actions,
            }

    if "mc_job_status" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_status(jobId: str) -> dict:
            status = training_service.status(jobId)
            mapped_status = _map_job_status(status["status"])
            if mapped_status == "running":
                next_actions = ["mc_job_pause to pause", "mc_job_cancel to stop"]
            elif mapped_status == "paused":
                next_actions = ["mc_job_resume to continue", "mc_job_cancel to stop"]
            elif mapped_status == "completed":
                next_actions = ["mc_checkpoint_export to deploy", "mc_infer to test"]
            elif mapped_status in {"failed", "canceled"}:
                next_actions = ["mc_train_start to retry"]
            else:
                next_actions = ["mc_job_status to check progress"]
            return {
                "_schema": "mc.job.status.v1",
                "jobId": status["jobId"],
                "status": mapped_status,
                "progress": (status["currentStep"] / status["totalSteps"]) if status["totalSteps"] else 0.0,
                "currentEpoch": status["currentEpoch"],
                "totalEpochs": status["totalEpochs"],
                "loss": status["loss"],
                "throughputTPS": None,
                "etaSeconds": None,
                "lastUpdate": status.get("updatedAt", status.get("createdAt")),
                "nextActions": next_actions,
            }

    if "mc_job_list" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_list(status: str | None = None, activeOnly: bool = False) -> dict:
            status_filter = status
            if status_filter == "queued":
                status_filter = "pending"
            if status_filter == "canceled":
                status_filter = "cancelled"
            jobs = job_service.list_jobs(status=status_filter, active_only=activeOnly)
            entries = []
            for job in jobs:
                progress = (
                    (job["currentStep"] / job["totalSteps"]) if job.get("totalSteps") else 0.0
                )
                entries.append(
                    {
                        "jobId": job["jobId"],
                        "status": _map_job_status(job["status"]),
                        "modelId": job["modelId"],
                        "progress": progress,
                    }
                )
            next_actions = (
                ["mc_train_start to create a job"]
                if not entries
                else ["mc_job_status for details", "mc_job_attach to stream"]
            )
            return {
                "_schema": "mc.job.list.v1",
                "jobs": entries,
                "count": len(entries),
                "nextActions": next_actions,
            }

    if "mc_job_detail" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_detail(jobId: str) -> dict:
            payload = job_service.show_job(jobId, include_loss_history=True)
            hyper = payload.get("hyperparameters", {}) or {}
            loss_history = payload.get("lossHistory", []) or []
            normalized_loss = []
            for entry in loss_history:
                if isinstance(entry, dict) and "step" in entry and "loss" in entry:
                    normalized_loss.append({"step": entry["step"], "loss": entry["loss"]})
            progress = payload.get("progress")
            if progress is None:
                total_steps = payload.get("totalSteps", 0) or 0
                current_step = payload.get("currentStep", 0) or 0
                progress = (current_step / total_steps) if total_steps else 0.0
            return {
                "_schema": "mc.job.detail.v1",
                "jobId": payload["jobId"],
                "status": _map_job_status(payload["status"]),
                "createdAt": payload["createdAt"],
                "startedAt": payload.get("startedAt"),
                "completedAt": payload.get("completedAt"),
                "modelId": payload["modelId"],
                "datasetPath": payload["datasetPath"],
                "progress": progress,
                "finalLoss": payload.get("finalLoss"),
                "checkpoints": payload.get("checkpoints", []),
                "hyperparameters": {
                    "learningRate": hyper.get("learningRate", hyper.get("learning_rate", 0.0)),
                    "batchSize": hyper.get("batchSize", hyper.get("batch_size", 0)),
                    "epochs": hyper.get("epochs", 0),
                    "sequenceLength": hyper.get("sequenceLength", hyper.get("sequence_length", 0)),
                },
                "lossHistory": normalized_loss,
                "nextActions": [f"mc_job_status with jobId={payload['jobId']}"],
            }

    if "mc_job_cancel" in tool_set:
        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_job_cancel(jobId: str) -> dict:
            training_service.cancel(jobId)
            return {
                "_schema": "mc.job.cancel.v1",
                "jobId": jobId,
                "status": "canceled",
                "nextActions": ["mc_train_start to restart", "mc_job_list to see other jobs"],
            }

    if "mc_job_pause" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_job_pause(jobId: str) -> dict:
            training_service.pause(jobId)
            return {
                "_schema": "mc.job.pause.v1",
                "jobId": jobId,
                "status": "paused",
                "nextActions": ["mc_job_resume to continue", "mc_job_status to check"],
            }

    if "mc_job_resume" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_job_resume(jobId: str) -> dict:
            training_service.resume(jobId)
            return {
                "_schema": "mc.job.resume.v1",
                "jobId": jobId,
                "status": "resumed",
                "nextActions": ["mc_job_status to check progress", "mc_job_pause to pause again"],
            }

    if "mc_model_list" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_list() -> dict:
            models = model_service.list_models()
            entries = [
                {
                    "id": model.id,
                    "alias": model.alias,
                    "path": model.path,
                    "architecture": model.architecture,
                    "format": model.format,
                    "sizeBytes": model.size_bytes,
                }
                for model in models
            ]
            next_actions = (
                ["mc_model_fetch to download a model"]
                if not entries
                else ["mc_train_start with model", "mc_infer with model"]
            )
            return {
                "_schema": "mc.model.list.v1",
                "models": entries,
                "count": len(entries),
                "nextActions": next_actions,
            }

    if "mc_model_search" in tool_set:
        @mcp.tool(annotations=NETWORK_ANNOTATIONS)
        def mc_model_search(
            query: str | None = None,
            author: str | None = None,
            library: str = "mlx",
            quant: str | None = None,
            sort: str = "downloads",
            limit: int = 20,
            cursor: str | None = None,
        ) -> dict:
            if limit <= 0:
                raise ValueError("limit must be a positive integer")

            library_key = library.lower()
            if library_key == "mlx":
                library_filter = ModelSearchLibraryFilter.mlx
            elif library_key == "safetensors":
                library_filter = ModelSearchLibraryFilter.safetensors
            elif library_key == "pytorch":
                library_filter = ModelSearchLibraryFilter.pytorch
            elif library_key == "any":
                library_filter = ModelSearchLibraryFilter.any
            else:
                raise ValueError("Invalid library filter. Use: mlx, safetensors, pytorch, any.")

            quant_filter: ModelSearchQuantization | None
            if quant is None:
                quant_filter = None
            else:
                quant_key = quant.lower()
                if quant_key == "4bit":
                    quant_filter = ModelSearchQuantization.four_bit
                elif quant_key == "8bit":
                    quant_filter = ModelSearchQuantization.eight_bit
                elif quant_key == "any":
                    quant_filter = ModelSearchQuantization.any
                else:
                    raise ValueError("Invalid quant filter. Use: 4bit, 8bit, any.")

            sort_key = sort.lower()
            if sort_key == "downloads":
                sort_option = ModelSearchSortOption.downloads
            elif sort_key == "likes":
                sort_option = ModelSearchSortOption.likes
            elif sort_key == "lastmodified":
                sort_option = ModelSearchSortOption.last_modified
            elif sort_key == "trending":
                sort_option = ModelSearchSortOption.trending
            else:
                raise ValueError("Invalid sort option. Use: downloads, likes, lastModified, trending.")

            filters = ModelSearchFilters(
                query=query,
                architecture=None,
                max_size_gb=None,
                author=author,
                library=library_filter,
                quantization=quant_filter,
                sort_by=sort_option,
                limit=min(limit, 100),
            )

            try:
                page = model_search_service.search(filters, cursor)
            except ModelSearchError as exc:
                raise ValueError(f"Search failed: {exc}") from exc

            models = [
                {
                    "id": model.id,
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "author": model.author,
                    "pipelineTag": model.pipeline_tag,
                    "tags": model.tags,
                    "isGated": model.is_gated,
                    "isPrivate": model.is_private,
                    "isRecommended": model.is_recommended,
                    "estimatedSizeGB": model.estimated_size_gb,
                    "memoryFitStatus": model.memory_fit_status.value if model.memory_fit_status else None,
                }
                for model in page.models
            ]
            next_actions = (
                ["Try a different search query"]
                if not models
                else ["mc_model_fetch with model ID to download", "mc_model_search with cursor for next page"]
            )
            return {
                "_schema": "mc.model.search.v1",
                "count": len(models),
                "hasMore": page.has_more,
                "nextCursor": page.next_cursor,
                "models": models,
                "nextActions": next_actions,
            }

    if "mc_model_probe" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_probe(modelPath: str) -> dict:
            """Probe a model for architecture details."""
            model_path = _require_existing_directory(modelPath)
            result = model_probe_service.probe(model_path)
            return {
                "_schema": "mc.model.probe.v1",
                "architecture": result.architecture,
                "parameterCount": result.parameter_count,
                "vocabSize": result.vocab_size,
                "hiddenSize": result.hidden_size,
                "numAttentionHeads": result.num_attention_heads,
                "quantization": result.quantization,
                "layerCount": len(result.layers),
                "layers": [
                    {
                        "name": layer.name,
                        "type": layer.type,
                        "parameters": layer.parameters,
                        "shape": layer.shape,
                    }
                    for layer in result.layers[:20]
                ],
                "nextActions": [
                    f"mc_model_validate_merge to check merge compatibility",
                    f"mc_model_analyze_alignment to analyze drift",
                ],
            }

    if "mc_model_validate_merge" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_validate_merge(source: str, target: str) -> dict:
            """Validate merge compatibility between two models."""
            source_path = _require_existing_directory(source)
            target_path = _require_existing_directory(target)
            result = model_probe_service.validate_merge(source_path, target_path)
            return {
                "_schema": "mc.model.validate_merge.v1",
                "compatible": result.compatible,
                "architectureMatch": result.architecture_match,
                "vocabMatch": result.vocab_match,
                "dimensionMatch": result.dimension_match,
                "warnings": result.warnings,
                "nextActions": (
                    ["mc_model_merge to perform the merge"]
                    if result.compatible
                    else ["Fix compatibility issues before merging"]
                ),
            }

    if "mc_model_analyze_alignment" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_analyze_alignment(modelA: str, modelB: str) -> dict:
            """Analyze alignment drift between two models."""
            path_a = _require_existing_directory(modelA)
            path_b = _require_existing_directory(modelB)
            result = model_probe_service.analyze_alignment(path_a, path_b)
            return {
                "_schema": "mc.model.analyze_alignment.v1",
                "driftMagnitude": result.drift_magnitude,
                "assessment": result.assessment,
                "interpretation": result.interpretation,
                "layerDrifts": [
                    {
                        "layerName": drift.layer_name,
                        "driftMagnitude": drift.drift_magnitude,
                        "direction": drift.direction,
                    }
                    for drift in result.layer_drifts[:20]
                ],
                "nextActions": [
                    "mc_model_validate_merge to check merge compatibility",
                    "mc_geometry_training_status for training metrics",
                ],
            }

    if "mc_infer" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer(
            model: str,
            prompt: str,
            maxTokens: int = 512,
            temperature: float = 0.7,
            topP: float = 0.95,
        ) -> dict:
            if maxTokens <= 0:
                raise ValueError("maxTokens must be a positive integer")
            if temperature < 0.0 or temperature > 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
            if topP < 0.0 or topP > 1.0:
                raise ValueError("topP must be between 0.0 and 1.0")
            model_path = _require_existing_directory(model)
            result = inference_engine.infer(model_path, prompt, maxTokens, temperature, topP)
            return {
                "_schema": "mc.infer.v1",
                "modelId": result["modelId"],
                "prompt": result["prompt"],
                "response": result["response"],
                "tokenCount": result["tokenCount"],
                "tokensPerSecondTPS": result["tokensPerSecond"],
                "timeToFirstTokenSeconds": result["timeToFirstToken"],
                "totalDurationSeconds": result["totalDuration"],
                "nextActions": ["mc_infer for more prompts", "mc_checkpoint_export to deploy"],
            }

    if "mc_system_status" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_system_status() -> dict:
            return _system_status_payload()

    if "mc_settings_snapshot" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_settings_snapshot() -> dict:
            snapshot = settings_service.snapshot()
            return {"_schema": "mc.settings.snapshot.v1", **snapshot.as_dict()}

    if "mc_geometry_validate" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_validate(includeFixtures: bool = False) -> dict:
            report = geometry_service.validate(include_fixtures=includeFixtures)
            return geometry_service.validation_payload(report, include_schema=True)

    if "mc_geometry_training_status" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_status(jobId: str, format: str = "full") -> dict:
            format_key = format.lower()
            if format_key not in {"full", "summary"}:
                raise ValueError("format must be 'full' or 'summary'")
            return geometry_training_service.training_status_payload(jobId, output_format=format_key)

    if "mc_geometry_training_history" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_training_history(jobId: str) -> dict:
            return geometry_training_service.training_history_payload(jobId)

    if "mc_safety_circuit_breaker" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_circuit_breaker(
            jobId: str | None = None,
            entropySignal: float | None = None,
            refusalDistance: float | None = None,
            personaDriftMagnitude: float | None = None,
            hasOscillation: bool = False,
        ) -> dict:
            state, signals = geometry_safety_service.evaluate_circuit_breaker(
                job_id=jobId,
                entropy_signal=entropySignal,
                refusal_distance=refusalDistance,
                persona_drift_magnitude=personaDriftMagnitude,
                has_oscillation=hasOscillation,
            )
            state_label = "tripped" if state.is_tripped else ("warning" if state.severity >= 0.5 else "nominal")
            return {
                "_schema": "mc.safety.circuit_breaker.v1",
                "jobId": jobId,
                "checkpointPath": None,
                "tripped": state.is_tripped,
                "severity": state.severity,
                "state": state_label,
                "signals": {
                    "refusalDistance": signals.refusal_distance,
                    "personaDrift": signals.persona_drift_magnitude,
                    "semanticEntropyDelta": signals.entropy_signal,
                    "activationAnomaly": None,
                    "gradientNormSpike": None,
                },
                "thresholds": {
                    "refusalWarning": 0.3,
                    "refusalCritical": 0.15,
                    "personaDriftWarning": 0.2,
                    "personaDriftCritical": 0.4,
                    "semanticEntropyWarning": 0.7,
                    "aggregateTripThreshold": 0.75,
                },
                "interpretation": state.interpretation,
                "recommendedAction": state.recommended_action.description,
                "nextActions": [
                    "mc_safety_persona_drift for detailed persona analysis",
                    "mc_job_pause if tripped=true",
                    "mc_geometry_training_status for full metrics",
                ],
            }

    if "mc_safety_persona_drift" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_safety_persona_drift(jobId: str) -> dict:
            drift_info = geometry_safety_service.persona_drift(jobId)
            if drift_info is None:
                raise ValueError(f"Job '{jobId}' not found or has no persona drift metrics")
            interpretation = geometry_safety_service.persona_interpretation(drift_info)
            trait_drifts = [
                {
                    "traitName": trait,
                    "driftMagnitude": drift_info.overall_drift_magnitude,
                    "direction": "unknown",
                    "baselineValue": None,
                    "currentValue": None,
                }
                for trait in drift_info.drifting_traits
            ]
            return {
                "_schema": "mc.safety.persona_drift.v1",
                "jobId": jobId,
                "checkpointPath": None,
                "baselineCheckpointPath": None,
                "overallDriftMagnitude": drift_info.overall_drift_magnitude,
                "driftAssessment": drift_info.assessment,
                "traitDrifts": trait_drifts or None,
                "refusalDirectionCorrelation": drift_info.refusal_distance,
                "helpfulnessCorrelation": None,
                "interpretation": interpretation,
                "nextActions": [
                    "mc_safety_circuit_breaker for combined safety assessment",
                    "mc_job_pause if assessment is 'critical'",
                    "mc_geometry_training_status for full metrics",
                ],
            }

    if "mc_geometry_dare_sparsity" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dare_sparsity(checkpointPath: str, basePath: str | None = None) -> dict:
            analysis = geometry_adapter_service.analyze_dare(checkpointPath, basePath)
            readiness = geometry_adapter_service.dare_merge_readiness(analysis.effective_sparsity)
            per_layer = []
            for name, metrics in analysis.per_layer_sparsity.items():
                importance = max(0.0, min(1.0, metrics.essential_fraction))
                per_layer.append(
                    {
                        "layerName": name,
                        "sparsity": metrics.sparsity,
                        "importance": importance,
                        "canDrop": metrics.sparsity >= analysis.recommended_drop_rate,
                    }
                )
            layer_ranking = [entry["layerName"] for entry in sorted(per_layer, key=lambda item: item["importance"], reverse=True)]
            interpretation = (
                f"Effective sparsity {analysis.effective_sparsity:.2%} "
                f"({analysis.quality_assessment.value}). Recommended drop rate "
                f"{analysis.recommended_drop_rate:.2f}."
            )
            return {
                "_schema": "mc.geometry.dare_sparsity.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "effectiveSparsity": analysis.effective_sparsity,
                "qualityAssessment": analysis.quality_assessment.value,
                "mergeReadiness": readiness,
                "perLayerSparsity": per_layer or None,
                "layerRanking": layer_ranking or None,
                "recommendedDropRate": analysis.recommended_drop_rate,
                "interpretation": interpretation,
                "nextActions": [
                    "mc_geometry_dora_decomposition for learning type",
                    "mc_checkpoint_score for quality assessment",
                    "mc_checkpoint_export for deployment",
                ],
            }

    if "mc_geometry_dora_decomposition" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_geometry_dora_decomposition(checkpointPath: str, basePath: str | None = None) -> dict:
            result = geometry_adapter_service.analyze_dora(checkpointPath, basePath)
            learning_type = geometry_adapter_service.dora_learning_type(result)
            learning_confidence = geometry_adapter_service.dora_learning_type_confidence(result)
            stability_score = geometry_adapter_service.dora_stability_score(result)
            overfit_risk = geometry_adapter_service.dora_overfit_risk(result)
            per_layer = []
            for name, metrics in result.per_layer_metrics.items():
                if metrics.interpretation.value in {"amplification", "attenuation"}:
                    dominant = "magnitude"
                elif metrics.interpretation.value == "rotation":
                    dominant = "direction"
                else:
                    dominant = "balanced"
                per_layer.append(
                    {
                        "layerName": name,
                        "magnitudeChange": metrics.relative_magnitude_change,
                        "directionalDrift": metrics.directional_drift,
                        "dominantType": dominant,
                    }
                )
            learning_type_value = learning_type if learning_type != "minimal" else "balanced"
            return {
                "_schema": "mc.geometry.dora_decomposition.v1",
                "checkpointPath": checkpointPath,
                "baseModelPath": basePath,
                "magnitudeChangeRatio": result.overall_magnitude_change,
                "directionalDrift": result.overall_directional_drift,
                "learningType": learning_type_value,
                "learningTypeConfidence": learning_confidence,
                "perLayerDecomposition": per_layer or None,
                "stabilityScore": stability_score,
                "overfitRisk": overfit_risk,
                "interpretation": geometry_adapter_service.dora_interpretation(result),
                "nextActions": [
                    "mc_geometry_dare_sparsity for sparsity assessment",
                    "mc_checkpoint_export for deployment",
                ],
            }

    if "mc_validate_train" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_validate_train(
            model: str,
            dataset: str,
            sequenceLength: int = 2048,
            batchSize: int | None = None,
        ) -> dict:
            if sequenceLength <= 0:
                raise ValueError("sequenceLength must be a positive integer")
            if batchSize is not None and batchSize <= 0:
                raise ValueError("batchSize must be a positive integer")
            dataset_path = _require_existing_path(dataset)
            resolved_batch = batchSize if batchSize is not None else 1
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                learning_rate=1e-5,
                batch_size=resolved_batch,
                epochs=1,
                sequence_length=sequenceLength,
            )
            result = training_service.preflight(config)
            valid = result["canProceed"]
            metal_available = system_service.status().get("metalAvailable", False)
            return {
                "_schema": "mc.validate.train.v1",
                "valid": valid,
                "metalAvailable": metal_available,
                "recommendedBatchSize": result["predictedBatchSize"],
                "nextActions": (
                    [f"mc_train_start with model={model}, dataset={dataset}"]
                    if valid
                    else ["Reduce batch size", "Check model availability"]
                ),
            }

    if "mc_estimate_train" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_estimate_train(
            model: str,
            dataset: str,
            batchSize: int = 1,
            sequenceLength: int = 2048,
        ) -> dict:
            if batchSize <= 0:
                raise ValueError("batchSize must be a positive integer")
            if sequenceLength <= 0:
                raise ValueError("sequenceLength must be a positive integer")
            dataset_path = _require_existing_path(dataset)
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                learning_rate=1e-5,
                batch_size=batchSize,
                epochs=1,
                sequence_length=sequenceLength,
            )
            result = training_service.preflight(config)
            will_fit = result["canProceed"]
            return {
                "_schema": "mc.estimate.train.v1",
                "willFit": will_fit,
                "recommendedBatchSize": result["predictedBatchSize"],
                "projectedPeakGB": result["estimatedVRAMUsageBytes"] / (1024**3),
                "availableGB": result["availableVRAMBytes"] / (1024**3),
                "tokensPerSecond": None,
                "etaSeconds": None,
                "confidence": "low",
                "nextActions": (
                    [f"mc_train_start with recommended batch size {result['predictedBatchSize']}"]
                    if will_fit
                    else ["Reduce batch size", "Reduce sequence length", "Use smaller model"]
                ),
            }

    if "mc_dataset_validate" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_validate(path: str) -> dict:
            dataset_path = _require_existing_path(path)
            result = dataset_service.validate_dataset(dataset_path)
            return {
                "_schema": "mc.dataset.validate.v1",
                "valid": result["valid"],
                "path": dataset_path,
                "exampleCount": result["totalExamples"],
                "tokenStats": {
                    "min": result["minTokens"],
                    "max": result["maxTokens"],
                    "average": result["averageTokens"],
                },
                "warnings": result["warnings"],
                "errors": result["errors"],
                "nextActions": (
                    [f"mc_train_start with dataset={dataset_path}"]
                    if result["valid"]
                    else ["Fix dataset issues", "mc_dataset_validate after fixes"]
                ),
            }

    if "mc_dataset_get_row" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dataset_get_row(path: str, lineNumber: int) -> dict:
            if lineNumber <= 0:
                raise ValueError("lineNumber must be a positive integer")
            dataset_path = _require_existing_path(path)
            row = dataset_editor_service.get_row(dataset_path, lineNumber)
            return _row_payload(row)

    if "mc_dataset_update_row" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_update_row(path: str, lineNumber: int, content: dict) -> dict:
            if lineNumber <= 0:
                raise ValueError("lineNumber must be a positive integer")
            dataset_path = _require_existing_path(path)
            result = dataset_editor_service.update_row(dataset_path, lineNumber, content)
            return {
                "_schema": "mc.dataset.edit.v1",
                "status": result.status,
                "lineNumber": result.line_number,
                "row": _row_payload(result.row) if result.row else None,
                "warnings": result.warnings,
            }

    if "mc_dataset_add_row" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_add_row(path: str, format: str, fields: dict) -> dict:
            parsed_format = _parse_dataset_format(format)
            result = dataset_editor_service.add_row(path, parsed_format, fields)
            return {
                "_schema": "mc.dataset.edit.v1",
                "status": result.status,
                "lineNumber": result.line_number,
                "row": _row_payload(result.row) if result.row else None,
                "warnings": result.warnings,
            }

    if "mc_dataset_delete_row" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_delete_row(path: str, lineNumber: int) -> dict:
            if lineNumber <= 0:
                raise ValueError("lineNumber must be a positive integer")
            dataset_path = _require_existing_path(path)
            result = dataset_editor_service.delete_row(dataset_path, lineNumber)
            return {
                "_schema": "mc.dataset.edit.v1",
                "status": result.status,
                "lineNumber": result.line_number,
                "row": None,
                "warnings": result.warnings,
            }

    if "mc_dataset_convert" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_dataset_convert(path: str, targetFormat: str, outputPath: str) -> dict:
            dataset_path = _require_existing_path(path)
            parsed_format = _parse_dataset_format(targetFormat)
            result = dataset_editor_service.convert_dataset(dataset_path, parsed_format, outputPath)
            return {
                "_schema": "mc.dataset.convert.v1",
                "sourcePath": result.source_path,
                "outputPath": result.output_path,
                "targetFormat": result.target_format.value,
                "lineCount": result.line_count,
                "warnings": result.warnings,
            }

    if "mc_model_fetch" in tool_set:
        @mcp.tool(annotations=NETWORK_ANNOTATIONS)
        def mc_model_fetch(
            modelId: str,
            revision: str = "main",
            idempotencyKey: str | None = None,
        ) -> dict:
            if idempotencyKey:
                previous = _get_idempotency("model_fetch", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.model.fetch.v1",
                        "wasExecuted": False,
                        "modelId": None,
                        "path": None,
                        "status": None,
                        "previousPath": previous,
                        "message": "Model already downloaded with this idempotency key",
                        "nextActions": ["mc_train_start with this model path"],
                    }

            result = model_service.fetch_model(modelId, revision, False, None, None)
            local_path = result["localPath"]
            if idempotencyKey:
                _set_idempotency("model_fetch", idempotencyKey, local_path)
            return {
                "_schema": "mc.model.fetch.v1",
                "wasExecuted": True,
                "modelId": modelId,
                "path": local_path,
                "status": "completed",
                "previousPath": None,
                "message": None,
                "nextActions": [f"mc_train_start with model={modelId}", f"mc_infer with model={modelId}"],
            }

    if "mc_checkpoint_export" in tool_set:
        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_checkpoint_export(
            checkpoint: str,
            format: str,
            outputPath: str,
            idempotencyKey: str | None = None,
        ) -> dict:
            checkpoint_path = _require_existing_directory(checkpoint)
            format_key = format.lower()
            supported_formats = ["gguf", "safetensors", "coreml", "ollama", "mlx", "npz"]
            if format_key not in supported_formats:
                supported = ", ".join(supported_formats)
                raise ValueError(f"Unsupported export format: {format}. Supported: {supported}")
            if idempotencyKey:
                previous = _get_idempotency("checkpoint_export", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.checkpoint.export.v1",
                        "wasExecuted": False,
                        "checkpoint": None,
                        "format": None,
                        "outputPath": None,
                        "status": None,
                        "previousOutputPath": previous,
                        "message": "Export already completed with this idempotency key",
                        "nextActions": ["mc_infer with the exported model"],
                    }

            result = checkpoint_service.export_checkpoint(checkpoint_path, format_key, outputPath)
            output_path = result["outputPath"]
            if idempotencyKey:
                _set_idempotency("checkpoint_export", idempotencyKey, output_path)
            return {
                "_schema": "mc.checkpoint.export.v1",
                "wasExecuted": True,
                "checkpoint": checkpoint_path,
                "format": format,
                "outputPath": output_path,
                "status": "completed",
                "previousOutputPath": None,
                "message": None,
                "nextActions": [f"mc_infer with model={output_path}"],
            }

    @mcp.resource("tc://models")
    def resource_models() -> str:
        models = model_service.list_models()
        entries = [
            {
                "id": model.id,
                "alias": model.alias,
                "path": model.path,
                "architecture": model.architecture,
                "format": model.format,
                "sizeBytes": model.size_bytes,
            }
            for model in models
        ]
        return dump_json(entries)

    @mcp.resource("tc://jobs")
    def resource_jobs() -> str:
        jobs = job_service.list_job_records()
        entries = []
        for job in jobs:
            progress = (job.current_step / job.total_steps) if job.total_steps else 0.0
            entries.append(
                {
                    "jobId": job.job_id,
                    "status": _map_job_status(job.status.value),
                    "createdAt": job.created_at.isoformat() + "Z",
                    "completedAt": job.completed_at.isoformat() + "Z" if job.completed_at else None,
                    "progress": progress,
                    "modelId": job.model_id,
                    "datasetPath": job.dataset_path,
                }
            )
        return dump_json(entries)

    @mcp.resource("tc://checkpoints")
    def resource_checkpoints() -> str:
        checkpoints = checkpoint_service.list_checkpoints().get("checkpoints", [])
        entries = [
            {
                "jobId": checkpoint.get("jobId"),
                "step": checkpoint.get("step"),
                "loss": checkpoint.get("loss"),
                "filePath": checkpoint.get("filePath"),
            }
            for checkpoint in checkpoints
        ]
        return dump_json(entries)

    @mcp.resource("tc://datasets")
    def resource_datasets() -> str:
        datasets = dataset_service.list_datasets()
        entries = [
            {
                "id": dataset.id,
                "name": dataset.name,
                "path": dataset.path,
                "sizeBytes": dataset.size_bytes,
                "exampleCount": dataset.example_count,
            }
            for dataset in datasets
        ]
        return dump_json(entries)

    @mcp.resource("tc://system")
    def resource_system() -> str:
        return dump_json(_system_status_payload())

    return mcp


def main() -> None:
    mcp = build_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
