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

"""Training and job management MCP tools.

Contains tools for:
- Training start, preflight, export
- Job status, list, detail, cancel, pause, resume, delete
- Training validation and estimation
"""

from __future__ import annotations

from pathlib import Path

from modelcypher.core.domain.training import Hyperparameters, TrainingConfig
from modelcypher.mcp.security import ConfirmationError, create_confirmation_response

from .common import (
    DESTRUCTIVE_ANNOTATIONS,
    MUTATING_ANNOTATIONS,
    READ_ONLY_ANNOTATIONS,
    ServiceContext,
    map_job_status,
    require_existing_path,
)


def register_training_tools(ctx: ServiceContext) -> None:
    """Register training and job management MCP tools."""
    mcp = ctx.mcp
    tool_set = ctx.tool_set

    def _parse_hyperparameters(hyperparameters: dict) -> Hyperparameters:
        required_hp = {
            "batchSize",
            "learningRate",
            "epochs",
            "sequenceLength",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "mixedPrecision",
            "computePrecision",
            "warmupSteps",
            "weightDecay",
            "seed",
            "deterministic",
            "optimizerType",
        }
        missing_hp = sorted(k for k in required_hp if k not in hyperparameters)
        if missing_hp:
            raise ValueError(f"hyperparameters missing required fields: {missing_hp}")

        batch_size = hyperparameters["batchSize"]
        learning_rate = hyperparameters["learningRate"]
        epochs = hyperparameters["epochs"]
        sequence_length = hyperparameters["sequenceLength"]
        grad_accum = hyperparameters["gradientAccumulationSteps"]
        gradient_checkpointing = hyperparameters["gradientCheckpointing"]
        mixed_precision = hyperparameters["mixedPrecision"]
        compute_precision = hyperparameters["computePrecision"]
        warmup_steps = hyperparameters["warmupSteps"]
        weight_decay = hyperparameters["weightDecay"]
        seed = hyperparameters["seed"]
        deterministic = hyperparameters["deterministic"]
        optimizer_type = hyperparameters["optimizerType"]

        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learningRate must be positive")
        if batch_size <= 0:
            raise ValueError("batchSize must be a positive integer")
        if sequence_length <= 0:
            raise ValueError("sequenceLength must be a positive integer")
        if grad_accum <= 0:
            raise ValueError("gradientAccumulationSteps must be a positive integer")
        if warmup_steps < 0:
            raise ValueError("warmupSteps must be >= 0")
        if weight_decay < 0:
            raise ValueError("weightDecay must be >= 0")

        from modelcypher.core.domain.training import ComputePrecision

        try:
            precision = ComputePrecision(compute_precision)
        except ValueError as exc:
            raise ValueError(f"Invalid computePrecision: {compute_precision}") from exc

        return Hyperparameters(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            sequence_length=sequence_length,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=gradient_checkpointing,
            mixed_precision=mixed_precision,
            compute_precision=precision,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            seed=seed,
            deterministic=deterministic,
            optimizer_type=optimizer_type,
        )

    if "mc_train_start" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_train_start(
            model: str,
            dataset: str,
            outputPath: str,
            hyperparameters: dict,
            lora: dict | None = None,
            idempotencyKey: str | None = None,
            autoEval: bool,
            evalDataset: str | None = None,
            evalMetrics: list[str] | None = None,
            evalBatchSize: int | None = None,
            evalMaxSamples: int | None = None,
            evalWait: bool | None = None,
        ) -> dict:
            dataset_path = require_existing_path(dataset)
            hyper = _parse_hyperparameters(hyperparameters)
            batch_size = hyper.batch_size
            if idempotencyKey:
                previous = ctx.get_idempotency("train_start", idempotencyKey)
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
                    }

            lora_config = None
            if lora is not None:
                required_lora = {"rank", "alpha", "dropout", "targetModules"}
                missing_lora = sorted(k for k in required_lora if k not in lora)
                if missing_lora:
                    raise ValueError(f"lora missing required fields: {missing_lora}")
                if lora["rank"] <= 0:
                    raise ValueError("lora.rank must be a positive integer")
                if lora["alpha"] <= 0:
                    raise ValueError("lora.alpha must be positive")
                if lora["dropout"] < 0:
                    raise ValueError("lora.dropout must be >= 0")
                from modelcypher.core.domain.training import LoRAConfig

                lora_config = LoRAConfig(
                    rank=lora["rank"],
                    alpha=lora["alpha"],
                    dropout=lora["dropout"],
                    target_modules=lora["targetModules"],
                )

            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                output_path=outputPath,
                hyperparameters=hyper,
                lora_config=lora_config,
            )
            result, _ = ctx.training_service.start(config, stream=False)
            job_id = result["jobId"]
            if idempotencyKey:
                ctx.set_idempotency("train_start", idempotencyKey, job_id)

            auto_eval_payload = None
            if autoEval:
                if evalDataset is None:
                    raise ValueError("evalDataset is required when autoEval is enabled")
                if evalBatchSize is None:
                    raise ValueError("evalBatchSize is required when autoEval is enabled")
                if evalWait is None:
                    raise ValueError("evalWait is required when autoEval is enabled")
                auto_eval_payload = {
                    "enabled": True,
                    "evalDataset": evalDataset,
                    "metrics": evalMetrics or [],
                    "batchSize": evalBatchSize,
                    "maxSamples": evalMaxSamples,
                    "waitForCompletion": evalWait,
                }

            return {
                "_schema": "mc.train.start.v1",
                "jobId": job_id,
                "status": "started",
                "batchSize": batch_size,
                "wasExecuted": True,
                "previousJobId": None,
                "message": "Training started with auto-evaluation enabled"
                if auto_eval_payload
                else None,
                "autoEval": auto_eval_payload,
            }

    if "mc_job_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_status(jobId: str) -> dict:
            status = ctx.training_service.status(jobId)
            mapped_status = map_job_status(status["status"])
            return {
                "_schema": "mc.job.status.v1",
                "jobId": status["jobId"],
                "status": mapped_status,
                "progress": (status["currentStep"] / status["totalSteps"])
                if status["totalSteps"]
                else 0.0,
                "currentEpoch": status["currentEpoch"],
                "totalEpochs": status["totalEpochs"],
                "loss": status["loss"],
                "throughputTPS": None,
                "etaSeconds": None,
                "lastUpdate": status.get("updatedAt", status.get("createdAt")),
            }

    if "mc_job_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_list(status: str | None = None, activeOnly: bool = False) -> dict:
            status_filter = status
            if status_filter == "queued":
                status_filter = "pending"
            if status_filter == "canceled":
                status_filter = "cancelled"
            jobs = ctx.job_service.list_jobs(status=status_filter, active_only=activeOnly)
            entries = []
            for job in jobs:
                progress = (
                    (job["currentStep"] / job["totalSteps"]) if job.get("totalSteps") else 0.0
                )
                entries.append(
                    {
                        "jobId": job["jobId"],
                        "status": map_job_status(job["status"]),
                        "modelId": job["modelId"],
                        "progress": progress,
                    }
                )
            return {
                "_schema": "mc.job.list.v1",
                "jobs": entries,
                "count": len(entries),
            }

    if "mc_job_detail" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_job_detail(jobId: str) -> dict:
            payload = ctx.job_service.show_job(jobId, include_loss_history=True)
            hyper = payload.get("hyperparameters")
            if not isinstance(hyper, dict):
                raise ValueError("Job details missing hyperparameters.")
            required_hp = {"learningRate", "batchSize", "epochs", "sequenceLength"}
            missing_hp = sorted(k for k in required_hp if k not in hyper)
            if missing_hp:
                raise ValueError(f"Job hyperparameters missing fields: {missing_hp}")
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
                "status": map_job_status(payload["status"]),
                "createdAt": payload["createdAt"],
                "startedAt": payload.get("startedAt"),
                "completedAt": payload.get("completedAt"),
                "modelId": payload["modelId"],
                "datasetPath": payload["datasetPath"],
                "progress": progress,
                "finalLoss": payload.get("finalLoss"),
                "checkpoints": payload.get("checkpoints", []),
                "hyperparameters": {
                    "learningRate": hyper["learningRate"],
                    "batchSize": hyper["batchSize"],
                    "epochs": hyper["epochs"],
                    "sequenceLength": hyper["sequenceLength"],
                },
                "lossHistory": normalized_loss,
            }

    if "mc_job_cancel" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_job_cancel(jobId: str) -> dict:
            ctx.training_service.cancel(jobId)
            return {
                "_schema": "mc.job.cancel.v1",
                "jobId": jobId,
                "status": "canceled",
            }

    if "mc_job_pause" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_job_pause(jobId: str) -> dict:
            ctx.training_service.pause(jobId)
            return {
                "_schema": "mc.job.pause.v1",
                "jobId": jobId,
                "status": "paused",
            }

    if "mc_job_resume" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_job_resume(jobId: str) -> dict:
            ctx.training_service.resume(jobId)
            return {
                "_schema": "mc.job.resume.v1",
                "jobId": jobId,
                "status": "resumed",
            }

    if "mc_job_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_job_delete(jobId: str, confirmationToken: str | None = None) -> dict:
            """Delete a training job. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                ctx.confirmation_manager.require_confirmation(
                    operation="delete_job",
                    tool_name="mc_job_delete",
                    parameters={"jobId": jobId},
                    description=f"Delete training job '{jobId}' and all associated data",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete training job '{jobId}' and all associated data",
                    timeout_seconds=ctx.security_config.confirmation_timeout_seconds,
                )
            ctx.job_service.delete_job(jobId)
            return {
                "_schema": "mc.job.delete.v1",
                "jobId": jobId,
                "status": "deleted",
            }

    if "mc_validate_train" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_validate_train(
            model: str,
            dataset: str,
            outputPath: str,
            hyperparameters: dict,
        ) -> dict:
            dataset_path = require_existing_path(dataset)
            hyper = _parse_hyperparameters(hyperparameters)
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                output_path=outputPath,
                hyperparameters=hyper,
            )
            result = ctx.training_service.preflight(config)
            valid = result["canProceed"]
            metal_available = ctx.system_service.status().get("metalAvailable", False)
            return {
                "_schema": "mc.validate.train.v1",
                "valid": valid,
                "metalAvailable": metal_available,
                "recommendedBatchSize": result["predictedBatchSize"],
            }

    if "mc_estimate_train" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_estimate_train(
            model: str,
            dataset: str,
            outputPath: str,
            hyperparameters: dict,
        ) -> dict:
            dataset_path = require_existing_path(dataset)
            hyper = _parse_hyperparameters(hyperparameters)
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                output_path=outputPath,
                hyperparameters=hyper,
            )
            result = ctx.training_service.preflight(config)
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
            }

    if "mc_train_preflight" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_train_preflight(
            model: str,
            dataset: str,
            outputPath: str,
            hyperparameters: dict,
            lora: dict | None = None,
        ) -> dict:
            """Check training feasibility."""
            dataset_path = require_existing_path(dataset)

            lora_config = None
            if lora is not None:
                required_lora = {"rank", "alpha", "dropout", "targetModules"}
                missing_lora = sorted(k for k in required_lora if k not in lora)
                if missing_lora:
                    raise ValueError(f"lora missing required fields: {missing_lora}")
                if lora["rank"] <= 0:
                    raise ValueError("lora.rank must be a positive integer")
                if lora["alpha"] <= 0:
                    raise ValueError("lora.alpha must be positive")
                if lora["dropout"] < 0:
                    raise ValueError("lora.dropout must be >= 0")
                from modelcypher.core.domain.training import LoRAConfig

                lora_config = LoRAConfig(
                    rank=lora["rank"],
                    alpha=lora["alpha"],
                    dropout=lora["dropout"],
                    target_modules=lora["targetModules"],
                )

            hyper = _parse_hyperparameters(hyperparameters)
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                output_path=outputPath,
                hyperparameters=hyper,
                lora_config=lora_config,
            )

            result = ctx.training_service.preflight(config)
            return {
                "_schema": "mc.train.preflight.v1",
                "predictedBatchSize": result["predictedBatchSize"],
                "estimatedVRAMUsageBytes": result["estimatedVRAMUsageBytes"],
                "availableVRAMBytes": result["availableVRAMBytes"],
                "canProceed": result["canProceed"],
            }

    if "mc_train_export" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_train_export(jobId: str, output: str, format: str = "safetensors") -> dict:
            """Export trained model from job."""
            status = ctx.training_service.status(jobId)
            current_step = status["currentStep"]

            output_path = Path(output).expanduser().resolve()
            ctx.checkpoint_service.export_checkpoint(
                job_id=jobId,
                step=current_step,
                output_path=str(output_path),
                format=format,
                fuse_adapters=True,
            )
            return {
                "_schema": "mc.train.export.v1",
                "jobId": jobId,
                "step": current_step,
                "outputPath": str(output_path),
                "status": "exported",
            }
