from __future__ import annotations

import os
from mcp.server.fastmcp import FastMCP

from modelcypher.adapters.local_inference import LocalInferenceEngine
from modelcypher.core.use_cases.checkpoint_service import CheckpointService
from modelcypher.core.use_cases.dataset_service import DatasetService
from modelcypher.core.use_cases.inventory_service import InventoryService
from modelcypher.core.use_cases.job_service import JobService
from modelcypher.core.use_cases.model_service import ModelService
from modelcypher.core.use_cases.system_service import SystemService
from modelcypher.core.use_cases.training_service import TrainingService
from modelcypher.core.domain.training import TrainingConfig
from modelcypher.utils.json import dump_json


TOOL_PROFILES = {
    "full": {
        "tc_inventory",
        "tc_train_start",
        "tc_job_status",
        "tc_job_list",
        "tc_job_cancel",
        "tc_job_pause",
        "tc_job_resume",
        "tc_system_status",
        "tc_validate_train",
        "tc_estimate_train",
        "tc_dataset_validate",
        "tc_model_fetch",
        "tc_model_list",
        "tc_checkpoint_export",
        "tc_infer",
    },
    "training": {
        "tc_inventory",
        "tc_train_start",
        "tc_job_status",
        "tc_job_list",
        "tc_job_cancel",
        "tc_job_pause",
        "tc_job_resume",
        "tc_system_status",
        "tc_validate_train",
        "tc_estimate_train",
        "tc_dataset_validate",
        "tc_model_fetch",
        "tc_model_list",
        "tc_checkpoint_export",
    },
    "inference": {
        "tc_inventory",
        "tc_model_list",
        "tc_infer",
        "tc_system_status",
    },
    "monitoring": {
        "tc_inventory",
        "tc_job_status",
        "tc_job_list",
        "tc_system_status",
    },
}


def build_server() -> FastMCP:
    profile = os.environ.get("TC_MCP_PROFILE", "full")
    tool_set = TOOL_PROFILES.get(profile, TOOL_PROFILES["full"])

    mcp = FastMCP("ModelCypher", json_response=True)
    inventory_service = InventoryService()
    training_service = TrainingService()
    job_service = JobService()
    model_service = ModelService()
    dataset_service = DatasetService()
    system_service = SystemService()
    checkpoint_service = CheckpointService()
    inference_engine = LocalInferenceEngine()

    if "tc_inventory" in tool_set:
        @mcp.tool()
        def tc_inventory() -> dict:
            return inventory_service.inventory()

    if "tc_train_start" in tool_set:
        @mcp.tool()
        def tc_train_start(
            model: str,
            dataset: str,
            epochs: int = 3,
            learningRate: float = 1e-5,
            batchSize: int = 4,
            sequenceLength: int = 2048,
            loraRank: int | None = None,
            loraAlpha: float | None = None,
        ) -> dict:
            lora = None
            if loraRank and loraAlpha:
                from modelcypher.core.domain.training import LoRAConfig

                lora = LoRAConfig(rank=loraRank, alpha=loraAlpha, dropout=0.0, targets=["q_proj", "v_proj"])
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset,
                learning_rate=learningRate,
                batch_size=batchSize,
                epochs=epochs,
                sequence_length=sequenceLength,
                lora=lora,
            )
            result, _ = training_service.start(config, stream=False)
            return {"jobId": result["jobId"], "status": "started", "batchSize": batchSize}

    if "tc_job_status" in tool_set:
        @mcp.tool()
        def tc_job_status(jobId: str) -> dict:
            status = training_service.status(jobId)
            return {
                "jobId": status["jobId"],
                "status": status["status"],
                "progress": (status["currentStep"] / status["totalSteps"]) if status["totalSteps"] else 0.0,
                "currentEpoch": status["currentEpoch"],
                "totalEpochs": status["totalEpochs"],
                "currentStep": status["currentStep"],
                "totalSteps": status["totalSteps"],
                "loss": status["loss"],
                "learningRate": status["learningRate"],
                "tokensPerSecond": None,
                "etaSeconds": None,
                "memoryUsageMB": None,
                "startedAt": status["createdAt"],
                "modelId": status["modelId"],
                "datasetPath": status["datasetPath"],
            }

    if "tc_job_list" in tool_set:
        @mcp.tool()
        def tc_job_list(status: str | None = None, activeOnly: bool = False) -> list[dict]:
            return job_service.list_jobs(status=status, active_only=activeOnly)

    if "tc_job_cancel" in tool_set:
        @mcp.tool()
        def tc_job_cancel(jobId: str) -> dict:
            result = training_service.cancel(jobId)
            return {"status": result["status"], "jobId": jobId}

    if "tc_job_pause" in tool_set:
        @mcp.tool()
        def tc_job_pause(jobId: str) -> dict:
            result = training_service.pause(jobId)
            return {"status": result["status"], "jobId": jobId}

    if "tc_job_resume" in tool_set:
        @mcp.tool()
        def tc_job_resume(jobId: str) -> dict:
            result = training_service.resume(jobId)
            return {"status": result["status"], "jobId": jobId}

    if "tc_model_list" in tool_set:
        @mcp.tool()
        def tc_model_list() -> list[dict]:
            models = model_service.list_models()
            return [
                {
                    "id": model.id,
                    "alias": model.alias,
                    "format": model.format,
                    "sizeBytes": model.size_bytes,
                    "path": model.path,
                    "architecture": model.architecture,
                    "parameterCount": model.parameter_count,
                }
                for model in models
            ]

    if "tc_infer" in tool_set:
        @mcp.tool()
        def tc_infer(
            model: str,
            prompt: str,
            maxTokens: int = 512,
            temperature: float = 0.7,
            topP: float = 0.95,
        ) -> dict:
            return inference_engine.infer(model, prompt, maxTokens, temperature, topP)

    if "tc_system_status" in tool_set:
        @mcp.tool()
        def tc_system_status() -> dict:
            return system_service.readiness()

    if "tc_validate_train" in tool_set:
        @mcp.tool()
        def tc_validate_train(
            model: str,
            dataset: str,
            learningRate: float = 1e-5,
            batchSize: int = 4,
            sequenceLength: int = 2048,
            epochs: int = 1,
        ) -> dict:
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset,
                learning_rate=learningRate,
                batch_size=batchSize,
                epochs=epochs,
                sequence_length=sequenceLength,
            )
            result = training_service.preflight(config)
            return {
                "valid": result["canProceed"],
                "model": {"id": model, "found": True, "architecture": None},
                "dataset": {"path": dataset, "exists": True, "readable": True},
                "memory": {
                    "willFit": result["canProceed"],
                    "recommendedBatchSize": result["predictedBatchSize"],
                    "projectedPeakGB": None,
                    "availableGB": None,
                },
                "config": {
                    "batchSize": batchSize,
                    "sequenceLength": sequenceLength,
                    "learningRate": learningRate,
                    "epochs": epochs,
                },
                "warnings": [],
                "errors": [] if result["canProceed"] else ["Configuration may not fit in memory"],
                "nextActions": [f"tc train start --model {model} --dataset {dataset}"],
            }

    if "tc_estimate_train" in tool_set:
        @mcp.tool()
        def tc_estimate_train(
            model: str,
            dataset: str,
            batchSize: int = 1,
            sequenceLength: int = 2048,
            dtype: str = "fp16",
        ) -> dict:
            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset,
                learning_rate=1e-5,
                batch_size=batchSize,
                epochs=1,
                sequence_length=sequenceLength,
            )
            result = training_service.preflight(config)
            return {
                "willFit": result["canProceed"],
                "recommendedBatchSize": result["predictedBatchSize"],
                "projectedPeakGB": result["estimatedVRAMUsageBytes"] / (1024**3),
                "availableGB": result["availableVRAMBytes"] / (1024**3),
                "ttftSeconds": None,
                "tokensPerSecond": None,
                "tokensPerSecondMin": None,
                "tokensPerSecondMax": None,
                "confidence": "low",
                "powerSource": "unknown",
                "thermalState": "unknown",
                "etaSeconds": None,
                "notes": [f"dtype={dtype}"],
                "nextActions": [f"tc train start --model {model} --dataset {dataset} --batch-size {batchSize}"],
            }

    if "tc_dataset_validate" in tool_set:
        @mcp.tool()
        def tc_dataset_validate(dataset: str) -> dict:
            return dataset_service.validate_dataset(dataset)

    if "tc_model_fetch" in tool_set:
        @mcp.tool()
        def tc_model_fetch(
            repoId: str,
            revision: str = "main",
            autoRegister: bool = False,
            alias: str | None = None,
            architecture: str | None = None,
        ) -> dict:
            return model_service.fetch_model(repoId, revision, autoRegister, alias, architecture)

    if "tc_checkpoint_export" in tool_set:
        @mcp.tool()
        def tc_checkpoint_export(checkpointPath: str, format: str, outputPath: str) -> dict:
            result = checkpoint_service.export_checkpoint(checkpointPath, format, outputPath)
            return result

    @mcp.resource("tc://models")
    def resource_models() -> str:
        return dump_json(model_service.list_models())

    @mcp.resource("tc://jobs")
    def resource_jobs() -> str:
        return dump_json(job_service.list_jobs())

    @mcp.resource("tc://checkpoints")
    def resource_checkpoints() -> str:
        return dump_json(checkpoint_service.list_checkpoints())

    @mcp.resource("tc://datasets")
    def resource_datasets() -> str:
        return dump_json(dataset_service.list_datasets())

    @mcp.resource("tc://system")
    def resource_system() -> str:
        return dump_json(system_service.readiness())

    return mcp


def main() -> None:
    mcp = build_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
