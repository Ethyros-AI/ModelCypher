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

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.domain.model_search import (
    ModelSearchError,
    ModelSearchFilters,
    ModelSearchLibraryFilter,
    ModelSearchQuantization,
    ModelSearchSortOption,
)
from modelcypher.core.domain.training import TrainingConfig
from modelcypher.core.use_cases.concept_response_matrix_service import (
    ConceptResponseMatrixService,
)
from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService
from modelcypher.core.use_cases.evaluation_service import (
    EvalConfig,
)
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.geometry_stitch_service import GeometryStitchService
from modelcypher.core.use_cases.merge_validation_service import (
    MergeValidationConfig,
)
from modelcypher.core.use_cases.unified_geometric_merge import UnifiedGeometricMerger
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService
from modelcypher.core.use_cases.settings_service import SettingsService
from modelcypher.infrastructure.container import PortRegistry
from modelcypher.infrastructure.service_factory import ServiceFactory
from modelcypher.mcp.security import (
    ConfirmationError,
    ConfirmationManager,
    create_confirmation_response,
    validate_security_config,
)
from modelcypher.utils.json import dump_json

IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60
DEFAULT_PATH_THRESHOLD = 0.55
DEFAULT_PATH_MAX_TOKENS = 200


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
        "mc_job_delete",
        "mc_system_status",
        "mc_validate_train",
        "mc_estimate_train",
        "mc_model_fetch",
        "mc_model_list",
        "mc_model_search",
        "mc_model_probe",
        "mc_model_validate_merge",
        "mc_model_analyze_alignment",
        "mc_model_merge",  # New
        "mc_model_register",  # New
        "mc_model_delete",  # New
        "mc_checkpoint_export",
        "mc_checkpoint_list",  # New
        "mc_checkpoint_delete",  # New
        "mc_geometry_training_status",
        "mc_geometry_training_history",
        "mc_geometry_validate",
        "mc_safety_circuit_breaker",
        "mc_safety_persona_drift",
        "mc_safety_redteam_scan",  # New
        "mc_safety_behavioral_probe",  # New
        "mc_entropy_analyze",  # New
        "mc_entropy_detect_distress",  # New
        "mc_entropy_verify_baseline",  # New
        "mc_geometry_safety_jailbreak_test",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
        "mc_geometry_primes_list",
        "mc_geometry_primes_probe",
        "mc_geometry_primes_compare",
        "mc_geometry_crm_build",
        "mc_geometry_crm_compare",
        "mc_geometry_crm_sequence_inventory",
        "mc_geometry_stitch_analyze",
        "mc_geometry_stitch_apply",
        "mc_geometry_path_detect",  # New
        "mc_geometry_path_compare",  # New
        "mc_geometry_concept_detect",  # New
        "mc_geometry_concept_compare",  # New
        "mc_geometry_cross_cultural_analyze",  # New
        "mc_geometry_gromov_wasserstein",  # New
        "mc_geometry_intrinsic_dimension",  # New
        "mc_geometry_topological_fingerprint",  # New
        "mc_geometry_sparse_domains",  # New
        "mc_geometry_sparse_locate",  # New
        "mc_geometry_refusal_pairs",  # New
        "mc_geometry_refusal_detect",  # New
        "mc_geometry_persona_traits",  # New
        "mc_geometry_persona_extract",  # New
        "mc_geometry_persona_drift",  # New
        "mc_geometry_manifold_cluster",  # New
        "mc_geometry_manifold_dimension",  # New
        "mc_geometry_manifold_query",  # New
        "mc_geometry_transport_merge",  # New
        "mc_geometry_transport_synthesize",  # New
        "mc_geometry_invariant_map_layers",  # New
        "mc_geometry_invariant_collapse_risk",  # New
        "mc_geometry_atlas_inventory",  # New - multi-atlas probe inventory
        "mc_infer",
        # New tools for CLI/MCP parity
        "mc_calibration_run",
        "mc_calibration_status",
        "mc_calibration_apply",
        "mc_stability_run",
        "mc_stability_report",
        "mc_agent_eval_run",
        "mc_agent_eval_results",
        "mc_dashboard_metrics",
        "mc_dashboard_export",
        "mc_help_ask",
        "mc_schema",
        "mc_infer_run",
        "mc_infer_batch",
        "mc_infer_suite",
        # Thermo tools
        "mc_thermo_measure",
        "mc_thermo_detect",
        "mc_thermo_detect_batch",
        "mc_thermo_analyze",  # New
        "mc_thermo_path",  # New
        "mc_thermo_path_integration",  # New
        "mc_thermo_entropy",  # New
        # Storage tools
        "mc_storage_usage",
        "mc_storage_cleanup",
        # Ensemble tools
        "mc_ensemble_create",
        "mc_ensemble_run",
        "mc_ensemble_list",  # New
        "mc_ensemble_delete",  # New
        # Research tools
        "mc_research_sparse_region",
        "mc_research_afm",
        # Adapter tools
        "mc_adapter_merge",
        "mc_adapter_inspect",  # New
        # Phase 2: Safety tools
        "mc_safety_adapter_probe",  # New - adapter delta feature probing
        # Phase 2: Entropy tools
        "mc_entropy_window",  # New - sliding window tracking
        "mc_entropy_conversation_track",  # New - conversation entropy
        "mc_entropy_dual_path",  # New - dual-path adapter analysis
        # Phase 2: Agent tools
        "mc_agent_trace_import",  # New - trace import
        "mc_agent_trace_analyze",  # New - trace analytics
        "mc_agent_validate_action",  # New - action validation
        # Eval tools
        "mc_eval_run",  # New
        "mc_eval_list",  # New
        "mc_eval_show",  # New
        "mc_train_preflight",  # New
        "mc_train_export",  # New
        # Geometry refinement and stitching tools
        "mc_geometry_refinement_analyze",  # New - RefinementDensityAnalyzer
        "mc_geometry_stitch_train",  # New - AffineStitchingLayer training
        "mc_geometry_domain_profile",  # New - DomainSignalProfile
        # Merge validation tools
        "mc_merge_validate",  # New - Full merge validation suite
        "mc_merge_coherence",  # New - Coherence scoring
        "mc_merge_probe",  # New - Task probes
        "mc_merge_diagnose",  # New - Geometric diagnosis
        # Merge entropy tools
        "mc_merge_entropy_profile",  # New - Model entropy profile for merge planning
        "mc_merge_entropy_guide",  # New - Entropy-aware merge recommendations
        "mc_merge_entropy_validate",  # New - Post-merge entropy validation
        # Phase 13: CLI/MCP Parity
        "mc_model_validate_knowledge",  # New - Knowledge transfer validation (Gap 1)
        "mc_geometry_sparse_neurons",  # New - Per-neuron sparsity analysis (Gap 2)
        # 3D Spatial Metrology
        "mc_geometry_spatial_anchors",  # New - Spatial Prime Atlas anchors
        "mc_geometry_spatial_euclidean",  # New - Euclidean consistency test
        "mc_geometry_spatial_gravity",  # New - Gravity gradient analysis
        "mc_geometry_spatial_density",  # New - Volumetric density probe
        "mc_geometry_spatial_analyze",  # New - Full 3D world model analysis
        "mc_geometry_spatial_probe_model",  # New - End-to-end model probing
        # Domain Geometry Baselines
        "mc_geometry_baseline_list",  # New - List available baselines
        "mc_geometry_baseline_extract",  # New - Extract baseline from model
        "mc_geometry_baseline_validate",  # New - Validate model against baselines
        "mc_geometry_baseline_compare",  # New - Compare two models
        # Task management (MCP 2025 Tasks framework)
        "mc_task_list",  # New - List async tasks
        "mc_task_status",  # New - Get task status
        "mc_task_cancel",  # New - Cancel running task
        "mc_task_result",  # New - Get task result
        "mc_task_delete",  # New - Delete completed task
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
        "mc_job_delete",
        "mc_system_status",
        "mc_validate_train",
        "mc_estimate_train",
        "mc_model_fetch",
        "mc_model_list",
        "mc_model_search",
        "mc_checkpoint_export",
        "mc_checkpoint_list",
        "mc_checkpoint_delete",
        "mc_geometry_training_status",
        "mc_geometry_training_history",
        "mc_geometry_validate",
        "mc_safety_circuit_breaker",
        "mc_safety_persona_drift",
        "mc_safety_redteam_scan",  # New
        "mc_safety_behavioral_probe",  # New
        "mc_entropy_analyze",  # New
        "mc_entropy_detect_distress",  # New
        "mc_entropy_verify_baseline",  # New
        "mc_geometry_safety_jailbreak_test",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
        "mc_geometry_crm_build",
        "mc_geometry_crm_compare",
        "mc_geometry_crm_sequence_inventory",
        "mc_calibration_run",
        "mc_calibration_status",
        "mc_calibration_apply",
        # Thermo tools
        "mc_thermo_measure",
        "mc_thermo_detect",
        "mc_thermo_detect_batch",
        # Storage tools
        "mc_storage_usage",
        "mc_storage_cleanup",
        # Research tools
        "mc_research_sparse_region",
        "mc_research_afm",
        # Adapter tools
        "mc_adapter_merge",
        "mc_eval_run",
        "mc_eval_list",
        "mc_eval_show",
        "mc_train_preflight",
        "mc_train_export",
        # Geometry refinement and merge validation
        "mc_geometry_refinement_analyze",
        "mc_geometry_stitch_train",
        "mc_merge_validate",
        "mc_merge_diagnose",
        # Phase 13: CLI/MCP Parity
        "mc_model_validate_knowledge",  # Knowledge transfer validation
        "mc_geometry_sparse_neurons",  # Per-neuron sparsity analysis
        # 3D Spatial Metrology (for merge quality verification)
        "mc_geometry_spatial_anchors",
        "mc_geometry_spatial_analyze",
        "mc_geometry_spatial_probe_model",
        # Domain Geometry Baselines (for merge validation)
        "mc_geometry_baseline_list",
        "mc_geometry_baseline_validate",
        # Task management (async training jobs)
        "mc_task_list",
        "mc_task_status",
        "mc_task_cancel",
        "mc_task_result",
        "mc_task_delete",
    },
    "inference": {
        "mc_inventory",
        "mc_settings_snapshot",
        "mc_model_list",
        "mc_infer",
        "mc_infer_run",
        "mc_infer_batch",
        "mc_infer_suite",
        "mc_system_status",
        # Ensemble tools
        "mc_ensemble_create",
        "mc_ensemble_run",
        "mc_ensemble_list",
        "mc_ensemble_delete",
        # Merge entropy validation
        "mc_merge_entropy_validate",  # Post-merge stability check
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
        "mc_safety_redteam_scan",  # New
        "mc_safety_behavioral_probe",  # New
        "mc_entropy_analyze",  # New
        "mc_entropy_detect_distress",  # New
        "mc_entropy_verify_baseline",  # New
        "mc_geometry_safety_jailbreak_test",
        "mc_geometry_dare_sparsity",
        "mc_geometry_dora_decomposition",
        # Geometry refinement and merge validation (monitoring)
        "mc_geometry_refinement_analyze",
        "mc_geometry_domain_profile",
        "mc_merge_validate",
        "mc_merge_diagnose",
        # Phase 13: CLI/MCP Parity
        "mc_model_validate_knowledge",  # Knowledge transfer validation
        "mc_geometry_sparse_neurons",  # Per-neuron sparsity analysis
        # 3D Spatial Metrology (model quality monitoring)
        "mc_geometry_spatial_anchors",
        "mc_geometry_spatial_analyze",
        # Domain Geometry Baselines (model health monitoring)
        "mc_geometry_baseline_list",
        "mc_geometry_baseline_validate",
        # Task monitoring (read-only status checks)
        "mc_task_list",
        "mc_task_status",
        "mc_task_result",
    },
}


def _map_job_status(status: str) -> str:
    if status == "pending":
        return "queued"
    if status == "cancelled":
        return "canceled"
    return status


READ_ONLY_ANNOTATIONS = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}
MUTATING_ANNOTATIONS = {"readOnlyHint": False, "idempotentHint": False, "openWorldHint": False}
IDEMPOTENT_MUTATING_ANNOTATIONS = {
    "readOnlyHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
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

    # Create PortRegistry and ServiceFactory for proper dependency injection
    registry = PortRegistry.create_production()
    factory = ServiceFactory(registry)

    # Services via factory (require port dependencies)
    inventory_service = factory.inventory_service()
    training_service = factory.training_service()
    job_service = factory.job_service()
    model_service = factory.model_service()
    model_search_service = factory.model_search_service()
    system_service = factory.system_service()
    checkpoint_service = factory.checkpoint_service()
    evaluation_service = factory.evaluation_service()
    geometry_training_service = factory.geometry_training_service()
    ensemble_service = factory.ensemble_service()

    # Services without port dependencies (direct instantiation)
    model_probe_service = ModelProbeService()
    settings_service = SettingsService()

    # Use inference engine from registry
    inference_engine = registry.inference_engine
    embedder = EmbeddingDefaults.make_default_embedder()
    GeometryService(embedder=embedder)
    # GeometrySafetyService requires calibration - constructed on-demand via service provider
    # GeometryAdapterService is instantiated with model_loader in tool handlers
    ConceptResponseMatrixService(engine=inference_engine)
    GeometryStitchService(model_loader=registry.model_loader)

    from modelcypher.core.use_cases.adapter_service import AdapterService
    from modelcypher.core.use_cases.doc_service import DocService
    from modelcypher.core.use_cases.thermo_service import ThermoService

    thermo_service = ThermoService(embedder=embedder)
    adapter_service = AdapterService()
    doc_service = DocService()

    # Security configuration (optional, enabled via environment variables)
    security_config, security_issues = validate_security_config()
    if security_issues:
        import logging

        logger = logging.getLogger(__name__)
        for issue in security_issues:
            logger.warning(f"MCP Security: {issue}")
    confirmation_manager = ConfirmationManager(security_config)

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
            expired = [
                cache_key for cache_key, entry in idempotency_cache.items() if entry.is_expired()
            ]
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

    def _system_status_payload() -> dict:
        readiness = system_service.readiness()
        readiness_score = readiness.get("readinessScore", 0)
        if readiness_score >= 80:
            next_actions = ["mc_geometry_validate for baseline geometry", "mc_merge_validate to vet merges"]
        elif readiness_score >= 60:
            next_actions = ["Address blockers first", "mc_system_status to recheck"]
        else:
            next_actions = ["Fix critical blockers", "mc_model_list to verify models"]
        return {
            "_schema": "mc.system.status.v1",
            "machineName": readiness.get("machineName", ""),
            "unifiedMemoryGB": readiness.get("unifiedMemoryGB", 0),
            "mlxVersion": readiness.get("mlxVersion"),
            "cudaVersion": readiness.get("cudaVersion"),
            "jaxVersion": readiness.get("jaxVersion"),
            "preferredBackend": readiness.get("preferredBackend"),
            "cudaAvailable": readiness.get("cudaAvailable", False),
            "jaxAvailable": readiness.get("jaxAvailable", False),
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
                    }
                )
            return {
                "models": inventory.get("models", []),
                "checkpoints": inventory.get("checkpoints", []),
                "jobs": jobs,
                "workspace": inventory.get("workspace", {}),
                "mlxVersion": inventory.get("mlxVersion"),
                "cudaVersion": inventory.get("cudaVersion"),
                "jaxVersion": inventory.get("jaxVersion"),
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

                lora = LoRAConfig(
                    rank=loraRank, alpha=loraAlpha, dropout=0.0, targets=["q_proj", "v_proj"]
                )

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
                "message": "Training started with auto-evaluation enabled"
                if auto_eval_payload
                else None,
                "autoEval": auto_eval_payload,
                "nextActions": next_actions,
            }

    if "mc_eval_run" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_run(
            model: str,
            dataset: str,
            metrics: list[str] | None = None,
            batchSize: int = 4,
            maxSamples: int | None = None,
        ) -> dict:
            """Run evaluation on a model."""
            model_path = _require_existing_directory(model)
            dataset_path = _require_existing_path(dataset)

            config = EvalConfig(
                metrics=metrics,
                batch_size=batchSize,
                max_samples=maxSamples,
            )

            result = evaluation_service.run(model_path, dataset_path, config)

            return {
                "_schema": "mc.eval.run.v1",
                "evalId": result.eval_id,
                "averageLoss": result.average_loss,
                "perplexity": result.perplexity,
                "sampleCount": result.sample_count,
                "nextActions": [
                    f"mc_eval_show with evalId={result.eval_id}",
                    "mc_model_merge if metric is good",
                ],
            }

    if "mc_eval_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_list(limit: int = 50) -> dict:
            return evaluation_service.list_evaluations(limit)

    if "mc_eval_show" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_eval_show(evalId: str) -> dict:
            return evaluation_service.results(evalId)

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
                "progress": (status["currentStep"] / status["totalSteps"])
                if status["totalSteps"]
                else 0.0,
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

    if "mc_job_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_job_delete(jobId: str, confirmationToken: str | None = None) -> dict:
            """Delete a training job. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                confirmation_manager.require_confirmation(
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
                    timeout_seconds=security_config.confirmation_timeout_seconds,
                )
            job_service.delete_job(jobId)
            return {
                "_schema": "mc.job.delete.v1",
                "jobId": jobId,
                "status": "deleted",
                "nextActions": ["mc_job_list to verify"],
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
                else ["mc_model_probe with model", "mc_infer with model"]
            )
            return {
                "_schema": "mc.model.list.v1",
                "models": entries,
                "count": len(entries),
                "nextActions": next_actions,
            }

    if "mc_model_register" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_model_register(path: str, alias: str | None = None) -> dict:
            """Register a local model."""
            model_path = _require_existing_directory(path)
            model = model_service.register_model(model_path, alias=alias)
            return {
                "_schema": "mc.model.register.v1",
                "modelId": model.id,
                "path": model.path,
                "alias": model.alias,
                "status": "registered",
                "nextActions": ["mc_model_list to verify"],
            }

    if "mc_model_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_model_delete(modelId: str, confirmationToken: str | None = None) -> dict:
            """Delete a model. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                confirmation_manager.require_confirmation(
                    operation="delete_model",
                    tool_name="mc_model_delete",
                    parameters={"modelId": modelId},
                    description=f"Delete model '{modelId}' from local registry",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete model '{modelId}' from local registry",
                    timeout_seconds=security_config.confirmation_timeout_seconds,
                )
            model_service.delete_model(modelId)
            return {
                "_schema": "mc.model.delete.v1",
                "modelId": modelId,
                "status": "deleted",
                "nextActions": ["mc_model_list to verify"],
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
                raise ValueError(
                    "Invalid sort option. Use: downloads, likes, lastModified, trending."
                )

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
                    "memoryFitStatus": model.memory_fit_status.value
                    if model.memory_fit_status
                    else None,
                }
                for model in page.models
            ]
            next_actions = (
                ["Try a different search query"]
                if not models
                else [
                    "mc_model_fetch with model ID to download",
                    "mc_model_search with cursor for next page",
                ]
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
                    "mc_model_validate_merge to check merge compatibility",
                    "mc_model_analyze_alignment to analyze drift",
                ],
            }

    if "mc_model_validate_merge" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_model_validate_merge(source: str, target: str) -> dict:
            """Validate merge effort between two models."""
            source_path = _require_existing_directory(source)
            target_path = _require_existing_directory(target)
            result = model_probe_service.validate_merge(source_path, target_path)
            return {
                "_schema": "mc.model.validate_merge.v1",
                "lowEffort": result.low_effort,
                "architectureMatch": result.architecture_match,
                "vocabMatch": result.vocab_match,
                "dimensionMatch": result.dimension_match,
                "warnings": result.warnings,
                "nextActions": (
                    ["mc_model_merge to perform the merge"]
                    if result.low_effort
                    else ["Use layer mapping for cross-architecture merge"]
                ),
            }

    if "mc_model_merge" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_model_merge(
            source: str,
            target: str,
            output: str,
            idempotencyKey: str | None = None,
        ) -> dict:
            """Merge two models using pure geometric alignment.

            Pipeline: VOCAB → PROBE → PERMUTE → ROTATE → BLEND → PROPAGATE → VALIDATE

            The geometry determines everything - per-layer blend coefficients,
            alignment rotations, neuron permutations. No configuration needed.
            """
            if idempotencyKey:
                previous = _get_idempotency("model_merge", idempotencyKey)
                if previous:
                    return {
                        "_schema": "mc.model.merge.v1",
                        "status": "duplicate",
                        "message": "Merge already completed with this idempotency key",
                        "outputPath": previous,
                    }

            _require_existing_directory(source)
            _require_existing_directory(target)
            output_path = Path(output).expanduser().resolve()

            merger = UnifiedGeometricMerger(
                model_loader=registry.model_loader,
            )
            result = merger.merge(
                source_path=source,
                target_path=target,
                output_dir=str(output_path),
            )

            if idempotencyKey:
                _set_idempotency("model_merge", idempotencyKey, str(output_path))

            return {
                "_schema": "mc.model.merge.v1",
                "status": "completed",
                "outputPath": result.output_path,
                "layerCount": result.layer_count,
                "weightCount": result.weight_count,
                "meanConfidence": result.mean_confidence,
                "vocabAligned": result.vocab_aligned,
                "metrics": {
                    "meanProcrustesError": result.mean_procrustes_error,
                },
                "nextActions": [
                    f"mc_eval_run using model={output}",
                    f"mc_infer using model={output}",
                ],
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

    # Geometry tools moved to modelcypher/mcp/tools/geometry.py
    # Safety tools moved to modelcypher/mcp/tools/safety_entropy.py
    # Entropy tools moved to modelcypher/mcp/tools/safety_entropy.py

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

    if "mc_train_preflight" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_train_preflight(
            model: str,
            dataset: str,
            sequenceLength: int = 2048,
            loraRank: int | None = None,
            loraAlpha: float | None = None,
            batchSize: int | None = None,
        ) -> dict:
            """Check training feasibility."""
            # Reuse similar logic to mc_validate_train but expose as preflight for CLI parity
            dataset_path = _require_existing_path(dataset)

            lora = None
            if loraRank is not None and loraAlpha is not None:
                from modelcypher.core.domain.training import LoRAConfig

                lora = LoRAConfig(
                    rank=loraRank, alpha=loraAlpha, dropout=0.0, targets=["q_proj", "v_proj"]
                )

            config = TrainingConfig(
                model_id=model,
                dataset_path=dataset_path,
                learning_rate=1e-5,  # Dummy for preflight
                batch_size=batchSize or 1,
                epochs=1,
                sequence_length=sequenceLength,
                lora=lora,
            )

            result = training_service.preflight(config)
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
            # Just an alias wrapper for checkpoint export of final step implies logic not exposed in base service
            # For now, we will assume it exports the latest checkpoint
            # Getting latest checkpoint step
            status = training_service.status(jobId)
            current_step = status["currentStep"]

            output_path = Path(output).expanduser().resolve()
            checkpoint_service.export_checkpoint(
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

    if "mc_doc_convert" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_doc_convert(
            inputs: list[str],
            outputPath: str,
            chunkSize: int = 2000,
            chunkOverlap: int = 200,
            textOnly: bool = True,
        ) -> dict:
            """Convert documents into a dataset for training."""
            result, _ = doc_service.convert(
                inputs=inputs,
                output_path=outputPath,
                chunk_size=chunkSize,
                chunk_overlap=chunkOverlap,
                text_only=textOnly,
                stream=False,
                update_manifest=False,
            )
            message = (
                f"Processed {result.files_processed} files into {result.sample_count} samples."
            )
            return {
                "_schema": "mc.doc.convert.v1",
                "taskId": result.job_id,
                "status": "completed",
                "outputPath": outputPath,
                "message": message,
                "nextActions": [
                    "mc_train_start to begin training",
                ],
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
                "nextActions": [
                    f"mc_train_start with model={modelId}",
                    f"mc_infer with model={modelId}",
                ],
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

    if "mc_checkpoint_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_checkpoint_list(jobId: str) -> dict:
            """List checkpoints for a job."""
            checkpoints = checkpoint_service.list_checkpoints(jobId)
            return {
                "_schema": "mc.checkpoint.list.v1",
                "jobId": jobId,
                "checkpoints": [{"step": cp.step, "metrics": cp.metrics} for cp in checkpoints],
                "count": len(checkpoints),
            }

    if "mc_checkpoint_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_checkpoint_delete(
            jobId: str, step: int, confirmationToken: str | None = None
        ) -> dict:
            """Delete a specific checkpoint. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                confirmation_manager.require_confirmation(
                    operation="delete_checkpoint",
                    tool_name="mc_checkpoint_delete",
                    parameters={"jobId": jobId, "step": step},
                    description=f"Delete checkpoint at step {step} for job '{jobId}'",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete checkpoint at step {step} for job '{jobId}'",
                    timeout_seconds=security_config.confirmation_timeout_seconds,
                )
            checkpoint_service.delete_checkpoint(jobId, step)
            return {
                "_schema": "mc.checkpoint.delete.v1",
                "jobId": jobId,
                "step": step,
                "status": "deleted",
            }

    @mcp.resource("mc://models")
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

    @mcp.resource("mc://jobs")
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

    @mcp.resource("mc://checkpoints")
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

    @mcp.resource("mc://system")
    def resource_system() -> str:
        return dump_json(_system_status_payload())

    # Geometry primes/CRM/stitch tools moved to modelcypher/mcp/tools/geometry.py

    # --- MERGE VALIDATION TOOLS ---

    merge_validation_service = factory.merge_validation_service()

    if "mc_merge_validate" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_validate(
            merged: str,
            source: str | None = None,
            target: str | None = None,
            coherencePrompts: list[str] | None = None,
            taskProbes: list[dict] | None = None,
            geometricDiagnosis: bool = True,
        ) -> dict:
            """
            Run full merge validation suite on a merged model.

            Validates model behavior using:
            - Coherence scoring (if prompts provided)
            - Task probes (if probes provided)
            - Geometric diagnosis (if source/target provided and issues detected)

            Returns overall status: healthy, degraded, or failed.
            """
            merged_path = _require_existing_directory(merged)
            source_path = _require_existing_directory(source) if source else None
            target_path = _require_existing_directory(target) if target else None

            config = MergeValidationConfig(
                coherence_prompts=coherencePrompts,
                task_probes=taskProbes,
                geometric_diagnosis=geometricDiagnosis,
            )

            result = merge_validation_service.validate(
                merged_model=merged_path,
                source_model=source_path,
                target_model=target_path,
                config=config,
            )

            payload = result.to_dict()
            payload["_schema"] = "mc.merge.validate.v1"

            # Add contextual next actions
            next_actions = []
            if result.overall_status == "failed":
                next_actions.append("mc_merge_diagnose for detailed geometric analysis")
                next_actions.append("Re-merge with lower alpha or different parameters")
            elif result.overall_status == "degraded":
                next_actions.append("mc_merge_diagnose to identify problematic layers")
                next_actions.append("Consider layer-wise alpha adjustment")
            else:
                next_actions.append("mc_infer to test the merged model")
                next_actions.append("mc_merge_entropy_validate for stability checks")

            payload["nextActions"] = next_actions
            return payload

    if "mc_merge_coherence" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_coherence(
            model: str,
            prompts: list[str],
            maxTokens: int = 50,
        ) -> dict:
            """
            Score coherence of model responses to given prompts.

            Higher score = more coherent sentence continuations.
            Useful for detecting attention layer issues.
            """
            model_path = _require_existing_directory(model)

            if not prompts or len(prompts) == 0:
                raise ValueError("At least one prompt required")

            score = merge_validation_service.compute_coherence(model_path, prompts, maxTokens)

            return {
                "_schema": "mc.merge.coherence.v1",
                "model": model_path,
                "promptCount": len(prompts),
                "coherenceScore": score,
                "interpretation": (
                    "excellent"
                    if score > 0.8
                    else "good"
                    if score > 0.6
                    else "acceptable"
                    if score > 0.4
                    else "degraded"
                ),
                "nextActions": [
                    "mc_merge_probe for task-specific testing",
                    "mc_merge_validate for full validation suite",
                ],
            }

    if "mc_merge_probe" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_probe(
            model: str,
            probes: list[dict],
        ) -> dict:
            """
            Run task probes to test specific model capabilities.

            Each probe should have:
            - name: Human-readable name
            - prompt: The prompt to send
            - expected_pattern: Regex pattern expected in output

            Example probes:
            - Code generation: {"name": "python_hello", "prompt": "Write Python code to print hello", "expected_pattern": "print"}
            - Math: {"name": "addition", "prompt": "2+2=", "expected_pattern": "4"}
            """
            model_path = _require_existing_directory(model)

            if not probes or len(probes) == 0:
                raise ValueError("At least one probe required")

            results = merge_validation_service.run_task_probes(model_path, probes)

            passed = sum(1 for r in results if r.passed)
            pass_rate = passed / len(results) if results else 0.0

            return {
                "_schema": "mc.merge.probe.v1",
                "model": model_path,
                "probeCount": len(probes),
                "passedCount": passed,
                "passRate": pass_rate,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "output": r.output[:200] if r.output else None,
                        "matchDetails": r.match_details,
                    }
                    for r in results
                ],
                "interpretation": (
                    "all_passed"
                    if pass_rate == 1.0
                    else "mostly_passed"
                    if pass_rate >= 0.7
                    else "partial"
                    if pass_rate >= 0.5
                    else "mostly_failed"
                ),
                "nextActions": [
                    "mc_merge_diagnose for geometric analysis of failures",
                    "mc_merge_validate for full validation suite",
                ],
            }

    if "mc_merge_diagnose" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_diagnose(
            merged: str,
            source: str,
            target: str,
        ) -> dict:
            """
            Diagnose geometric issues in a merged model.

            Compares merged model against source to identify:
            - Layers with high drift (diverged significantly)
            - Recommended fixes (alpha adjustment, etc.)

            Use this when merge validation shows degradation.
            """
            merged_path = _require_existing_directory(merged)
            source_path = _require_existing_directory(source)
            target_path = _require_existing_directory(target)

            diagnosis = merge_validation_service.diagnose_geometry(
                merged_path, source_path, target_path
            )

            return {
                "_schema": "mc.merge.diagnose.v1",
                "mergedModel": merged_path,
                "sourceModel": source_path,
                "targetModel": target_path,
                "divergedLayers": diagnosis.diverged_layers,
                "highDriftLayers": diagnosis.high_drift_layers,
                "meanDrift": diagnosis.mean_drift,
                "maxDrift": diagnosis.max_drift,
                "recommendations": diagnosis.recommendations,
                "severity": (
                    "critical"
                    if len(diagnosis.high_drift_layers) > 5
                    else "high"
                    if len(diagnosis.high_drift_layers) > 0
                    else "moderate"
                    if len(diagnosis.diverged_layers) > 5
                    else "low"
                    if len(diagnosis.diverged_layers) > 0
                    else "minimal"
                ),
                "nextActions": [
                    "Re-merge with layer-wise alpha using divergedLayers",
                    "mc_geometry_refinement_analyze for detailed layer analysis",
                    "mc_model_merge with lower alpha for problematic layers",
                ],
            }

    # Calibration tools
    if "mc_calibration_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_calibration_run(
            model: str,
            dataset: str,
            batchSize: int = 4,
            maxSamples: int | None = None,
            method: str = "minmax",
        ) -> dict:
            """Execute calibration on a model with a dataset."""
            from modelcypher.core.use_cases.calibration_service import (
                CalibrationConfig,
                CalibrationService,
            )

            model_path = _require_existing_directory(model)
            dataset_path = _require_existing_path(dataset)
            config = CalibrationConfig(
                batch_size=batchSize,
                max_samples=maxSamples,
                calibration_method=method,
            )
            service = CalibrationService()
            result = service.run(model_path, dataset_path, config)
            return {
                "_schema": "mc.calibration.run.v1",
                "calibrationId": result.calibration_id,
                "modelPath": result.model_path,
                "datasetPath": result.dataset_path,
                "status": result.status,
                "startedAt": result.started_at,
                "config": result.config,
                "metrics": result.metrics,
                "nextActions": [
                    f"mc_calibration_status with calibrationId={result.calibration_id}",
                    "mc_calibration_apply to apply calibration",
                ],
            }

    if "mc_calibration_status" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_calibration_status(calibrationId: str) -> dict:
            """Get status of a calibration operation."""
            from modelcypher.core.use_cases.calibration_service import CalibrationService

            service = CalibrationService()
            result = service.status(calibrationId)
            return {
                "_schema": "mc.calibration.status.v1",
                "calibrationId": result.calibration_id,
                "status": result.status,
                "progress": result.progress,
                "currentStep": result.current_step,
                "totalSteps": result.total_steps,
                "metrics": result.metrics,
                "error": result.error,
                "nextActions": [
                    "mc_calibration_apply if status is completed",
                ],
            }

    if "mc_calibration_apply" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_calibration_apply(
            calibrationId: str,
            model: str,
            outputPath: str | None = None,
        ) -> dict:
            """Apply calibration results to a model."""
            from modelcypher.core.use_cases.calibration_service import CalibrationService

            model_path = _require_existing_directory(model)
            service = CalibrationService()
            result = service.apply(calibrationId, model_path, outputPath)
            return {
                "_schema": "mc.calibration.apply.v1",
                "calibrationId": result.calibration_id,
                "modelPath": result.model_path,
                "outputPath": result.output_path,
                "appliedAt": result.applied_at,
                "metrics": result.metrics,
                "nextActions": [
                    f"mc_infer with model={result.output_path}",
                ],
            }

    # Stability tools
    if "mc_stability_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_stability_run(
            model: str,
            numRuns: int = 10,
            promptVariations: int = 5,
            seed: int | None = None,
        ) -> dict:
            """Execute stability suite on a model."""
            from modelcypher.core.use_cases.stability_service import (
                StabilityConfig,
                StabilityService,
            )

            model_path = _require_existing_directory(model)
            config = StabilityConfig(
                num_runs=numRuns,
                prompt_variations=promptVariations,
                seed=seed,
            )
            service = StabilityService()
            result = service.run(model_path, config)
            return {
                "_schema": "mc.stability.run.v1",
                "suiteId": result.suite_id,
                "modelPath": result.model_path,
                "status": result.status,
                "startedAt": result.started_at,
                "config": result.config,
                "summary": result.summary,
                "nextActions": [
                    f"mc_stability_report with suiteId={result.suite_id}",
                ],
            }

    if "mc_stability_report" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_stability_report(suiteId: str) -> dict:
            """Get detailed stability report."""
            from modelcypher.core.use_cases.stability_service import StabilityService

            service = StabilityService()
            result = service.report(suiteId)
            return {
                "_schema": "mc.stability.report.v1",
                "suiteId": result.suite_id,
                "modelPath": result.model_path,
                "status": result.status,
                "startedAt": result.started_at,
                "completedAt": result.completed_at,
                "config": result.config,
                "metrics": result.metrics,
                "perPromptResults": result.per_prompt_results,
                "interpretation": result.interpretation,
                "recommendations": result.recommendations,
                "nextActions": [
                    "mc_stability_run to run another suite",
                ],
            }

    # Agent eval tools
    if "mc_agent_eval_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_agent_eval_run(
            model: str,
            evalSuite: str = "default",
            maxTurns: int = 10,
            timeout: int = 300,
            seed: int | None = None,
        ) -> dict:
            """Execute agent evaluation."""
            from modelcypher.core.use_cases.agent_eval_service import (
                AgentEvalConfig,
                AgentEvalService,
            )

            model_path = _require_existing_directory(model)
            config = AgentEvalConfig(
                model_path=model_path,
                eval_suite=evalSuite,
                max_turns=maxTurns,
                timeout_seconds=timeout,
                seed=seed,
            )
            service = AgentEvalService()
            result = service.run(config)
            return {
                "_schema": "mc.agent_eval.run.v1",
                "evalId": result.eval_id,
                "modelPath": result.model_path,
                "evalSuite": result.eval_suite,
                "status": result.status,
                "startedAt": result.started_at,
                "config": result.config,
                "summary": result.summary,
                "nextActions": [
                    f"mc_agent_eval_results with evalId={result.eval_id}",
                ],
            }

    if "mc_agent_eval_results" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_agent_eval_results(evalId: str) -> dict:
            """Get agent evaluation results."""
            from modelcypher.core.use_cases.agent_eval_service import AgentEvalService

            service = AgentEvalService()
            result = service.results(evalId)
            return {
                "_schema": "mc.agent_eval.results.v1",
                "evalId": result.eval_id,
                "modelPath": result.model_path,
                "evalSuite": result.eval_suite,
                "status": result.status,
                "startedAt": result.started_at,
                "completedAt": result.completed_at,
                "config": result.config,
                "metrics": result.metrics,
                "taskResults": result.task_results,
                "interpretation": result.interpretation,
                "overallScore": result.overall_score,
                "nextActions": [
                    "mc_agent_eval_run to run another evaluation",
                ],
            }

    # Dashboard tools
    if "mc_dashboard_metrics" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dashboard_metrics() -> dict:
            """Return current metrics in Prometheus format."""
            from modelcypher.core.use_cases.dashboard_service import DashboardService

            service = DashboardService()
            metrics = service.metrics()
            # Parse prometheus format to dict
            lines = metrics.strip().split("\n")
            metric_dict = {}
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split(" ")
                if len(parts) >= 2:
                    metric_dict[parts[0]] = parts[1]
            return {
                "_schema": "mc.dashboard.metrics.v1",
                "metrics": metric_dict,
                "format": "prometheus",
                "nextActions": [
                    "mc_dashboard_export to export in different formats",
                ],
            }

    if "mc_dashboard_export" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_dashboard_export(format: str = "prometheus", outputPath: str | None = None) -> dict:
            """Export dashboard data."""
            from modelcypher.core.use_cases.dashboard_service import DashboardService

            service = DashboardService()
            result = service.export(format, outputPath)
            return {
                "_schema": "mc.dashboard.export.v1",
                "format": result.format,
                "exportPath": result.export_path,
                "exportedAt": result.exported_at,
                "metricsCount": result.metrics_count,
                "nextActions": [
                    "mc_dashboard_metrics for live metrics",
                ],
            }

    # Help tools
    if "mc_help_ask" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_help_ask(question: str) -> dict:
            """Get contextual help for a question."""
            from modelcypher.core.use_cases.help_service import HelpService

            service = HelpService()
            result = service.ask(question)
            return {
                "_schema": "mc.help.ask.v1",
                "question": result.question,
                "answer": result.answer,
                "relatedCommands": result.related_commands,
                "examples": result.examples,
                "docsUrl": result.docs_url,
                "nextActions": result.related_commands[:3],
            }

    if "mc_schema" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_schema(command: str) -> dict:
            """Return JSON schema for command output."""
            from modelcypher.core.use_cases.help_service import HelpService

            service = HelpService()
            schema = service.schema(command)
            return {
                "_schema": "mc.schema.v1",
                "command": command,
                "outputSchema": schema,
                "nextActions": [
                    "mc_help_ask for more help",
                ],
            }

    # Inference suite tools
    if "mc_infer_run" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_run(
            model: str,
            prompt: str,
            adapter: str | None = None,
            securityScan: bool = False,
            maxTokens: int = 512,
            temperature: float = 0.7,
            topP: float = 0.95,
        ) -> dict:
            """Execute inference with optional adapter and security scanning."""
            model_path = _require_existing_directory(model)
            adapter_path = _require_existing_directory(adapter) if adapter else None

            result = inference_engine.run(
                model=model_path,
                prompt=prompt,
                adapter=adapter_path,
                security_scan=securityScan,
                max_tokens=maxTokens,
                temperature=temperature,
                top_p=topP,
            )

            payload = {
                "_schema": "mc.infer.run.v1",
                "model": result.model,
                "prompt": result.prompt,
                "response": result.response,
                "tokenCount": result.token_count,
                "tokensPerSecond": result.tokens_per_second,
                "timeToFirstToken": result.time_to_first_token,
                "totalDuration": result.total_duration,
                "stopReason": result.stop_reason,
                "adapter": result.adapter,
                "nextActions": [
                    "mc_infer_run for more prompts",
                    "mc_infer_suite for batch testing",
                ],
            }

            if result.security:
                payload["security"] = {
                    "hasSecurityFlags": result.security.has_security_flags,
                    "anomalyCount": result.security.anomaly_count,
                    "maxAnomalyScore": result.security.max_anomaly_score,
                    "avgDelta": result.security.avg_delta,
                    "disagreementRate": result.security.disagreement_rate,
                    "circuitBreakerTripped": result.security.circuit_breaker_tripped,
                    "circuitBreakerTripIndex": result.security.circuit_breaker_trip_index,
                }

            return payload

    if "mc_infer_batch" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_batch(
            model: str,
            promptsFile: str,
            maxTokens: int = 512,
            temperature: float = 0.7,
            topP: float = 0.95,
        ) -> dict:
            """Execute batched inference from a prompts file."""
            model_path = _require_existing_directory(model)
            prompts_path = _require_existing_path(promptsFile)
            result = inference_engine.run_batch(
                model_path, prompts_path, maxTokens, temperature, topP
            )
            return {
                "_schema": "mc.infer.batch.v1",
                "modelId": result.model_id,
                "promptsFile": result.prompts_file,
                "totalPrompts": result.total_prompts,
                "successful": result.successful,
                "failed": result.failed,
                "totalTokens": result.total_tokens,
                "totalDuration": result.total_duration,
                "averageTokensPerSecond": result.average_tokens_per_second,
                "results": result.results[:10],
                "nextActions": [
                    "mc_infer_suite for structured testing",
                    "mc_infer for single prompts",
                ],
            }

    if "mc_infer_suite" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_infer_suite(
            model: str,
            suiteFile: str,
            adapter: str | None = None,
            securityScan: bool = False,
            maxTokens: int = 512,
            temperature: float = 0.7,
        ) -> dict:
            """Execute batched inference over a suite of prompts."""
            model_path = _require_existing_directory(model)
            suite_path = _require_existing_path(suiteFile)
            adapter_path = _require_existing_directory(adapter) if adapter else None

            result = inference_engine.suite(
                model=model_path,
                suite_file=suite_path,
                adapter=adapter_path,
                security_scan=securityScan,
                max_tokens=maxTokens,
                temperature=temperature,
            )

            # Convert cases to dict format
            cases_payload = []
            for case in result.cases:
                case_dict = {
                    "name": case.name,
                    "prompt": case.prompt,
                    "response": case.response,
                    "tokenCount": case.token_count,
                    "duration": case.duration,
                    "passed": case.passed,
                    "expected": case.expected,
                }
                if case.error:
                    case_dict["error"] = case.error
                cases_payload.append(case_dict)

            return {
                "_schema": "mc.infer.suite.v1",
                "model": result.model,
                "adapter": result.adapter,
                "suite": result.suite,
                "totalCases": result.total_cases,
                "passed": result.passed,
                "failed": result.failed,
                "totalDuration": result.total_duration,
                "summary": result.summary,
                "cases": cases_payload[:10],
                "nextActions": [
                    "mc_infer_batch for batch inference",
                    "mc_infer_run for single prompts",
                ],
            }

    # Thermo tools
    if "mc_thermo_analyze" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_analyze(jobId: str) -> dict:
            result = thermo_service.analyze(jobId)
            return {
                "_schema": "mc.thermo.analyze.v1",
                "jobId": result.job_id,
                "entropy": result.entropy,
                "temperature": result.temperature,
                "freeEnergy": result.free_energy,
                "interpretation": result.interpretation,
                "nextActions": [
                    "mc_thermo_entropy for entropy history",
                    "mc_thermo_path for checkpoint path analysis",
                ],
            }

    if "mc_thermo_path" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_path(checkpoints: list[str]) -> dict:
            resolved = [_require_existing_path(path) for path in checkpoints]
            result = thermo_service.path(resolved)
            return {
                "_schema": "mc.thermo.path.v1",
                "checkpoints": result.checkpoints,
                "pathLength": result.path_length,
                "curvature": result.curvature,
                "interpretation": result.interpretation,
                "nextActions": [
                    "mc_thermo_analyze for job-level metrics",
                    "mc_geometry_stitch_analyze for geometry stitching",
                ],
            }

    if "mc_thermo_path_integration" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_path_integration(
            prompt: str,
            model: str,
            gateThreshold: float = 0.55,
            maxTokens: int = 200,
            temperature: float = 0.0,
            captureTrajectory: bool = True,
        ) -> dict:
            model_path = _require_existing_directory(model)
            result = thermo_service.path_integration(
                prompt=prompt,
                model_path=model_path,
                gate_threshold=gateThreshold,
                max_tokens=maxTokens,
                temperature=temperature,
                capture_trajectory=captureTrajectory,
            )
            measurement = result.measurement
            assessment = measurement.assessment
            return {
                "_schema": "mc.thermo.path_integration.v1",
                "modelId": result.model_id,
                "prompt": result.prompt,
                "responseText": result.response_text,
                "meanEntropy": measurement.mean_entropy,
                "entropyVariance": measurement.entropy_variance,
                "firstTokenEntropy": measurement.first_token_entropy,
                "entropyTrajectory": measurement.entropy_trajectory,
                "gateSequence": measurement.gate_sequence,
                "gateCount": measurement.gate_count,
                "gateDetails": [
                    {
                        "gateId": gate.gate_id,
                        "gateName": gate.gate_name,
                        "localEntropy": gate.local_entropy,
                        "confidence": gate.confidence,
                    }
                    for gate in measurement.gate_details
                ],
                "entropyPathCorrelation": measurement.entropy_path_correlation,
                "gateTransitionEntropies": [
                    {
                        "fromGate": item.from_gate,
                        "toGate": item.to_gate,
                        "entropyDelta": item.entropy_delta,
                        "isSpike": item.is_spike,
                    }
                    for item in measurement.gate_transition_entropies
                ],
                "assessment": {
                    "h3Supported": assessment.h3_supported,
                    "correlation": assessment.correlation,
                    "spikeRate": assessment.spike_rate,
                    "measurementCount": assessment.measurement_count,
                    "strength": assessment.strength_for_thresholds(),
                    "rationale": assessment.rationale,
                },
                "nextActions": [
                    "mc_geometry_path_compare to compare gate trajectories",
                    "mc_thermo_measure for modifier sweeps",
                ],
            }

    if "mc_thermo_entropy" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_entropy(jobId: str) -> dict:
            result = thermo_service.entropy(jobId)
            return {
                "_schema": "mc.thermo.entropy.v1",
                "jobId": result.job_id,
                "entropyHistory": result.entropy_history,
                "finalEntropy": result.final_entropy,
                "entropyTrend": result.entropy_trend,
                "nextActions": [
                    "mc_thermo_analyze for thermodynamic summary",
                    "mc_geometry_training_status for live metrics",
                ],
            }

    if "mc_thermo_measure" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_measure(
            prompt: str,
            model: str,
            modifiers: list[str] | None = None,
        ) -> dict:
            """Measure entropy across linguistic modifiers for a prompt."""
            model_path = _require_existing_directory(model)
            result = thermo_service.measure(prompt, model_path, modifiers)

            return {
                "_schema": "mc.thermo.measure.v1",
                "basePrompt": result.base_prompt,
                "measurements": [
                    {
                        "modifier": m.modifier,
                        "meanEntropy": m.mean_entropy,
                        "deltaH": m.delta_h,
                        "ridgeCrossed": m.ridge_crossed,
                        "behavioralOutcome": m.behavioral_outcome,
                    }
                    for m in result.measurements
                ],
                "statistics": {
                    "meanEntropy": result.statistics.mean_entropy,
                    "stdEntropy": result.statistics.std_entropy,
                    "minEntropy": result.statistics.min_entropy,
                    "maxEntropy": result.statistics.max_entropy,
                    "meanDeltaH": result.statistics.mean_delta_h,
                    "intensityCorrelation": result.statistics.intensity_correlation,
                },
                "timestamp": result.timestamp.isoformat(),
                "nextActions": [
                    "mc_thermo_detect for unsafe prompt detection",
                    "mc_thermo_detect_batch for batch analysis",
                ],
            }

    if "mc_thermo_detect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_detect(
            prompt: str,
            model: str,
            preset: str = "default",
        ) -> dict:
            """Detect unsafe prompt patterns via entropy differential."""
            model_path = _require_existing_directory(model)
            result = thermo_service.detect(prompt, model_path, preset)

            return {
                "_schema": "mc.thermo.detect.v1",
                "prompt": result.prompt,
                "classification": result.classification,
                "riskLevel": result.risk_level,
                "confidence": result.confidence,
                "baselineEntropy": result.baseline_entropy,
                "intensityEntropy": result.intensity_entropy,
                "deltaH": result.delta_h,
                "processingTime": result.processing_time,
                "nextActions": [
                    "mc_thermo_measure for detailed entropy analysis",
                    "mc_thermo_detect_batch for batch detection",
                    "mc_safety_circuit_breaker for safety assessment",
                ],
            }

    if "mc_thermo_detect_batch" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_thermo_detect_batch(
            promptsFile: str,
            model: str,
            preset: str = "default",
        ) -> dict:
            """Batch detect unsafe patterns across multiple prompts."""
            model_path = _require_existing_directory(model)
            prompts_path = _require_existing_path(promptsFile)
            results = thermo_service.detect_batch(prompts_path, model_path, preset)

            return {
                "_schema": "mc.thermo.detect_batch.v1",
                "promptsFile": promptsFile,
                "totalPrompts": len(results),
                "results": [
                    {
                        "prompt": r.prompt,
                        "classification": r.classification,
                        "riskLevel": r.risk_level,
                        "confidence": r.confidence,
                        "deltaH": r.delta_h,
                    }
                    for r in results
                ],
                "summary": {
                    "safe": sum(1 for r in results if r.classification == "safe"),
                    "unsafe": sum(1 for r in results if r.classification == "unsafe"),
                    "ambiguous": sum(1 for r in results if r.classification == "ambiguous"),
                },
                "nextActions": [
                    "mc_thermo_detect for individual prompt analysis",
                    "mc_thermo_measure for detailed entropy analysis",
                ],
            }

    # Storage tools
    if "mc_storage_usage" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_storage_usage() -> dict:
            """Return storage usage breakdown by category."""
            service = factory.storage_service()
            snapshot = service.compute_snapshot()
            usage = snapshot.usage
            disk = snapshot.disk

            return {
                "_schema": "mc.storage.usage.v1",
                "totalGb": usage.total_gb,
                "modelsGb": usage.models_gb,
                "checkpointsGb": usage.checkpoints_gb,
                "otherGb": usage.other_gb,
                "disk": {
                    "totalBytes": disk.total_bytes,
                    "freeBytes": disk.free_bytes,
                },
                "nextActions": [
                    "mc_storage_cleanup to free space",
                    "mc_inventory to see all resources",
                ],
            }

    if "mc_storage_cleanup" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_storage_cleanup(
            targets: list[str],
            dryRun: bool = False,
            confirmationToken: str | None = None,
        ) -> dict:
            """Remove old artifacts and return freed space. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            service = factory.storage_service()

            if dryRun:
                return {
                    "_schema": "mc.storage.cleanup.v1",
                    "dryRun": True,
                    "targets": targets,
                    "freedBytes": 0,
                    "freedGb": 0.0,
                    "categoriesCleaned": [],
                    "message": "Dry run - no files deleted",
                    "nextActions": [
                        "mc_storage_cleanup with dryRun=false to execute",
                        "mc_storage_usage to check current usage",
                    ],
                }

            # Require confirmation for actual cleanup (not dry run)
            try:
                confirmation_manager.require_confirmation(
                    operation="storage_cleanup",
                    tool_name="mc_storage_cleanup",
                    parameters={"targets": targets, "dryRun": dryRun},
                    description=f"Clean up storage artifacts: {', '.join(targets)}",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Clean up storage artifacts: {', '.join(targets)}",
                    timeout_seconds=security_config.confirmation_timeout_seconds,
                )

            # Get before snapshot for comparison
            before_snapshot = service.compute_snapshot()

            cleared = service.cleanup(targets)

            # Get after snapshot
            after_snapshot = service.compute_snapshot()
            freed_bytes = max(0, after_snapshot.disk.free_bytes - before_snapshot.disk.free_bytes)

            return {
                "_schema": "mc.storage.cleanup.v1",
                "dryRun": False,
                "targets": targets,
                "freedBytes": freed_bytes,
                "freedGb": freed_bytes / (1024**3),
                "categoriesCleaned": cleared,
                "message": None,
                "nextActions": [
                    "mc_storage_usage to verify cleanup",
                    "mc_inventory to see remaining resources",
                ],
            }

    # Ensemble tools
    if "mc_ensemble_create" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_ensemble_create(
            models: list[str],
            strategy: str = "weighted",
            weights: list[float] | None = None,
        ) -> dict:
            """Create an ensemble configuration from multiple models."""
            # Validate model paths
            validated_models = [_require_existing_directory(m) for m in models]

            result = ensemble_service.create(
                model_paths=validated_models,
                strategy=strategy,
                weights=weights,
            )

            return {
                "_schema": "mc.ensemble.create.v1",
                "ensembleId": result.ensemble_id,
                "models": result.models,
                "routingStrategy": result.routing_strategy,
                "weights": result.weights,
                "createdAt": result.created_at,
                "configPath": result.config_path,
                "nextActions": [
                    f"mc_ensemble_run with ensembleId={result.ensemble_id}",
                    "mc_ensemble_create to create another ensemble",
                ],
            }

    if "mc_ensemble_run" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_ensemble_run(
            ensembleId: str,
            prompt: str,
            maxTokens: int = 512,
            temperature: float = 0.7,
        ) -> dict:
            """Execute ensemble inference."""
            result = ensemble_service.run(
                ensemble_id=ensembleId,
                prompt=prompt,
                max_tokens=maxTokens,
                temperature=temperature,
            )

            return {
                "_schema": "mc.ensemble.run.v1",
                "ensembleId": result.ensemble_id,
                "prompt": result.prompt[:100] if len(result.prompt) > 100 else result.prompt,
                "response": result.response,
                "modelContributions": result.model_contributions,
                "totalDuration": result.total_duration,
                "strategy": result.strategy,
                "modelsUsed": result.models_used,
                "aggregationMethod": result.aggregation_method,
                "nextActions": [
                    "mc_ensemble_run with different prompt",
                    "mc_ensemble_create to create new ensemble",
                ],
            }

    if "mc_ensemble_list" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_ensemble_list(limit: int = 50) -> dict:
            ensembles = ensemble_service.list_ensembles(limit=limit)
            return {
                "_schema": "mc.ensemble.list.v1",
                "ensembles": [
                    {
                        "ensembleId": ensemble.ensemble_id,
                        "models": len(ensemble.models),
                        "strategy": ensemble.routing_strategy,
                        "createdAt": ensemble.created_at,
                    }
                    for ensemble in ensembles
                ],
                "count": len(ensembles),
                "nextActions": [
                    "mc_ensemble_run to execute inference",
                    "mc_ensemble_delete to remove an ensemble",
                ],
            }

    if "mc_ensemble_delete" in tool_set:

        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_ensemble_delete(ensembleId: str, confirmationToken: str | None = None) -> dict:
            """Delete an ensemble configuration. Requires confirmation if MC_MCP_REQUIRE_CONFIRMATION=1."""
            try:
                confirmation_manager.require_confirmation(
                    operation="delete_ensemble",
                    tool_name="mc_ensemble_delete",
                    parameters={"ensembleId": ensembleId},
                    description=f"Delete ensemble configuration '{ensembleId}'",
                    confirmation_token=confirmationToken,
                )
            except ConfirmationError as e:
                return create_confirmation_response(
                    e,
                    description=f"Delete ensemble configuration '{ensembleId}'",
                    timeout_seconds=security_config.confirmation_timeout_seconds,
                )
            deleted = ensemble_service.delete(ensembleId)
            if not deleted:
                return {
                    "_schema": "mc.ensemble.delete.v1",
                    "deleted": None,
                    "message": f"Ensemble not found: {ensembleId}",
                    "nextActions": ["mc_ensemble_list to view ensembles"],
                }
            return {
                "_schema": "mc.ensemble.delete.v1",
                "deleted": ensembleId,
                "nextActions": ["mc_ensemble_list to verify deletion"],
            }

    # Research tools
    if "mc_research_sparse_region" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_research_sparse_region(
            modelPath: str,
        ) -> dict:
            """Analyze sparse activation regions in a model."""
            from modelcypher.core.use_cases.research_service import ResearchService

            model_path = _require_existing_directory(modelPath)
            service = ResearchService()
            result = service.sparse_region(model_path)

            return {
                "_schema": "mc.research.sparse_region.v1",
                "modelPath": result.model_path,
                "totalSparsity": result.total_sparsity,
                "layerCount": result.layer_count,
                "regions": [
                    {
                        "layerName": r.layer_name,
                        "startIndex": r.start_index,
                        "endIndex": r.end_index,
                        "sparsityRatio": r.sparsity_ratio,
                        "activationPattern": r.activation_pattern,
                    }
                    for r in result.regions[:20]  # Limit to first 20 for response size
                ],
                "interpretation": result.interpretation,
                "nextActions": [
                    "mc_research_afm for activation function mapping",
                    "mc_model_probe for architecture details",
                ],
            }

    if "mc_research_afm" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_research_afm(
            modelPath: str,
        ) -> dict:
            """Run activation function mapping analysis."""
            from modelcypher.core.use_cases.research_service import ResearchService

            model_path = _require_existing_directory(modelPath)
            service = ResearchService()
            result = service.afm(model_path)

            return {
                "_schema": "mc.research.afm.v1",
                "modelPath": result.model_path,
                "dominantPatterns": result.dominant_patterns,
                "layerSummaries": [
                    {
                        "layerName": s.layer_name,
                        "dominantPattern": s.dominant_pattern,
                        "meanActivation": s.mean_activation,
                        "maxActivation": s.max_activation,
                    }
                    for s in result.layer_summaries[:20]  # Limit to first 20 for response size
                ],
                "interpretation": result.interpretation,
                "nextActions": [
                    "mc_research_sparse_region for sparsity analysis",
                    "mc_model_probe for architecture details",
                ],
            }

    if "mc_adapter_inspect" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_adapter_inspect(adapterPath: str) -> dict:
            """Inspect a LoRA adapter configuration and weights."""
            adapter_path = _require_existing_directory(adapterPath)
            result = adapter_service.inspect(adapter_path)
            return {
                "_schema": "mc.adapter.inspect.v1",
                "rank": result.rank,
                "alpha": result.alpha,
                "targetModules": result.target_modules,
                "sparsity": result.sparsity,
                "parameterCount": result.parameter_count,
                "layerAnalysis": [
                    {
                        "name": layer.name,
                        "rank": layer.rank,
                        "alpha": layer.alpha,
                        "parameters": layer.parameters,
                    }
                    for layer in result.layer_analysis
                ],
                "nextActions": [
                    "mc_adapter_merge to merge adapters",
                    "mc_geometry_dare_sparsity to analyze sparsity",
                ],
            }

    if "mc_adapter_merge" in tool_set:

        @mcp.tool(annotations=MUTATING_ANNOTATIONS)
        def mc_adapter_merge(
            adapterPaths: list[str],
            outputDir: str,
            recommendEnsemble: bool = False,
        ) -> dict:
            """Merge multiple LoRA adapters using geometric alignment.

            Uses Procrustes rotation and permutation re-basin for mathematically
            correct manifold alignment. No heuristic options - one correct way.
            """
            # Validate adapter paths exist
            resolved_paths = []
            for adapter_path in adapterPaths:
                resolved_paths.append(_require_existing_directory(adapter_path))

            result = adapter_service.merge(
                adapter_paths=resolved_paths,
                output_dir=outputDir,
                recommend_ensemble=recommendEnsemble,
            )

            return {
                "_schema": "mc.adapter.merge.v2",
                "outputPath": result.output_path,
                "mergedModules": result.merged_modules,
                "procrustesError": result.procrustes_error,
                "permutationQuality": result.permutation_quality,
                "mergeConfidence": result.merge_confidence,
                "ensembleRecommendation": result.ensemble_recommendation,
                "nextActions": [
                    f"mc_infer with adapter={result.output_path} to test merged adapter",
                    "mc_adapter_merge to merge with additional adapters",
                ],
            }

    # Register modular tools (extracted from this file for maintainability)
    from modelcypher.mcp.tools.agent import register_agent_tools
    from modelcypher.mcp.tools.common import ServiceContext
    from modelcypher.mcp.tools.geometry import (
        register_geometry_baseline_tools,
        register_geometry_crm_tools,
        register_geometry_interference_tools,
        register_geometry_invariant_tools,
        register_geometry_primes_tools,
        register_geometry_safety_tools,
        register_geometry_spatial_tools,
        register_geometry_stitch_tools,
        register_geometry_tools,
    )
    from modelcypher.mcp.tools.merge_entropy import register_merge_entropy_tools
    from modelcypher.mcp.tools.safety_entropy import register_entropy_tools, register_safety_tools
    from modelcypher.mcp.tools.tasks import register_task_tools

    service_context = ServiceContext(
        mcp=mcp,
        tool_set=tool_set,
        security_config=security_config,
        confirmation_manager=confirmation_manager,
        registry=registry,
        factory=factory,
    )
    register_safety_tools(service_context)
    register_entropy_tools(service_context)
    register_agent_tools(service_context)
    register_geometry_tools(service_context)
    register_geometry_invariant_tools(service_context)
    register_geometry_safety_tools(service_context)
    register_geometry_primes_tools(service_context)
    register_geometry_crm_tools(service_context)
    register_geometry_stitch_tools(service_context)
    register_geometry_spatial_tools(service_context)
    register_geometry_interference_tools(service_context)
    register_geometry_baseline_tools(service_context)
    register_merge_entropy_tools(service_context)
    register_task_tools(service_context)

    return mcp


def main() -> None:
    mcp = build_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
