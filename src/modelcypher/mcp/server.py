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

import os
import time
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from modelcypher.adapters.embedding_defaults import EmbeddingDefaults
from modelcypher.core.use_cases.concept_response_matrix_service import (
    ConceptResponseMatrixService,
)
from modelcypher.core.use_cases.geometry_service import GeometryService
from modelcypher.core.use_cases.geometry_stitch_service import GeometryStitchService
from modelcypher.core.use_cases.merge_validation_service import (
    MergeValidationConfig,
)
from modelcypher.core.use_cases.model_probe_service import ModelProbeService
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
        "mc_program_run",  # Multi-donor transplant
        "mc_program_status",
        "mc_program_list",
        "mc_program_show",
        "mc_program_generate",  # Auto-generate from density profiles
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
        "mc_geometry_spectral_signature",  # New
        "mc_geometry_dimension_constraint_invariance",  # New
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
        # Model Geometry Profiles
        "mc_geometry_baseline_list",  # List available profiles
        "mc_geometry_baseline_extract",  # Extract profile from model
        "mc_geometry_baseline_compare",  # Compare two models
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
        # Model Geometry Profiles (for merge validation)
        "mc_geometry_baseline_list",
        "mc_geometry_baseline_compare",
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
        # Model Geometry Profiles (model geometry monitoring)
        "mc_geometry_baseline_list",
        "mc_geometry_baseline_compare",
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
    # Bootstrap atlas inventories for concept detection and geometry tools
    from modelcypher.core.use_cases.atlas_bootstrap import register_default_atlas_inventories

    register_default_atlas_inventories()

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
    factory.geometry_training_service()
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

    # Training tools moved to modelcypher/mcp/tools/training.py
    # Evaluation tools moved to modelcypher/mcp/tools/evaluation.py
    # Job tools moved to modelcypher/mcp/tools/training.py
    # Inference tools moved to modelcypher/mcp/tools/inference.py

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
    # Training validation tools moved to modelcypher/mcp/tools/training.py

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
            return payload

    if "mc_merge_coherence" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_merge_coherence(
            model: str,
            prompts: list[str],
        ) -> dict:
            """
            Score coherence of model responses to given prompts.

            Higher score = more coherent sentence continuations.
            Useful for detecting attention layer issues.
            """
            model_path = _require_existing_directory(model)

            if not prompts or len(prompts) == 0:
                raise ValueError("At least one prompt required")

            score = merge_validation_service.compute_coherence(model_path, prompts)

            return {
                "_schema": "mc.merge.coherence.v1",
                "model": model_path,
                "promptCount": len(prompts),
                "coherenceScore": score,
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
                "overallScore": result.overall_score,
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
            }

    # Inference suite tools moved to modelcypher/mcp/tools/inference.py
    # Thermo tools moved to modelcypher/mcp/tools/thermo.py

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
            }

    if "mc_ensemble_run" in tool_set:

        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_ensemble_run(
            ensembleId: str,
            prompt: str,
        ) -> dict:
            """Execute ensemble inference."""
            result = ensemble_service.run(
                ensemble_id=ensembleId,
                prompt=prompt,
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
                }
            return {
                "_schema": "mc.ensemble.delete.v1",
                "deleted": ensembleId,
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
                "ensembleRouting": result.ensemble_recommendation,
            }

    # Register modular tools (extracted from this file for maintainability)
    from modelcypher.mcp.tools.agent import register_agent_tools
    from modelcypher.mcp.tools.common import ServiceContext
    from modelcypher.mcp.tools.evaluation import register_evaluation_tools
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
    from modelcypher.mcp.tools.inference import register_inference_tools
    from modelcypher.mcp.tools.merge_entropy import register_merge_entropy_tools
    from modelcypher.mcp.tools.model import register_model_tools
    from modelcypher.mcp.tools.program import register_program_tools
    from modelcypher.mcp.tools.safety_entropy import register_entropy_tools, register_safety_tools
    from modelcypher.mcp.tools.tasks import register_task_tools
    from modelcypher.mcp.tools.thermo import register_thermo_tools
    from modelcypher.mcp.tools.training import register_training_tools

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
    register_model_tools(service_context)
    register_training_tools(service_context)
    register_evaluation_tools(service_context)
    register_thermo_tools(service_context)
    register_inference_tools(service_context)
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
    register_program_tools(service_context)
    register_task_tools(service_context)

    return mcp


def main() -> None:
    mcp = build_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
