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

"""Common utilities and types for MCP tools."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory
    from modelcypher.mcp.security import ConfirmationManager, SecurityConfig

# Tool annotations
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


def require_existing_path(path: str) -> str:
    """Resolve and validate that a path exists."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")
    return str(resolved)


def require_existing_directory(path: str) -> str:
    """Resolve and validate that a directory exists."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Directory does not exist: {resolved}")
    return str(resolved)


def parse_dataset_format(value: str):
    """Parse dataset format string to enum."""
    from modelcypher.core.domain.dataset_validation import DatasetContentFormat

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


def map_job_status(status: str) -> str:
    """Map internal job status to external representation."""
    if status == "pending":
        return "queued"
    if status == "cancelled":
        return "canceled"
    return status


def expand_rag_paths(paths: list[str]) -> list[str]:
    """Expand directory paths to individual files."""
    expanded: list[str] = []
    for raw_path in paths:
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        if resolved.is_dir():
            for candidate in resolved.rglob("*"):
                if candidate.is_file():
                    expanded.append(str(candidate))
        else:
            expanded.append(str(resolved))
    if not expanded:
        raise ValueError("No files found to index.")
    return expanded


def row_payload(row) -> dict:
    """Convert a dataset row to payload dict."""
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


IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


@dataclass
class IdempotencyEntry:
    """Cache entry for idempotent operations."""

    value: str
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at


@dataclass
class ServiceContext:
    """Container for all MCP services.

    Uses PortRegistry and ServiceFactory for proper dependency injection.
    Services that need ports are created via the factory; services without
    port dependencies are created directly.
    """

    mcp: "FastMCP"
    tool_set: set[str]
    security_config: "SecurityConfig"
    confirmation_manager: "ConfirmationManager"
    registry: "PortRegistry"
    factory: "ServiceFactory"
    idempotency_cache: dict[str, IdempotencyEntry] = field(default_factory=dict)

    # Lazy-loaded services (set to None initially)
    _inventory_service: object = None
    _training_service: object = None
    _job_service: object = None
    _model_service: object = None
    _model_search_service: object = None
    _model_probe_service: object = None
    _dataset_service: object = None
    _dataset_editor_service: object = None
    _system_service: object = None
    _settings_service: object = None
    _checkpoint_service: object = None
    _geometry_service: object = None
    _geometry_training_service: object = None
    _geometry_safety_service: object = None
    _geometry_adapter_service: object = None
    _geometry_crm_service: object = None
    _geometry_stitch_service: object = None
    _geometry_metrics_service: object = None
    _geometry_sparse_service: object = None
    _geometry_persona_service: object = None
    _geometry_transport_service: object = None
    _invariant_mapping_service: object = None
    _evaluation_service: object = None
    _thermo_service: object = None
    _ensemble_service: object = None
    _adapter_service: object = None
    _doc_service: object = None
    _safety_probe_service: object = None
    _entropy_probe_service: object = None

    @property
    def inference_engine(self):
        """Get inference engine from registry."""
        return self.registry.inference_engine

    @property
    def inventory_service(self):
        if self._inventory_service is None:
            self._inventory_service = self.factory.inventory_service()
        return self._inventory_service

    @property
    def training_service(self):
        if self._training_service is None:
            self._training_service = self.factory.training_service()
        return self._training_service

    @property
    def job_service(self):
        if self._job_service is None:
            self._job_service = self.factory.job_service()
        return self._job_service

    @property
    def model_service(self):
        if self._model_service is None:
            self._model_service = self.factory.model_service()
        return self._model_service

    @property
    def model_search_service(self):
        if self._model_search_service is None:
            self._model_search_service = self.factory.model_search_service()
        return self._model_search_service

    @property
    def model_probe_service(self):
        if self._model_probe_service is None:
            from modelcypher.core.use_cases.model_probe_service import ModelProbeService

            self._model_probe_service = ModelProbeService()
        return self._model_probe_service

    @property
    def dataset_service(self):
        if self._dataset_service is None:
            self._dataset_service = self.factory.dataset_service()
        return self._dataset_service

    @property
    def dataset_editor_service(self):
        if self._dataset_editor_service is None:
            from modelcypher.core.use_cases.dataset_editor_service import DatasetEditorService

            self._dataset_editor_service = DatasetEditorService(job_service=self.job_service)
        return self._dataset_editor_service

    @property
    def system_service(self):
        if self._system_service is None:
            self._system_service = self.factory.system_service()
        return self._system_service

    @property
    def settings_service(self):
        if self._settings_service is None:
            from modelcypher.core.use_cases.settings_service import SettingsService

            self._settings_service = SettingsService()
        return self._settings_service

    @property
    def checkpoint_service(self):
        if self._checkpoint_service is None:
            self._checkpoint_service = self.factory.checkpoint_service()
        return self._checkpoint_service

    @property
    def geometry_service(self):
        if self._geometry_service is None:
            from modelcypher.core.use_cases.geometry_service import GeometryService

            self._geometry_service = GeometryService()
        return self._geometry_service

    @property
    def geometry_training_service(self):
        if self._geometry_training_service is None:
            self._geometry_training_service = self.factory.geometry_training_service()
        return self._geometry_training_service

    @property
    def geometry_safety_service(self):
        if self._geometry_safety_service is None:
            raise ValueError(
                "GeometrySafetyService requires calibration data. "
                "Call set_safety_calibration() first with baseline measurements."
            )
        return self._geometry_safety_service

    def set_safety_calibration(
        self,
        drift_samples: list[float],
        safe_delta_h_samples: list[float],
        attack_entropy_samples: list[float],
    ) -> None:
        """Set calibration data for geometry safety service."""
        from modelcypher.core.use_cases.geometry_safety_service import (
            GeometrySafetyConfig,
            GeometrySafetyService,
        )

        config = GeometrySafetyConfig.from_calibration_data(
            drift_samples=drift_samples,
            safe_delta_h_samples=safe_delta_h_samples,
            attack_entropy_samples=attack_entropy_samples,
        )
        self._geometry_safety_service = GeometrySafetyService(
            self.geometry_training_service,
            config=config,
        )

    @property
    def geometry_adapter_service(self):
        if self._geometry_adapter_service is None:
            from modelcypher.adapters.mlx_model_loader import MLXModelLoader
            from modelcypher.core.use_cases.geometry_adapter_service import GeometryAdapterService

            model_loader = MLXModelLoader()
            self._geometry_adapter_service = GeometryAdapterService(model_loader=model_loader)
        return self._geometry_adapter_service

    @property
    def geometry_crm_service(self):
        if self._geometry_crm_service is None:
            from modelcypher.core.use_cases.concept_response_matrix_service import (
                ConceptResponseMatrixService,
            )

            self._geometry_crm_service = ConceptResponseMatrixService(engine=self.inference_engine)
        return self._geometry_crm_service

    @property
    def geometry_stitch_service(self):
        if self._geometry_stitch_service is None:
            from modelcypher.core.use_cases.geometry_stitch_service import GeometryStitchService

            self._geometry_stitch_service = GeometryStitchService()
        return self._geometry_stitch_service

    @property
    def geometry_metrics_service(self):
        if self._geometry_metrics_service is None:
            from modelcypher.core.use_cases.geometry_metrics_service import GeometryMetricsService

            self._geometry_metrics_service = GeometryMetricsService()
        return self._geometry_metrics_service

    @property
    def geometry_sparse_service(self):
        if self._geometry_sparse_service is None:
            from modelcypher.core.use_cases.geometry_sparse_service import GeometrySparseService

            self._geometry_sparse_service = GeometrySparseService()
        return self._geometry_sparse_service

    @property
    def geometry_persona_service(self):
        if self._geometry_persona_service is None:
            from modelcypher.core.use_cases.geometry_persona_service import GeometryPersonaService

            self._geometry_persona_service = GeometryPersonaService()
        return self._geometry_persona_service

    @property
    def geometry_transport_service(self):
        if self._geometry_transport_service is None:
            from modelcypher.core.use_cases.geometry_transport_service import (
                GeometryTransportService,
            )

            self._geometry_transport_service = GeometryTransportService()
        return self._geometry_transport_service

    @property
    def invariant_mapping_service(self):
        if self._invariant_mapping_service is None:
            from modelcypher.core.use_cases.invariant_layer_mapping_service import (
                InvariantLayerMappingService,
            )

            self._invariant_mapping_service = InvariantLayerMappingService()
        return self._invariant_mapping_service

    @property
    def evaluation_service(self):
        if self._evaluation_service is None:
            self._evaluation_service = self.factory.evaluation_service()
        return self._evaluation_service

    @property
    def thermo_service(self):
        if self._thermo_service is None:
            from modelcypher.core.use_cases.thermo_service import ThermoService

            self._thermo_service = ThermoService()
        return self._thermo_service

    @property
    def ensemble_service(self):
        if self._ensemble_service is None:
            self._ensemble_service = self.factory.ensemble_service()
        return self._ensemble_service

    @property
    def adapter_service(self):
        if self._adapter_service is None:
            from modelcypher.core.use_cases.adapter_service import AdapterService

            self._adapter_service = AdapterService()
        return self._adapter_service

    @property
    def doc_service(self):
        if self._doc_service is None:
            from modelcypher.core.use_cases.doc_service import DocService

            self._doc_service = DocService()
        return self._doc_service

    @property
    def safety_probe_service(self):
        if self._safety_probe_service is None:
            from modelcypher.core.use_cases.safety_probe_service import SafetyProbeService

            self._safety_probe_service = SafetyProbeService()
        return self._safety_probe_service

    @property
    def entropy_probe_service(self):
        if self._entropy_probe_service is None:
            from modelcypher.core.use_cases.entropy_probe_service import EntropyProbeService

            self._entropy_probe_service = EntropyProbeService()
        return self._entropy_probe_service

    def namespaced_key(self, operation: str, key: str) -> str:
        return f"{operation}:{key}"

    def get_idempotency(self, operation: str, key: str) -> str | None:
        entry = self.idempotency_cache.get(self.namespaced_key(operation, key))
        if entry is None:
            return None
        if entry.is_expired():
            self.idempotency_cache.pop(self.namespaced_key(operation, key), None)
            return None
        return entry.value

    def set_idempotency(self, operation: str, key: str, value: str) -> None:
        self.idempotency_cache[self.namespaced_key(operation, key)] = IdempotencyEntry(
            value=value,
            expires_at=time.time() + IDEMPOTENCY_TTL_SECONDS,
        )
        if len(self.idempotency_cache) % 100 == 0:
            expired = [
                cache_key
                for cache_key, entry in self.idempotency_cache.items()
                if entry.is_expired()
            ]
            for cache_key in expired:
                self.idempotency_cache.pop(cache_key, None)

    def system_status_payload(self) -> dict:
        """Generate system status payload."""
        readiness = self.system_service.readiness()
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
