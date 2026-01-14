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

"""CLI Composition Root.

This module provides service factory functions for CLI commands.
All services are created with proper dependency injection via PortRegistry.

Usage in CLI commands:
    from modelcypher.cli.composition import get_model_service

    service = get_model_service()
    models = service.list_models()
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.use_cases.bridge_service import BridgeService
    from modelcypher.core.use_cases.checkpoint_service import CheckpointService
    from modelcypher.core.use_cases.compare_service import CompareService
    from modelcypher.core.use_cases.entropy_calibration_service import (
        EntropyCalibrationService,
    )
    from modelcypher.core.use_cases.evaluation_service import EvaluationService
    from modelcypher.core.use_cases.export_service import ExportService
    from modelcypher.core.use_cases.invariant_layer_mapping_service import (
        InvariantLayerMappingService,
    )
    from modelcypher.core.use_cases.inventory_service import InventoryService
    from modelcypher.core.use_cases.job_service import JobService
    from modelcypher.core.use_cases.model_probe_service import ModelProbeService
    from modelcypher.core.use_cases.model_search_service import ModelSearchService
    from modelcypher.core.use_cases.model_service import ModelService
    from modelcypher.core.use_cases.storage_service import StorageService
    from modelcypher.core.use_cases.training_service import TrainingService
    from modelcypher.core.use_cases.merge import UnifiedGeometricMerger
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory
    from modelcypher.ports.inference import InferenceEngine
    from modelcypher.ports.model_loader import ModelLoaderPort


@lru_cache(maxsize=1)
def _get_registry() -> "PortRegistry":
    """Get the singleton PortRegistry instance."""
    from modelcypher.infrastructure.container import PortRegistry

    return PortRegistry.create_production()


@lru_cache(maxsize=1)
def _get_factory() -> "ServiceFactory":
    """Get the singleton ServiceFactory instance."""
    from modelcypher.infrastructure.service_factory import ServiceFactory

    return ServiceFactory(_get_registry())


# --- Service Factory Functions ---


def get_model_service() -> "ModelService":
    """Get ModelService with proper dependency injection."""
    return _get_factory().model_service()


def get_model_search_service() -> "ModelSearchService":
    """Get ModelSearchService with proper dependency injection."""
    return _get_factory().model_search_service()

def get_model_probe_service() -> "ModelProbeService":
    """Get ModelProbeService with proper dependency injection."""
    return _get_factory().model_probe_service()

def get_entropy_calibration_service() -> "EntropyCalibrationService":
    """Get EntropyCalibrationService with proper dependency injection."""
    return _get_factory().entropy_calibration_service()

def get_bridge_service() -> "BridgeService":
    """Get BridgeService with proper dependency injection."""
    return _get_factory().bridge_service()

def get_invariant_mapping_service() -> "InvariantLayerMappingService":
    """Get InvariantLayerMappingService with proper dependency injection."""
    return _get_factory().invariant_mapping_service()

def get_model_loader() -> "ModelLoaderPort":
    """Get ModelLoaderPort from the registry."""
    return _get_registry().model_loader

def get_inference_engine() -> "InferenceEngine":
    """Get InferenceEngine from the registry."""
    return _get_registry().inference_engine


def get_geometric_merger() -> "UnifiedGeometricMerger":
    """Get UnifiedGeometricMerger with proper dependency injection."""
    from modelcypher.core.use_cases.merge import UnifiedGeometricMerger

    registry = _get_registry()
    return UnifiedGeometricMerger(
        model_loader=registry.model_loader,
        activation_provider=registry.activation_provider,
    )


def get_storage_service() -> "StorageService":
    """Get StorageService with proper dependency injection."""
    registry = _get_registry()
    from modelcypher.core.use_cases.storage_service import StorageService

    return StorageService(
        model_store=registry.model_store,
        job_store=registry.job_store,
        base_dir=registry.base_dir,
        logs_dir=registry.logs_dir,
    )


def get_inventory_service() -> "InventoryService":
    """Get InventoryService with proper dependency injection."""
    return _get_factory().inventory_service()


def get_training_service() -> "TrainingService":
    """Get TrainingService with proper dependency injection."""
    return _get_factory().training_service()


def get_job_service() -> "JobService":
    """Get JobService with proper dependency injection."""
    return _get_factory().job_service()


def get_export_service() -> "ExportService":
    """Get ExportService with proper dependency injection."""
    return _get_factory().export_service()


def get_checkpoint_service() -> "CheckpointService":
    """Get CheckpointService with proper dependency injection."""
    return _get_factory().checkpoint_service()


def get_evaluation_service() -> "EvaluationService":
    """Get EvaluationService with proper dependency injection."""
    return _get_factory().evaluation_service()


def get_compare_service() -> "CompareService":
    """Get CompareService with proper dependency injection."""
    return _get_factory().compare_service()


def get_geometry_training_service():
    """Get GeometryTrainingService with proper dependency injection."""
    from modelcypher.adapters.filesystem_storage import FileSystemStore
    from modelcypher.core.use_cases.geometry_training_service import GeometryTrainingService

    store = FileSystemStore()
    return GeometryTrainingService(store)


def get_geometry_safety_service(
    drift_samples: list[float] | None = None,
    safe_delta_h_samples: list[float] | None = None,
    attack_entropy_samples: list[float] | None = None,
):
    """Get GeometrySafetyService with calibration-derived thresholds.

    Args:
        drift_samples: Historical persona drift magnitudes from baseline runs.
        safe_delta_h_samples: Delta-H values from safe prompt baseline.
        attack_entropy_samples: Attack entropy values from safe prompt baseline.
    """
    from modelcypher.core.use_cases.geometry_safety_service import (
        DriftThresholds,
        GeometrySafetyService,
        VulnerabilityThresholds,
    )

    if drift_samples is None or safe_delta_h_samples is None or attack_entropy_samples is None:
        raise ValueError(
            "Provide all calibration samples; geometry safety requires calibration-derived thresholds."
        )
    drift_thresholds = DriftThresholds.from_calibration_data(drift_samples)
    vulnerability_thresholds = VulnerabilityThresholds.from_calibration_data(
        safe_delta_h_samples,
        attack_entropy_samples,
    )
    return GeometrySafetyService(
        training_service=get_geometry_training_service(),
        drift_thresholds=drift_thresholds,
        vulnerability_thresholds=vulnerability_thresholds,
    )


def get_domain_geometry_waypoint_service():
    """Get DomainGeometryWaypointService with proper dependency injection."""
    from modelcypher.core.domain.geometry.domain_geometry_waypoints import (
        DomainGeometryWaypointService,
    )

    registry = _get_registry()
    return DomainGeometryWaypointService(
        backend=registry.backend,
        model_loader=registry.model_loader,
    )


def get_merge_pipeline_service():
    """Get MergePipelineService with proper dependency injection."""
    return _get_factory().merge_pipeline_service()


def get_system_service():
    """Get SystemService with proper dependency injection."""
    from modelcypher.core.use_cases.system_service import SystemService
    from modelcypher.utils.paths import get_modelcypher_home

    @dataclass(frozen=True)
    class _SystemPaths:
        base: Path

    @dataclass(frozen=True)
    class _SystemStore:
        paths: _SystemPaths

    store = _SystemStore(paths=_SystemPaths(base=get_modelcypher_home()))
    return SystemService(model_store=store)


# --- Utility Functions ---


def get_registry() -> "PortRegistry":
    """Get the PortRegistry for direct access to ports.

    Use this when you need direct access to a port adapter
    rather than a service.
    """
    return _get_registry()
