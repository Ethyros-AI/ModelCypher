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

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.core.use_cases.checkpoint_service import CheckpointService
    from modelcypher.core.use_cases.compare_service import CompareService
    from modelcypher.core.use_cases.ensemble_service import EnsembleService
    from modelcypher.core.use_cases.evaluation_service import EvaluationService
    from modelcypher.core.use_cases.export_service import ExportService
    from modelcypher.core.use_cases.inventory_service import InventoryService
    from modelcypher.core.use_cases.job_service import JobService
    from modelcypher.core.use_cases.model_search_service import ModelSearchService
    from modelcypher.core.use_cases.unified_geometric_merge import UnifiedGeometricMerger
    from modelcypher.core.use_cases.model_service import ModelService
    from modelcypher.core.use_cases.storage_service import StorageService
    from modelcypher.core.use_cases.training_service import TrainingService
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory


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


def get_geometric_merger() -> "UnifiedGeometricMerger":
    """Get UnifiedGeometricMerger with proper dependency injection."""
    from modelcypher.core.use_cases.unified_geometric_merge import UnifiedGeometricMerger

    registry = _get_registry()
    return UnifiedGeometricMerger(
        model_loader=registry.model_loader,
    )


def get_storage_service() -> "StorageService":
    """Get StorageService with proper dependency injection."""
    registry = _get_registry()
    from modelcypher.core.use_cases.storage_service import StorageService

    return StorageService(
        model_store=registry.model_store,
        job_store=registry.job_store,
        dataset_store=registry.dataset_store,
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


def get_ensemble_service() -> "EnsembleService":
    """Get EnsembleService with proper dependency injection."""
    return _get_factory().ensemble_service()


def get_evaluation_service() -> "EvaluationService":
    """Get EvaluationService with proper dependency injection."""
    return _get_factory().evaluation_service()


def get_compare_service() -> "CompareService":
    """Get CompareService with proper dependency injection."""
    return _get_factory().compare_service()


def get_geometry_training_service():
    """Get GeometryTrainingService with proper dependency injection."""
    return _get_factory().geometry_training_service()


def get_geometry_safety_service(
    drift_samples: list[float] | None = None,
    safe_delta_h_samples: list[float] | None = None,
    attack_entropy_samples: list[float] | None = None,
):
    """Get GeometrySafetyService with calibration-derived config.

    Args:
        drift_samples: Historical persona drift magnitudes from baseline runs.
        safe_delta_h_samples: Delta-H values from safe prompt baseline.
        attack_entropy_samples: Attack entropy values from safe prompt baseline.
    """
    from modelcypher.core.use_cases.geometry_safety_service import (
        GeometrySafetyConfig,
        GeometrySafetyService,
    )

    if drift_samples is None and safe_delta_h_samples is None and attack_entropy_samples is None:
        config = GeometrySafetyConfig.default()
    elif drift_samples is None or safe_delta_h_samples is None or attack_entropy_samples is None:
        raise ValueError(
            "Provide all calibration samples or none for default calibration."
        )
    else:
        config = GeometrySafetyConfig.from_calibration_data(
            drift_samples=drift_samples,
            safe_delta_h_samples=safe_delta_h_samples,
            attack_entropy_samples=attack_entropy_samples,
        )
    return GeometrySafetyService(
        training_service=get_geometry_training_service(),
        config=config,
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


def get_system_service():
    """Get SystemService with proper dependency injection."""
    from modelcypher.core.use_cases.system_service import SystemService

    registry = _get_registry()
    return SystemService(
        model_store=registry.model_store,
        dataset_store=registry.dataset_store,
    )


# --- Utility Functions ---


def get_registry() -> "PortRegistry":
    """Get the PortRegistry for direct access to ports.

    Use this when you need direct access to a port adapter
    rather than a service.
    """
    return _get_registry()
