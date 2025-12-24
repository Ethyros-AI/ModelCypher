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
    from modelcypher.core.use_cases.model_merge_service import ModelMergeService
    from modelcypher.core.use_cases.model_service import ModelService
    from modelcypher.core.use_cases.model_search_service import ModelSearchService
    from modelcypher.core.use_cases.storage_service import StorageService
    from modelcypher.core.use_cases.inventory_service import InventoryService
    from modelcypher.core.use_cases.training_service import TrainingService
    from modelcypher.core.use_cases.job_service import JobService
    from modelcypher.core.use_cases.export_service import ExportService
    from modelcypher.core.use_cases.checkpoint_service import CheckpointService
    from modelcypher.core.use_cases.dataset_service import DatasetService
    from modelcypher.core.use_cases.ensemble_service import EnsembleService
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


def get_model_merge_service() -> "ModelMergeService":
    """Get ModelMergeService with proper dependency injection."""
    from modelcypher.core.use_cases.model_merge_service import ModelMergeService

    registry = _get_registry()
    return ModelMergeService(
        store=registry.model_store,
        model_loader=registry.model_loader,
    )


def get_storage_service() -> "StorageService":
    """Get StorageService with proper dependency injection."""
    return _get_factory().storage_service()


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


def get_dataset_service() -> "DatasetService":
    """Get DatasetService with proper dependency injection."""
    return _get_factory().dataset_service()


def get_ensemble_service() -> "EnsembleService":
    """Get EnsembleService with proper dependency injection."""
    return _get_factory().ensemble_service()


# --- Utility Functions ---


def get_registry() -> "PortRegistry":
    """Get the PortRegistry for direct access to ports.

    Use this when you need direct access to a port adapter
    rather than a service.
    """
    return _get_registry()
