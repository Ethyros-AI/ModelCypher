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

"""ServiceFactory - Creates services with proper dependency injection.

This factory uses the PortRegistry to create services with their required
port dependencies. All services are created with REQUIRED parameters -
no optional defaults.

This ensures:
- Tests must explicitly wire mock dependencies
- Production code uses the registry's adapters
- No hidden coupling to concrete implementations
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.infrastructure.container import PortRegistry

class ServiceFactory:
    """Factory for creating services with proper dependency injection.

    All services are created with REQUIRED parameters from the PortRegistry.
    This ensures tests and production both explicitly wire dependencies.
    """

    def __init__(self, registry: "PortRegistry") -> None:
        self._registry = registry
        self._cache: dict[str, object] = {}

    # --- Storage Services ---

    def storage_service(self):
        """Create StorageService with injected stores and paths."""
        from modelcypher.core.use_cases.storage_service import StorageService

        return StorageService(
            model_store=self._registry.model_store,
            job_store=self._registry.job_store,
            dataset_store=self._registry.dataset_store,
            base_dir=self._registry.base_dir,
            logs_dir=self._registry.logs_dir,
        )

    def dataset_service(self):
        """Create DatasetService with injected DatasetStore."""
        from modelcypher.core.use_cases.dataset_service import DatasetService

        return DatasetService(store=self._registry.dataset_store)

    def job_service(self):
        """Create JobService with injected JobStore and logs_dir."""
        from modelcypher.core.use_cases.job_service import JobService

        return JobService(
            store=self._registry.job_store,
            logs_dir=self._registry.logs_dir,
        )

    def evaluation_service(self):
        """Create EvaluationService with injected EvaluationStore."""
        from modelcypher.core.use_cases.evaluation_service import EvaluationService

        return EvaluationService(store=self._registry.evaluation_store)

    def compare_service(self):
        """Create CompareService with injected CompareStore and JobStore."""
        from modelcypher.core.use_cases.compare_service import CompareService

        return CompareService(
            store=self._registry.compare_store,
            job_store=self._registry.job_store,
        )

    def manifold_profile_service(self):
        """Create ManifoldProfileService with injected ManifoldProfileStore."""
        from modelcypher.core.use_cases.manifold_profile_service import (
            ManifoldProfileService,
        )

        return ManifoldProfileService(store=self._registry.manifold_profile_store)

    def inventory_service(self):
        """Create InventoryService with injected store and system service."""
        from modelcypher.core.use_cases.inventory_service import InventoryService

        return InventoryService(
            store=self._registry.model_store,
            system=self.system_service(),
        )

    def system_service(self):
        """Create SystemService with injected stores."""
        from modelcypher.core.use_cases.system_service import SystemService

        return SystemService(
            model_store=self._registry.model_store,
            dataset_store=self._registry.dataset_store,
        )

    # --- Model Services ---

    def model_service(self):
        """Create ModelService with injected stores, hub, and model loader."""
        from modelcypher.core.use_cases.model_service import ModelService

        return ModelService(
            store=self._registry.model_store,
            hub=self._registry.hub_adapter,
            model_loader=self._registry.model_loader,
        )

    def model_search_service(self):
        """Create ModelSearchService with injected search adapter."""
        from modelcypher.core.use_cases.model_search_service import ModelSearchService

        return ModelSearchService(adapter=self._registry.model_search)

    # --- Training Services ---

    def training_service(self):
        """Create TrainingService with injected TrainingEngine."""
        from modelcypher.core.use_cases.training_service import TrainingService

        return TrainingService(engine=self._registry.training_engine)

    def geometry_training_service(self):
        """Create GeometryTrainingService with injected JobStore."""
        from modelcypher.core.use_cases.geometry_training_service import (
            GeometryTrainingService,
        )

        return GeometryTrainingService(store=self._registry.job_store)

    # --- Export Services ---

    def export_service(self):
        """Create ExportService with injected store and exporter."""
        from modelcypher.core.use_cases.export_service import ExportService

        return ExportService(
            store=self._registry.model_store,
            exporter=self._registry.exporter,
        )

    def checkpoint_service(self):
        """Create CheckpointService with injected store and exporter."""
        from modelcypher.core.use_cases.checkpoint_service import CheckpointService

        return CheckpointService(
            store=self._registry.job_store,
            exporter=self._registry.exporter,
        )

    # --- Inference Services ---

    def ensemble_service(self):
        """Create EnsembleService with injected stores and inference."""
        from modelcypher.core.use_cases.ensemble_service import EnsembleService

        return EnsembleService(
            store=self._registry.model_store,
            inference_engine=self._registry.inference_engine,
        )

    def merge_validation_service(self):
        """Create MergeValidationService with injected inference engine."""
        from modelcypher.core.use_cases.merge_validation_service import (
            MergeValidationService,
        )

        return MergeValidationService(inference_engine=self._registry.inference_engine)

    def knowledge_transfer_service(self):
        """Create KnowledgeTransferService with injected inference engine."""
        from modelcypher.core.use_cases.knowledge_transfer_service import (
            KnowledgeTransferService,
        )

        return KnowledgeTransferService(
            inference_engine=self._registry.inference_engine
        )
