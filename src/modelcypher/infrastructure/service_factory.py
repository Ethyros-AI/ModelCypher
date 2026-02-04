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

from typing import TYPE_CHECKING, Any

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

    # --- Core Services ---

    def system_service(self):
        """Create SystemService."""
        from modelcypher.core.use_cases.system_service import SystemService

        return SystemService(model_store=self._registry.model_store)

    def model_probe_service(self):
        """Create ModelProbeService with injected probe port."""
        from modelcypher.core.use_cases.model_probe_service import ModelProbeService

        return ModelProbeService(probe=self._registry.model_probe)

    def entropy_calibration_service(self):
        """Create EntropyCalibrationService with injected model loader."""
        from modelcypher.core.use_cases.entropy_calibration_service import (
            EntropyCalibrationService,
        )

        return EntropyCalibrationService(model_loader=self._registry.model_loader)

    def model_service(self):
        """Create ModelService with injected store and model loader."""
        from modelcypher.core.use_cases.model_service import ModelService

        return ModelService(
            store=self._registry.model_store,
            model_loader=self._registry.model_loader,
        )

    # --- Training Services ---

    def geometry_training_service(self):
        """Create GeometryTrainingService with injected JobStore."""
        from modelcypher.core.use_cases.geometry_training_service import (
            GeometryTrainingService,
        )

        return GeometryTrainingService(store=self._registry.job_store)

    def thermo_service(self):
        """Create ThermoService with injected model loader."""
        from modelcypher.core.use_cases.thermo_service import ThermoService

        return ThermoService(
            model_loader=self._registry.model_loader,
        )

    # --- Geometry Services ---

    def manifold_profile_service(self):
        """Create ManifoldProfileService with injected ManifoldProfileStore."""
        from modelcypher.core.use_cases.manifold_profile_service import (
            ManifoldProfileService,
        )

        return ManifoldProfileService(store=self._registry.manifold_profile_store)

    def model_profiler_service(self):
        """Create ModelProfilerService with proper probe dependency."""
        from modelcypher.core.use_cases.model_profiler_service import (
            ModelProfilerService,
        )

        return ModelProfilerService(probe=self._registry.model_probe)

    def geometry_service(self):
        """Create GeometryService with needed dependencies."""
        from modelcypher.core.use_cases.geometry_service import GeometryService

        return GeometryService(
            backend=self._registry.backend,
            detector=self.gate_detector(),
        )

    def geometry_metrics_service(self):
        """Create GPU-accelerated GeometryMetricsService."""
        from modelcypher.core.use_cases.geometry_metrics_service import (
            GeometryMetricsService,
        )

        return GeometryMetricsService(backend=self._registry.backend)

    def stability_service(self):
        """Create StabilityService."""
        from modelcypher.core.use_cases.stability_service import StabilityService

        return StabilityService(
            backend=self._registry.backend,
            inference_engine=self._registry.inference_engine,
        )

    def consolidation_service(self, model: Any, n_layers: int, hidden_dim: int):
        """Create ConsolidationService for a specific model."""
        from modelcypher.core.domain.geometry.null_space_tracker import (
            NullSpaceTracker,
        )
        from modelcypher.core.use_cases.consolidation_service import (
            ConsolidationService,
        )

        tracker = NullSpaceTracker(
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            backend=self._registry.backend,
        )
        return ConsolidationService(
            model=model,
            null_space_tracker=tracker,
            backend=self._registry.backend,
        )

    def gate_detector(self):
        """Create GateDetector with proper dependencies."""
        from modelcypher.adapters.embedding_defaults import make_default_embedder
        from modelcypher.core.domain.geometry.gate_detector import GateDetector

        embedder = make_default_embedder()
        if embedder is None:
            return None
        return GateDetector(
            embedder=embedder,
            backend=self._registry.backend,
        )

    def geometry_adapter_service(self):
        """Create GeometryAdapterService with proper dependencies."""
        from modelcypher.core.use_cases.geometry_adapter_service import (
            GeometryAdapterService,
        )

        return GeometryAdapterService(
            model_loader=self._registry.model_loader,
            backend=self._registry.backend,
        )

    def geometry_persona_service(self):
        """Create GeometryPersonaService with proper dependencies."""
        from modelcypher.core.use_cases.geometry_persona_service import (
            GeometryPersonaService,
        )

        return GeometryPersonaService(backend=self._registry.backend)
