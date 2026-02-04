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
    from modelcypher.core.use_cases.entropy_calibration_service import (
        EntropyCalibrationService,
    )
    from modelcypher.core.use_cases.model_probe_service import ModelProbeService
    from modelcypher.core.use_cases.model_service import ModelService
    from modelcypher.infrastructure.container import PortRegistry
    from modelcypher.infrastructure.service_factory import ServiceFactory
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Backend
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


def get_model_probe_service() -> "ModelProbeService":
    """Get ModelProbeService with proper dependency injection."""
    return _get_factory().model_probe_service()


def get_entropy_calibration_service() -> "EntropyCalibrationService":
    """Get EntropyCalibrationService with proper dependency injection."""
    return _get_factory().entropy_calibration_service()


def get_backend() -> "Backend":
    """Get the compute backend from the registry.

    This centralizes backend access through the composition root,
    avoiding direct imports from core/domain/_backend.

    Returns:
        The configured Backend instance.
    """
    return _get_registry().backend


def get_model_loader() -> "ModelLoaderPort":
    """Get ModelLoaderPort from the registry."""
    return _get_registry().model_loader


def get_activation_provider() -> "ActivationProvider":
    """Get ActivationProvider from the registry."""
    return _get_registry().activation_provider


def get_inference_engine() -> "InferenceEngine":
    """Get InferenceEngine from the registry."""
    return _get_registry().inference_engine


def get_geometry_training_service():
    """Get GeometryTrainingService with proper dependency injection."""
    from modelcypher.adapters.filesystem_storage import FileSystemStore
    from modelcypher.core.use_cases.geometry_training_service import GeometryTrainingService

    store = FileSystemStore()
    return GeometryTrainingService(store)


def get_geometry_analysis_service():
    """Get GeometryAnalysisService with proper dependency injection.

    Returns a service for geometric analysis operations (intrinsic dimension,
    manifold entropy, reasoning flow, Jacobian analysis).
    """
    from modelcypher.core.use_cases.geometry_analysis_service import GeometryAnalysisService

    registry = _get_registry()
    return GeometryAnalysisService(
        backend=registry.backend,
        activation_provider=registry.activation_provider,
    )


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


def get_system_service():
    """Get SystemService."""
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


def get_lora_safety_service():
    """Get LoRASafetyService for LoRA safety analysis.

    This service provides:
    - Fisher-guided module targeting (exp15: r=-0.864)
    - Mode connectivity barrier check (exp16: r=0.989)
    - Goldilocks quality scoring for curriculum (exp17: r=-0.955)
    """
    from modelcypher.core.use_cases.lora_safety_service import LoRASafetyService

    return LoRASafetyService()


def get_registry() -> "PortRegistry":
    """Get the PortRegistry for direct access to ports.

    Use this when you need direct access to a port adapter
    rather than a service.
    """
    return _get_registry()
