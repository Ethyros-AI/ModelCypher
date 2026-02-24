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
    from modelcypher.core.use_cases.adapter_analysis_service import AdapterAnalysisService
    from modelcypher.core.use_cases.capacity_analysis_service import (
        CapacityAnalysisService,
    )
    from modelcypher.core.use_cases.entropy_calibration_service import (
        EntropyCalibrationService,
    )
    from modelcypher.core.use_cases.model_probe_service import ModelProbeService
    from modelcypher.core.use_cases.model_service import ModelService
    from modelcypher.core.use_cases.verification_depth_profile_service import (
        VerificationDepthProfileService,
    )
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


def get_verification_depth_profile_service() -> "VerificationDepthProfileService":
    """Get VerificationDepthProfileService with proper dependency injection."""
    from modelcypher.core.use_cases.verification_depth_profile_service import (
        VerificationDepthProfileService,
    )

    registry = _get_registry()
    return VerificationDepthProfileService(
        backend=registry.backend,
        activation_provider=registry.activation_provider,
    )


def get_geometry_safety_service(
    safe_drift_samples: list[float] | None = None,
    attack_drift_samples: list[float] | None = None,
    safe_delta_h_samples: list[float] | None = None,
    attack_delta_h_samples: list[float] | None = None,
    safe_entropy_samples: list[float] | None = None,
    attack_entropy_samples: list[float] | None = None,
    n_bootstrap: int = 1000,
    seed: int | None = None,
):
    """Get GeometrySafetyService with calibration-derived thresholds.

    Thresholds are derived from distribution crossings between safe and
    attack measurements (Neyman-Pearson). Both safe AND attack data are
    required — you need a reference distribution to set a boundary.

    Args:
        safe_drift_samples: Drift magnitudes from safe/normal operation.
        attack_drift_samples: Drift magnitudes from attack/adversarial operation.
        safe_delta_h_samples: Delta-H values from safe prompts.
        attack_delta_h_samples: Delta-H values from attack prompts.
        safe_entropy_samples: Absolute entropy from safe prompts.
        attack_entropy_samples: Absolute entropy from attack prompts.
        n_bootstrap: Bootstrap resamples for CI.
        seed: RNG seed for reproducibility.
    """
    from modelcypher.core.use_cases.geometry_safety_service import (
        DriftThresholds,
        GeometrySafetyService,
        VulnerabilityThresholds,
    )

    if any(
        s is None for s in (
            safe_drift_samples, attack_drift_samples,
            safe_delta_h_samples, attack_delta_h_samples,
            safe_entropy_samples, attack_entropy_samples,
        )
    ):
        raise ValueError(
            "All calibration samples required (safe + attack for drift, delta-H, and entropy). "
            "Geometry safety requires two-group calibration data to derive decision boundaries."
        )
    drift_thresholds, _ = DriftThresholds.from_calibration_data(
        safe_drift_samples, attack_drift_samples,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    vulnerability_thresholds, _ = VulnerabilityThresholds.from_calibration_data(
        safe_delta_h_samples, attack_delta_h_samples,
        safe_entropy_samples, attack_entropy_samples,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    return GeometrySafetyService(
        training_service=get_geometry_training_service(),
        drift_thresholds=drift_thresholds,
        vulnerability_thresholds=vulnerability_thresholds,
        model_loader=get_model_loader(),
        backend=get_backend(),
    )


def get_concept_volume_service():
    """Get ConceptVolumeService for multi-concept activation probing."""
    from modelcypher.core.use_cases.concept_volume_service import ConceptVolumeService

    registry = _get_registry()
    return ConceptVolumeService(
        activation_provider=registry.activation_provider,
        backend=registry.backend,
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


def get_system_cache_service():
    """Get SystemCacheService with injected backend."""
    from modelcypher.core.use_cases.system_cache_service import SystemCacheService

    registry = _get_registry()
    return SystemCacheService(backend=registry.backend)


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


# --- Geometry Services (from factory) ---


def get_geometry_adapter_service():
    """Get GeometryAdapterService with proper dependency injection."""
    return _get_factory().geometry_adapter_service()


def get_geometry_metrics_service():
    """Get GPU-accelerated GeometryMetricsService."""
    return _get_factory().geometry_metrics_service()


def get_geometry_persona_service():
    """Get GeometryPersonaService with proper dependency injection."""
    return _get_factory().geometry_persona_service()


def get_geometry_service():
    """Get GeometryService with needed dependencies."""
    return _get_factory().geometry_service()


def get_manifold_profile_service():
    """Get ManifoldProfileService with injected store."""
    return _get_factory().manifold_profile_service()


def get_model_profiler_service():
    """Get ModelProfilerService with proper probe dependency."""
    return _get_factory().model_profiler_service()


def get_stability_service():
    """Get StabilityService with backend and inference engine."""
    return _get_factory().stability_service()


def get_thermo_service():
    """Get ThermoService with injected model loader."""
    return _get_factory().thermo_service()


# --- Benchmark & Analysis Services ---


def get_benchmark_service():
    """Get BenchmarkService for running benchmarks with geometric metrics."""
    return _get_factory().benchmark_service()


def get_adapter_analysis_service() -> "AdapterAnalysisService":
    """Get AdapterAnalysisService with injected backend and loaders."""
    from modelcypher.adapters.adapter_weights_loader import AutoAdapterWeightsLoader
    from modelcypher.core.use_cases.adapter_analysis_service import AdapterAnalysisService
    from modelcypher.core.use_cases.lora_baseline_artifact import (
        load_reference_baseline,
    )

    registry = _get_registry()
    return AdapterAnalysisService(
        backend=registry.backend,
        weights_loader=AutoAdapterWeightsLoader(),
        baseline_loader=load_reference_baseline,
    )


def get_entropy_probe_service():
    """Get EntropyProbeService for entropy baseline verification."""
    return _get_factory().entropy_probe_service()


def get_geometry_sparse_service():
    """Get GeometrySparseService for sparse region analysis."""
    return _get_factory().geometry_sparse_service()


def get_safety_probe_service():
    """Get SafetyProbeService with default embedder."""
    return _get_factory().safety_probe_service()


def get_lora_memory_service():
    """Get LoRAMemoryService for LoRA adapter memory management."""
    return _get_factory().lora_memory_service()


def get_dataset_training_service():
    """Get DatasetTrainingService for dataset-driven LoRA training."""
    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.use_cases.dataset_training_service import DatasetTrainingService

    backend = _get_registry().backend
    adapter = MLXTrainingAdapter(backend)
    return DatasetTrainingService(adapter=adapter, backend=backend)


def get_star_training_service():
    """Get StarTrainingService for STaR round orchestration."""
    from modelcypher.backends.mlx_training_adapter import MLXTrainingAdapter
    from modelcypher.core.use_cases.dataset_training_service import DatasetTrainingService
    from modelcypher.core.use_cases.star_training_service import StarTrainingService

    backend = _get_registry().backend
    adapter = MLXTrainingAdapter(backend)
    dataset_service = DatasetTrainingService(adapter=adapter, backend=backend)
    return StarTrainingService(
        backend=backend,
        dataset_training_service=dataset_service,
        training_adapter=adapter,
    )


def get_knowledge_analyzer():
    """Get KnowledgeAnalyzer for factual knowledge detection."""
    return _get_factory().knowledge_analyzer()


def get_curriculum_profiler(model, tokenizer, layer_idx: int | None = None):
    """Get CurriculumProfiler for geometric difficulty measurement.

    Args:
        model: The loaded model to profile
        tokenizer: Tokenizer for the model
        layer_idx: Optional layer index for profiling (defaults to final layer)
    """
    return _get_factory().curriculum_profiler(model, tokenizer, layer_idx)


def get_concept_response_matrix_service():
    """Get ConceptResponseMatrixService for CRM operations."""
    return _get_factory().concept_response_matrix_service()


def get_bilm_probe_service():
    """Get BiLMProbeService for bidirectional LM probing."""
    return _get_factory().bilm_probe_service()


def get_quantization_service():
    """Get QuantizationService for model quantization."""
    return _get_factory().quantization_service()


def get_capacity_analysis_service() -> "CapacityAnalysisService":
    """Get CapacityAnalysisService for per-layer spectral capacity analysis."""
    from modelcypher.core.use_cases.capacity_analysis_service import (
        CapacityAnalysisService,
    )

    registry = _get_registry()
    return CapacityAnalysisService(
        backend=registry.backend,
        model_loader=registry.model_loader,
    )


def get_merge_service():
    """Get UnifiedGeometricMerger for model merging."""
    from modelcypher.experimental.merge.merger import UnifiedGeometricMerger

    registry = _get_registry()
    return UnifiedGeometricMerger(
        model_loader=registry.model_loader,
        activation_provider=registry.activation_provider,
        inference_engine=registry.inference_engine,
        backend=registry.backend,
    )


def get_geodesic_trajectory_service():
    """Get GeodesicTrajectoryService for CoT geodesic measurement.

    Experimental. Measures whether activation trajectories follow
    geodesics on the manifold or cut through voids.
    """
    from modelcypher.core.use_cases.geodesic_trajectory_service import (
        GeodesicTrajectoryService,
    )

    registry = _get_registry()
    return GeodesicTrajectoryService(
        backend=registry.backend,
        activation_provider=registry.activation_provider,
    )
