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
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from modelcypher.ports.backend import Backend

BackendType = Literal["mlx", "jax", "cuda"]


@dataclass(frozen=True)
class BackendDescriptor:
    """Summary of backend availability and system info."""

    key: str
    display_name: str
    available: bool
    error: str | None = None
    system_info: dict[str, Any] = field(default_factory=dict)


def _try_mlx_available() -> tuple[bool, str | None]:
    """Check if MLX is available via Backend."""
    try:
        from modelcypher.backends.mlx_backend import MLXBackend
        info = MLXBackend().get_system_info()
        return info.get("available", False), None
    except Exception as e:
        return False, str(e)


def _try_cuda_available() -> bool:
    """Check if CUDA is available via Backend."""
    try:
        from modelcypher.backends.cuda_backend import CUDABackend
        info = CUDABackend().get_system_info()
        return info.get("available", False)
    except Exception:
        return False


def _try_jax_available() -> bool:
    """Check if JAX is available via Backend."""
    try:
        from modelcypher.backends.jax_backend import JAXBackend
        info = JAXBackend().get_system_info()
        return info.get("available", False)
    except Exception:
        return False


def _probe_availability(backend_key: BackendType) -> tuple[bool, str | None]:
    if backend_key == "mlx":
        return _try_mlx_available()
    if backend_key == "cuda":
        return (_try_cuda_available(), None)
    if backend_key == "jax":
        return (_try_jax_available(), None)
    return (False, f"Unknown backend key: {backend_key}")


def _backend_specs() -> list[tuple[BackendType, str, Callable[[], Backend]]]:
    return [
        ("mlx", "MLX", lambda: get_backend("mlx")),
        ("cuda", "CUDA", lambda: get_backend("cuda")),
        ("jax", "JAX", lambda: get_backend("jax")),
    ]


def probe_backends(explicit: bool = False) -> list[BackendDescriptor]:
    """Probe available backends and return descriptors.

    Args:
        explicit: If True, run full probes even when optional backends
            may be skipped by platform heuristics.
    """
    descriptors: list[BackendDescriptor] = []
    for key, display_name, loader in _backend_specs():
        available, error = _probe_availability(key)
        system_info: dict[str, Any] = {}
        if available:
            try:
                backend = loader()
                system_info = backend.get_system_info()
                available = bool(system_info.get("available", True))
            except Exception as exc:
                available = False
                error = str(exc)
        descriptors.append(
            BackendDescriptor(
                key=key,
                display_name=display_name,
                available=available,
                error=error,
                system_info=system_info,
            )
        )
    return descriptors


def detect_default_backend_type() -> BackendType:
    """Detect the preferred backend for the current platform.

    Priority:
        1. MC_BACKEND (or MODELCYPHER_BACKEND) environment variable
        2. MLX on macOS (Apple Silicon)
        3. CUDA if available
        4. JAX if available

    Returns:
        The backend type string to use.

    Raises:
        RuntimeError: If an explicitly requested backend is unavailable.
    """
    # Check environment variable override
    env_backend = os.environ.get("MC_BACKEND", "").lower()
    if not env_backend:
        env_backend = os.environ.get("MODELCYPHER_BACKEND", "").lower()

    if env_backend in ("mlx", "jax", "cuda"):
        if env_backend == "mlx":
            available, error = _try_mlx_available()
            if not available:
                raise RuntimeError(
                    f"MC_BACKEND=mlx requested but MLX failed to initialize. {error or 'Unknown error'}."
                )
        return env_backend  # type: ignore[return-value]

    # Auto-detect best available backend
    mlx_available, mlx_error = _try_mlx_available()

    if sys.platform == "darwin":
        if mlx_available:
            return "mlx"
        message = "MLX backend unavailable on macOS."
        if mlx_error:
            message = f"{message} {mlx_error}."
        raise RuntimeError(message)

    if mlx_available:
        return "mlx"

    if _try_cuda_available():
        return "cuda"

    if _try_jax_available():
        return "jax"

    message = "No GPU backend available. ModelCypher requires GPU acceleration."
    if mlx_error:
        message = f"{message} MLX probe error: {mlx_error}."
    raise RuntimeError(f"{message} Install MLX (macOS), CUDA (NVIDIA), or JAX (TPU/GPU).")


def get_backend(backend_type: BackendType) -> Backend:
    """Get a specific backend by type.

    Args:
        backend_type: One of "mlx", "jax", "cuda"

    Returns:
        The requested backend instance.

    Raises:
        ImportError: If the backend's dependencies are not installed.
        ValueError: If the backend type is not recognized.
        RuntimeError: If the backend failed to initialize.
    """
    if backend_type == "mlx":
        available, error = _try_mlx_available()
        if not available:
            raise RuntimeError(
                f"MLX backend requested but failed to initialize. {error or 'Unknown error'}."
            )
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_type == "jax":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend()
    elif backend_type == "cuda":
        from modelcypher.backends.cuda_backend import CUDABackend
        return CUDABackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def initialize_default_backend() -> Backend:
    """Initialize the default backend based on platform detection.

    Returns:
        The initialized backend instance.
    """
    from modelcypher.core.domain._backend import get_default_backend, set_default_backend

    try:
        return get_default_backend()
    except RuntimeError:
        pass

    backend_type = detect_default_backend_type()
    backend = get_backend(backend_type)
    set_default_backend(backend)
    return backend


def default_backend() -> Backend:
    """Get the default backend."""
    from modelcypher.core.domain._backend import get_default_backend
    return get_default_backend()


def get_activation_provider(
    backend: Backend | None = None,
    model_path: str | None = None,
    pooling: str = "auto",
):
    """Get an activation provider implementation.

    This returns a Backend-based activation provider that works with
    any supported runtime through the Backend protocol.

    Args:
        backend: Optional Backend instance. Uses default if not provided.
        model_path: Optional path to model (for model-path-based loading).
        pooling: Pooling strategy ("auto", "mean", "frechet").

    Returns:
        An ActivationProvider implementation.
    """
    from modelcypher.adapters.activation_provider import ActivationProviderAdapter

    if backend is None:
        backend = initialize_default_backend()

    return ActivationProviderAdapter(backend=backend, model_path=model_path, pooling=pooling)


def get_inference_engine(backend: Backend | None = None):
    """Get the unified inference engine."""
    from modelcypher.adapters.inference_engine import InferenceEngine

    return InferenceEngine(backend=backend)


def get_model_probe(backend: Backend | None = None):
    """Get the unified model probe."""
    from modelcypher.adapters.model_probe import BackendModelProbe

    return BackendModelProbe(backend=backend)


def get_training_engine():
    """Get the backend-selected training engine."""
    from modelcypher.adapters.training_engine_stub import BackendTrainingEngine

    return BackendTrainingEngine()


def get_training_checkpoint_manager(max_checkpoints: int = 3):
    """Get the backend-selected checkpoint manager."""
    from modelcypher.adapters.training_engine_stub import BackendCheckpointManager

    return BackendCheckpointManager(max_checkpoints=max_checkpoints)


def get_training_loss_landscape_computer():
    """Get the backend-selected loss landscape computer."""
    from modelcypher.adapters.training_engine_stub import BackendLossLandscapeComputer

    return BackendLossLandscapeComputer()


def get_multimodal_embedding_extractor(backend: Backend | None = None):
    """Get the multimodal embedding extractor."""
    from modelcypher.adapters.multimodal_embedding_extractor import (
        MultiModalEmbeddingExtractor,
    )

    return MultiModalEmbeddingExtractor(backend=backend)


def get_embedding_provider(model_path: str | None = None, backend: Backend | None = None):
    """Get the backend-based embedding provider."""
    from modelcypher.adapters.backend_embedding_provider import get_embedding_provider

    return get_embedding_provider(model_path=model_path, backend=backend)


def get_dual_path_generator_class() -> type:
    """Get the dual-path generator class."""
    from modelcypher.adapters.security_scan_stub import DualPathGenerator

    return DualPathGenerator


def get_security_scan_metrics_class() -> type:
    """Get the security scan metrics class."""
    from modelcypher.adapters.security_scan_stub import SecurityScanMetrics

    return SecurityScanMetrics


__all__ = [
    "Backend",
    "BackendType",
    "BackendDescriptor",
    "default_backend",
    "detect_default_backend_type",
    "get_activation_provider",
    "get_backend",
    "get_inference_engine",
    "get_model_probe",
    "get_training_engine",
    "get_training_checkpoint_manager",
    "get_training_loss_landscape_computer",
    "get_multimodal_embedding_extractor",
    "get_embedding_provider",
    "get_dual_path_generator_class",
    "get_security_scan_metrics_class",
    "initialize_default_backend",
    "probe_backends",
]


def __getattr__(name: str):
    """Lazy import backends to avoid import errors when dependencies missing."""
    if name == "MLXBackend":
        from modelcypher.backends.mlx_backend import MLXBackend
        return MLXBackend
    if name == "JAXBackend":
        from modelcypher.backends.jax_backend import JAXBackend
        return JAXBackend
    if name == "CUDABackend":
        from modelcypher.backends.cuda_backend import CUDABackend
        return CUDABackend
    if name == "LazyBackend":
        from modelcypher.backends.lazy_backend import LazyBackend
        return LazyBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
