"""Model probe service - orchestrates backend-specific probe implementations.

This service automatically selects the appropriate backend for the current platform:
- macOS (Darwin): MLXModelProbe - uses mx.load() for bfloat16 support
- Linux with CUDA: CUDAModelProbe - uses PyTorch/safetensors
- Linux with TPU/JAX: JAXModelProbe - uses JAX/Flax

Weight loading is inherently backend-specific and cannot be abstracted,
so each backend provides its own implementation.
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from modelcypher.ports.model_probe import (
    AlignmentAnalysisResult,
    LayerDrift,
    LayerInfo,
    MergeValidationResult,
    ModelProbePort,
    ModelProbeResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Re-export dataclasses for backwards compatibility
__all__ = [
    "AlignmentAnalysisResult",
    "LayerDrift",
    "LayerInfo",
    "MergeValidationResult",
    "ModelProbeResult",
    "ModelProbeService",
    "get_model_probe",
]


def get_model_probe() -> ModelProbePort:
    """
    Get the appropriate model probe for the current platform.

    Returns:
        ModelProbePort implementation for the current backend.

    Platform selection:
        - macOS (Darwin): MLXModelProbe
        - Linux + CUDA available: CUDAModelProbe
        - Linux + JAX available: JAXModelProbe
        - Fallback: CUDAModelProbe (requires PyTorch)

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    if sys.platform == "darwin":
        try:
            from modelcypher.backends.mlx_model_probe import MLXModelProbe
            return MLXModelProbe()
        except ImportError as exc:
            raise RuntimeError(
                "MLX not available on macOS. Install with: pip install mlx"
            ) from exc

    # Linux: try CUDA first, then JAX
    try:
        from modelcypher.backends.cuda_model_probe import CUDAModelProbe
        probe = CUDAModelProbe()
        if probe.available:
            return probe
    except ImportError:
        pass

    try:
        from modelcypher.backends.jax_model_probe import JAXModelProbe
        probe = JAXModelProbe()
        if probe.available:
            return probe
    except ImportError:
        pass

    raise RuntimeError(
        "No suitable backend available. Install one of:\n"
        "  - macOS: pip install mlx\n"
        "  - Linux/CUDA: pip install torch\n"
        "  - Linux/TPU: pip install jax jaxlib"
    )


class ModelProbeService:
    """
    Service for probing and analyzing model architecture and compatibility.

    This is a facade that delegates to the appropriate backend-specific probe.
    Use get_model_probe() directly for more control over backend selection.
    """

    def __init__(self, probe: ModelProbePort | None = None) -> None:
        """
        Initialize the service.

        Args:
            probe: Optional probe implementation. If None, auto-selects based on platform.
        """
        self._probe = probe or get_model_probe()

    def probe(self, model_path: str) -> ModelProbeResult:
        """Probe model for architecture details.

        Args:
            model_path: Path to the model directory containing config.json and weight files.

        Returns:
            ModelProbeResult with architecture details.

        Raises:
            ValueError: If model path is invalid or config.json is missing.
        """
        return self._probe.probe(model_path)

    def validate_merge(self, source: str, target: str) -> MergeValidationResult:
        """Validate merge compatibility between two models.

        Args:
            source: Path to the source model directory.
            target: Path to the target model directory.

        Returns:
            MergeValidationResult with compatibility assessment.
        """
        return self._probe.validate_merge(source, target)

    def analyze_alignment(self, model_a: str, model_b: str) -> AlignmentAnalysisResult:
        """Analyze alignment drift between two models.

        Computes layer-wise drift using weight comparison.

        Args:
            model_a: Path to the first model directory.
            model_b: Path to the second model directory.

        Returns:
            AlignmentAnalysisResult with drift metrics bounded in [0.0, 1.0].
        """
        return self._probe.analyze_alignment(model_a, model_b)
