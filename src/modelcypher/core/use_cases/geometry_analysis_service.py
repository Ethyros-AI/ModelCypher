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

"""Geometry Analysis Service.

Provides a service layer for geometric analysis operations on model representations.
This service wraps domain geometry modules to ensure proper hexagonal architecture
separation between CLI commands and domain logic.

Operations provided:
- Intrinsic dimension estimation (TwoNN)
- Manifold entropy measurement
- Reasoning flow analysis (velocities, curvatures)
- Jacobian spectrum analysis

Example:
    from modelcypher.cli.composition import get_geometry_analysis_service

    service = get_geometry_analysis_service()
    result = service.compute_intrinsic_dimension(activations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.activation_provider import ActivationProvider
    from modelcypher.ports.backend import Array, Backend


# Re-export result types for consumers
from modelcypher.core.domain.geometry.intrinsic_dimension import (
    ConfidenceInterval,
    IntrinsicDimension,
    LocalDimensionMap,
    TwoNNEstimate,
)
from modelcypher.core.domain.geometry.jacobian_analyzer import (
    JacobianProfile,
    JacobianTraceResult,
    trace_jacobian_spectrum,
)
from modelcypher.core.domain.geometry.manifold_entropy import (
    LayerEntropyResult,
    ManifoldEntropy,
    ManifoldEntropyResult,
)
from modelcypher.core.domain.geometry.reasoning_flow import (
    FlowMetrics,
    LayerFlowProfile,
    ReasoningFlowAnalyzer,
    TokenCurvatureProfile,
    analyze_multilayer_flow,
    analyze_token_curvature,
)

__all__ = [
    # Service class
    "GeometryAnalysisService",
    # Intrinsic dimension types
    "TwoNNEstimate",
    "ConfidenceInterval",
    "LocalDimensionMap",
    # Manifold entropy types
    "ManifoldEntropyResult",
    "LayerEntropyResult",
    # Reasoning flow types
    "FlowMetrics",
    "LayerFlowProfile",
    "TokenCurvatureProfile",
    # Jacobian types
    "JacobianProfile",
    "JacobianTraceResult",
]


class GeometryAnalysisService:
    """Service for geometric analysis of model representations.

    Encapsulates domain geometry modules behind a service interface,
    maintaining proper hexagonal architecture boundaries.

    This service is stateless - all operations take inputs and return results.
    Backend and activation provider are injected at construction time.
    """

    def __init__(
        self,
        backend: "Backend",
        activation_provider: "ActivationProvider | None" = None,
    ) -> None:
        """Initialize the geometry analysis service.

        Args:
            backend: Compute backend for tensor operations.
            activation_provider: Optional activation provider for model operations.
        """
        self._backend = backend
        self._activation_provider = activation_provider

    @property
    def backend(self) -> "Backend":
        """Get the compute backend."""
        return self._backend

    @property
    def activation_provider(self) -> "ActivationProvider | None":
        """Get the activation provider."""
        return self._activation_provider

    # --- Intrinsic Dimension ---

    def compute_intrinsic_dimension(
        self,
        activations: "Array",
        *,
        with_ci: bool = False,
    ) -> TwoNNEstimate:
        """Estimate intrinsic dimension using TwoNN method.

        Uses geodesic distances and the regression variant from Facco et al. (2017)
        for robust estimation.

        Args:
            activations: Activation matrix [n_samples, hidden_dim].
            with_ci: If True, compute confidence intervals via bootstrap.

        Returns:
            TwoNNEstimate with intrinsic dimension and sample counts.
        """
        estimator = IntrinsicDimension(backend=self._backend)
        return estimator.compute(activations, with_ci=with_ci)

    def compute_local_dimension_map(
        self,
        activations: "Array",
        *,
        n_neighbors: int | None = None,
    ) -> LocalDimensionMap:
        """Compute per-point intrinsic dimension map.

        Estimates local intrinsic dimension for each point using k-nearest
        neighbors. Useful for identifying regions of varying complexity.

        Args:
            activations: Activation matrix [n_samples, hidden_dim].
            n_neighbors: Number of neighbors for local estimation.
                If None, derived from sample size.

        Returns:
            LocalDimensionMap with per-point dimension estimates.
        """
        estimator = IntrinsicDimension(backend=self._backend)
        return estimator.local_dimension_map(activations, n_neighbors=n_neighbors)

    # --- Manifold Entropy ---

    def compute_manifold_entropy(
        self,
        layer_activations: dict[int, list["Array"]],
    ) -> ManifoldEntropyResult:
        """Compute manifold entropy across layers.

        Aggregates intrinsic dimension, effective rank, and spectral entropy
        measurements across all layers.

        Args:
            layer_activations: Dict mapping layer index to list of activation
                arrays (one per probe text).

        Returns:
            ManifoldEntropyResult with total entropy and per-layer breakdown.
        """
        analyzer = ManifoldEntropy(backend=self._backend)
        return analyzer.compute(layer_activations)

    def compute_layer_entropy(
        self,
        activations: "Array",
        layer_idx: int,
    ) -> LayerEntropyResult:
        """Compute entropy for a single layer's activations.

        Computes intrinsic dimension (TwoNN), effective rank (Shannon entropy-based),
        and spectral entropy for a single layer.

        Args:
            activations: Activation matrix [n_samples, hidden_dim].
            layer_idx: Layer index for identification in results.

        Returns:
            LayerEntropyResult with intrinsic dimension, effective rank, etc.
        """
        analyzer = ManifoldEntropy(backend=self._backend)
        return analyzer.compute_layer_entropy(activations, layer_idx)

    # --- Reasoning Flow ---

    def analyze_reasoning_flow(
        self,
        layer_positions: dict[int, "Array"],
    ) -> list[LayerFlowProfile]:
        """Analyze reasoning flow geometry across layers.

        Computes velocities, curvatures, and flow metrics for token trajectories
        at each layer. Based on Zhou et al. (ICLR 2026) reasoning flow framework.

        Args:
            layer_positions: Dict mapping layer index to position array
                [n_tokens, hidden_dim].

        Returns:
            List of LayerFlowProfile with geometric metrics per layer.
        """
        return analyze_multilayer_flow(self._backend, layer_positions)

    def analyze_token_curvature(
        self,
        layer_positions: dict[int, "Array"],
    ) -> TokenCurvatureProfile:
        """Analyze per-token curvature patterns.

        Identifies which tokens have highest curvature (sharpest reasoning turns)
        and computes curvature statistics.

        Args:
            layer_positions: List of position arrays [n_tokens, hidden_dim],
                one per layer.

        Returns:
            TokenCurvatureResult with per-token and per-layer curvature analysis.
        """
        return analyze_token_curvature(self._backend, layer_positions)

    def create_flow_analyzer(self) -> ReasoningFlowAnalyzer:
        """Create a ReasoningFlowAnalyzer for fine-grained control.

        Use this when you need direct access to velocity/curvature computation
        methods rather than the high-level analyze_* functions.

        Returns:
            ReasoningFlowAnalyzer instance configured with this service's backend.
        """
        return ReasoningFlowAnalyzer(self._backend)

    # --- Jacobian Analysis ---

    def trace_jacobian_spectrum(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        model_path: str,
        *,
        num_probes: int = 16,
    ) -> JacobianTraceResult:
        """Analyze Jacobian spectrum across model layers.

        Computes singular value spectrum of layer Jacobians using randomized
        methods. Reveals amplification/compression patterns and numerical stability.

        Args:
            model: The loaded model.
            tokenizer: The model's tokenizer.
            prompt: Input prompt for analysis.
            model_path: Path to model (for metadata).
            num_probes: Number of random probe vectors for approximation.

        Returns:
            JacobianTraceResult with per-layer profiles and summary statistics.
        """
        return trace_jacobian_spectrum(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_path=model_path,
            num_probes=num_probes,
            backend=self._backend,
        )
