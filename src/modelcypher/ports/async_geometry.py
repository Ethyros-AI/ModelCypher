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


from typing import Any, Protocol, runtime_checkable

from modelcypher.core.domain.geometry.types import (
    AlignmentConfig,
    BatchMergerResult,
    ClusteringConfiguration,
    ClusteringResult,
    CompositionAnalysis,
    CompositionProbe,
    ConsistencyResult,
    IntrinsicDimensionResult,
    ManifoldPoint,
    MergerConfig,
    MergerResult,
    ModelFingerprints,
    PermutationAlignmentResult,
    ProcrustesConfig,
    ProcrustesResult,
    ProjectionMethod,
    ProjectionResult,
    RebasinResult,
    RefusalConfig,
    RefusalDirection,
    RefusalDistanceMetrics,
)


@runtime_checkable
class GeometryPort(Protocol):
    """
    Abstract interface for high-dimensional geometry operations.
    Adapters (MLX, CUDA) must implement this.
    """

    # --- Permutation Alignment ---

    async def align_permutations(
        self, source_weight: Any, target_weight: Any, anchors: Any | None, config: AlignmentConfig
    ) -> PermutationAlignmentResult: ...

    async def align_via_anchor_projection(
        self, source_weight: Any, target_weight: Any, anchors: Any, config: AlignmentConfig
    ) -> PermutationAlignmentResult: ...

    async def rebasin_mlp(
        self,
        source_weights: dict[str, Any],
        target_weights: dict[str, Any],
        anchors: Any,
        config: AlignmentConfig,
    ) -> RebasinResult: ...

    # --- Safety / Refusal Geometry ---

    async def compute_refusal_direction(
        self,
        harmful_activations: Any,  # [N, D]
        harmless_activations: Any,  # [N, D]
        config: RefusalConfig,
        metadata: dict[str, Any],  # e.g. layer_id, model_id
    ) -> RefusalDirection | None: ...

    async def measure_refusal_distance(
        self,
        activation: Any,  # [D]
        direction: RefusalDirection,
        token_index: int,
        previous_projection: float | None = None,
    ) -> RefusalDistanceMetrics: ...

    # --- Transport Guided Merger ---

    async def merge_models_transport(
        self,
        source_weights: Any,  # Matrix or Dict of matrices
        target_weights: Any,
        source_activations: Any,
        target_activations: Any,
        config: MergerConfig,
    ) -> MergerResult | BatchMergerResult: ...

    # --- Manifold Analysis ---

    async def cluster_manifold(
        self, points: list[ManifoldPoint], config: ClusteringConfiguration
    ) -> ClusteringResult: ...

    async def estimate_intrinsic_dimension(
        self,
        points: list[Any],  # Vectors
        method: str = "mle",
    ) -> IntrinsicDimensionResult: ...

    # --- Alignment & Projection ---

    async def project_fingerprints(
        self,
        fingerprints: ModelFingerprints,
        method: ProjectionMethod = ProjectionMethod.PCA,
        max_features: int = 1200,
        layers: set[int] | None = None,
        seed: int = 42,
    ) -> ProjectionResult: ...

    async def align_procrustes(
        self, activations: list[list[list[float]]], config: ProcrustesConfig
    ) -> ProcrustesResult | None: ...

    # --- Compositional Analysis ---

    async def analyze_composition(
        self,
        composition_embedding: Any,
        component_embeddings: Any,  # Array/Tensor [N, D]
        probe: CompositionProbe,
    ) -> CompositionAnalysis: ...

    async def check_consistency(
        self, analyses_a: list[CompositionAnalysis], analyses_b: list[CompositionAnalysis]
    ) -> ConsistencyResult: ...
