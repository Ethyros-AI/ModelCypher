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

"""Low-rank adaptation derived from geometric specifications.

Computes LoRA weight matrices directly from target activation positions
without requiring training data. Given a TransferPoint specifying where
a concept should appear in the target model's latent space, this module
solves for the minimal-rank weight perturbation that achieves that geometry.

Mathematical Framework:
    Given target activation y* and representative input x, find ΔW such that:
        (W + ΔW) @ x ≈ y*

    Subject to rank constraint: rank(ΔW) ≤ r

    Solution via SVD:
        ΔW_full = (y* - W@x) ⊗ x / ||x||²
        U, S, V^T = SVD(ΔW_full)
        ΔW_r = U[:,:r] @ diag(S[:r]) @ V[:r,:]

    Factored form for LoRA: ΔW = B @ A where
        B = U[:,:r] @ sqrt(diag(S[:r]))
        A = sqrt(diag(S[:r])) @ V[:r,:]

References:
    - Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language
      Models. arXiv:2106.09685.
    - Eckart, C., & Young, G. (1936). The approximation of one matrix by
      another of lower rank. Psychometrika, 1(3), 211-218.
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
      Johns Hopkins University Press. Chapter 2: Matrix Analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .manifold_transfer import TransferPoint, AnchorDistanceProfile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AdaptationQuality(str, Enum):
    """Quality assessment of geometric LoRA based on reconstruction error."""
    OPTIMAL = "optimal"  # Geometric loss < 0.1
    COMPRESSED = "compressed"  # Geometric loss < 0.3, some approximation
    MINIMAL = "minimal"  # Rank 1, maximum compression
    DEGRADED = "degraded"  # Geometric loss >= 0.3


@dataclass(frozen=True)
class GeometricLoRAConfig:
    """Configuration for geometric LoRA generation.

    Attributes:
        target_rank: Target rank for LoRA matrices.
        auto_rank: Automatically determine rank from singular values.
        singular_value_threshold: Fraction of max for significant values.
        max_rank: Maximum allowed rank.
        min_rank: Minimum rank.
        regularization: Numerical stability regularization.
        scale_factor: LoRA alpha scaling factor.
        target_layers: Which layers to generate LoRA for.
        target_projections: Which projection types to target.
    """
    target_rank: int = 4
    auto_rank: bool = True
    singular_value_threshold: float = 0.01
    max_rank: int = 64
    min_rank: int = 1
    regularization: float = 1e-6
    scale_factor: float = 1.0
    target_layers: list[int] | None = None
    target_projections: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


@dataclass
class LayerLoRAWeights:
    """Low-rank weight matrices for a single layer projection.

    The weight perturbation is factored as: ΔW = B @ A
    where A ∈ R^{r×d_in} and B ∈ R^{d_out×r}.

    Attributes:
        layer_idx: Layer index in the model.
        projection_name: Name of the projection (q_proj, v_proj, etc.).
        A: Down projection matrix (rank, in_features).
        B: Up projection matrix (out_features, rank).
        rank: Rank of the factorization.
        singular_values: Full singular value spectrum for diagnostics.
        geometric_loss: Relative error in achieving target geometry.
    """
    layer_idx: int
    projection_name: str
    A: np.ndarray
    B: np.ndarray
    rank: int
    singular_values: np.ndarray
    geometric_loss: float

    @property
    def in_features(self) -> int:
        return self.A.shape[1]

    @property
    def out_features(self) -> int:
        return self.B.shape[0]

    @property
    def delta_W(self) -> np.ndarray:
        """Compute full weight delta (for verification)."""
        return self.B @ self.A

    @property
    def effective_rank(self) -> float:
        """Compute effective rank from singular value decay."""
        if len(self.singular_values) == 0:
            return 0.0
        normalized = self.singular_values / (self.singular_values[0] + 1e-10)
        return float(np.sum(normalized > 0.01))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "layer": self.layer_idx,
            "projection": self.projection_name,
            "rank": self.rank,
            "inFeatures": self.in_features,
            "outFeatures": self.out_features,
            "geometricLoss": self.geometric_loss,
            "effectiveRank": self.effective_rank,
            "topSingularValues": self.singular_values[:5].tolist(),
        }


@dataclass
class GeometricLoRA:
    """Complete geometric LoRA adapter for a model.

    Contains factored weight matrices for all targeted layers,
    derived from a TransferPoint geometric specification.

    Attributes:
        transfer_point: The geometric specification.
        weights: List of per-layer LoRA weights.
        config: Generation configuration.
        mean_geometric_loss: Average reconstruction error.
        total_rank: Sum of ranks across all layers.
        quality: Overall quality assessment.
    """
    transfer_point: TransferPoint
    weights: list[LayerLoRAWeights]
    config: GeometricLoRAConfig
    mean_geometric_loss: float
    total_rank: int
    quality: AdaptationQuality

    @property
    def num_layers(self) -> int:
        return len(set(w.layer_idx for w in self.weights))

    @property
    def num_parameters(self) -> int:
        """Total number of LoRA parameters."""
        return sum(w.A.size + w.B.size for w in self.weights)

    def get_weights_for_layer(self, layer_idx: int) -> list[LayerLoRAWeights]:
        """Get all weights for a specific layer."""
        return [w for w in self.weights if w.layer_idx == layer_idx]

    def to_safetensors_dict(self) -> dict[str, np.ndarray]:
        """Convert to safetensors-compatible dictionary.

        Uses standard LoRA naming convention:
        base_model.model.layers.{layer}.{proj}.lora_A.weight
        base_model.model.layers.{layer}.{proj}.lora_B.weight
        """
        result = {}
        for w in self.weights:
            prefix = f"base_model.model.layers.{w.layer_idx}.self_attn.{w.projection_name}"
            result[f"{prefix}.lora_A.weight"] = w.A
            result[f"{prefix}.lora_B.weight"] = w.B
        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "conceptId": self.transfer_point.concept_id,
            "numLayers": self.num_layers,
            "totalRank": self.total_rank,
            "numParameters": self.num_parameters,
            "meanGeometricLoss": self.mean_geometric_loss,
            "quality": self.quality.value,
            "transferConfidence": self.transfer_point.confidence,
            "weights": [w.to_dict() for w in self.weights],
        }


class GeometricLoRAGenerator:
    """Generates LoRA weights from geometric specifications.

    Given a TransferPoint specifying the target position of a concept
    in the model's latent space, computes the low-rank weight perturbation
    that achieves that geometry.

    The key insight is that LoRA weights can be computed analytically
    by solving for the perturbation that maps a representative input
    to the target activation, then applying rank truncation via SVD.
    """

    def __init__(self, config: GeometricLoRAConfig | None = None):
        self.config = config or GeometricLoRAConfig()

    def generate(
        self,
        transfer_point: TransferPoint,
        model_weights: dict[int, dict[str, np.ndarray]],
        anchor_activations: dict[str, np.ndarray],
    ) -> GeometricLoRA:
        """Generate geometric LoRA from transfer point specification.

        Args:
            transfer_point: The target geometry specification.
            model_weights: Current weights by layer/projection.
            anchor_activations: Anchor activations for input estimation.

        Returns:
            GeometricLoRA with factored weight matrices.
        """
        weights_list = []

        target_layers = self.config.target_layers
        if target_layers is None:
            target_layers = sorted(model_weights.keys())

        for layer_idx in target_layers:
            if layer_idx not in model_weights:
                continue

            layer_weights = model_weights[layer_idx]

            for proj_name in self.config.target_projections:
                if proj_name not in layer_weights:
                    continue

                W = layer_weights[proj_name]

                lora_weights = self._compute_layer_lora(
                    transfer_point=transfer_point,
                    current_weight=W,
                    layer_idx=layer_idx,
                    proj_name=proj_name,
                    anchor_activations=anchor_activations,
                )

                if lora_weights is not None:
                    weights_list.append(lora_weights)

        if weights_list:
            mean_loss = float(np.mean([w.geometric_loss for w in weights_list]))
            total_rank = sum(w.rank for w in weights_list)
        else:
            mean_loss = 1.0
            total_rank = 0

        quality = self._assess_quality(weights_list)

        return GeometricLoRA(
            transfer_point=transfer_point,
            weights=weights_list,
            config=self.config,
            mean_geometric_loss=mean_loss,
            total_rank=total_rank,
            quality=quality,
        )

    def _compute_layer_lora(
        self,
        transfer_point: TransferPoint,
        current_weight: np.ndarray,
        layer_idx: int,
        proj_name: str,
        anchor_activations: dict[str, np.ndarray],
    ) -> LayerLoRAWeights | None:
        """Compute LoRA weights for a single layer projection.

        Solves: find ΔW such that (W + ΔW) @ x ≈ y*
        where y* is the target activation from transfer_point.
        """
        out_features, in_features = current_weight.shape
        target_position = transfer_point.coordinates

        # Estimate representative input from anchor activations
        anchor_inputs = []
        weights = []

        profile = transfer_point.source_profile
        for i, anchor_id in enumerate(profile.anchor_ids):
            if anchor_id in anchor_activations:
                anchor_inputs.append(np.mean(anchor_activations[anchor_id], axis=0))
                weights.append(profile.weights[i])

        if not anchor_inputs:
            logger.warning(f"No anchor activations for layer {layer_idx}")
            return None

        anchor_inputs = np.array(anchor_inputs)
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        representative_input = np.average(anchor_inputs, axis=0, weights=weights)

        if len(representative_input) != in_features:
            logger.warning(
                f"Input dimension mismatch: {len(representative_input)} vs {in_features}"
            )
            return None

        if len(target_position) != out_features:
            logger.warning(
                f"Output dimension mismatch: {len(target_position)} vs {out_features}"
            )
            return None

        # Current output and target delta
        current_output = current_weight @ representative_input
        output_delta = target_position - current_output

        # Compute full-rank delta: ΔW = output_delta ⊗ input / ||input||²
        input_norm_sq = np.dot(representative_input, representative_input)
        if input_norm_sq < 1e-10:
            logger.warning(f"Near-zero input for layer {layer_idx}")
            return None

        delta_W_full = np.outer(output_delta, representative_input) / input_norm_sq
        delta_W_full = delta_W_full + self.config.regularization * np.eye(
            out_features, in_features
        )

        # SVD for low-rank approximation
        U, S, Vt = np.linalg.svd(delta_W_full, full_matrices=False)

        # Determine rank
        rank = self._determine_rank(S)

        # Truncate and factor
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        sqrt_S = np.sqrt(S_r)
        B = U_r * sqrt_S
        A = sqrt_S[:, np.newaxis] * Vt_r

        B = B * self.config.scale_factor

        # Compute geometric loss
        reconstructed = (B @ A) @ representative_input
        geometric_loss = float(
            np.linalg.norm(reconstructed - output_delta) /
            (np.linalg.norm(output_delta) + 1e-10)
        )

        return LayerLoRAWeights(
            layer_idx=layer_idx,
            projection_name=proj_name,
            A=A,
            B=B,
            rank=rank,
            singular_values=S,
            geometric_loss=geometric_loss,
        )

    def _determine_rank(self, singular_values: np.ndarray) -> int:
        """Determine appropriate rank from singular value spectrum."""
        if not self.config.auto_rank:
            return min(self.config.target_rank, len(singular_values))

        threshold = self.config.singular_value_threshold * singular_values[0]
        significant = np.sum(singular_values > threshold)

        rank = max(self.config.min_rank, min(significant, self.config.max_rank))
        return int(rank)

    def _assess_quality(self, weights: list[LayerLoRAWeights]) -> AdaptationQuality:
        """Assess overall quality of generated LoRA."""
        if not weights:
            return AdaptationQuality.MINIMAL

        losses = [w.geometric_loss for w in weights]
        ranks = [w.rank for w in weights]
        mean_loss = np.mean(losses)
        mean_rank = np.mean(ranks)

        if mean_loss < 0.1 and mean_rank <= self.config.target_rank:
            return AdaptationQuality.OPTIMAL
        elif mean_loss < 0.3:
            return AdaptationQuality.COMPRESSED
        elif mean_rank <= 1:
            return AdaptationQuality.MINIMAL
        else:
            return AdaptationQuality.DEGRADED


def generate_geometric_lora(
    transfer_point: TransferPoint,
    model_weights: dict[int, dict[str, np.ndarray]],
    anchor_activations: dict[str, np.ndarray],
    config: GeometricLoRAConfig | None = None,
) -> GeometricLoRA:
    """Convenience function for geometric LoRA generation.

    Args:
        transfer_point: Target geometry specification.
        model_weights: Model weights by layer/projection.
        anchor_activations: Anchor activations.
        config: Optional configuration.

    Returns:
        GeometricLoRA with factored weight matrices.
    """
    generator = GeometricLoRAGenerator(config)
    return generator.generate(transfer_point, model_weights, anchor_activations)


def save_geometric_lora(
    lora: GeometricLoRA,
    output_path: str,
    include_metadata: bool = True,
) -> None:
    """Save geometric LoRA to safetensors format.

    Args:
        lora: The geometric LoRA to save.
        output_path: Path to save (should end in .safetensors).
        include_metadata: Whether to include generation metadata.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError:
        logger.error("safetensors not installed, cannot save LoRA")
        raise

    tensors = lora.to_safetensors_dict()

    metadata = {}
    if include_metadata:
        metadata = {
            "concept_id": lora.transfer_point.concept_id,
            "quality": lora.quality.value,
            "mean_geometric_loss": str(lora.mean_geometric_loss),
            "total_rank": str(lora.total_rank),
            "num_parameters": str(lora.num_parameters),
            "transfer_confidence": str(lora.transfer_point.confidence),
            "generator": "ModelCypher Geometric LoRA",
        }

    save_file(tensors, output_path, metadata=metadata)
    logger.info(f"Saved geometric LoRA to {output_path}")
