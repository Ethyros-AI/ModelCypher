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
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

from .manifold_transfer import TransferPoint

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
    target_projections: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class LayerLoRAWeights:
    """Low-rank weight matrices for a single layer projection.

    The weight perturbation is factored as: ΔW = B @ A
    where A ∈ R^{r×d_in} and B ∈ R^{d_out×r}.

    Attributes:
        layer_idx: Layer index in the model.
        projection_name: Name of the projection (q_proj, v_proj, etc.).
        A: Down projection matrix (rank, in_features) - stored as list for backend agnostic.
        B: Up projection matrix (out_features, rank) - stored as list for backend agnostic.
        rank: Rank of the factorization.
        singular_values: Full singular value spectrum for diagnostics.
        geometric_loss: Relative error in achieving target geometry.
    """

    layer_idx: int
    projection_name: str
    A: "object"  # Backend array
    B: "object"  # Backend array
    rank: int
    singular_values: "object"  # Backend array
    geometric_loss: float

    @property
    def in_features(self) -> int:
        backend = get_default_backend()
        return backend.to_numpy(self.A).shape[1]

    @property
    def out_features(self) -> int:
        backend = get_default_backend()
        return backend.to_numpy(self.B).shape[0]

    @property
    def delta_W(self) -> "object":
        """Compute full weight delta (for verification)."""
        backend = get_default_backend()
        result = backend.matmul(self.B, self.A)
        backend.eval(result)
        return result

    @property
    def effective_rank(self) -> float:
        """Compute effective rank from singular value decay."""
        backend = get_default_backend()
        sv_np = backend.to_numpy(self.singular_values)
        if len(sv_np) == 0:
            return 0.0
        normalized = sv_np / (sv_np[0] + 1e-10)
        return float((normalized > 0.01).sum())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        backend = get_default_backend()
        sv_np = backend.to_numpy(self.singular_values)
        return {
            "layer": self.layer_idx,
            "projection": self.projection_name,
            "rank": self.rank,
            "inFeatures": self.in_features,
            "outFeatures": self.out_features,
            "geometricLoss": self.geometric_loss,
            "effectiveRank": self.effective_rank,
            "topSingularValues": sv_np[:5].tolist(),
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
        backend = get_default_backend()
        total = 0
        for w in self.weights:
            A_np = backend.to_numpy(w.A)
            B_np = backend.to_numpy(w.B)
            total += A_np.size + B_np.size
        return total

    def get_weights_for_layer(self, layer_idx: int) -> list[LayerLoRAWeights]:
        """Get all weights for a specific layer."""
        return [w for w in self.weights if w.layer_idx == layer_idx]

    def to_safetensors_dict(self) -> dict[str, "object"]:
        """Convert to safetensors-compatible dictionary.

        Uses standard LoRA naming convention:
        base_model.model.layers.{layer}.{proj}.lora_A.weight
        base_model.model.layers.{layer}.{proj}.lora_B.weight
        """
        backend = get_default_backend()
        result = {}
        for w in self.weights:
            prefix = f"base_model.model.layers.{w.layer_idx}.self_attn.{w.projection_name}"
            result[f"{prefix}.lora_A.weight"] = backend.to_numpy(w.A)
            result[f"{prefix}.lora_B.weight"] = backend.to_numpy(w.B)
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
        model_weights: dict[int, dict[str, "object"]],
        anchor_activations: dict[str, "object"],
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
            losses = [w.geometric_loss for w in weights_list]
            mean_loss = sum(losses) / len(losses)
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
        current_weight: "object",
        layer_idx: int,
        proj_name: str,
        anchor_activations: dict[str, "object"],
    ) -> LayerLoRAWeights | None:
        """Compute LoRA weights for a single layer projection.

        Solves: find ΔW such that (W + ΔW) @ x ≈ y*
        where y* is the target activation from transfer_point.
        """
        backend = get_default_backend()
        weight_shape = backend.shape(current_weight)
        out_features, in_features = weight_shape[0], weight_shape[1]
        target_position = transfer_point.coordinates

        # Estimate representative input from anchor activations
        anchor_inputs = []
        weights_list = []

        profile = transfer_point.source_profile
        for i, anchor_id in enumerate(profile.anchor_ids):
            if anchor_id in anchor_activations:
                act = backend.to_numpy(anchor_activations[anchor_id])
                if len(act.shape) > 1:
                    anchor_inputs.append(act.mean(axis=0))
                else:
                    anchor_inputs.append(act)
                weights_list.append(profile.weights[i])

        if not anchor_inputs:
            logger.warning(f"No anchor activations for layer {layer_idx}")
            return None

        # Stack anchor inputs and compute weighted average using backend
        anchor_stack = backend.stack([backend.array(a) for a in anchor_inputs], axis=0)
        weights_arr = backend.array(weights_list)
        weights_arr = weights_arr / backend.sum(weights_arr)

        # Weighted average: sum(weights * inputs) along axis 0
        weighted = anchor_stack * backend.expand_dims(weights_arr, axis=1)
        representative_input = backend.sum(weighted, axis=0)
        backend.eval(representative_input)

        rep_shape = backend.shape(representative_input)
        if rep_shape[0] != in_features:
            logger.warning(
                f"Input dimension mismatch: {rep_shape[0]} vs {in_features}"
            )
            return None

        if len(target_position) != out_features:
            logger.warning(f"Output dimension mismatch: {len(target_position)} vs {out_features}")
            return None

        # Current output and target delta using backend
        current_output = backend.matmul(current_weight, representative_input)
        target_arr = backend.array(target_position)
        output_delta = target_arr - current_output
        backend.eval(current_output, output_delta)

        # Compute full-rank delta: ΔW = output_delta ⊗ input / ||input||²
        input_norm_sq = backend.sum(representative_input * representative_input)
        backend.eval(input_norm_sq)
        input_norm_sq_val = float(backend.to_numpy(input_norm_sq))
        if input_norm_sq_val < 1e-10:
            logger.warning(f"Near-zero input for layer {layer_idx}")
            return None

        # Outer product: output_delta[:, None] @ representative_input[None, :]
        output_delta_col = backend.expand_dims(output_delta, axis=1)
        rep_input_row = backend.expand_dims(representative_input, axis=0)
        delta_W_full = backend.matmul(output_delta_col, rep_input_row) / input_norm_sq
        reg_matrix = self.config.regularization * backend.eye(out_features, in_features)
        delta_W_full = delta_W_full + reg_matrix
        backend.eval(delta_W_full)

        # SVD for low-rank approximation
        U, S, Vt = backend.svd(delta_W_full, full_matrices=False)
        backend.eval(U, S, Vt)

        # Determine rank
        rank = self._determine_rank(S, backend)

        # Truncate and factor
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        sqrt_S = backend.sqrt(S_r)
        B = U_r * sqrt_S  # Broadcasting: (out, rank) * (rank,)
        sqrt_S_col = backend.expand_dims(sqrt_S, axis=1)
        A = sqrt_S_col * Vt_r  # (rank, 1) * (rank, in) = (rank, in)

        B = B * self.config.scale_factor
        backend.eval(A, B)

        # Compute geometric loss
        reconstructed = backend.matmul(backend.matmul(B, A), representative_input)
        diff_norm = backend.norm(reconstructed - output_delta)
        delta_norm = backend.norm(output_delta) + 1e-10
        backend.eval(diff_norm, delta_norm)
        geometric_loss = float(backend.to_numpy(diff_norm) / backend.to_numpy(delta_norm))

        # A, B, S are already backend arrays
        A_backend = A
        B_backend = B
        S_backend = S

        return LayerLoRAWeights(
            layer_idx=layer_idx,
            projection_name=proj_name,
            A=A_backend,
            B=B_backend,
            rank=rank,
            singular_values=S_backend,
            geometric_loss=geometric_loss,
        )

    def _determine_rank(self, singular_values: "object", backend: "object") -> int:
        """Determine appropriate rank from singular value spectrum."""
        sv_shape = backend.shape(singular_values)
        sv_len = sv_shape[0]

        if not self.config.auto_rank:
            return min(self.config.target_rank, sv_len)

        # Get first singular value for threshold
        first_sv = float(backend.to_numpy(singular_values[0]))
        threshold = self.config.singular_value_threshold * first_sv

        # Count significant singular values
        significant_mask = singular_values > threshold
        significant = int(backend.sum(significant_mask))

        rank = max(self.config.min_rank, min(significant, self.config.max_rank))
        return int(rank)

    def _assess_quality(self, weights: list[LayerLoRAWeights]) -> AdaptationQuality:
        """Assess overall quality of generated LoRA."""
        if not weights:
            return AdaptationQuality.MINIMAL

        losses = [w.geometric_loss for w in weights]
        ranks = [w.rank for w in weights]
        mean_loss = sum(losses) / len(losses)
        mean_rank = sum(ranks) / len(ranks)

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
    model_weights: dict[int, dict[str, "object"]],
    anchor_activations: dict[str, "object"],
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
