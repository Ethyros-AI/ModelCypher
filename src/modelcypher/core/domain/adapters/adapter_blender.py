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

"""Utilities for blending multiple LoRA adapters via weight interpolation.

Supports linear blending of LoRA A and B matrices:
- A_blend = Σ αᵢ * Aᵢ
- B_blend = Σ αᵢ * Bᵢ

Also provides geometric weight computation based on compatibility scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from uuid import UUID

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlendResult:
    """Result of a weight blending operation."""

    weights: dict[str, NDArray[np.float32]]
    """Blended weight matrices keyed by module path."""

    adapter_weights: dict[UUID, float]
    """Normalized blend weights for each adapter."""

    modules_blended: int
    """Number of modules that were blended."""


class AdapterBlender:
    """Utilities for blending multiple LoRA adapters via weight interpolation."""

    @staticmethod
    def blend_weights(
        weights: list[tuple[NDArray[np.float32], float]],
    ) -> NDArray[np.float32] | None:
        """Blend multiple weight matrices with given weights.

        Computes the weighted sum: W_blend = Σ αᵢ * Wᵢ

        Args:
            weights: Array of (weight matrix, blend weight) pairs.

        Returns:
            Blended weight matrix, or None if empty input.
        """
        if not weights:
            return None

        if len(weights) == 1:
            matrix, weight = weights[0]
            return matrix * weight

        # Start with first weighted matrix
        result = weights[0][0] * weights[0][1]

        # Add remaining weighted matrices
        for matrix, weight in weights[1:]:
            result = result + matrix * weight

        return result.astype(np.float32)

    @staticmethod
    def blend_lora_matrices(
        a_matrices: list[dict[str, NDArray[np.float32]]],
        weights: list[float],
    ) -> dict[str, NDArray[np.float32]]:
        """Blend LoRA A matrices from multiple adapters.

        Args:
            a_matrices: A matrices keyed by module path, one dict per adapter.
            weights: Blend weight for each adapter.

        Returns:
            Blended A matrices by module path.
        """
        if len(a_matrices) != len(weights) or not a_matrices:
            return {}

        # Collect all module paths
        all_paths: set[str] = set()
        for matrices in a_matrices:
            all_paths.update(matrices.keys())

        blended: dict[str, NDArray[np.float32]] = {}

        for path in all_paths:
            # Collect matrices for this path with their weights
            path_weights: list[tuple[NDArray[np.float32], float]] = []

            for index, matrices in enumerate(a_matrices):
                if path in matrices:
                    path_weights.append((matrices[path], weights[index]))

            # Only blend if all adapters have this module
            if len(path_weights) == len(a_matrices):
                blended_matrix = AdapterBlender.blend_weights(path_weights)
                if blended_matrix is not None:
                    blended[path] = blended_matrix

        return blended

    @staticmethod
    def normalize_weights(weights: dict[UUID, float]) -> dict[UUID, float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Raw weights.

        Returns:
            Normalized weights.
        """
        total = sum(weights.values())
        if total <= 0:
            return weights

        return {id_: weight / total for id_, weight in weights.items()}

    @staticmethod
    def softmax_weights(
        weights: dict[UUID, float],
        temperature: float = 1.0,
    ) -> dict[UUID, float]:
        """Softmax normalization for sharper weight distribution.

        Args:
            weights: Raw weights.
            temperature: Softmax temperature (lower = sharper).

        Returns:
            Softmax-normalized weights.
        """
        if not weights:
            return {}

        scaled_values = [math.exp(w / temperature) for w in weights.values()]
        total = sum(scaled_values)

        result: dict[UUID, float] = {}
        for (id_, _), scaled in zip(weights.items(), scaled_values):
            result[id_] = scaled / total
        return result

    @staticmethod
    def apply_weight_floor(
        weights: dict[UUID, float],
        floor: float,
    ) -> dict[UUID, float]:
        """Apply weight floor to ensure minimum participation.

        Args:
            weights: Input weights.
            floor: Minimum weight value.

        Returns:
            Weights with floor applied and renormalized.
        """
        floored = {id_: max(weight, floor) for id_, weight in weights.items()}
        return AdapterBlender.normalize_weights(floored)

    @staticmethod
    def separate_lora_weights(
        weights: dict[str, NDArray[np.float32]],
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]:
        """Separates a weight dictionary into A and B matrices by key suffix.

        LoRA weights use keys like:
        - `model.layers.0.self_attn.q_proj.lora_a` for A matrices
        - `model.layers.0.self_attn.q_proj.lora_b` for B matrices

        Args:
            weights: Full weight dictionary with lora_a and lora_b keys.

        Returns:
            Tuple of (A matrices, B matrices) keyed by base path.
        """
        a_matrices: dict[str, NDArray[np.float32]] = {}
        b_matrices: dict[str, NDArray[np.float32]] = {}

        for key, matrix in weights.items():
            if key.endswith(".lora_a"):
                base_path = key[: -len(".lora_a")]
                a_matrices[base_path] = matrix
            elif key.endswith(".lora_b"):
                base_path = key[: -len(".lora_b")]
                b_matrices[base_path] = matrix

        return a_matrices, b_matrices

    @staticmethod
    def recombine_lora_weights(
        a_matrices: dict[str, NDArray[np.float32]],
        b_matrices: dict[str, NDArray[np.float32]],
    ) -> dict[str, NDArray[np.float32]]:
        """Recombines separated A and B matrices into a full weight dictionary.

        Args:
            a_matrices: A matrices keyed by base path.
            b_matrices: B matrices keyed by base path.

        Returns:
            Full weight dictionary with lora_a and lora_b suffixes.
        """
        combined: dict[str, NDArray[np.float32]] = {}

        for base_path, matrix in a_matrices.items():
            combined[f"{base_path}.lora_a"] = matrix

        for base_path, matrix in b_matrices.items():
            combined[f"{base_path}.lora_b"] = matrix

        return combined

    @staticmethod
    def blend_complete_adapters(
        adapters: list[tuple[dict[str, NDArray[np.float32]], float]],
    ) -> dict[str, NDArray[np.float32]] | None:
        """Blends multiple complete LoRA weight dictionaries with geometric weights.

        This is the main entry point for ensemble weight blending. It:
        1. Separates each adapter's weights into A and B matrices
        2. Blends A matrices: A_blend = Σ αᵢ * Aᵢ
        3. Blends B matrices: B_blend = Σ αᵢ * Bᵢ
        4. Recombines into a single weight dictionary

        The result is a mathematically coherent blended adapter that can be
        loaded into the model for a single inference pass.

        Args:
            adapters: Array of (weight dictionary, blend weight) pairs.

        Returns:
            Blended weight dictionary, or None if blending fails.
        """
        if not adapters:
            return None

        # Single adapter - just scale weights
        if len(adapters) == 1:
            weights, scale = adapters[0]
            return {key: matrix * scale for key, matrix in weights.items()}

        # Separate A and B matrices from each adapter
        all_a_matrices: list[dict[str, NDArray[np.float32]]] = []
        all_b_matrices: list[dict[str, NDArray[np.float32]]] = []
        blend_weights: list[float] = []

        for weights, weight in adapters:
            a_matrices, b_matrices = AdapterBlender.separate_lora_weights(weights)
            all_a_matrices.append(a_matrices)
            all_b_matrices.append(b_matrices)
            blend_weights.append(weight)

        # Blend A matrices
        blended_a = AdapterBlender.blend_lora_matrices(
            a_matrices=all_a_matrices,
            weights=blend_weights,
        )

        # Blend B matrices
        blended_b = AdapterBlender.blend_lora_matrices(
            a_matrices=all_b_matrices,
            weights=blend_weights,
        )

        if not blended_a or not blended_b:
            return None

        # Recombine into full weight dictionary
        return AdapterBlender.recombine_lora_weights(
            a_matrices=blended_a,
            b_matrices=blended_b,
        )

    @staticmethod
    def compute_geometric_weights(
        compatibility_scores: dict[UUID, float],
    ) -> dict[UUID, float]:
        """Compute blend weights from compatibility scores.

        Weights are proportional to how well each adapter's fingerprint
        aligns with the target model.

        Args:
            compatibility_scores: Compatibility score for each adapter.

        Returns:
            Normalized weights (sum to 1.0) for each adapter.
        """
        return AdapterBlender.normalize_weights(compatibility_scores)

    @staticmethod
    def compute_fidelity_weights(
        fidelity_scores: dict[UUID, float],
        fallback: float = 0.5,
    ) -> dict[UUID, float]:
        """Compute blend weights from transfer fidelity predictions.

        Uses the expected fidelity as weight.

        Args:
            fidelity_scores: Fidelity score for each adapter.
            fallback: Fallback weight for adapters without fidelity data.

        Returns:
            Normalized weights for each adapter.
        """
        scores = {
            id_: score if score is not None else fallback
            for id_, score in fidelity_scores.items()
        }
        return AdapterBlender.normalize_weights(scores)
