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

"""ScaledGD preconditioning for LoRA training (Riemannian GD on rank-r manifold).

Backend-agnostic. Operates on pre-flattened dicts so the caller handles
framework-specific tree_flatten/tree_unflatten.

For each LoRA layer, preconditions:
    grad_A = grad_A @ (B B^T + eI)^{-1}    (normalize out B's scale)
    grad_B = (A^T A + eI)^{-1} @ grad_B     (normalize out A's scale)

At initialization (spectral init with ||A|| = ||B|| = sqrt(sigma_k/2)),
B B^T has eigenvalues ~ sigma_k/2 and e ~ sigma_k^2. The preconditioner
(B B^T + eI)^{-1} gracefully degrades toward (1/e)I (uniform scaling
= standard GD) when factors are small. As training progresses and
factors develop spectral structure, the preconditioner adapts.

References:
    Tong, Ma, Chi (JMLR 2021): condition-number-free convergence.
    Hayou et al. (ICML 2024): A and B need different effective rates.
    Mu & Klabjan (Dec 2025): step size must decrease as adapters grow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.core.domain.training.geometric_optimizer import (
        OptimizerGeometryConfig,
    )
    from modelcypher.ports.backend import Backend
    from modelcypher.ports.training import LoRALayerConfig


def precondition_lora_gradients(
    grad_flat: dict[str, Any],
    param_flat: dict[str, Any],
    lora_configs: list["LoRALayerConfig"],
    opt_config: "OptimizerGeometryConfig",
    backend: "Backend",
) -> dict[str, Any]:
    """Apply ScaledGD preconditioning to LoRA gradients.

    Operates on pre-flattened dicts (caller does tree_flatten).
    Returns modified flat dict (caller does tree_unflatten if needed).

    Args:
        grad_flat: Flattened gradient dict (key -> array).
        param_flat: Flattened trainable parameter dict (key -> array).
        lora_configs: List of LoRALayerConfig from derive_lora_configs().
        opt_config: OptimizerGeometryConfig (provides epsilon, decay per layer).
        backend: Backend for matrix operations.

    Returns:
        Preconditioned gradient flat dict (same keys as grad_flat).
    """
    # Work on a copy so caller's dict is not mutated
    result = dict(grad_flat)

    for cfg in lora_configs:
        if cfg.rank <= 0:
            continue

        prefix = cfg.layer_key.replace(".weight", "")
        a_key = prefix + ".lora_a"
        b_key = prefix + ".lora_b"

        if a_key not in param_flat or b_key not in param_flat:
            continue
        if a_key not in result or b_key not in result:
            continue

        A = param_flat[a_key]  # [in, rank]
        B = param_flat[b_key]  # [rank, out]

        # Get epsilon for numerical stability of inverse
        layer_opt = opt_config.layer_configs.get(cfg.layer_key)
        eps = layer_opt.epsilon if layer_opt else 1e-7

        rank = B.shape[0]

        # Precondition grad_A by (B B^T + eI)^{-1}
        # B: [rank, out], B^T: [out, rank], B @ B^T: [rank, rank]
        BBt = backend.matmul(B, backend.transpose(B)) + eps * backend.eye(rank)
        BBt_inv = backend.inv(BBt)
        result[a_key] = backend.matmul(result[a_key], BBt_inv)  # [in, rank] @ [rank, rank]

        # Precondition grad_B by (A^T A + eI)^{-1}
        # A: [in, rank], A^T: [rank, in], A^T @ A: [rank, rank]
        AtA = backend.matmul(backend.transpose(A), A) + eps * backend.eye(A.shape[1])
        AtA_inv = backend.inv(AtA)
        result[b_key] = backend.matmul(AtA_inv, result[b_key])  # [rank, rank] @ [rank, out]

        # Weight decay (condition-aware, separate from LR)
        decay = layer_opt.decay_scale if layer_opt else 0.0
        if decay > 0:
            result[a_key] = result[a_key] + decay * param_flat[a_key]
            result[b_key] = result[b_key] + decay * param_flat[b_key]

    return result


__all__ = [
    "precondition_lora_gradients",
]
