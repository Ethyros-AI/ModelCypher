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

"""Trajectory-based null-space discovery.

Trajectories capture the DYNAMICS of information flow through the model.
A forward pass through text produces a trajectory - a sequence of activations
as context builds. The trajectory's geometry (positions, velocities) spans
far more of the activation space than individual token activations.

Key insight: The trajectory's tangent space (velocities) captures directions
the model USES to process information flow. These are directions we should
NOT project into. The orthogonal complement is the true null-space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision_float32 as _promote_precision,
)

from modelcypher.core.domain.geometry.null_space import (
    _get_model_architecture,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryResult:
    """Result of collecting a trajectory through the model.

    A trajectory is the sequence of activations as context builds token-by-token.
    Includes positions (raw activations) and velocities (first differences).
    """

    positions: "Array"  # [seq_len, hidden_dim] - raw activations per position
    velocities: "Array"  # [seq_len-1, hidden_dim] - first differences
    accelerations: "Array | None"  # [seq_len-2, hidden_dim] - second differences (optional)
    seq_len: int
    hidden_dim: int
    text: str  # Original text (for debugging)


@dataclass
class TrajectorySubspaceResult:
    """Result of computing the subspace spanned by trajectories."""

    Vt: "Array"  # Right singular vectors [min(n,d), hidden_dim]
    singular_values: "Array"  # Singular values
    rank: int  # Numerical rank
    hidden_dim: int
    total_samples: int  # Total trajectory points used
    position_contribution: int  # Samples from positions
    velocity_contribution: int  # Samples from velocities


def collect_trajectory(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int,
    backend: "Backend",
    include_accelerations: bool = False,
) -> TrajectoryResult | None:
    """Collect activation trajectory for a full text sequence at a specific layer.

    A single forward pass gives the full trajectory - the sequence of hidden states
    as context accumulates. This captures HOW the model processes information flow,
    not just WHERE it lands.

    Args:
        model: The model to collect activations from.
        tokenizer: Tokenizer for encoding text.
        text: Input text to process.
        layer_idx: Layer index to collect activations from.
        backend: Backend for tensor operations.
        include_accelerations: If True, also compute second differences.

    Returns:
        TrajectoryResult with positions, velocities, and optional accelerations.
        Returns None if collection fails.
    """
    b = backend
    logger.debug("TRAJECTORY: Collecting for layer %d, text='%s...'", layer_idx, text[:30])

    try:
        # Get model architecture via protocol
        arch = _get_model_architecture(model)

        if arch.num_layers == 0:
            logger.warning("TRAJECTORY: Model has no layers")
            return None

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)

        if len(token_ids) < 2:
            logger.debug("TRAJECTORY: Text too short (need >= 2 tokens)")
            return None

        # Create input tensor
        input_ids = b.array([token_ids])
        b.eval(input_ids)

        # Get embeddings via architecture protocol
        embed_module = arch.embed_module
        if embed_module is None:
            logger.warning("TRAJECTORY: Cannot find embedding layer")
            return None
        h = embed_module(input_ids)

        b.eval(h)

        # Forward through layers up to target
        for idx, layer in enumerate(arch.layers):
            if idx > layer_idx:
                break
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

        b.eval(h)

        # h is now [batch=1, seq_len, hidden_dim]
        # Squeeze batch dimension to get [seq_len, hidden_dim]
        positions = b.squeeze(h, axis=0)
        b.eval(positions)

        seq_len = int(b.shape(positions)[0])
        hidden_dim = int(b.shape(positions)[1])

        # Compute velocities: first differences along sequence
        # velocities[i] = positions[i+1] - positions[i]
        pos_shifted = positions[1:, :]  # [seq_len-1, hidden_dim]
        pos_base = positions[:-1, :]  # [seq_len-1, hidden_dim]
        velocities = pos_shifted - pos_base
        b.eval(velocities)

        # Optionally compute accelerations
        accelerations = None
        if include_accelerations and seq_len >= 3:
            vel_shifted = velocities[1:, :]  # [seq_len-2, hidden_dim]
            vel_base = velocities[:-1, :]  # [seq_len-2, hidden_dim]
            accelerations = vel_shifted - vel_base
            b.eval(accelerations)

        return TrajectoryResult(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            text=text,
        )
    except Exception as e:
        logger.warning("TRAJECTORY: Collection failed for '%s...': %s", text[:30], e)
        return None


def _model_max_seq_len(model: Any) -> int | None:
    """Extract maximum sequence length from model metadata, if available."""
    # Try to get config from model (similar to _get_model_architecture helper)
    config: dict = {}
    if hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "to_dict"):
            config = model_config.to_dict()
        elif isinstance(model_config, dict):
            config = model_config
    elif hasattr(model, "model") and hasattr(model.model, "config"):
        model_config = model.model.config
        if hasattr(model_config, "to_dict"):
            config = model_config.to_dict()
        elif isinstance(model_config, dict):
            config = model_config

    # Check config for common max seq len keys
    candidates = [
        config.get("max_position_embeddings"),
        config.get("max_seq_len"),
        config.get("max_seq_length"),
        config.get("n_positions"),  # GPT-2 style
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return None


def compute_trajectory_subspace(
    trajectories: list[TrajectoryResult],
    backend: "Backend",
    include_velocities: bool = True,
    include_accelerations: bool = False,
) -> TrajectorySubspaceResult | None:
    """Compute the subspace spanned by trajectory dynamics.

    This captures not just WHERE the model goes, but HOW it gets there.
    Velocities capture the directions the model uses to flow between concepts.

    Args:
        trajectories: List of TrajectoryResult objects.
        backend: Backend for tensor operations.
        include_velocities: If True, include first differences in subspace.
        include_accelerations: If True, include second differences in subspace.

    Returns:
        TrajectorySubspaceResult with SVD decomposition and rank.
        Returns None if computation fails.
    """
    b = backend

    if not trajectories:
        logger.warning("TRAJECTORY SUBSPACE: No trajectories provided")
        return None

    logger.info(
        "TRAJECTORY SUBSPACE: Computing from %d trajectories (velocities=%s, accelerations=%s)",
        len(trajectories), include_velocities, include_accelerations
    )

    try:
        # Collect all trajectory features
        all_features: list["Array"] = []
        position_count = 0
        velocity_count = 0

        for traj in trajectories:
            # Always include positions
            all_features.append(traj.positions)
            position_count += int(b.shape(traj.positions)[0])

            if include_velocities:
                all_features.append(traj.velocities)
                velocity_count += int(b.shape(traj.velocities)[0])

            if include_accelerations and traj.accelerations is not None:
                all_features.append(traj.accelerations)

        # Stack all features: [total_points, hidden_dim]
        X = b.concatenate(all_features, axis=0)
        X = _promote_precision(X, b)
        b.eval(X)

        total_samples = int(b.shape(X)[0])
        hidden_dim = int(b.shape(X)[1])

        logger.info(
            "TRAJECTORY SUBSPACE: X shape [%d, %d] (positions=%d, velocities=%d)",
            total_samples, hidden_dim, position_count, velocity_count
        )

        # For tall-skinny matrices (m >> n), use Gram matrix trick:
        # G = X.T @ X has shape [n, n] instead of [m, n]
        # Eigenvalues of G = singular_values^2 of X
        # Eigenvectors of G = right singular vectors (V) of X
        # This avoids computing the full m x m U matrix which would exceed memory
        if total_samples > hidden_dim * 2:
            logger.info(
                "TRAJECTORY SUBSPACE: Using Gram matrix trick (m=%d >> n=%d)",
                total_samples, hidden_dim
            )
            # G = X.T @ X is [hidden_dim, hidden_dim]
            G = b.matmul(b.transpose(X), X)
            b.eval(G)

            # Eigendecomposition: G = V @ diag(eigenvalues) @ V.T
            eigenvalues, V = b.eigh(G)
            b.eval(eigenvalues, V)

            # Singular values = sqrt(eigenvalues), sorted descending
            # eigenvalues from eigh are in ascending order, so reverse
            n = int(b.shape(eigenvalues)[0])
            reverse_idx = b.arange(n - 1, -1, -1)
            eigenvalues = b.take(eigenvalues, reverse_idx, axis=0)
            V = b.take(V, reverse_idx, axis=1)
            b.eval(eigenvalues, V)

            # Clamp negative eigenvalues to zero (numerical noise)
            eigenvalues = b.maximum(eigenvalues, b.zeros_like(eigenvalues))
            S = b.sqrt(eigenvalues)
            b.eval(S)

            # V is [hidden_dim, hidden_dim], Vt = V.T
            Vt = b.transpose(V)
            b.eval(Vt)
        else:
            # Standard SVD for wide or square matrices
            logger.info("TRAJECTORY SUBSPACE: Using standard SVD")
            _, S, Vt = b.svd(X, full_matrices=False)
            b.eval(S, Vt)

        # Compute numerical rank using threshold sigma_max * sqrt(eps)
        eps = machine_epsilon(b, X)
        threshold_factor = sqrt_scalar(eps, b)

        max_s_arr = b.max(S)
        b.eval(max_s_arr)
        max_s = float(b.to_scalar(max_s_arr))
        threshold = max_s * threshold_factor

        # Count singular values above threshold
        rank_mask = S > threshold
        rank_arr = b.sum(b.astype(rank_mask, "int32"))
        b.eval(rank_arr)
        rank = int(b.to_scalar(rank_arr))

        logger.info(
            "TRAJECTORY SUBSPACE: rank=%d/%d (%.1f%% coverage), sigma_max=%.4e, threshold=%.4e",
            rank, hidden_dim, 100.0 * rank / hidden_dim, max_s, threshold
        )

        return TrajectorySubspaceResult(
            Vt=Vt,
            singular_values=S,
            rank=rank,
            hidden_dim=hidden_dim,
            total_samples=total_samples,
            position_contribution=position_count,
            velocity_contribution=velocity_count,
        )

    except Exception as e:
        logger.error("TRAJECTORY SUBSPACE: Computation failed: %s", e)
        import traceback
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        return None


def compute_trajectory_null_space(
    subspace_result: TrajectorySubspaceResult,
    backend: "Backend",
) -> "Array | None":
    """Compute the null space from trajectory subspace.

    The null space consists of directions orthogonal to ALL trajectory dynamics.
    These are the directions the model doesn't use for information flow -
    the safe directions for null-space projection.

    Args:
        subspace_result: Result from compute_trajectory_subspace.
        backend: Backend for tensor operations.

    Returns:
        Null space basis [hidden_dim, null_rank], or None if already full rank.
    """
    b = backend

    rank = subspace_result.rank
    hidden_dim = subspace_result.hidden_dim
    Vt = subspace_result.Vt

    null_rank = hidden_dim - rank

    if null_rank <= 0:
        logger.info("TRAJECTORY NULL SPACE: Trajectories already span full space")
        return None

    logger.info(
        "TRAJECTORY NULL SPACE: rank=%d, null_rank=%d (%.1f%% available)",
        rank, null_rank, 100.0 * null_rank / hidden_dim
    )

    # Null space is spanned by Vt[rank:].T
    # Vt has shape [min(n,d), hidden_dim]
    # We want the directions NOT covered by the first `rank` singular vectors
    Vt_null = Vt[rank:, :]  # [null_rank, hidden_dim]
    b.eval(Vt_null)

    # Return as [hidden_dim, null_rank] to match compute_null_space_basis convention
    U_null = b.transpose(Vt_null)  # [hidden_dim, null_rank]
    b.eval(U_null)

    return U_null


def collect_trajectories_batch(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer_idx: int,
    backend: "Backend",
    include_accelerations: bool = False,
) -> list[TrajectoryResult]:
    """Collect trajectories for multiple texts efficiently.

    Args:
        model: The model to collect activations from.
        tokenizer: Tokenizer for encoding text.
        texts: List of input texts to process.
        layer_idx: Layer index to collect activations from.
        backend: Backend for tensor operations.
        include_accelerations: If True, also compute second differences.

    Returns:
        List of TrajectoryResult objects (may be shorter than texts if some fail).
    """
    results: list[TrajectoryResult] = []
    max_seq_len = _model_max_seq_len(model)

    for text in texts:
        # Truncate very long texts
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)

        if max_seq_len is not None and len(token_ids) > max_seq_len:
            # Truncate and decode back to text
            truncated_ids = token_ids[:max_seq_len]
            text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

        traj = collect_trajectory(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layer_idx=layer_idx,
            backend=backend,
            include_accelerations=include_accelerations,
        )

        if traj is not None:
            results.append(traj)

    logger.info(
        "TRAJECTORY BATCH: Collected %d/%d trajectories for layer %d",
        len(results), len(texts), layer_idx
    )

    return results


def find_trajectory_null_space(
    model: Any,
    tokenizer: Any,
    probe_texts: list[str],
    layer_idx: int,
    backend: "Backend",
    include_velocities: bool = True,
    include_accelerations: bool = False,
) -> tuple["Array | None", TrajectorySubspaceResult | None]:
    """High-level function to find null space from trajectory probing.

    This is the main entry point for trajectory-based null-space discovery.
    It collects trajectories, computes the subspace they span, and returns
    the orthogonal complement (null space).

    Args:
        model: The model to analyze.
        tokenizer: Tokenizer for encoding text.
        probe_texts: List of probe texts to use for trajectory collection.
        layer_idx: Layer index to analyze.
        backend: Backend for tensor operations.
        include_velocities: If True, include first differences.
        include_accelerations: If True, include second differences.

    Returns:
        Tuple of (U_null, subspace_result) where:
        - U_null: Null space basis [hidden_dim, null_rank], or None if full rank
        - subspace_result: TrajectorySubspaceResult with diagnostic info
    """
    b = backend

    logger.info(
        "TRAJECTORY NULL SPACE: Finding for layer %d with %d probes",
        layer_idx, len(probe_texts)
    )

    # Collect trajectories
    trajectories = collect_trajectories_batch(
        model=model,
        tokenizer=tokenizer,
        texts=probe_texts,
        layer_idx=layer_idx,
        backend=b,
        include_accelerations=include_accelerations,
    )

    if not trajectories:
        logger.warning("TRAJECTORY NULL SPACE: No trajectories collected")
        return None, None

    # Compute subspace
    subspace_result = compute_trajectory_subspace(
        trajectories=trajectories,
        backend=b,
        include_velocities=include_velocities,
        include_accelerations=include_accelerations,
    )

    if subspace_result is None:
        logger.warning("TRAJECTORY NULL SPACE: Subspace computation failed")
        return None, None

    # Compute null space
    U_null = compute_trajectory_null_space(subspace_result, b)

    return U_null, subspace_result


__all__ = [
    "TrajectoryResult",
    "TrajectorySubspaceResult",
    "collect_trajectory",
    "compute_trajectory_subspace",
    "compute_trajectory_null_space",
    "collect_trajectories_batch",
    "find_trajectory_null_space",
]
