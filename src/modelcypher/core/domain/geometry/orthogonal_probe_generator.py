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

"""Orthogonal probe generation for full-rank alignment.

Generates probes via gradient ascent to maximize activation in directions
orthogonal to the current probe subspace. This enables systematic rank
augmentation until activations span the full hidden dimension.

The algorithm:
1. Compute current activation rank via SVD
2. Find null space basis (eigenvectors of A^T @ A with small eigenvalues)
3. Gradient ascent on embeddings to maximize ||U_null^T @ activation||
4. Discretize continuous embeddings to nearest token IDs

Reference: Exp14 validation (experiments/validation_protocol/exp14_gradient_probe_generation/)
showed ~1.0 rank per generated probe (theoretical optimum).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.precision_utils import (
    _promote_precision_float32 as _promote_precision,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalProbeResult:
    """Result of generating a single orthogonal probe."""

    token_ids: list[int]
    text: str
    orthogonal_component_norm: float


@dataclass
class RankAugmentationResult:
    """Result of iterative rank augmentation."""

    initial_rank: int
    final_rank: int
    hidden_dim: int
    probes_generated: int
    iterations: int
    full_rank_achieved: bool
    generated_probes: list[OrthogonalProbeResult]


def compute_numerical_rank(
    activations: "Array",
    backend: "Backend",
    debug_layer: int | None = None,
) -> tuple[int, int]:
    """Compute numerical rank of activation matrix via SVD.

    Uses threshold sigma_max * sqrt(eps) for numerical stability.

    Args:
        activations: Activation matrix [n_samples, hidden_dim].
        backend: Backend for tensor operations.
        debug_layer: If set, log detailed SVD info for this layer.

    Returns:
        Tuple (rank, hidden_dim).
    """
    b = backend
    acts = _promote_precision(activations, b)
    b.eval(acts)

    shape = b.shape(acts)
    n_samples = int(shape[0])
    hidden_dim = int(shape[1])

    if n_samples == 0 or hidden_dim == 0:
        return 0, hidden_dim

    # SVD to get singular values
    _, S, _ = b.svd(acts, compute_uv=True)
    b.eval(S)

    # Threshold: sigma_max * sqrt(eps)
    eps = machine_epsilon(b, acts)
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

    # DEBUG: Log SVD spectrum for specific layers
    if debug_layer is not None:
        # Use tolist() to get values without numpy
        s_len = int(b.shape(S)[0])
        top_10_indices = min(10, s_len)
        s_top = b.take(S, b.arange(top_10_indices, dtype="int32"), axis=0)
        b.eval(s_top)
        top_10_vals = b.tolist(s_top)

        # Bottom 10 (smallest singular values)
        if s_len >= 10:
            start_idx = s_len - 10
            s_bottom = b.take(S, b.arange(start_idx, s_len, dtype="int32"), axis=0)
            b.eval(s_bottom)
            bottom_10_vals = b.tolist(s_bottom)
        else:
            bottom_10_vals = []

        logger.info(
            "SVD DEBUG Layer %d: n=%d, d=%d, rank=%d, sigma_max=%.6e, threshold=%.6e, "
            "top_10=%s, bottom_10=%s",
            debug_layer, n_samples, hidden_dim, rank, max_s, threshold,
            [f"{v:.4e}" for v in top_10_vals],
            [f"{v:.4e}" for v in bottom_10_vals],
        )

    return rank, hidden_dim


def compute_null_space_basis(
    activations: "Array",
    rank: int,
    backend: "Backend",
) -> "Array | None":
    """Compute basis for orthogonal complement of probe subspace.

    Uses eigendecomposition of A^T @ A to find directions with smallest
    eigenvalues (the null space of the current probe coverage).

    Args:
        activations: Activation matrix [n_samples, hidden_dim].
        rank: Current numerical rank of activations.
        backend: Backend for tensor operations.

    Returns:
        Null space basis [hidden_dim, null_rank], or None if already full rank.
    """
    b = backend
    logger.info("NULL SPACE: Computing basis for rank=%d activations...", rank)

    try:
        acts = _promote_precision(activations, b)
        b.eval(acts)

        hidden_dim = int(b.shape(acts)[1])
        null_rank = hidden_dim - rank
        logger.info("NULL SPACE: hidden_dim=%d, null_rank=%d", hidden_dim, null_rank)

        if null_rank <= 0:
            logger.info("NULL SPACE: Already full rank, returning None")
            return None  # Already full rank

        # Covariance in hidden_dim space: A^T @ A
        logger.info("NULL SPACE: Computing covariance matrix...")
        cov = b.matmul(b.transpose(acts), acts)
        b.eval(cov)

        # Eigendecomposition (eigh returns ascending eigenvalues)
        logger.info("NULL SPACE: Computing eigendecomposition...")
        eigvals, eigvecs = b.eigh(cov)
        b.eval(eigvals, eigvecs)

        # First null_rank eigenvectors (smallest eigenvalues) form the null space
        U_null = eigvecs[:, :null_rank]
        b.eval(U_null)

        logger.info("NULL SPACE: Computed basis with shape %s", b.shape(U_null))
        return U_null

    except Exception as e:
        logger.error("NULL SPACE: Computation FAILED: %s: %s", type(e).__name__, e)
        import traceback
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise


class OrthogonalProbeGenerator:
    """Generates probes that activate in orthogonal directions.

    Uses gradient ascent on continuous embeddings to find token sequences
    that maximize activation in the null space of current probes.
    """

    def __init__(
        self,
        backend: "Backend",
        n_steps: int = 30,
        learning_rate: float = 0.05,
        seq_len: int = 10,
    ) -> None:
        """Initialize generator.

        Args:
            backend: Backend for tensor operations.
            n_steps: Number of gradient ascent steps per probe.
            learning_rate: Step size for gradient ascent.
            seq_len: Length of generated token sequences.
        """
        self.backend = backend
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.seq_len = seq_len

    def generate_orthogonal_probe(
        self,
        model: Any,
        tokenizer: Any,
        U_null: "Array",
        layer_idx: int,
        get_layer_activation_fn: Callable[..., "Array"],
        seed_tokens: list[int] | None = None,
    ) -> OrthogonalProbeResult | None:
        """Generate a probe that activates in the null space.

        Uses gradient ascent on continuous embeddings, then discretizes
        to nearest tokens.

        Args:
            model: The model to generate probes for.
            tokenizer: Tokenizer for decoding.
            U_null: Null space basis [hidden_dim, null_rank].
            layer_idx: Target layer index for activation.
            get_layer_activation_fn: Function(model, input_ids, layer_idx) -> activation.
            seed_tokens: Optional initial token IDs.

        Returns:
            OrthogonalProbeResult or None if generation fails.
        """
        b = self.backend

        # Get inner model and embedding weights
        inner = model.model if hasattr(model, "model") else model

        if hasattr(inner, "embed_tokens"):
            embed_weight = inner.embed_tokens.weight
        elif hasattr(inner, "wte"):
            embed_weight = inner.wte.weight
        else:
            logger.warning("Cannot find embedding layer for probe generation")
            return None

        vocab_size = int(b.shape(embed_weight)[0])
        embed_dim = int(b.shape(embed_weight)[1])

        # Initialize token IDs
        if seed_tokens is not None:
            init_ids = b.array(seed_tokens[: self.seq_len])
            current_len = int(b.shape(init_ids)[0])
            if current_len < self.seq_len:
                # Pad with zeros
                pad = b.zeros((self.seq_len - current_len,), dtype="int32")
                init_ids = b.concatenate([init_ids, pad], axis=0)
        else:
            # Random initialization avoiding edge tokens
            init_ids = b.random_randint(100, vocab_size - 100, (self.seq_len,))
        b.eval(init_ids)

        # Get initial embeddings
        embeds = b.take(embed_weight, init_ids, axis=0)  # [seq_len, embed_dim]
        b.eval(embeds)

        # Gradient ascent to maximize projection onto null space
        for step in range(self.n_steps):
            # Objective: maximize ||U_null^T @ activation||
            # We use value_and_grad to compute gradient

            def objective(e: "Array") -> "Array":
                # Forward through layers to get activation
                h = b.expand_dims(e, axis=0)  # [1, seq_len, embed_dim]

                # Forward through layers up to target
                for idx, layer in enumerate(inner.layers):
                    if idx > layer_idx:
                        break
                    result = layer(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result

                # Mean pool over sequence
                activation = b.mean(h, axis=(0, 1))  # [hidden_dim]

                # Project onto null space
                proj = b.matmul(U_null, b.matmul(b.transpose(U_null), activation))

                # Return negative norm (we want to maximize)
                norm_sq = b.sum(proj * proj)
                return -b.sqrt(norm_sq + 1e-8)

            # Compute gradient
            loss_and_grad = b.value_and_grad(objective)
            loss, grad = loss_and_grad(embeds)
            b.eval(loss, grad)

            # Gradient descent (actually ascent since objective is negated)
            embeds = embeds - self.learning_rate * grad
            b.eval(embeds)

            if step % 10 == 0:
                logger.debug(
                    "Step %d: null_proj_norm = %.4f", step, -float(b.to_scalar(loss))
                )

        # Discretize: find nearest tokens for each position
        final_ids = []
        for pos in range(self.seq_len):
            embed_pos = embeds[pos]  # [embed_dim]
            # Distance to all vocab embeddings
            diff = embed_weight - b.expand_dims(embed_pos, axis=0)
            dists = b.sum(diff * diff, axis=1)
            b.eval(dists)
            nearest_id = int(b.to_scalar(b.argmin(dists)))
            final_ids.append(nearest_id)

        # Decode to text
        try:
            text = tokenizer.decode(final_ids)
        except Exception:
            text = "<decode-failed>"

        # Compute final orthogonal component norm
        final_input = b.array([final_ids])
        b.eval(final_input)
        activation = get_layer_activation_fn(model, final_input, layer_idx)
        if activation is not None:
            b.eval(activation)
            proj = b.matmul(U_null, b.matmul(b.transpose(U_null), activation))
            norm = b.sqrt(b.sum(proj * proj))
            b.eval(norm)
            orthogonal_norm = float(b.to_scalar(norm))
        else:
            orthogonal_norm = 0.0

        return OrthogonalProbeResult(
            token_ids=final_ids,
            text=text,
            orthogonal_component_norm=orthogonal_norm,
        )

    def augment_rank_iteratively(
        self,
        model: Any,
        tokenizer: Any,
        activations: "Array",
        layer_idx: int,
        get_layer_activation_fn: Callable[..., "Array"],
        batch_size: int = 10,
        max_iterations: int = 100,
    ) -> RankAugmentationResult:
        """Generate probes in batches until full rank is achieved.

        Args:
            model: The model to generate probes for.
            tokenizer: Tokenizer for decoding.
            activations: Current activation matrix [n_samples, hidden_dim].
            layer_idx: Target layer index.
            get_layer_activation_fn: Function(model, input_ids, layer_idx) -> activation.
            batch_size: Number of probes to generate per iteration.
            max_iterations: Maximum number of augmentation iterations.

        Returns:
            RankAugmentationResult with generation statistics.
        """
        b = self.backend
        current_acts = _promote_precision(activations, b)
        b.eval(current_acts)

        initial_rank, hidden_dim = compute_numerical_rank(current_acts, b)
        logger.info(
            "RANK AUGMENTATION: Starting at rank %d/%d (%.1f%% coverage)",
            initial_rank,
            hidden_dim,
            100.0 * initial_rank / hidden_dim,
        )

        if initial_rank >= hidden_dim:
            logger.info("RANK AUGMENTATION: Already at full rank!")
            return RankAugmentationResult(
                initial_rank=initial_rank,
                final_rank=initial_rank,
                hidden_dim=hidden_dim,
                probes_generated=0,
                iterations=0,
                full_rank_achieved=True,
                generated_probes=[],
            )

        generated_probes: list[OrthogonalProbeResult] = []
        total_generated = 0

        for iteration in range(max_iterations):
            # Compute current rank
            current_rank, _ = compute_numerical_rank(current_acts, b)

            if current_rank >= hidden_dim:
                logger.info(
                    "RANK AUGMENTATION: Full rank achieved at iteration %d", iteration
                )
                break

            # Compute null space basis
            U_null = compute_null_space_basis(current_acts, current_rank, b)
            if U_null is None:
                logger.info("RANK AUGMENTATION: No null space (full rank)")
                break

            null_dim = int(b.shape(U_null)[1])
            logger.info(
                "RANK AUGMENTATION: Iteration %d, rank=%d/%d, null_dim=%d",
                iteration,
                current_rank,
                hidden_dim,
                null_dim,
            )

            # Generate batch of probes
            batch_activations = []
            for _ in range(min(batch_size, null_dim)):
                probe_result = self.generate_orthogonal_probe(
                    model=model,
                    tokenizer=tokenizer,
                    U_null=U_null,
                    layer_idx=layer_idx,
                    get_layer_activation_fn=get_layer_activation_fn,
                )

                if probe_result is not None:
                    generated_probes.append(probe_result)
                    total_generated += 1

                    # Get activation for this probe
                    input_ids = b.array([probe_result.token_ids])
                    b.eval(input_ids)
                    activation = get_layer_activation_fn(model, input_ids, layer_idx)
                    if activation is not None:
                        b.eval(activation)
                        batch_activations.append(activation)

            if batch_activations:
                # Stack new activations and concatenate with current
                new_acts = b.stack(batch_activations, axis=0)
                b.eval(new_acts)
                current_acts = b.concatenate([current_acts, new_acts], axis=0)
                b.eval(current_acts)

            # Progress check
            new_rank, _ = compute_numerical_rank(current_acts, b)
            rank_increase = new_rank - current_rank
            logger.info(
                "RANK AUGMENTATION: Generated %d probes, rank increased by %d (%d -> %d)",
                len(batch_activations),
                rank_increase,
                current_rank,
                new_rank,
            )

            if rank_increase == 0:
                logger.warning(
                    "RANK AUGMENTATION: No rank increase in iteration %d, may be stuck",
                    iteration,
                )

        final_rank, _ = compute_numerical_rank(current_acts, b)
        full_rank_achieved = final_rank >= hidden_dim

        logger.info(
            "RANK AUGMENTATION: Complete. Final rank=%d/%d (%.1f%%), "
            "generated=%d probes in %d iterations",
            final_rank,
            hidden_dim,
            100.0 * final_rank / hidden_dim,
            total_generated,
            iteration + 1,
        )

        return RankAugmentationResult(
            initial_rank=initial_rank,
            final_rank=final_rank,
            hidden_dim=hidden_dim,
            probes_generated=total_generated,
            iterations=iteration + 1,
            full_rank_achieved=full_rank_achieved,
            generated_probes=generated_probes,
        )


def validate_full_rank_coverage(
    source_activations: dict[int, "Array"],
    target_activations: dict[int, "Array"],
    backend: "Backend",
) -> dict[int, dict]:
    """Validate rank coverage per layer.

    Full rank is achieved when both source and target have matching numerical rank.
    The "effective dimensionality" is bounded by the target's numerical rank, which
    reflects the layer's intrinsic geometry. Middle layers may have lower effective
    dimensionality due to representation compression.

    Args:
        source_activations: Source activations by layer.
        target_activations: Target activations by layer.
        backend: Backend for tensor operations.

    Returns:
        Dict mapping layer -> {rank, dim, deficit, full_rank_achieved}.
    """
    b = backend
    results: dict[int, dict] = {}

    # Get common layers
    common_layers = set(source_activations.keys()) & set(target_activations.keys())

    for layer_idx in sorted(common_layers):
        src_acts = source_activations[layer_idx]
        tgt_acts = target_activations[layer_idx]

        # Stack if list
        if isinstance(src_acts, list):
            src_acts = b.stack(src_acts, axis=0)
        if isinstance(tgt_acts, list):
            tgt_acts = b.stack(tgt_acts, axis=0)

        b.eval(src_acts, tgt_acts)

        # Compute ranks - pass debug_layer for layers 0 and 6 to diagnose rank bug
        debug_layer = layer_idx if layer_idx in (0, 6) else None
        src_rank, src_dim = compute_numerical_rank(src_acts, b, debug_layer=debug_layer)
        tgt_rank, tgt_dim = compute_numerical_rank(tgt_acts, b)

        # =====================================================================
        # FULL RANK REQUIREMENT: BOTH models must have full rank
        # =====================================================================
        # For F = pinv(A_src) @ A_tgt:
        # - If A_src has rank < src_dim, then F is undefined for (src_dim - src_rank) directions
        # - If A_tgt has rank < tgt_dim, then we can't span all of target's space
        #
        # Cross-dimensional alignment (e.g., 4096 → 2048) still requires:
        # - Source: full rank (src_rank == src_dim) so F covers all source directions
        # - Target: full rank (tgt_rank == tgt_dim) so we span the full target space
        #
        # "Dark" dimensions in source (where probes don't activate) get arbitrary
        # coefficients in F, which causes garbage in the transplanted weights.
        # =====================================================================
        src_full_rank = src_rank >= src_dim
        tgt_full_rank = tgt_rank >= tgt_dim
        full_rank = src_full_rank and tgt_full_rank

        # Deficits for each model
        src_deficit = src_dim - src_rank
        tgt_deficit = tgt_dim - tgt_rank
        total_deficit = src_deficit + tgt_deficit

        # For backward compatibility, keep alignment_rank as min
        alignment_rank = min(src_rank, tgt_rank)

        results[layer_idx] = {
            "source_rank": src_rank,
            "source_dim": src_dim,
            "source_deficit": src_deficit,
            "source_full_rank": src_full_rank,
            "target_rank": tgt_rank,
            "target_dim": tgt_dim,
            "target_deficit": tgt_deficit,
            "target_full_rank": tgt_full_rank,
            "alignment_rank": alignment_rank,
            "deficit": total_deficit,
            "full_rank_achieved": full_rank,
            "coverage_ratio": (src_rank / src_dim * tgt_rank / tgt_dim) if src_dim > 0 and tgt_dim > 0 else 1.0,
        }

        if not full_rank:
            logger.info(
                "RANK COVERAGE Layer %d: src=%d/%d (%s), tgt=%d/%d (%s), total_deficit=%d",
                layer_idx,
                src_rank, src_dim, "FULL" if src_full_rank else "DEFICIT",
                tgt_rank, tgt_dim, "FULL" if tgt_full_rank else "DEFICIT",
                total_deficit,
            )
        else:
            logger.debug(
                "RANK COVERAGE Layer %d: FULL RANK (src=%d/%d, tgt=%d/%d)",
                layer_idx,
                src_rank, src_dim,
                tgt_rank, tgt_dim,
            )

    return results


def score_tokens_for_null_space(
    activations_by_token: "Array",
    U_null: "Array",
    backend: "Backend",
    normalize: bool = True,
) -> "Array":
    """Score tokens by how much they activate null space directions.

    This is the closed-form approach: no iteration, just matrix multiply.

    Args:
        activations_by_token: Activations for each token [vocab_size, hidden_dim].
        U_null: Null space basis [hidden_dim, null_rank].
        backend: Backend for tensor operations.
        normalize: If True, normalize activations to unit vectors first.
            This scores by DIRECTION not magnitude.

    Returns:
        Scores [vocab_size] where higher = activates more null space directions.
    """
    b = backend

    acts = activations_by_token
    if normalize:
        # Normalize each activation to unit norm (direction only)
        norms = b.sqrt(b.sum(acts * acts, axis=1, keepdims=True) + 1e-8)
        acts = acts / norms
        b.eval(acts)

    # Project each token's activation onto null space
    # projections[i] = U_null.T @ activations[i] -> [null_rank]
    # We want: activations @ U_null -> [vocab_size, null_rank]
    projections = b.matmul(acts, U_null)
    b.eval(projections)

    # Score = L2 norm of projection (how much this token activates null directions)
    scores = b.sqrt(b.sum(projections * projections, axis=1))
    b.eval(scores)

    return scores


def find_null_space_tokens_closed_form(
    model: Any,
    U_null: "Array",
    layer_idx: int,
    backend: "Backend",
    top_k: int = 100,
    batch_size: int = 512,
    normalize: bool = True,
) -> list[tuple[int, float]]:
    """Find tokens that activate null space directions - CLOSED FORM.

    Instead of gradient ascent, we:
    1. Forward ALL tokens through the model to target layer
    2. Normalize activations to unit vectors (direction only)
    3. Score each token by ||U_null.T @ activation||
    4. Select top-k tokens

    This is O(vocab_size) forward passes (batched), but:
    - Deterministic
    - No iteration
    - Guaranteed to find best tokens in vocabulary

    Args:
        model: The model.
        U_null: Null space basis [hidden_dim, null_rank].
        layer_idx: Target layer for activations.
        backend: Backend for tensor operations.
        top_k: Number of top tokens to return.
        batch_size: Batch size for forward passes.
        normalize: If True, normalize activations to unit vectors (recommended).

    Returns:
        List of (token_id, score) tuples, sorted by score descending.
    """
    b = backend
    logger.info("CLOSED-FORM TOKEN SCORING: Starting for layer %d, top_k=%d...", layer_idx, top_k)

    try:
        # Get model internals
        inner = model.model if hasattr(model, "model") else model

        if hasattr(inner, "embed_tokens"):
            embed_weight = inner.embed_tokens.weight
            logger.info("CLOSED-FORM: Found embed_tokens")
        elif hasattr(inner, "wte"):
            embed_weight = inner.wte.weight
            logger.info("CLOSED-FORM: Found wte")
        else:
            logger.error("CLOSED-FORM: Cannot find embedding layer")
            raise RuntimeError("Cannot find embedding layer")

        vocab_size = int(b.shape(embed_weight)[0])

        logger.info(
            "CLOSED-FORM TOKEN SCORING: Scoring %d tokens for layer %d",
            vocab_size,
            layer_idx,
        )

        all_scores: list[float] = []

        # Process vocabulary in batches
        for start in range(0, vocab_size, batch_size):
            end = min(start + batch_size, vocab_size)
            batch_tokens = list(range(start, end))

            # Get embeddings for this batch
            token_indices = b.array(batch_tokens, dtype="int32")
            embeddings = b.take(embed_weight, token_indices, axis=0)  # [batch, embed_dim]
            b.eval(embeddings)

            # Forward through layers to target layer
            # We need to process each token as a single-token sequence
            h = b.expand_dims(embeddings, axis=1)  # [batch, 1, embed_dim]

            for idx, layer in enumerate(inner.layers):
                if idx > layer_idx:
                    break
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

            # h is now [batch, 1, hidden_dim]
            # Squeeze to [batch, hidden_dim]
            activations = b.squeeze(h, axis=1)
            b.eval(activations)

            # Score against null space (normalized to compare directions, not magnitudes)
            scores = score_tokens_for_null_space(activations, U_null, b, normalize=normalize)
            b.eval(scores)

            batch_scores = b.tolist(scores)
            all_scores.extend(batch_scores)

            if (end - start) == batch_size and end < vocab_size:
                logger.debug(
                    "CLOSED-FORM: Processed %d/%d tokens (%.1f%%)",
                    end,
                    vocab_size,
                    100.0 * end / vocab_size,
                )

        # Sort by score and return top-k
        indexed_scores = [(i, s) for i, s in enumerate(all_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_tokens = indexed_scores[:top_k]

        logger.info(
            "CLOSED-FORM TOKEN SCORING: Top token score=%.4f, %dth token score=%.4f",
            top_tokens[0][1] if top_tokens else 0.0,
            min(top_k, len(top_tokens)),
            top_tokens[-1][1] if top_tokens else 0.0,
        )

        return top_tokens

    except Exception as e:
        logger.error("CLOSED-FORM TOKEN SCORING FAILED: %s: %s", type(e).__name__, e)
        import traceback
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        raise


def augment_rank_closed_form(
    model: Any,
    tokenizer: Any,
    activations: "Array",
    layer_idx: int,
    backend: "Backend",
    target_rank: int | None = None,
    batch_size: int = 512,
) -> RankAugmentationResult:
    """Augment activation rank using closed-form token selection.

    This replaces gradient ascent with:
    1. Compute null space of current activations
    2. Score ALL tokens by null space projection (closed-form)
    3. Add best tokens until target rank achieved

    Args:
        model: The model.
        tokenizer: Tokenizer for decoding.
        activations: Current activations [n_samples, hidden_dim].
        layer_idx: Target layer.
        backend: Backend for tensor operations.
        target_rank: Target rank (default: full rank = hidden_dim).
        batch_size: Batch size for token scoring.

    Returns:
        RankAugmentationResult with generation statistics.
    """
    b = backend
    current_acts = _promote_precision(activations, b)
    b.eval(current_acts)

    initial_rank, hidden_dim = compute_numerical_rank(current_acts, b)

    if target_rank is None:
        target_rank = hidden_dim

    logger.info(
        "CLOSED-FORM RANK AUGMENTATION: Starting at rank %d/%d, target=%d",
        initial_rank,
        hidden_dim,
        target_rank,
    )

    if initial_rank >= target_rank:
        logger.info("CLOSED-FORM: Already at target rank!")
        return RankAugmentationResult(
            initial_rank=initial_rank,
            final_rank=initial_rank,
            hidden_dim=hidden_dim,
            probes_generated=0,
            iterations=1,
            full_rank_achieved=initial_rank >= hidden_dim,
            generated_probes=[],
        )

    generated_probes: list[OrthogonalProbeResult] = []
    current_rank = initial_rank
    iteration = 0
    max_iterations = 100  # Safety limit

    while current_rank < target_rank and iteration < max_iterations:
        iteration += 1

        # Compute null space
        U_null = compute_null_space_basis(current_acts, current_rank, b)
        if U_null is None:
            logger.info("CLOSED-FORM: No null space remaining")
            break

        null_dim = int(b.shape(U_null)[1])
        logger.info(
            "CLOSED-FORM: Iteration %d, rank=%d/%d, null_dim=%d",
            iteration,
            current_rank,
            target_rank,
            null_dim,
        )

        # Find best tokens for null space (closed-form)
        top_tokens = find_null_space_tokens_closed_form(
            model=model,
            U_null=U_null,
            layer_idx=layer_idx,
            backend=b,
            top_k=min(null_dim, 50),  # Don't need more than null_dim tokens
            batch_size=batch_size,
        )

        if not top_tokens:
            logger.warning("CLOSED-FORM: No tokens found")
            break

        # Add top tokens and their activations
        new_activations = []
        for token_id, score in top_tokens:
            # Get activation for this single token
            token_input = b.array([[token_id]], dtype="int32")
            b.eval(token_input)

            # Forward to get activation
            inner = model.model if hasattr(model, "model") else model
            if hasattr(inner, "embed_tokens"):
                h = inner.embed_tokens(token_input)
            elif hasattr(inner, "wte"):
                h = inner.wte(token_input)
            else:
                continue

            for idx, layer in enumerate(inner.layers):
                if idx > layer_idx:
                    break
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

            # Mean pool (single token, so just squeeze)
            activation = b.squeeze(h, axis=(0, 1))
            b.eval(activation)
            new_activations.append(activation)

            # Decode token
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                text = f"<token-{token_id}>"

            generated_probes.append(
                OrthogonalProbeResult(
                    token_ids=[token_id],
                    text=text,
                    orthogonal_component_norm=score,
                )
            )

            # Check if we've reached target rank
            if len(new_activations) >= null_dim:
                break

        if new_activations:
            # Stack and concatenate
            new_acts_stacked = b.stack(new_activations, axis=0)
            b.eval(new_acts_stacked)
            current_acts = b.concatenate([current_acts, new_acts_stacked], axis=0)
            b.eval(current_acts)

        # Check new rank
        new_rank, _ = compute_numerical_rank(current_acts, b)
        rank_increase = new_rank - current_rank

        logger.info(
            "CLOSED-FORM: Added %d tokens, rank %d -> %d (+%d)",
            len(new_activations),
            current_rank,
            new_rank,
            rank_increase,
        )

        if rank_increase == 0:
            logger.warning("CLOSED-FORM: No rank increase, stopping")
            break

        current_rank = new_rank

    final_rank, _ = compute_numerical_rank(current_acts, b)
    full_rank_achieved = final_rank >= hidden_dim

    logger.info(
        "CLOSED-FORM RANK AUGMENTATION: Complete. rank=%d/%d (%.1f%%), "
        "generated=%d probes in %d iterations",
        final_rank,
        hidden_dim,
        100.0 * final_rank / hidden_dim,
        len(generated_probes),
        iteration,
    )

    return RankAugmentationResult(
        initial_rank=initial_rank,
        final_rank=final_rank,
        hidden_dim=hidden_dim,
        probes_generated=len(generated_probes),
        iterations=iteration,
        full_rank_achieved=full_rank_achieved,
        generated_probes=generated_probes,
    )


__all__ = [
    "OrthogonalProbeGenerator",
    "OrthogonalProbeResult",
    "RankAugmentationResult",
    "compute_numerical_rank",
    "compute_null_space_basis",
    "validate_full_rank_coverage",
    "score_tokens_for_null_space",
    "find_null_space_tokens_closed_form",
    "augment_rank_closed_form",
]
