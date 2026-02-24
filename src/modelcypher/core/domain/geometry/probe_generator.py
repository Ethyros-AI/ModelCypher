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

"""Orthogonal probe generation for alignment coverage.

Provides the OrthogonalProbeGenerator class and token-finding functions
for null-space activation scoring.

Algorithm:
1. Compute current activation rank via SVD
2. Find null space basis (eigenvectors of A^T @ A with small eigenvalues)
3. Score vocabulary tokens by ||U_null^T @ activation||
4. Select numerically significant tokens up to null-space rank
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain.geometry.null_space import (
    OrthogonalProbeResult,
    RankAugmentationResult,
    SequenceProbeResult,
    _get_model_architecture,
    compute_null_space_basis,
    compute_numerical_rank,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision_float32 as _promote_precision,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    regularization_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


class OrthogonalProbeGenerator:
    """Generates probes that activate in orthogonal directions.

    Uses null-space scoring to avoid iterative optimization.
    """

    def __init__(self, backend: "Backend") -> None:
        """Initialize generator.

        Args:
            backend: Backend for tensor operations.
        """
        self.backend = backend

    def generate_orthogonal_probe(
        self,
        model: Any,
        tokenizer: Any,
        U_null: "Array",
        layer_idx: int,
    ) -> OrthogonalProbeResult | None:
        """Generate a probe that activates in the null space.

        Args:
            model: The model to generate probes for.
            tokenizer: Tokenizer for decoding.
            U_null: Null space basis [hidden_dim, null_rank].
            layer_idx: Target layer index for activation.

        Returns:
            OrthogonalProbeResult or None if generation fails.
        """
        top_tokens = find_null_space_tokens_closed_form(
            model=model,
            U_null=U_null,
            layer_idx=layer_idx,
            backend=self.backend,
        )

        if not top_tokens:
            logger.warning("ORTHOGONAL PROBE: No tokens found for null space")
            return None

        token_id, score = top_tokens[0]
        try:
            text = tokenizer.decode([token_id], skip_special_tokens=True)
        except Exception:
            text = "<decode-failed>"

        return OrthogonalProbeResult(
            token_ids=[token_id],
            text=text,
            orthogonal_component_norm=score,
        )

    def augment_rank_iteratively(
        self,
        model: Any,
        tokenizer: Any,
        activations: "Array",
        layer_idx: int,
    ) -> RankAugmentationResult:
        """Augment rank using projection-based token selection."""
        return augment_rank_closed_form(
            model=model,
            tokenizer=tokenizer,
            activations=activations,
            layer_idx=layer_idx,
            backend=self.backend,
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

        # Compute ranks
        src_rank, src_dim = compute_numerical_rank(src_acts, b)
        tgt_rank, tgt_dim = compute_numerical_rank(tgt_acts, b)

        # =====================================================================
        # FULL RANK REQUIREMENT: BOTH models must have full rank
        # =====================================================================
        # For F = pinv(A_src) @ A_tgt:
        # - If A_src has rank < src_dim, then F is undefined for (src_dim - src_rank) directions
        # - If A_tgt has rank < tgt_dim, then we can't span all of target's space
        #
        # Cross-dimensional alignment (e.g., 4096 -> 2048) still requires:
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
) -> "Array":
    """Score tokens by how much they activate null space directions.

    This is the closed-form approach: no iteration, just matrix multiply.

    Args:
        activations_by_token: Activations for each token [vocab_size, hidden_dim].
        U_null: Null space basis [hidden_dim, null_rank].
        backend: Backend for tensor operations.

    Returns:
        Scores [vocab_size] where higher = activates more null space directions.
    """
    b = backend

    acts = activations_by_token
    # Normalize each activation to unit norm (direction only)
    eps = regularization_epsilon(b, acts)
    norms = b.sqrt(b.sum(acts * acts, axis=1, keepdims=True) + eps)
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

    Returns:
        List of (token_id, score) tuples, sorted by score descending.
    """
    b = backend
    logger.info("CLOSED-FORM TOKEN SCORING: Starting for layer %d...", layer_idx)

    try:
        # Get model architecture via protocol
        arch = _get_model_architecture(model)
        embed_module = arch.embed_module
        if embed_module is None:
            logger.error("CLOSED-FORM: Cannot find embedding layer")
            raise RuntimeError("Cannot find embedding layer")

        embed_weight = embed_module.weight
        logger.info("CLOSED-FORM: Found embed module via architecture protocol")

        vocab_size = int(b.shape(embed_weight)[0])
        hidden_dim = int(b.shape(U_null)[0])
        null_rank = int(b.shape(U_null)[1])
        if null_rank <= 0:
            logger.info("CLOSED-FORM TOKEN SCORING: No null space available")
            return []

        batch_size = min(vocab_size, max(1, hidden_dim))

        logger.info(
            "CLOSED-FORM TOKEN SCORING: Scoring %d tokens for layer %d (batch=%d, null_rank=%d)",
            vocab_size,
            layer_idx,
            batch_size,
            null_rank,
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

            for idx, layer in enumerate(arch.layers):
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
            scores = score_tokens_for_null_space(activations, U_null, b)
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

        if not all_scores:
            logger.warning("CLOSED-FORM TOKEN SCORING: No scores computed")
            return []

        max_score = max(all_scores)
        score_eps = regularization_epsilon(b, U_null)
        if max_score <= score_eps:
            logger.info("CLOSED-FORM TOKEN SCORING: No numerically significant scores")
            return []

        threshold = max_score * sqrt_scalar(machine_epsilon(b, U_null), b)
        scored_tokens = [
            (token_id, score)
            for token_id, score in enumerate(all_scores)
            if score >= threshold
        ]
        scored_tokens.sort(key=lambda item: item[1], reverse=True)

        limit = min(null_rank, len(scored_tokens))
        if limit <= 0:
            return []

        top_tokens = scored_tokens[:limit]

        logger.info(
            "CLOSED-FORM TOKEN SCORING: Selected %d tokens (threshold=%.3e, max=%.3e)",
            len(top_tokens),
            threshold,
            max_score,
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
) -> RankAugmentationResult:
    """Augment activation rank using projection-based token selection.

    This replaces gradient ascent with:
    1. Compute null space of current activations
    2. Score ALL tokens by null space projection
    3. Add best tokens until target rank achieved

    Args:
        model: The model.
        tokenizer: Tokenizer for decoding.
        activations: Current activations [n_samples, hidden_dim].
        layer_idx: Target layer.
        backend: Backend for tensor operations.

    Returns:
        RankAugmentationResult with generation statistics.
    """
    b = backend
    current_acts = _promote_precision(activations, b)
    b.eval(current_acts)

    initial_rank, hidden_dim = compute_numerical_rank(current_acts, b)

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
    max_iterations = hidden_dim  # At most one rank gain per iteration

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
        )

        if not top_tokens:
            logger.warning("CLOSED-FORM: No tokens found")
            break

        # Add top tokens and their activations
        new_activations = []
        arch = _get_model_architecture(model)
        for token_id, score in top_tokens:
            # Get activation for this single token
            token_input = b.array([[token_id]], dtype="int32")
            b.eval(token_input)

            # Forward to get activation via architecture protocol
            embed_module = arch.embed_module
            if embed_module is None:
                continue
            h = embed_module(token_input)

            for idx, layer in enumerate(arch.layers):
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
        if new_rank < current_rank:
            logger.warning(
                "CLOSED-FORM: Numerical rank decreased (%d -> %d); clamping to preserve monotonic rank",
                current_rank,
                new_rank,
            )
            new_rank = current_rank
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


def find_null_space_sequences(
    model: Any,
    tokenizer: Any,
    U_null: "Array",
    layer_idx: int,
    backend: "Backend",
) -> list[SequenceProbeResult]:
    """Find token sequences that activate null-space directions.

    Closed-form approach:
    1. Score all vocab tokens by ||U_null^T @ activation||
    2. Select numerically significant tokens up to null-space rank
    3. Return single-token sequences (deterministic, no optimization)

    Args:
        model: The model (must have .model with .embed_tokens/.wte and .layers).
        tokenizer: Tokenizer for decoding.
        U_null: Null space basis [hidden_dim, null_rank] from compute_null_space_basis.
        layer_idx: Target layer for null-space activation.
        backend: Backend for tensor operations.

    Returns:
        List of SequenceProbeResult, sorted by null_space_norm descending.
        Each result contains the token sequence and activation metrics.

    Example:
        >>> U_null = compute_null_space_basis(activations, rank, backend)
        >>> sequences = find_null_space_sequences(
        ...     model, tokenizer, U_null, layer_idx=0, backend=backend
        ... )
        >>> for seq in sequences[:10]:
        ...     print(f"'{seq.text}': null_norm={seq.null_space_norm:.4f}")
    """
    b = backend
    logger.info(
        "SEQUENCE PROBE: Closed-form token scoring for layer %d",
        layer_idx,
    )

    top_tokens = find_null_space_tokens_closed_form(
        model=model,
        U_null=U_null,
        layer_idx=layer_idx,
        backend=b,
    )

    if not top_tokens:
        logger.warning("SEQUENCE PROBE: No tokens found in vocabulary scan")
        return []

    results: list[SequenceProbeResult] = []
    for seed_token_id, seed_score in top_tokens:
        try:
            text = tokenizer.decode([seed_token_id], skip_special_tokens=True)
        except Exception:
            text = f"<tokens:{[seed_token_id]}>"

        results.append(
            SequenceProbeResult(
                token_ids=[seed_token_id],
                text=text,
                null_space_norm=seed_score,
                seed_token_id=seed_token_id,
                gradient_steps=0,
                improvement_ratio=1.0,
            )
        )

    if results:
        best = results[0].null_space_norm
        worst = results[-1].null_space_norm
        mean_norm = sum(r.null_space_norm for r in results) / len(results)
        logger.info(
            "SEQUENCE PROBE: %d tokens selected (best=%.4f, worst=%.4f, mean=%.4f)",
            len(results),
            best,
            worst,
            mean_norm,
        )

    return results


def find_null_space_texts(
    model: Any,
    tokenizer: Any,
    U_null: "Array",
    layer_idx: int,
    backend: "Backend",
) -> list[str]:
    """Convenience wrapper that returns decoded text strings.

    Same as find_null_space_sequences but returns just the text strings,
    suitable for direct use in rank augmentation.

    Args:
        model: The model.
        tokenizer: Tokenizer for decoding.
        U_null: Null space basis [hidden_dim, null_rank].
        layer_idx: Target layer for null-space activation.
        backend: Backend for tensor operations.

    Returns:
        List of decoded text strings that activate null-space directions.
    """
    sequences = find_null_space_sequences(
        model=model,
        tokenizer=tokenizer,
        U_null=U_null,
        layer_idx=layer_idx,
        backend=backend,
    )

    # Return unique, non-empty texts
    texts: list[str] = []
    seen: set[str] = set()

    for seq in sequences:
        text = seq.text.strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)

    return texts


__all__ = [
    "OrthogonalProbeGenerator",
    "validate_full_rank_coverage",
    "score_tokens_for_null_space",
    "find_null_space_tokens_closed_form",
    "augment_rank_closed_form",
    "find_null_space_sequences",
    "find_null_space_texts",
]
