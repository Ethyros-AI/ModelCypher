#!/usr/bin/env python3
"""Experiment 16: Full Rank at ALL Layers - No Exceptions.

This experiment achieves full rank (rank = hidden_dim) at EVERY layer.
No partial coverage. No moving forward until the map is complete.

Strategy:
1. Single tokens first (closed-form, fast)
2. When single tokens are exhausted, use gradient-based sequence generation
3. Continue until rank = hidden_dim

We will run the whole tokenizer if we have to.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import mlx.core as mx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str):
    """Load MLX model and tokenizer."""
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())
    return model, tokenizer


def normalize_activation(act):
    """Normalize activation to unit norm."""
    norm = mx.sqrt(mx.sum(act * act) + 1e-8)
    return act / norm


def get_layer_activation(model, input_ids, layer_idx):
    """Get mean-pooled activation at specific layer."""
    inner = model.model if hasattr(model, "model") else model

    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        return None

    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    return pooled


def get_sequence_activation(model, token_ids_list, layer_idx):
    """Get activation for a sequence of tokens."""
    input_ids = mx.array([token_ids_list])
    mx.eval(input_ids)
    return get_layer_activation(model, input_ids, layer_idx)


def compute_rank(activations, backend):
    """Compute numerical rank via SVD."""
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import compute_numerical_rank
    return compute_numerical_rank(activations, backend)


def compute_null_space(activations, rank, backend):
    """Compute null space basis."""
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import compute_null_space_basis
    return compute_null_space_basis(activations, rank, backend)


def score_single_tokens_batch(model, U_null, layer_idx, backend, batch_size=512):
    """Score all single tokens against null space."""
    inner = model.model if hasattr(model, "model") else model
    embed_weight = inner.embed_tokens.weight
    vocab_size = int(embed_weight.shape[0])

    all_scores = []

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        batch_tokens = list(range(start, end))

        token_indices = mx.array(batch_tokens)
        embeddings = mx.take(embed_weight, token_indices, axis=0)
        mx.eval(embeddings)

        h = mx.expand_dims(embeddings, axis=1)

        for idx, layer in enumerate(inner.layers):
            if idx > layer_idx:
                break
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

        activations = mx.squeeze(h, axis=1)
        mx.eval(activations)

        # Normalize
        norms = mx.sqrt(mx.sum(activations * activations, axis=1, keepdims=True) + 1e-8)
        activations = activations / norms
        mx.eval(activations)

        # Project onto null space
        projections = mx.matmul(activations, U_null)
        scores = mx.sqrt(mx.sum(projections * projections, axis=1))
        mx.eval(scores)

        all_scores.extend(scores.tolist())

    return all_scores


def generate_sequence_gradient(model, U_null, layer_idx, seq_len=5, n_steps=50, lr=0.1):
    """Generate a token sequence via gradient ascent to maximize null space activation."""
    inner = model.model if hasattr(model, "model") else model
    embed_weight = inner.embed_tokens.weight
    vocab_size = int(embed_weight.shape[0])
    embed_dim = int(embed_weight.shape[1])

    # Random initialization
    init_ids = mx.random.randint(100, vocab_size - 100, shape=(seq_len,))
    mx.eval(init_ids)
    embeds = mx.take(embed_weight, init_ids, axis=0)
    mx.eval(embeds)

    for step in range(n_steps):
        def objective(e):
            h = mx.expand_dims(e, axis=0)
            for idx, layer in enumerate(inner.layers):
                if idx > layer_idx:
                    break
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result

            activation = mx.mean(h, axis=(0, 1))
            # Normalize
            activation = activation / (mx.sqrt(mx.sum(activation * activation)) + 1e-8)
            # Project onto null space
            proj = mx.matmul(activation, U_null)
            norm_sq = mx.sum(proj * proj)
            return -mx.sqrt(norm_sq + 1e-8)

        loss_and_grad = mx.value_and_grad(objective)
        loss, grad = loss_and_grad(embeds)
        mx.eval(loss, grad)

        embeds = embeds - lr * grad
        mx.eval(embeds)

    # Discretize to nearest tokens
    final_ids = []
    for pos in range(seq_len):
        embed_pos = embeds[pos]
        diff = embed_weight - mx.expand_dims(embed_pos, axis=0)
        dists = mx.sum(diff * diff, axis=1)
        mx.eval(dists)
        nearest_id = int(mx.argmin(dists).item())
        final_ids.append(nearest_id)

    return final_ids


def achieve_full_rank_layer(
    model,
    tokenizer,
    layer_idx,
    initial_activations,
    backend,
    max_single_token_rounds=50,
    max_sequence_rounds=200,
):
    """Achieve full rank at a single layer. No stopping until done."""

    current_acts = initial_activations
    mx.eval(current_acts)

    rank, hidden_dim = compute_rank(current_acts, backend)
    logger.info("Layer %d: Starting at rank %d/%d", layer_idx, rank, hidden_dim)

    if rank >= hidden_dim:
        return current_acts, {"tokens_used": 0, "sequences_used": 0, "final_rank": rank}

    used_tokens = set()
    tokens_added = 0
    sequences_added = 0

    # Phase 1: Single tokens (closed-form)
    logger.info("Layer %d: Phase 1 - Single token scoring", layer_idx)

    for round_num in range(max_single_token_rounds):
        rank, _ = compute_rank(current_acts, backend)
        if rank >= hidden_dim:
            logger.info("Layer %d: Full rank achieved with single tokens!", layer_idx)
            break

        deficit = hidden_dim - rank
        U_null = compute_null_space(current_acts, rank, backend)
        if U_null is None:
            break

        # Score all tokens
        all_scores = score_single_tokens_batch(model, U_null, layer_idx, backend)

        # Get top tokens not yet used
        indexed = [(i, s) for i, s in enumerate(all_scores) if i not in used_tokens]
        indexed.sort(key=lambda x: x[1], reverse=True)

        if not indexed or indexed[0][1] < 0.01:
            logger.info("Layer %d: Single tokens exhausted (best score %.4f)", layer_idx, indexed[0][1] if indexed else 0)
            break

        # Add top tokens
        added_this_round = 0
        for token_id, score in indexed[:min(50, deficit)]:
            act = get_layer_activation(model, mx.array([[token_id]]), layer_idx)
            if act is None:
                continue

            act = normalize_activation(act)
            mx.eval(act)

            current_acts = mx.concatenate([current_acts, mx.expand_dims(act, 0)], axis=0)
            mx.eval(current_acts)

            used_tokens.add(token_id)
            tokens_added += 1
            added_this_round += 1

        new_rank, _ = compute_rank(current_acts, backend)
        logger.info(
            "Layer %d: Round %d, added %d tokens, rank %d -> %d/%d (%.1f%%)",
            layer_idx, round_num + 1, added_this_round, rank, new_rank, hidden_dim,
            100 * new_rank / hidden_dim
        )

        if new_rank == rank:
            logger.info("Layer %d: No rank increase, moving to sequences", layer_idx)
            break

    # Check if done
    rank, _ = compute_rank(current_acts, backend)
    if rank >= hidden_dim:
        return current_acts, {
            "tokens_used": tokens_added,
            "sequences_used": 0,
            "final_rank": rank,
            "method": "single_tokens",
        }

    # Phase 2: Multi-token sequences via gradient
    logger.info("Layer %d: Phase 2 - Gradient sequence generation (rank=%d/%d)", layer_idx, rank, hidden_dim)

    stall_count = 0
    max_stall = 20

    for seq_round in range(max_sequence_rounds):
        rank, _ = compute_rank(current_acts, backend)
        if rank >= hidden_dim:
            logger.info("Layer %d: Full rank achieved with sequences!", layer_idx)
            break

        U_null = compute_null_space(current_acts, rank, backend)
        if U_null is None:
            break

        # Try different sequence lengths
        for seq_len in [2, 3, 5, 7, 10]:
            sequence = generate_sequence_gradient(
                model, U_null, layer_idx,
                seq_len=seq_len, n_steps=30, lr=0.1
            )

            act = get_sequence_activation(model, sequence, layer_idx)
            if act is None:
                continue

            act = normalize_activation(act)
            mx.eval(act)

            # Check if this actually helps
            test_acts = mx.concatenate([current_acts, mx.expand_dims(act, 0)], axis=0)
            mx.eval(test_acts)
            test_rank, _ = compute_rank(test_acts, backend)

            if test_rank > rank:
                current_acts = test_acts
                sequences_added += 1
                logger.info(
                    "Layer %d: Seq round %d, len=%d, rank %d -> %d/%d (%.1f%%)",
                    layer_idx, seq_round + 1, seq_len, rank, test_rank, hidden_dim,
                    100 * test_rank / hidden_dim
                )
                stall_count = 0
                break
        else:
            stall_count += 1
            if stall_count >= max_stall:
                logger.warning("Layer %d: Stalled at rank %d/%d after %d rounds", layer_idx, rank, hidden_dim, stall_count)
                # Don't give up - try with longer sequences and more steps
                logger.info("Layer %d: Trying harder with longer sequences...", layer_idx)
                for seq_len in [15, 20, 30]:
                    sequence = generate_sequence_gradient(
                        model, U_null, layer_idx,
                        seq_len=seq_len, n_steps=100, lr=0.05
                    )
                    act = get_sequence_activation(model, sequence, layer_idx)
                    if act is None:
                        continue
                    act = normalize_activation(act)
                    mx.eval(act)
                    test_acts = mx.concatenate([current_acts, mx.expand_dims(act, 0)], axis=0)
                    mx.eval(test_acts)
                    test_rank, _ = compute_rank(test_acts, backend)
                    if test_rank > rank:
                        current_acts = test_acts
                        sequences_added += 1
                        logger.info("Layer %d: Breakthrough! len=%d, rank %d -> %d", layer_idx, seq_len, rank, test_rank)
                        stall_count = 0
                        break
                else:
                    logger.error("Layer %d: FAILED to achieve full rank. Stuck at %d/%d", layer_idx, rank, hidden_dim)
                    break

    final_rank, _ = compute_rank(current_acts, backend)
    return current_acts, {
        "tokens_used": tokens_added,
        "sequences_used": sequences_added,
        "final_rank": final_rank,
        "method": "single_tokens" if sequences_added == 0 else "hybrid",
        "full_rank": final_rank >= hidden_dim,
    }


def run_experiment(model_path: str, output_dir: Path):
    """Run full rank experiment on all layers."""
    from modelcypher.backends.mlx_backend import MLXBackend

    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_path)
    backend = MLXBackend()

    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)
    hidden_dim = int(inner.layers[0].self_attn.q_proj.weight.shape[0])
    vocab_size = int(inner.embed_tokens.weight.shape[0])

    logger.info("Model: %d layers, hidden_dim=%d, vocab_size=%d", n_layers, hidden_dim, vocab_size)

    results = {
        "model_path": model_path,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "layers": {},
    }

    # Test on middle layers that failed before (15, 22) plus a control (7)
    test_layers = [7, 15, 22]

    for layer_idx in test_layers:
        logger.info("=" * 70)
        logger.info("LAYER %d - ACHIEVING FULL RANK", layer_idx)
        logger.info("=" * 70)

        # Collect initial activations (100 probes)
        probe_texts = [f"The quick brown fox {i}" for i in range(100)]
        initial_acts = []
        for text in probe_texts:
            tokens = tokenizer.encode(text)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)
            act = get_layer_activation(model, input_ids, layer_idx)
            if act is not None:
                act = normalize_activation(act)
                mx.eval(act)
                initial_acts.append(act)

        initial_acts = mx.stack(initial_acts, axis=0)
        mx.eval(initial_acts)

        start_time = time.time()
        final_acts, layer_result = achieve_full_rank_layer(
            model, tokenizer, layer_idx, initial_acts, backend
        )
        elapsed = time.time() - start_time

        layer_result["elapsed_seconds"] = elapsed
        results["layers"][layer_idx] = layer_result

        logger.info(
            "LAYER %d COMPLETE: rank=%d/%d, tokens=%d, sequences=%d, time=%.1fs, FULL_RANK=%s",
            layer_idx,
            layer_result["final_rank"],
            hidden_dim,
            layer_result["tokens_used"],
            layer_result["sequences_used"],
            elapsed,
            layer_result.get("full_rank", False),
        )

    # Summary
    all_full_rank = all(r.get("full_rank", False) for r in results["layers"].values())
    results["all_layers_full_rank"] = all_full_rank

    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    for layer_idx, r in results["layers"].items():
        status = "✓ FULL RANK" if r.get("full_rank", False) else "✗ INCOMPLETE"
        logger.info(
            "  Layer %d: %d/%d (%s) - %d tokens, %d sequences",
            layer_idx, r["final_rank"], hidden_dim, status,
            r["tokens_used"], r["sequences_used"]
        )

    if all_full_rank:
        logger.info("\n✓ ALL LAYERS ACHIEVED FULL RANK")
    else:
        logger.info("\n✗ SOME LAYERS FAILED - INVESTIGATION NEEDED")

    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    model_path = "HuggingFaceTB/SmolLM-135M"
    output_dir = Path(__file__).parent / "results"
    run_experiment(model_path, output_dir)
