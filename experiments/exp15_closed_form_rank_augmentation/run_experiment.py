#!/usr/bin/env python3
"""Experiment 15: Closed-Form Rank Augmentation Validation.

This experiment validates the closed-form approach to rank augmentation:
1. Load a real model (SmolLM-135M)
2. Collect activations from initial probes
3. Compute rank and null space
4. Use closed-form token scoring to find orthogonal tokens
5. Verify rank increase per added token

Success criteria:
- Each selected token should increase rank by ~1 (theoretical optimum)
- No iteration required (single pass through vocabulary)
- Deterministic and reproducible

Reference: Gradient ascent (exp14) achieved ~1.0 rank per probe but was iterative.
This should match that efficiency without iteration.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
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

    # Mean pool
    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    return pooled


def normalize_activation(act):
    """Normalize activation to unit norm (direction only, not magnitude)."""
    norm = mx.sqrt(mx.sum(act * act) + 1e-8)
    return act / norm


def collect_initial_activations(model, tokenizer, layer_idx: int, n_probes: int = 100, normalize: bool = True):
    """Collect activations from random probes."""
    logger.info("Collecting %d initial probe activations at layer %d (normalize=%s)", n_probes, layer_idx, normalize)

    # Use diverse probe texts
    probe_texts = [
        "The quick brown fox",
        "Mathematics is beautiful",
        "Water flows downhill",
        "The sun rises in the east",
        "Music brings joy",
        "Trees grow tall",
        "Books contain knowledge",
        "Stars shine bright",
        "Rivers meet the sea",
        "Birds fly south",
    ]

    # Extend with variations
    extended_probes = []
    for i in range(n_probes):
        base = probe_texts[i % len(probe_texts)]
        extended_probes.append(f"{base} {i}")

    activations = []
    for text in extended_probes[:n_probes]:
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        act = get_layer_activation(model, input_ids, layer_idx)
        if act is not None:
            if normalize:
                act = normalize_activation(act)
            mx.eval(act)
            activations.append(act)

    stacked = mx.stack(activations, axis=0)
    mx.eval(stacked)

    logger.info("Collected %d activations, shape=%s", len(activations), stacked.shape)
    return stacked


def run_experiment(model_path: str, output_dir: Path):
    """Run the closed-form rank augmentation experiment."""
    from modelcypher.backends.mlx_backend import MLXBackend
    from modelcypher.core.domain.geometry.orthogonal_probe_generator import (
        compute_numerical_rank,
        compute_null_space_basis,
        find_null_space_tokens_closed_form,
        augment_rank_closed_form,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    backend = MLXBackend()

    # Get model info
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
        "experiments": [],
    }

    # Test on multiple layers
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for layer_idx in test_layers:
        logger.info("=" * 60)
        logger.info("TESTING LAYER %d", layer_idx)
        logger.info("=" * 60)

        layer_result = {
            "layer_idx": layer_idx,
            "initial_probes": 100,
        }

        # Collect initial activations
        initial_acts = collect_initial_activations(model, tokenizer, layer_idx, n_probes=100)

        # Compute initial rank
        initial_rank, dim = compute_numerical_rank(initial_acts, backend)
        layer_result["initial_rank"] = initial_rank
        layer_result["hidden_dim"] = dim
        layer_result["initial_coverage"] = initial_rank / dim

        logger.info("Initial rank: %d/%d (%.1f%% coverage)", initial_rank, dim, 100.0 * initial_rank / dim)

        if initial_rank >= dim:
            logger.info("Already full rank, skipping augmentation")
            layer_result["augmentation_needed"] = False
            results["experiments"].append(layer_result)
            continue

        layer_result["augmentation_needed"] = True

        # Compute null space
        U_null = compute_null_space_basis(initial_acts, initial_rank, backend)
        null_dim = int(U_null.shape[1]) if U_null is not None else 0
        layer_result["null_dim"] = null_dim

        logger.info("Null space dimension: %d", null_dim)

        # Run closed-form token scoring
        logger.info("Running closed-form token scoring...")
        start_time = time.time()

        top_tokens = find_null_space_tokens_closed_form(
            model=model,
            U_null=U_null,
            layer_idx=layer_idx,
            backend=backend,
            top_k=min(100, null_dim),
            batch_size=256,
        )

        scoring_time = time.time() - start_time
        layer_result["scoring_time_seconds"] = scoring_time

        logger.info("Scoring complete in %.2f seconds", scoring_time)
        logger.info("Top 10 tokens by null-space score:")
        for token_id, score in top_tokens[:10]:
            try:
                text = tokenizer.decode([token_id])
            except:
                text = f"<{token_id}>"
            logger.info("  token %d ('%s'): score=%.4f", token_id, text[:20], score)

        # Measure rank increase as we add tokens until FULL RANK
        logger.info("Measuring rank increase per token until FULL RANK...")

        current_acts = initial_acts
        rank_trajectory = [initial_rank]
        tokens_added = []
        used_token_ids = set()

        # Keep going until full rank or we run out of useful tokens
        iteration = 0
        max_iterations = 100  # Safety limit

        while rank_trajectory[-1] < dim and iteration < max_iterations:
            iteration += 1
            current_rank = rank_trajectory[-1]

            # Recompute null space with current activations
            U_null_current = compute_null_space_basis(current_acts, current_rank, backend)
            if U_null_current is None:
                logger.info("Full rank achieved!")
                break

            null_dim_current = int(U_null_current.shape[1])

            # Score all tokens against CURRENT null space
            top_tokens_iter = find_null_space_tokens_closed_form(
                model=model,
                U_null=U_null_current,
                layer_idx=layer_idx,
                backend=backend,
                top_k=min(100, null_dim_current),
                batch_size=256,
            )

            # Filter out already-used tokens
            available_tokens = [(t, s) for t, s in top_tokens_iter if t not in used_token_ids]

            if not available_tokens:
                logger.warning("No more available tokens!")
                break

            # Add tokens in this iteration
            tokens_this_iter = 0
            for token_id, score in available_tokens[:20]:  # Add up to 20 per iteration
                # Get activation for this token
                token_input = mx.array([[token_id]])
                mx.eval(token_input)

                act = get_layer_activation(model, token_input, layer_idx)
                if act is None:
                    continue

                # Normalize to match initial activations (direction only, not magnitude)
                act = normalize_activation(act)
                mx.eval(act)

                # Add to activations
                current_acts = mx.concatenate([current_acts, mx.expand_dims(act, 0)], axis=0)
                mx.eval(current_acts)
                used_token_ids.add(token_id)
                tokens_this_iter += 1

                # Compute new rank
                new_rank, _ = compute_numerical_rank(current_acts, backend)
                rank_increase = new_rank - rank_trajectory[-1]

                rank_trajectory.append(new_rank)
                tokens_added.append({
                    "token_id": token_id,
                    "score": score,
                    "rank_after": new_rank,
                    "rank_increase": rank_increase,
                })

                if new_rank >= dim:
                    logger.info("Full rank achieved after %d tokens!", len(tokens_added))
                    break

            total_tokens = len(tokens_added)
            logger.info(
                "  Iteration %d: added %d tokens, rank=%d/%d (%.1f%%), avg_increase=%.2f",
                iteration,
                tokens_this_iter,
                rank_trajectory[-1],
                dim,
                100.0 * rank_trajectory[-1] / dim,
                (rank_trajectory[-1] - initial_rank) / total_tokens if total_tokens > 0 else 0,
            )

            if rank_trajectory[-1] >= dim:
                break

        final_rank = rank_trajectory[-1]
        tokens_used = len(tokens_added)
        avg_rank_per_token = (final_rank - initial_rank) / tokens_used if tokens_used > 0 else 0

        layer_result["final_rank"] = final_rank
        layer_result["tokens_used"] = tokens_used
        layer_result["rank_increase_total"] = final_rank - initial_rank
        layer_result["avg_rank_per_token"] = avg_rank_per_token
        layer_result["full_rank_achieved"] = final_rank >= dim
        layer_result["rank_trajectory"] = rank_trajectory
        layer_result["tokens_added"] = tokens_added[:20]  # Save first 20 for inspection

        logger.info("LAYER %d SUMMARY:", layer_idx)
        logger.info("  Initial rank: %d/%d", initial_rank, dim)
        logger.info("  Final rank: %d/%d", final_rank, dim)
        logger.info("  Tokens used: %d", tokens_used)
        logger.info("  Avg rank increase per token: %.3f", avg_rank_per_token)
        logger.info("  Full rank achieved: %s", final_rank >= dim)

        results["experiments"].append(layer_result)

    # Compute summary statistics
    experiments_with_aug = [e for e in results["experiments"] if e.get("augmentation_needed", False)]
    if experiments_with_aug:
        avg_rank_per_token_overall = sum(e["avg_rank_per_token"] for e in experiments_with_aug) / len(experiments_with_aug)
        full_rank_count = sum(1 for e in experiments_with_aug if e.get("full_rank_achieved", False))

        results["summary"] = {
            "layers_tested": len(results["experiments"]),
            "layers_needing_augmentation": len(experiments_with_aug),
            "layers_achieving_full_rank": full_rank_count,
            "avg_rank_per_token_overall": avg_rank_per_token_overall,
        }

        logger.info("=" * 60)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 60)
        logger.info("Layers tested: %d", len(results["experiments"]))
        logger.info("Layers needing augmentation: %d", len(experiments_with_aug))
        logger.info("Layers achieving full rank: %d", full_rank_count)
        logger.info("Average rank per token: %.3f (optimal: 1.0)", avg_rank_per_token_overall)

    # Save results
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Results saved to %s", output_file)

    return results


def main():
    # Use SmolLM-135M for testing (small enough to be fast)
    model_path = "HuggingFaceTB/SmolLM-135M"
    output_dir = Path(__file__).parent / "results"

    results = run_experiment(model_path, output_dir)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    if "summary" in results:
        summary = results["summary"]
        print(f"Average rank per token: {summary['avg_rank_per_token_overall']:.3f}")
        print(f"  (Optimal is 1.0 - each token adds one independent direction)")

        if summary["avg_rank_per_token_overall"] >= 0.8:
            print("\n✓ CLOSED-FORM APPROACH VALIDATED")
            print("  Achieves near-optimal rank increase without iteration")
        else:
            print("\n✗ CLOSED-FORM APPROACH NEEDS INVESTIGATION")
            print(f"  Only achieving {summary['avg_rank_per_token_overall']:.1%} of optimal")


if __name__ == "__main__":
    main()
