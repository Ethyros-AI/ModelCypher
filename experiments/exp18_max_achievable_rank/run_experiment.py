#!/usr/bin/env python3
"""Experiment 18: Maximum Achievable Rank Per Layer.

Question: What is the TRUE maximum rank achievable at each layer
using the entire vocabulary?

Strategy:
1. Forward ALL 49152 tokens through the model to each layer
2. Normalize activations
3. Compute rank via SVD at different thresholds
4. This tells us the TRUE ceiling for each layer
"""

import json
import logging
import sys
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


def get_all_token_activations_batched(model, layer_idx, batch_size=256):
    """Get activations for ALL tokens at a specific layer."""
    inner = model.model if hasattr(model, "model") else model
    embed_weight = inner.embed_tokens.weight
    vocab_size = int(embed_weight.shape[0])
    hidden_dim = int(embed_weight.shape[1])

    logger.info("Processing %d tokens at layer %d (batch_size=%d)", vocab_size, layer_idx, batch_size)

    all_activations = []

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        batch_tokens = list(range(start, end))

        token_indices = mx.array(batch_tokens)
        embeddings = mx.take(embed_weight, token_indices, axis=0)
        mx.eval(embeddings)

        # Forward through layers
        h = mx.expand_dims(embeddings, axis=1)

        for idx, layer in enumerate(inner.layers):
            if idx > layer_idx:
                break
            result = layer(h)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result

        # [batch, 1, hidden] -> [batch, hidden]
        activations = mx.squeeze(h, axis=1)
        mx.eval(activations)

        all_activations.append(activations)

        if start % 10000 == 0:
            logger.info("  Processed %d/%d tokens...", start, vocab_size)

    # Concatenate all
    all_acts = mx.concatenate(all_activations, axis=0)
    mx.eval(all_acts)

    logger.info("Collected activations shape: %s", all_acts.shape)
    return all_acts


def normalize_activations(acts):
    """Normalize each activation to unit norm."""
    norms = mx.sqrt(mx.sum(acts * acts, axis=1, keepdims=True) + 1e-8)
    return acts / norms


def compute_rank_via_svd(activations, thresholds_relative):
    """Compute rank at different relative thresholds."""
    n_samples, hidden_dim = activations.shape

    # SVD - need CPU stream
    cpu_stream = mx.cpu
    U, S, Vt = mx.linalg.svd(activations, stream=cpu_stream)
    mx.eval(S)

    # Singular values
    singular_values = S.tolist()
    max_sv = max(singular_values)

    ranks = {}
    for name, relative_threshold in thresholds_relative.items():
        threshold = max_sv * relative_threshold
        rank = sum(1 for sv in singular_values if sv > threshold)
        ranks[name] = rank

    return {
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
        "max_singular_value": max_sv,
        "min_singular_value": min(singular_values),
        "n_singular_values": len(singular_values),
        "ranks_by_threshold": ranks,
    }


def run_experiment(model_path: str, output_dir: Path):
    """Run maximum achievable rank analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_path)

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
        "layer_analysis": {},
    }

    # Thresholds to test
    thresholds = {
        "sqrt_eps": 3.16e-4,  # sqrt(eps) for float32
        "1e-3": 1e-3,
        "1e-4": 1e-4,
        "1e-5": 1e-5,
        "1e-6": 1e-6,
    }

    # Test key layers
    test_layers = [0, 7, 15, 22, 29]

    for layer_idx in test_layers:
        if layer_idx >= n_layers:
            continue

        logger.info("=" * 60)
        logger.info("ANALYZING LAYER %d", layer_idx)
        logger.info("=" * 60)

        # Get ALL token activations
        all_acts = get_all_token_activations_batched(model, layer_idx, batch_size=512)

        # Normalize
        all_acts = normalize_activations(all_acts)
        mx.eval(all_acts)

        logger.info("Computing SVD for %d x %d matrix...", all_acts.shape[0], all_acts.shape[1])

        # Since vocab_size (49152) >> hidden_dim (576), we can subsample
        # The rank is bounded by min(n_samples, hidden_dim) = hidden_dim = 576
        # So using more than ~2000 samples gives diminishing returns

        # Use stratified sample: every 25th token = 49152/25 ≈ 1966 tokens
        sample_stride = 25
        sampled_acts = all_acts[::sample_stride]
        mx.eval(sampled_acts)

        logger.info("Sampled to %d activations for SVD", sampled_acts.shape[0])

        # Compute rank
        analysis = compute_rank_via_svd(sampled_acts, thresholds)
        analysis["layer_idx"] = layer_idx
        results["layer_analysis"][layer_idx] = analysis

        logger.info("Results for layer %d:", layer_idx)
        logger.info("  Max singular value: %.4f", analysis["max_singular_value"])
        logger.info("  Ranks by threshold: %s", analysis["ranks_by_threshold"])

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: Maximum achievable rank by layer")
    logger.info("=" * 60)
    for layer_idx, analysis in sorted(results["layer_analysis"].items()):
        dim = hidden_dim
        for threshold_name, rank in analysis["ranks_by_threshold"].items():
            logger.info("  Layer %2d @ %s: rank=%3d/%d (%.1f%%)",
                        layer_idx, threshold_name, rank, dim, 100 * rank / dim)
        logger.info("")

    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_file)
    return results


if __name__ == "__main__":
    model_path = "HuggingFaceTB/SmolLM-135M"
    output_dir = Path(__file__).parent / "results"
    run_experiment(model_path, output_dir)
