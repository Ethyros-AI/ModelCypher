#!/usr/bin/env python3
"""Experiment 17: Eigenvalue Analysis Across Layers.

Understand WHY middle layers can't reach full rank:
1. What's the eigenvalue distribution at each layer?
2. How many eigenvalues are "near zero" but above threshold?
3. Is this a numerical threshold issue or a fundamental geometry issue?
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


def normalize_activation(act):
    """Normalize activation to unit norm."""
    norm = mx.sqrt(mx.sum(act * act) + 1e-8)
    return act / norm


def collect_activations(model, tokenizer, layer_idx, n_probes=500):
    """Collect many activations at a layer."""
    probe_texts = [f"The quick brown fox {i}" for i in range(n_probes)]

    activations = []
    for text in probe_texts:
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        act = get_layer_activation(model, input_ids, layer_idx)
        if act is not None:
            act = normalize_activation(act)
            mx.eval(act)
            activations.append(act)

    stacked = mx.stack(activations, axis=0)
    mx.eval(stacked)
    return stacked


def analyze_eigenvalues(activations, layer_idx):
    """Analyze eigenvalue structure via SVD (GPU compatible)."""
    # SVD of activations: A = U @ S @ V^T
    # Singular values squared = eigenvalues of A^T @ A
    # Need CPU stream for SVD in MLX
    cpu_stream = mx.cpu
    U, S, Vt = mx.linalg.svd(activations, stream=cpu_stream)
    mx.eval(S)

    # Eigenvalues are squared singular values (already sorted descending by SVD)
    eigenvalues = S * S
    mx.eval(eigenvalues)

    eigenvalues_list = eigenvalues.tolist()

    # Compute rank at different thresholds
    max_eigenvalue = max(eigenvalues_list)
    eps = 1e-7  # float32 machine epsilon

    thresholds = {
        "sqrt_eps": max_eigenvalue * (eps ** 0.5),
        "eps": max_eigenvalue * eps,
        "1e-3": max_eigenvalue * 1e-3,
        "1e-4": max_eigenvalue * 1e-4,
        "1e-5": max_eigenvalue * 1e-5,
        "1e-6": max_eigenvalue * 1e-6,
    }

    ranks = {}
    for name, threshold in thresholds.items():
        rank = sum(1 for e in eigenvalues_list if e > threshold)
        ranks[name] = rank

    # Analyze eigenvalue distribution
    n = len(eigenvalues_list)

    analysis = {
        "layer_idx": layer_idx,
        "hidden_dim": n,
        "max_eigenvalue": max_eigenvalue,
        "min_eigenvalue": min(eigenvalues_list),
        "condition_number": max_eigenvalue / max(min(eigenvalues_list), 1e-30),
        "ranks_by_threshold": ranks,
        "eigenvalue_percentiles": {
            "p50": eigenvalues_list[n // 2],
            "p75": eigenvalues_list[n // 4],
            "p90": eigenvalues_list[n // 10],
            "p95": eigenvalues_list[n // 20],
            "p99": eigenvalues_list[n // 100] if n >= 100 else eigenvalues_list[-1],
        },
        # Count eigenvalues in different ranges
        "eigenvalue_counts": {
            "> 1e-2 * max": sum(1 for e in eigenvalues_list if e > max_eigenvalue * 1e-2),
            "1e-2 to 1e-4 * max": sum(1 for e in eigenvalues_list if max_eigenvalue * 1e-4 < e <= max_eigenvalue * 1e-2),
            "1e-4 to 1e-6 * max": sum(1 for e in eigenvalues_list if max_eigenvalue * 1e-6 < e <= max_eigenvalue * 1e-4),
            "< 1e-6 * max": sum(1 for e in eigenvalues_list if e <= max_eigenvalue * 1e-6),
        },
        # First 20 and last 20 eigenvalues (normalized to max)
        "top_20_eigenvalues": [e / max_eigenvalue for e in eigenvalues_list[:20]],
        "bottom_20_eigenvalues": [e / max_eigenvalue for e in eigenvalues_list[-20:]],
    }

    return analysis


def run_experiment(model_path: str, output_dir: Path):
    """Run eigenvalue analysis across all layers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_path)

    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)
    hidden_dim = int(inner.layers[0].self_attn.q_proj.weight.shape[0])

    logger.info("Model: %d layers, hidden_dim=%d", n_layers, hidden_dim)

    results = {
        "model_path": model_path,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_analysis": {},
    }

    # Analyze every 5 layers including layer 0, 7, 15, 22, 29
    test_layers = [0, 5, 7, 10, 15, 20, 22, 25, 29]

    for layer_idx in test_layers:
        if layer_idx >= n_layers:
            continue

        logger.info("=" * 60)
        logger.info("ANALYZING LAYER %d", layer_idx)
        logger.info("=" * 60)

        # Collect activations
        activations = collect_activations(model, tokenizer, layer_idx, n_probes=500)
        logger.info("Collected %d activations, shape=%s", activations.shape[0], activations.shape)

        # Analyze eigenvalues
        analysis = analyze_eigenvalues(activations, layer_idx)
        results["layer_analysis"][layer_idx] = analysis

        logger.info("Max eigenvalue: %.6f", analysis["max_eigenvalue"])
        logger.info("Min eigenvalue: %.10f", analysis["min_eigenvalue"])
        logger.info("Condition number: %.2e", analysis["condition_number"])
        logger.info("Ranks by threshold: %s", analysis["ranks_by_threshold"])
        logger.info("Eigenvalue counts: %s", analysis["eigenvalue_counts"])

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: Rank at sqrt(eps) threshold by layer")
    logger.info("=" * 60)
    for layer_idx, analysis in sorted(results["layer_analysis"].items()):
        rank = analysis["ranks_by_threshold"]["sqrt_eps"]
        dim = analysis["hidden_dim"]
        logger.info("  Layer %2d: rank=%3d/%d (%.1f%%)",
                    layer_idx, rank, dim, 100 * rank / dim)

    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_file)
    return results


if __name__ == "__main__":
    model_path = "HuggingFaceTB/SmolLM-135M"
    output_dir = Path(__file__).parent / "results"
    run_experiment(model_path, output_dir)
