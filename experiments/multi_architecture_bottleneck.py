#!/usr/bin/env python3
"""Multi-Architecture Bottleneck Comparison.

Tests the universal bottleneck hypothesis across diverse architectures:
- SmolLM (HuggingFace transformer)
- LFM2 (Liquid Foundation Model)
- Qwen (Alibaba)
- Gemma (Google)
- Granite (IBM)

Key question: Do all architectures compress to similar low-dimensional
subspaces at middle layers?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
from itertools import combinations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "SmolLM-135M": {
        "path": str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M"),
        "architecture": "transformer",
    },
    "LFM2-350M": {
        "path": "/path/to/models/mlx-community/LFM2-350M-MLX-bf16",
        "architecture": "liquid",
    },
    "Qwen2.5-0.5B": {
        "path": "/path/to/models/mlx-community/Qwen2.5-Coder-0.5B-Instruct-bf16",
        "architecture": "qwen",
    },
    "Gemma3n-E2B": {
        "path": "/path/to/models/mlx-community/gemma-3n-E2B-it-bf16",
        "architecture": "gemma",
    },
    "Granite-3B": {
        "path": "/path/to/models/mlx-community/granite-3b-code-instruct-128k-mlx",
        "architecture": "granite",
    },
}


def load_model(name: str, config: dict):
    """Load a model with its tokenizer."""
    from mlx_lm import load
    import mlx.core as mx

    logger.info(f"Loading {name}...")
    model, tokenizer = load(config["path"])
    mx.eval(model.parameters())

    # Get layer count
    inner = model.model if hasattr(model, "model") else model
    n_layers = len(inner.layers)

    # Get hidden dim from first layer
    first_layer = inner.layers[0]
    if hasattr(first_layer, "self_attn"):
        if hasattr(first_layer.self_attn, "hidden_size"):
            hidden_dim = first_layer.self_attn.hidden_size
        elif hasattr(first_layer.self_attn, "q_proj"):
            hidden_dim = first_layer.self_attn.q_proj.weight.shape[0]
        else:
            hidden_dim = None
    elif hasattr(first_layer, "input_layernorm"):
        hidden_dim = first_layer.input_layernorm.weight.shape[0]
    else:
        hidden_dim = None

    logger.info(f"  {name}: {n_layers} layers, hidden_dim={hidden_dim}")

    return model, tokenizer, n_layers, hidden_dim


def get_layer_activation(model, tokenizer, text: str, layer_idx: int):
    """Get mean-pooled activation at specific layer."""
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

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


def get_activations(model, tokenizer, probes: list[str], layer_idx: int):
    """Get activations for probes at a specific layer."""
    import mlx.core as mx

    activations = []
    for probe in probes:
        act = get_layer_activation(model, tokenizer, probe, layer_idx)
        if act is not None:
            activations.append(act)

    if not activations:
        return None

    stacked = mx.stack(activations, axis=0)
    stacked = stacked.astype(mx.float32)
    mx.eval(stacked)
    return np.array(stacked)


def compute_gram_rank(activations: np.ndarray, threshold_ratio: float = 3.45e-4):
    """Compute effective rank of Gram matrix."""
    G = activations @ activations.T
    _, S, _ = np.linalg.svd(G, full_matrices=False)
    max_sv = S[0] if len(S) > 0 else 1.0
    threshold = max_sv * threshold_ratio
    return int(np.sum(S > threshold))


def principal_angles(V1: np.ndarray, V2: np.ndarray):
    """Compute principal angles between two subspaces."""
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    M = Q1.T @ Q2
    _, S, _ = np.linalg.svd(M)
    S = np.clip(S, -1.0, 1.0)
    return np.arccos(S)


def subspace_overlap(G1: np.ndarray, G2: np.ndarray, threshold_ratio: float = 3.45e-4):
    """Compute overlap between effective Gram subspaces."""
    # SVD of both Gram matrices
    U1, S1, _ = np.linalg.svd(G1, full_matrices=False)
    U2, S2, _ = np.linalg.svd(G2, full_matrices=False)

    # Effective ranks
    rank1 = int(np.sum(S1 > S1[0] * threshold_ratio))
    rank2 = int(np.sum(S2 > S2[0] * threshold_ratio))

    # Get effective subspaces
    U1_eff = U1[:, :rank1]
    U2_eff = U2[:, :rank2]

    # Principal angles
    angles = principal_angles(U1_eff, U2_eff)

    mean_cos = np.mean(np.cos(angles))
    shared_dim = int(np.sum(np.cos(angles) > 0.9))

    return {
        "rank1": rank1,
        "rank2": rank2,
        "mean_cos": float(mean_cos),
        "shared_dim": shared_dim,
        "min_angle_deg": float(np.degrees(angles.min())) if len(angles) > 0 else 0,
        "max_angle_deg": float(np.degrees(angles.max())) if len(angles) > 0 else 0,
    }


def main():
    from tests.fixtures.models import get_atlas_probes

    # Load probes
    probe_texts = get_atlas_probes(n_samples=500)  # Smaller for speed
    logger.info(f"Using {len(probe_texts)} probes")

    # Load all models
    models_data = {}
    for name, config in MODELS.items():
        try:
            model, tokenizer, n_layers, hidden_dim = load_model(name, config)
            models_data[name] = {
                "model": model,
                "tokenizer": tokenizer,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "architecture": config["architecture"],
            }
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue

    if len(models_data) < 2:
        logger.error("Need at least 2 models for comparison")
        return

    logger.info(f"Loaded {len(models_data)} models: {list(models_data.keys())}")

    # Test at multiple depths
    depths = [0.25, 0.50, 0.75]

    results = {
        "n_probes": len(probe_texts),
        "models": {name: {"n_layers": d["n_layers"], "hidden_dim": d["hidden_dim"], "architecture": d["architecture"]}
                   for name, d in models_data.items()},
        "depths": [],
    }

    for depth in depths:
        logger.info("=" * 60)
        logger.info(f"DEPTH {depth:.0%}")
        logger.info("=" * 60)

        # Collect activations for all models at this depth
        model_activations = {}
        model_gram_matrices = {}
        model_gram_ranks = {}

        for name, data in models_data.items():
            layer_idx = int(depth * (data["n_layers"] - 1))
            logger.info(f"  {name}: layer {layer_idx}/{data['n_layers']}")

            acts = get_activations(data["model"], data["tokenizer"], probe_texts, layer_idx)
            if acts is None:
                logger.warning(f"  Failed to get activations for {name}")
                continue

            model_activations[name] = acts
            G = acts @ acts.T
            model_gram_matrices[name] = G
            model_gram_ranks[name] = compute_gram_rank(acts)

            logger.info(f"    Shape: {acts.shape}, Gram rank: {model_gram_ranks[name]}")

        # Pairwise comparisons
        depth_result = {
            "depth": depth,
            "gram_ranks": model_gram_ranks,
            "pairwise": [],
        }

        model_names = list(model_gram_matrices.keys())
        for name1, name2 in combinations(model_names, 2):
            G1 = model_gram_matrices[name1]
            G2 = model_gram_matrices[name2]

            overlap = subspace_overlap(G1, G2)

            logger.info(f"  {name1} vs {name2}:")
            logger.info(f"    Ranks: {overlap['rank1']} vs {overlap['rank2']}")
            logger.info(f"    Mean cos: {overlap['mean_cos']:.4f}, Shared dims: {overlap['shared_dim']}")

            depth_result["pairwise"].append({
                "model1": name1,
                "model2": name2,
                **overlap,
            })

        results["depths"].append(depth_result)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: Gram Ranks at Depth 50% (The Bottleneck)")
    logger.info("=" * 60)

    for depth_data in results["depths"]:
        if depth_data["depth"] == 0.5:
            for name, rank in depth_data["gram_ranks"].items():
                arch = models_data[name]["architecture"]
                logger.info(f"  {name} ({arch}): Gram rank = {rank}")

            logger.info("\nPairwise Mean Cosine at Bottleneck:")
            for pair in depth_data["pairwise"]:
                logger.info(f"  {pair['model1']} vs {pair['model2']}: {pair['mean_cos']:.4f}")

    # Save results
    output_path = Path(__file__).parent / "multi_architecture_bottleneck_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
