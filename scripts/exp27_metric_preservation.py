#!/usr/bin/env python3
"""Experiment 27: Metric Preservation Under Compression.

User insight: "if we were truly compressing correctly, entropy would drop.
Entropy increasing means we're distorting topology, not compressing."

Hypothesis: Our T matrices don't preserve the METRIC STRUCTURE of representations.
True compression would maintain pairwise distances/angles between points.

Method:
1. Compute pairwise distances BEFORE MLP (input space)
2. Compute pairwise distances AFTER original MLP (output space)
3. Compute pairwise distances AFTER compressed MLP (T @ input)
4. Check: does T preserve the distance RATIOS?

If distances are distorted, we're breaking topology → entropy increases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_pairwise_distances(points):
    """Compute matrix of pairwise Euclidean distances."""
    import mlx.core as mx

    n = points.shape[0]
    distances = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                diff = points[i] - points[j]
                dist = mx.sqrt(mx.sum(diff * diff))
                mx.eval(dist)
                row.append(float(dist.item()))
        distances.append(row)

    return distances


def compute_pairwise_cosines(points):
    """Compute matrix of pairwise cosine similarities."""
    import mlx.core as mx

    n = points.shape[0]
    cosines = []

    # Normalize
    norms = mx.sqrt(mx.sum(points * points, axis=1, keepdims=True))
    normalized = points / (norms + 1e-10)
    mx.eval(normalized)

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                cos = mx.sum(normalized[i] * normalized[j])
                mx.eval(cos)
                row.append(float(cos.item()))
        cosines.append(row)

    return cosines


def distance_correlation(d1, d2):
    """Compute correlation between two distance matrices (flattened upper triangle)."""
    n = len(d1)
    vals1 = []
    vals2 = []

    for i in range(n):
        for j in range(i+1, n):
            vals1.append(d1[i][j])
            vals2.append(d2[i][j])

    if len(vals1) < 2:
        return 0.0

    mean1 = sum(vals1) / len(vals1)
    mean2 = sum(vals2) / len(vals2)

    num = sum((vals1[k] - mean1) * (vals2[k] - mean2) for k in range(len(vals1)))
    den1 = sum((v - mean1)**2 for v in vals1)
    den2 = sum((v - mean2)**2 for v in vals2)

    if den1 < 1e-10 or den2 < 1e-10:
        return 0.0

    return num / math.sqrt(den1 * den2)


def run_experiment():
    """Test whether compression preserves metric structure."""
    import mlx.core as mx
    import numpy as np

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)

    # Test prompts - need enough for meaningful pairwise distances
    test_prompts = [
        "The capital of France is",
        "Water freezes at",
        "The largest planet is",
        "DNA stands for",
        "The speed of light is",
        "Photosynthesis occurs in",
        "The periodic table organizes",
        "Machine learning algorithms",
        "The theory of relativity",
        "Quantum mechanics describes",
        "Shakespeare wrote plays",
        "The human brain contains",
    ]

    test_tokens = [tokenizer.encode(p) for p in test_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Test a few layers
    test_layers = [8, 15, 20, 25]

    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"LAYER {layer_idx}: METRIC PRESERVATION ANALYSIS")
        logger.info(f"{'='*60}")

        # Collect activations
        inputs = []
        orig_outputs = []

        for tokens in test_tokens:
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            layer.mlp = MLPHook(original_mlp)
            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(mlp_input[0, -1, :])
                orig_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X = mx.stack(inputs).astype(mx.float32)
        Y_orig = mx.stack(orig_outputs).astype(mx.float32)
        mx.eval(X, Y_orig)

        # Compress
        X_backend = backend.array(X)
        Y_backend = backend.array(Y_orig)

        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T)

        # Compressed output
        Y_comp = mx.matmul(X, T.T)
        mx.eval(Y_comp)

        # Compute distance matrices
        logger.info("\nComputing pairwise distances...")

        D_input = compute_pairwise_distances(X)
        D_orig = compute_pairwise_distances(Y_orig)
        D_comp = compute_pairwise_distances(Y_comp)

        # Compute cosine matrices
        C_input = compute_pairwise_cosines(X)
        C_orig = compute_pairwise_cosines(Y_orig)
        C_comp = compute_pairwise_cosines(Y_comp)

        # Distance preservation
        corr_orig_input = distance_correlation(D_orig, D_input)
        corr_comp_input = distance_correlation(D_comp, D_input)
        corr_comp_orig = distance_correlation(D_comp, D_orig)

        logger.info(f"\n--- Distance Preservation ---")
        logger.info(f"Original MLP preserves input distances: r = {corr_orig_input:.4f}")
        logger.info(f"Compressed MLP preserves input distances: r = {corr_comp_input:.4f}")
        logger.info(f"Compressed matches original output distances: r = {corr_comp_orig:.4f}")

        # Cosine preservation
        corr_cos_orig_input = distance_correlation(C_orig, C_input)
        corr_cos_comp_input = distance_correlation(C_comp, C_input)
        corr_cos_comp_orig = distance_correlation(C_comp, C_orig)

        logger.info(f"\n--- Angular Preservation ---")
        logger.info(f"Original MLP preserves input angles: r = {corr_cos_orig_input:.4f}")
        logger.info(f"Compressed MLP preserves input angles: r = {corr_cos_comp_input:.4f}")
        logger.info(f"Compressed matches original angles: r = {corr_cos_comp_orig:.4f}")

        # Key question: does compression DISTORT more than the original MLP?
        logger.info(f"\n--- Distortion Analysis ---")

        dist_loss = corr_orig_input - corr_comp_input
        logger.info(f"Distance preservation loss: {dist_loss:.4f}")

        if dist_loss > 0.1:
            logger.info(">>> COMPRESSION DISTORTS DISTANCES significantly")
        elif dist_loss < -0.1:
            logger.info(">>> COMPRESSION PRESERVES BETTER than original (?)")
        else:
            logger.info(">>> Similar distance preservation")

        # Check specific distance ratios
        logger.info(f"\n--- Sample Distance Ratios ---")
        for i in range(min(3, len(test_prompts))):
            for j in range(i+1, min(4, len(test_prompts))):
                d_orig = D_orig[i][j]
                d_comp = D_comp[i][j]
                ratio = d_comp / d_orig if d_orig > 0 else 0

                logger.info(f"Prompts ({i},{j}): orig={d_orig:.2f}, comp={d_comp:.2f}, ratio={ratio:.4f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*60}")

    logger.info("""
TRUE COMPRESSION would:
1. Preserve pairwise distances (isometric)
2. Preserve angles (conformal)
3. Maintain the metric tensor of the manifold

DISTORTION causes:
1. Distances change → topology breaks
2. Angles change → directional relationships break
3. Entropy increases because structure is lost

The key insight:
- MLP transforms ARE NOT isometric - they change distances
- Compression (T) approximates this non-isometric map
- But approximation errors compound → MORE distortion
- Each layer adds distortion → entropy grows

For TRUE lossless compression:
- Need ISOMETRIC projection, not least-squares fit
- Preserve the METRIC, not the Euclidean reconstruction
- This is a fundamentally different objective

Possible approaches:
1. Procrustes with SCALING constraint
2. Geodesic-preserving embeddings
3. Metric learning objective
""")


if __name__ == "__main__":
    run_experiment()
