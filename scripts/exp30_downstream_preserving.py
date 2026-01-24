#!/usr/bin/env python3
"""Experiment 30: Downstream-Preserving Compression.

Insight: We've been trying to match the MLP's output.
But what matters is whether DOWNSTREAM layers work correctly.

The MLP output feeds into:
1. Residual connection (added to input)
2. Next layer's attention (Q, K, V projections)

Hypothesis: Only certain DIRECTIONS of the MLP output matter.
Those are the directions that attention uses.

Method:
1. Collect MLP outputs across many prompts
2. Identify which output directions have HIGH VARIANCE
3. Compress to preserve only those high-variance directions
4. See if this reduces entropy while maintaining accuracy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_entropy(logits):
    """Compute entropy of softmax distribution."""
    import mlx.core as mx

    max_logit = mx.max(logits)
    shifted = logits - max_logit
    exp_logits = mx.exp(shifted)
    sum_exp = mx.sum(exp_logits)
    probs = exp_logits / sum_exp
    mx.eval(probs)

    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    mx.eval(entropy)

    return float(entropy.item())


def run_experiment():
    """Test downstream-preserving compression."""
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

    # Lots of calibration prompts for better statistics
    cal_prompts = [
        "The capital of France is Paris",
        "Water freezes at zero degrees",
        "The largest planet is Jupiter",
        "DNA stands for deoxyribonucleic acid",
        "The speed of light is very fast",
        "Photosynthesis occurs in plants",
        "The periodic table organizes elements",
        "Machine learning uses algorithms",
        "The theory of relativity was proposed",
        "Quantum mechanics describes particles",
        "Shakespeare wrote many plays",
        "The human brain has neurons",
        "Evolution explains species change",
        "Gravity attracts masses together",
        "The internet connects computers worldwide",
        "Vaccines prevent diseases effectively",
        "Mountains are formed by tectonics",
        "Rivers flow towards the ocean",
        "Stars are made of plasma",
        "Cells are the basic unit of life",
        "Electricity powers modern devices",
        "Sound travels through air as waves",
        "Chemistry studies matter and reactions",
        "History records past events accurately",
    ]

    held_prompts = [
        "The moon orbits Earth",
        "Birds can fly south",
        "Music has rhythm",
        "Plants need water",
        "Fire requires oxygen",
        "Ice is frozen water",
        "Math uses numbers",
        "Art expresses ideas",
    ]

    cal_tokens = [tokenizer.encode(p) for p in cal_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    layer_idx = 15

    logger.info(f"\n{'='*60}")
    logger.info(f"LAYER {layer_idx}: DOWNSTREAM-PRESERVING COMPRESSION")
    logger.info(f"{'='*60}")

    # Collect activations
    inputs = []
    outputs = []

    for tok in cal_tokens:
        input_ids = mx.array([tok])
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
            outputs.append(mlp_output[0, -1, :])
        finally:
            layer.mlp = original_mlp

    X = mx.stack(inputs).astype(mx.float32)
    Y = mx.stack(outputs).astype(mx.float32)
    mx.eval(X, Y)

    # Analyze output variance structure
    logger.info(f"\n--- Output Variance Analysis ---")

    Y_np = np.array(Y.tolist())

    # SVD of outputs
    U, S, Vh = np.linalg.svd(Y_np - Y_np.mean(axis=0), full_matrices=False)

    total_var = np.sum(S**2)
    cumvar = np.cumsum(S**2) / total_var

    logger.info(f"Singular values (top 10): {S[:10].round(2)}")
    logger.info(f"Cumulative variance: {cumvar[:10].round(3)}")

    # How many components for 90%, 95%, 99%?
    for thresh in [0.5, 0.9, 0.95, 0.99]:
        n_comp = np.argmax(cumvar >= thresh) + 1
        logger.info(f"Components for {thresh*100:.0f}% variance: {n_comp}")

    # The key insight: MLP output lives in a LOW-DIMENSIONAL subspace
    # Most variance is in a few directions

    # Standard compression
    logger.info(f"\n--- Standard Compression ---")

    X_backend = backend.array(X)
    Y_backend = backend.array(Y)

    rmt_result = compressor.compress_layer(X_backend, Y_backend)
    T_std = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
    mx.eval(T_std)

    # Try projecting to top-k principal components of Y
    logger.info(f"\n--- Low-Rank Projection Experiment ---")

    results = []

    for k in [4, 8, 12, 16, 20]:
        # Project Y to top-k components
        # Y_proj = Y_mean + (Y - Y_mean) @ Vh[:k].T @ Vh[:k]

        Y_mean = mx.mean(Y, axis=0, keepdims=True)
        mx.eval(Y_mean)

        Vh_k = mx.array(Vh[:k, :].T).astype(mx.float32)  # (d, k)
        mx.eval(Vh_k)

        Y_centered = Y - Y_mean
        mx.eval(Y_centered)

        # Project to k-dim and back
        Y_proj_k = mx.matmul(Y_centered, Vh_k)  # (n, k)
        Y_proj = mx.matmul(Y_proj_k, Vh_k.T) + Y_mean  # (n, d)
        mx.eval(Y_proj)

        # Now fit T to the projected outputs
        Y_proj_backend = backend.array(Y_proj)
        rmt_result_proj = compressor.compress_layer(X_backend, Y_proj_backend)
        T_proj = mx.array(backend.tolist(rmt_result_proj.T)).astype(mx.float32)
        mx.eval(T_proj)

        # Test accuracy
        def test_with_T(T_test):
            correct = 0
            total = 0
            entropies_orig = []
            entropies_comp = []

            for tok in held_tokens:
                input_ids = mx.array([tok])

                orig_logits = model(input_ids)
                mx.eval(orig_logits)
                orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
                entropies_orig.append(compute_entropy(orig_logits[0, -1, :]))

                layer = model.model.layers[layer_idx]
                original_mlp = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T
                    def __call__(self, x):
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(T_test)
                try:
                    comp_logits = model(input_ids)
                    mx.eval(comp_logits)
                    comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())
                    entropies_comp.append(compute_entropy(comp_logits[0, -1, :]))

                    if comp_top == orig_top:
                        correct += 1
                    total += 1
                finally:
                    layer.mlp = original_mlp

            acc = correct / total if total > 0 else 0.0
            entropy_delta = sum(entropies_comp) / len(entropies_comp) - sum(entropies_orig) / len(entropies_orig)
            return acc, entropy_delta

        acc, entropy_delta = test_with_T(T_proj)

        results.append({
            "k": k,
            "var_captured": cumvar[k-1] if k <= len(cumvar) else 1.0,
            "accuracy": acc,
            "entropy_delta": entropy_delta,
        })

        direction = "↑" if entropy_delta > 0.01 else ("↓" if entropy_delta < -0.01 else "→")
        logger.info(f"k={k:2d}: acc={acc:.1%}, entropy Δ={entropy_delta:+.4f} {direction}, var={cumvar[k-1]*100:.1f}%")

    # Compare to standard
    std_acc, std_entropy = test_with_T(T_std)
    logger.info(f"\nStandard: acc={std_acc:.1%}, entropy Δ={std_entropy:+.4f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*60}")

    # Check if lower entropy correlates with better accuracy
    best_by_entropy = min(results, key=lambda r: r["entropy_delta"])
    best_by_accuracy = max(results, key=lambda r: r["accuracy"])

    logger.info(f"\nLowest entropy increase: k={best_by_entropy['k']} (Δ={best_by_entropy['entropy_delta']:+.4f})")
    logger.info(f"Best accuracy: k={best_by_accuracy['k']} ({best_by_accuracy['accuracy']:.1%})")

    if best_by_entropy["k"] == best_by_accuracy["k"]:
        logger.info(">>> ENTROPY AND ACCURACY ALIGNED!")
    else:
        logger.info(">>> Entropy and accuracy optimized at different k")

    logger.info("""
Key questions:
1. Does projecting to fewer dimensions REDUCE entropy?
2. Is there a k where accuracy is high AND entropy is low?
3. Can we find the "essential" subspace of MLP computation?

If low-rank projection helps:
- MLP output is redundant in most directions
- We only need to preserve the high-variance subspace
- This is TRUE compression (removing redundancy)

If it doesn't help:
- All directions matter for downstream computation
- Or the projection loses essential information
""")


if __name__ == "__main__":
    run_experiment()
