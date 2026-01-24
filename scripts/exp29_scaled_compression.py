#!/usr/bin/env python3
"""Experiment 29: Scaled Compression.

Key insight from exp28: MLPs EXPAND all distances (ratio 1.5-2.0).
Standard compression has mean ratio ≈ 1.0, losing the expansion.

Simple fix: Scale T by the mean expansion ratio.

If original MLP expands by factor α on average,
and compressed T gives ratio ≈ 1.0,
then T_scaled = α * T should match the expansion.

This is the simplest topological fix: preserve the SCALE of distortion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_mean_expansion(X, Y):
    """Compute mean ratio of output to input pairwise distances."""
    import mlx.core as mx

    n = X.shape[0]
    ratios = []

    for i in range(n):
        for j in range(i+1, n):
            d_in = mx.sqrt(mx.sum((X[i] - X[j]) ** 2))
            d_out = mx.sqrt(mx.sum((Y[i] - Y[j]) ** 2))
            mx.eval(d_in, d_out)

            if float(d_in.item()) > 1e-10:
                ratio = float(d_out.item()) / float(d_in.item())
                ratios.append(ratio)

    return sum(ratios) / len(ratios) if ratios else 1.0


def run_experiment():
    """Test scaled compression."""
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

    # Calibration prompts
    cal_prompts = [
        "The capital of France is",
        "Water freezes at zero degrees",
        "The largest planet is Jupiter",
        "DNA stands for deoxyribonucleic acid",
        "The speed of light is fast",
        "Photosynthesis occurs in plants",
        "The periodic table organizes elements",
        "Machine learning uses algorithms",
        "The theory of relativity was proposed",
        "Quantum mechanics describes particles",
        "Shakespeare wrote many plays",
        "The human brain has neurons",
        "Evolution explains species change",
        "Gravity attracts masses together",
        "The internet connects computers",
        "Vaccines prevent diseases",
    ]

    # Held-out prompts
    held_prompts = [
        "The moon orbits Earth",
        "Birds can fly south",
        "Chemistry studies matter",
        "Music has rhythm",
        "Mountains are tall",
        "Rivers flow downhill",
        "Stars emit light",
        "Plants need water",
    ]

    cal_tokens = [tokenizer.encode(p) for p in cal_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Test on multiple layers
    test_layers = [8, 12, 15, 20, 25]

    results = []

    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"LAYER {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect calibration activations
        cal_inputs = []
        cal_outputs = []

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
                cal_inputs.append(mlp_input[0, -1, :])
                cal_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X = mx.stack(cal_inputs).astype(mx.float32)
        Y = mx.stack(cal_outputs).astype(mx.float32)
        mx.eval(X, Y)

        # Compute mean expansion ratio
        mean_expansion = compute_mean_expansion(X, Y)
        logger.info(f"Mean expansion ratio: {mean_expansion:.4f}")

        # Standard compression
        X_backend = backend.array(X)
        Y_backend = backend.array(Y)

        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T)

        # Compute expansion of T
        Y_comp = mx.matmul(X, T.T)
        mx.eval(Y_comp)
        comp_expansion = compute_mean_expansion(X, Y_comp)
        logger.info(f"Compression expansion ratio: {comp_expansion:.4f}")

        # Scale factor needed
        scale = mean_expansion / comp_expansion if comp_expansion > 0 else 1.0
        logger.info(f"Scale factor: {scale:.4f}")

        # Scaled T
        T_scaled = T * scale
        mx.eval(T_scaled)

        # Verify scaled expansion
        Y_scaled = mx.matmul(X, T_scaled.T)
        mx.eval(Y_scaled)
        scaled_expansion = compute_mean_expansion(X, Y_scaled)
        logger.info(f"Scaled expansion ratio: {scaled_expansion:.4f}")

        # Test accuracy
        def test_accuracy(T_test):
            correct = 0
            total = 0

            for tok in held_tokens:
                input_ids = mx.array([tok])

                orig_logits = model(input_ids)
                mx.eval(orig_logits)
                orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

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

                    if comp_top == orig_top:
                        correct += 1
                    total += 1
                finally:
                    layer.mlp = original_mlp

            return correct / total if total > 0 else 0.0

        std_acc = test_accuracy(T)
        scaled_acc = test_accuracy(T_scaled)

        logger.info(f"\nStandard T accuracy: {std_acc:.1%}")
        logger.info(f"Scaled T accuracy: {scaled_acc:.1%}")

        improvement = scaled_acc - std_acc

        if scaled_acc > std_acc:
            logger.info(f">>> SCALING HELPS: +{improvement*100:.1f}pp")
        elif scaled_acc < std_acc:
            logger.info(f">>> SCALING HURTS: {improvement*100:.1f}pp")
        else:
            logger.info(f">>> NO CHANGE")

        results.append({
            "layer": layer_idx,
            "mean_expansion": mean_expansion,
            "comp_expansion": comp_expansion,
            "scale": scale,
            "std_acc": std_acc,
            "scaled_acc": scaled_acc,
        })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Layer':<8} {'Expansion':<12} {'Scale':<10} {'Std Acc':<10} {'Scaled Acc':<12}")
    logger.info("-" * 55)

    for r in results:
        logger.info(
            f"{r['layer']:<8} {r['mean_expansion']:<12.3f} {r['scale']:<10.3f} "
            f"{r['std_acc']*100:<10.1f} {r['scaled_acc']*100:<12.1f}"
        )

    # Average improvement
    avg_std = sum(r["std_acc"] for r in results) / len(results)
    avg_scaled = sum(r["scaled_acc"] for r in results) / len(results)

    logger.info(f"\nAverage standard accuracy: {avg_std:.1%}")
    logger.info(f"Average scaled accuracy: {avg_scaled:.1%}")

    if avg_scaled > avg_std:
        logger.info(f">>> SCALING IMPROVES by {(avg_scaled - avg_std)*100:.1f}pp on average")
    else:
        logger.info(f">>> SCALING DOES NOT HELP (or hurts)")

    # Interpretation
    logger.info(f"\n{'='*60}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*60}")

    logger.info("""
If scaling helps:
  - The MLP's MAGNITUDE of distortion matters
  - MSE compression preserves direction but not scale
  - We need to match the expansion ratio

If scaling doesn't help:
  - It's not just about uniform scaling
  - The MLP does SELECTIVE distortion (some pairs more than others)
  - We need to preserve the PATTERN, not just the average

Next step: Preserve distortion PATTERN, not just mean scale.
This requires identifying which concept pairs should expand more.
""")


if __name__ == "__main__":
    run_experiment()
