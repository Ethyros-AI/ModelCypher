#!/usr/bin/env python3
"""Experiment 12: Geodesic Rank vs Euclidean Rank.

Tests the new compression modules:
1. GeodesicLayerAnalyzer - compute geodesic vs euclidean rank
2. RMTAwareCompressor - compress using RMT signal/noise separation

Hypothesis: Geodesic rank < Euclidean rank, revealing true manifold structure
where compression is possible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Compare geodesic vs Euclidean rank across layers."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import (
        GeodesicLayerAnalyzer,
        RMTAwareCompressor,
    )

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Prompts
    calibration_prompts = [
        "The capital of France is",
        "In mathematics, the derivative of",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
        "Photosynthesis is the process by which",
        "The human heart has",
        "DNA stands for",
        "The chemical symbol for gold is",
        "Shakespeare wrote",
        "The Great Wall of China was built",
        "E = mc² was discovered by",
        "The mitochondria is",
        "Python is a programming language that",
        "Machine learning algorithms",
        "The stock market",
        "Climate change refers to",
        "Quantum mechanics describes",
        "The Renaissance was a period",
        "Artificial intelligence",
    ]

    held_out_prompts = [
        "The theory of relativity states",
        "Neurons in the brain",
        "The periodic table",
        "Evolution by natural selection",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    # Initialize modules
    analyzer = GeodesicLayerAnalyzer(backend=backend)
    compressor = RMTAwareCompressor(backend=backend)

    # Test layers
    test_layers = [1, 2, 5, 6, 7, 10, 14]

    results = []

    for layer_idx in test_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect calibration activations
        cal_inputs = []
        cal_outputs = []

        for tokens in cal_tokens:
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
                cal_inputs.append(mlp_input[0, -1, :])
                cal_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X_cal = mx.stack(cal_inputs).astype(mx.float32)
        Y_cal = mx.stack(cal_outputs).astype(mx.float32)
        mx.eval(X_cal, Y_cal)

        # Convert to backend arrays
        X_backend = backend.array(X_cal)
        Y_backend = backend.array(Y_cal)

        # Step 1: Analyze geodesic structure
        logger.info("\n--- Geodesic Analysis ---")
        try:
            profile = analyzer.analyze(X_backend)
            logger.info(f"Euclidean rank: {profile.euclidean_rank}")
            logger.info(f"Geodesic rank: {profile.geodesic_rank}")
            logger.info(f"RMT signal rank: {profile.rmt_signal_rank}")
            logger.info(f"Null space dim: {profile.null_space_dimension}")
            logger.info(f"Compressibility: {profile.compressibility_score:.3f}")
        except Exception as e:
            logger.warning(f"Geodesic analysis failed: {e}")
            # Fallback: just use RMT
            profile = None

        # Step 2: Compress using RMT
        logger.info("\n--- RMT Compression ---")
        rmt_result, naive_result = compressor.compress_with_naive_comparison(
            X_backend, Y_backend
        )

        logger.info(f"RMT signal_rank: {rmt_result.signal_rank}/{rmt_result.total_rank}")
        logger.info(f"RMT recon error: {rmt_result.reconstruction_error:.4f}")
        logger.info(f"Naive recon error: {naive_result.reconstruction_error:.4f}")

        # Step 3: Evaluate on held-out (token accuracy via actual model)
        logger.info("\n--- Held-Out Evaluation ---")

        def evaluate_with_model(T):
            """Evaluate compression with actual model inference."""
            correct = 0
            total = 0

            T_mx = mx.array(backend.tolist(T)).astype(mx.float32)
            mx.eval(T_mx)

            for tokens in held_tokens:
                input_ids = mx.array([tokens])

                # Original logits
                orig_logits = model(input_ids)
                mx.eval(orig_logits)
                orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

                # Compressed
                layer = model.model.layers[layer_idx]
                original_mlp = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T
                    def __call__(self, x):
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(T_mx)
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

        rmt_accuracy = evaluate_with_model(rmt_result.T)
        naive_accuracy = evaluate_with_model(naive_result.T)

        logger.info(f"RMT token accuracy: {rmt_accuracy:.1%}")
        logger.info(f"Naive token accuracy: {naive_accuracy:.1%}")

        results.append({
            "layer": layer_idx,
            "euclidean_rank": profile.euclidean_rank if profile else 0,
            "geodesic_rank": profile.geodesic_rank if profile else 0,
            "rmt_signal_rank": rmt_result.signal_rank,
            "compressibility": profile.compressibility_score if profile else 0,
            "rmt_accuracy": rmt_accuracy,
            "naive_accuracy": naive_accuracy,
        })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Layer':<8} {'Euc Rank':<10} {'Geo Rank':<10} {'RMT Rank':<10} "
                f"{'Compress':<10} {'RMT Acc':<10} {'Naive Acc':<10}")
    logger.info("-" * 70)

    for r in results:
        logger.info(
            f"{r['layer']:<8} {r['euclidean_rank']:<10} {r['geodesic_rank']:<10} "
            f"{r['rmt_signal_rank']:<10} {r['compressibility']:<10.3f} "
            f"{r['rmt_accuracy']*100:<10.1f} {r['naive_accuracy']*100:<10.1f}"
        )

    # Check hypothesis: geodesic_rank < euclidean_rank
    geo_smaller_count = sum(
        1 for r in results
        if r['geodesic_rank'] < r['euclidean_rank'] and r['geodesic_rank'] > 0
    )

    logger.info(f"\n--- Hypothesis Check ---")
    logger.info(f"Layers where geodesic_rank < euclidean_rank: {geo_smaller_count}/{len(results)}")

    if geo_smaller_count > len(results) // 2:
        logger.info(">>> HYPOTHESIS SUPPORTED: Geodesic rank reveals sparser structure")
    else:
        logger.info(">>> HYPOTHESIS NOT SUPPORTED: Geodesic rank not consistently smaller")

    # Check correlation between compressibility and accuracy
    from statistics import correlation

    compress_scores = [r['compressibility'] for r in results if r['compressibility'] > 0]
    rmt_accuracies = [r['rmt_accuracy'] for r in results if r['compressibility'] > 0]

    if len(compress_scores) >= 3:
        try:
            corr = correlation(compress_scores, rmt_accuracies)
            logger.info(f"\nCorrelation(compressibility, rmt_accuracy) = {corr:.4f}")
            if corr > 0.5:
                logger.info(">>> COMPRESSIBILITY SCORE PREDICTS SUCCESS")
            elif corr < -0.5:
                logger.info(">>> COMPRESSIBILITY SCORE INVERSELY PREDICTS (unexpected)")
            else:
                logger.info(">>> WEAK CORRELATION")
        except Exception as e:
            logger.warning(f"Correlation failed: {e}")


if __name__ == "__main__":
    run_experiment()
