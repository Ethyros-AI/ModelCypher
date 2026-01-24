#!/usr/bin/env python3
"""Experiment 14: Entropy Coverage Prediction.

Tests whether calibration entropy predicts held-out generalization.

Hypothesis: Compression generalizes when calibration entropy >= held-out entropy.
The entropy measures how much of the activation manifold we've covered.

Connection to Energy-Based Models (LeCun):
- Energy = geodesic distance from manifold
- Low entropy calibration = narrow energy well = overfitting
- High entropy calibration = broad coverage = generalization

Entropy metrics:
1. Spectral entropy: H(λ) = -Σ p(λ) log p(λ) where p(λ) = λ_i / Σλ
2. Geodesic coverage: average pairwise geodesic distance
3. RMT signal entropy: entropy of singular values above MP edge
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_spectral_entropy(activations, backend):
    """Compute spectral entropy of activation covariance.

    H = -Σ p_i log(p_i) where p_i = λ_i / Σλ (normalized eigenvalues)

    High entropy = spread across many dimensions = good coverage
    Low entropy = concentrated in few dimensions = narrow coverage
    """
    b = backend

    # Center activations
    mean = b.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    b.eval(centered)

    # Compute covariance
    n = int(activations.shape[0])
    cov = b.matmul(b.transpose(centered), centered) / n
    b.eval(cov)

    # Eigenvalues of covariance
    eigenvalues = b.eigvalsh(cov)
    b.eval(eigenvalues)

    # Normalize to probabilities
    eigenvalues = b.maximum(eigenvalues, b.zeros_like(eigenvalues))  # Ensure non-negative
    total = b.sum(eigenvalues)
    b.eval(total)
    total_val = float(b.to_scalar(total))

    if total_val < 1e-10:
        return 0.0

    probs = eigenvalues / total_val
    b.eval(probs)

    # Compute entropy: H = -Σ p log(p)
    # Avoid log(0) by masking small values
    eps = 1e-10
    probs_safe = b.maximum(probs, b.full(probs.shape, eps))
    log_probs = b.log(probs_safe)
    entropy_terms = -probs * log_probs
    b.eval(entropy_terms)

    # Mask out terms where p was effectively 0
    mask = probs > eps
    entropy_terms_masked = b.where(mask, entropy_terms, b.zeros_like(entropy_terms))

    entropy = b.sum(entropy_terms_masked)
    b.eval(entropy)

    return float(b.to_scalar(entropy))


def compute_geodesic_coverage(activations, backend):
    """Compute geodesic coverage: mean pairwise geodesic distance.

    High coverage = activations spread across manifold
    Low coverage = activations clustered together
    """
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_distance_matrix

    b = backend

    # Compute geodesic distances
    geo_dist = geodesic_distance_matrix(activations, backend=b)
    b.eval(geo_dist)

    # Mean of off-diagonal elements
    n = int(activations.shape[0])
    mask = 1.0 - b.eye(n)
    masked_dist = geo_dist * mask

    total_dist = b.sum(masked_dist)
    n_pairs = n * (n - 1)
    b.eval(total_dist)

    mean_dist = float(b.to_scalar(total_dist)) / n_pairs if n_pairs > 0 else 0.0

    return mean_dist


def compute_rmt_signal_entropy(activations, backend):
    """Compute entropy of RMT signal singular values.

    Uses Marchenko-Pastur to identify signal, then computes
    entropy of the signal component distribution.
    """
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )

    b = backend

    # Center and SVD
    mean = b.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    b.eval(centered)

    _, S, _ = b.svd(centered)
    b.eval(S)

    n_samples = int(activations.shape[0])
    n_features = int(activations.shape[1])

    # RMT signal/noise separation
    mp_result = compute_signal_rank_from_singular_values(
        S, n_samples=n_samples, n_features=n_features, backend=b
    )

    signal_rank = max(1, int(mp_result.signal_rank))

    # Get signal singular values
    S_signal = S[:signal_rank]
    b.eval(S_signal)

    # Convert to variance (squared singular values)
    S_sq = S_signal * S_signal
    total = b.sum(S_sq)
    b.eval(total)
    total_val = float(b.to_scalar(total))

    if total_val < 1e-10:
        return 0.0, signal_rank

    probs = S_sq / total_val
    b.eval(probs)

    # Entropy
    eps = 1e-10
    probs_safe = b.maximum(probs, b.full(probs.shape, eps))
    log_probs = b.log(probs_safe)
    entropy_terms = -probs * log_probs

    entropy = b.sum(entropy_terms)
    b.eval(entropy)

    return float(b.to_scalar(entropy)), signal_rank


def run_experiment():
    """Correlate calibration entropy with held-out accuracy."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Calibration prompts - varying diversity
    diverse_prompts = [
        # Science
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "DNA replication occurs when",
        "The periodic table organizes",
        # History
        "The Roman Empire fell because",
        "World War II began when",
        "The Renaissance was a period of",
        "Ancient Egypt developed",
        # Technology
        "Machine learning algorithms",
        "The internet was invented",
        "Artificial intelligence",
        "Computer programming involves",
        # Philosophy
        "Plato's theory of forms",
        "Kant's categorical imperative",
        "Existentialism emphasizes",
        "The problem of consciousness",
        # Math
        "The derivative of a function",
        "Prime numbers are",
        "Calculus was invented by",
        "Statistical inference",
        # Literature
        "Shakespeare wrote",
        "The novel as a form",
        "Poetry differs from prose",
        "Narrative structure",
        # Biology
        "Evolution by natural selection",
        "Cells divide through",
        "The human brain contains",
        "Photosynthesis converts",
        # Physics
        "Newton's laws state",
        "Thermodynamics describes",
        "Electromagnetic waves",
        "The speed of light",
    ]

    held_out_prompts = [
        "The capital of Japan is",
        "Water boils at",
        "The largest ocean is",
        "Gravity causes objects to",
    ]

    # Vary calibration size to test entropy correlation
    calibration_sizes = [8, 16, 24, 32]

    compressor = RMTAwareCompressor(backend=backend)

    test_layers = [1, 5, 6, 7]

    all_results = []

    for cal_size in calibration_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"CALIBRATION SIZE: {cal_size}")
        logger.info(f"{'='*60}")

        cal_prompts = diverse_prompts[:cal_size]
        cal_tokens = [tokenizer.encode(p) for p in cal_prompts]
        held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

        for layer_idx in test_layers:
            logger.info(f"\n--- Layer {layer_idx} ---")

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

            # Collect held-out activations
            held_inputs = []
            for tokens in held_tokens:
                input_ids = mx.array([tokens])
                mlp_input = None

                layer = model.model.layers[layer_idx]
                original_mlp = layer.mlp

                class MLPHook:
                    def __init__(self, mlp):
                        self.mlp = mlp
                    def __call__(self, x):
                        nonlocal mlp_input
                        mlp_input = x
                        return self.mlp(x)

                layer.mlp = MLPHook(original_mlp)
                try:
                    _ = model(input_ids)
                    mx.eval(mlp_input)
                    held_inputs.append(mlp_input[0, -1, :])
                finally:
                    layer.mlp = original_mlp

            X_cal = mx.stack(cal_inputs).astype(mx.float32)
            Y_cal = mx.stack(cal_outputs).astype(mx.float32)
            X_held = mx.stack(held_inputs).astype(mx.float32)
            mx.eval(X_cal, Y_cal, X_held)

            X_cal_backend = backend.array(X_cal)
            Y_cal_backend = backend.array(Y_cal)
            X_held_backend = backend.array(X_held)

            # Compute entropy metrics
            logger.info("Computing entropy metrics...")

            cal_spectral_entropy = compute_spectral_entropy(X_cal_backend, backend)
            cal_rmt_entropy, cal_signal_rank = compute_rmt_signal_entropy(X_cal_backend, backend)
            cal_geodesic_coverage = compute_geodesic_coverage(X_cal_backend, backend)

            held_spectral_entropy = compute_spectral_entropy(X_held_backend, backend)
            held_rmt_entropy, held_signal_rank = compute_rmt_signal_entropy(X_held_backend, backend)
            held_geodesic_coverage = compute_geodesic_coverage(X_held_backend, backend)

            logger.info(f"Calibration: spectral_H={cal_spectral_entropy:.2f}, rmt_H={cal_rmt_entropy:.2f}, geo_cov={cal_geodesic_coverage:.4f}")
            logger.info(f"Held-out:    spectral_H={held_spectral_entropy:.2f}, rmt_H={held_rmt_entropy:.2f}, geo_cov={held_geodesic_coverage:.4f}")

            # Entropy ratios
            spectral_ratio = cal_spectral_entropy / (held_spectral_entropy + 1e-6)
            rmt_ratio = cal_rmt_entropy / (held_rmt_entropy + 1e-6)
            geo_ratio = cal_geodesic_coverage / (held_geodesic_coverage + 1e-6)

            logger.info(f"Ratios: spectral={spectral_ratio:.2f}, rmt={rmt_ratio:.2f}, geo={geo_ratio:.2f}")

            # Compress and evaluate
            rmt_result = compressor.compress_layer(X_cal_backend, Y_cal_backend)

            # Evaluate on held-out
            T_mx = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
            mx.eval(T_mx)

            correct = 0
            total = 0

            for tokens in held_tokens:
                input_ids = mx.array([tokens])

                # Original
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

            held_out_accuracy = correct / total if total > 0 else 0.0
            logger.info(f"Held-out accuracy: {held_out_accuracy:.1%}")

            all_results.append({
                "cal_size": cal_size,
                "layer": layer_idx,
                "cal_spectral_H": cal_spectral_entropy,
                "held_spectral_H": held_spectral_entropy,
                "spectral_ratio": spectral_ratio,
                "cal_rmt_H": cal_rmt_entropy,
                "held_rmt_H": held_rmt_entropy,
                "rmt_ratio": rmt_ratio,
                "cal_geo_cov": cal_geodesic_coverage,
                "held_geo_cov": held_geodesic_coverage,
                "geo_ratio": geo_ratio,
                "accuracy": held_out_accuracy,
            })

    # Correlation analysis
    logger.info(f"\n{'='*60}")
    logger.info("CORRELATION ANALYSIS")
    logger.info(f"{'='*60}")

    # Group by layer
    from statistics import correlation

    for layer_idx in test_layers:
        layer_results = [r for r in all_results if r["layer"] == layer_idx]

        if len(layer_results) >= 3:
            spectral_ratios = [r["spectral_ratio"] for r in layer_results]
            rmt_ratios = [r["rmt_ratio"] for r in layer_results]
            geo_ratios = [r["geo_ratio"] for r in layer_results]
            accuracies = [r["accuracy"] for r in layer_results]

            try:
                corr_spectral = correlation(spectral_ratios, accuracies)
                corr_rmt = correlation(rmt_ratios, accuracies)
                corr_geo = correlation(geo_ratios, accuracies)

                logger.info(f"\nLayer {layer_idx}:")
                logger.info(f"  Correlation(spectral_ratio, accuracy) = {corr_spectral:.4f}")
                logger.info(f"  Correlation(rmt_ratio, accuracy) = {corr_rmt:.4f}")
                logger.info(f"  Correlation(geo_ratio, accuracy) = {corr_geo:.4f}")
            except Exception as e:
                logger.warning(f"Correlation failed for layer {layer_idx}: {e}")

    # Overall summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Cal Size':<10} {'Layer':<8} {'Spectral R':<12} {'RMT R':<10} {'Geo R':<10} {'Accuracy':<10}")
    logger.info("-" * 70)

    for r in all_results:
        logger.info(
            f"{r['cal_size']:<10} {r['layer']:<8} {r['spectral_ratio']:<12.2f} "
            f"{r['rmt_ratio']:<10.2f} {r['geo_ratio']:<10.2f} {r['accuracy']*100:<10.1f}"
        )

    # Check if higher entropy ratio predicts higher accuracy
    high_ratio_results = [r for r in all_results if r["spectral_ratio"] > 1.0]
    low_ratio_results = [r for r in all_results if r["spectral_ratio"] <= 1.0]

    if high_ratio_results and low_ratio_results:
        avg_high = sum(r["accuracy"] for r in high_ratio_results) / len(high_ratio_results)
        avg_low = sum(r["accuracy"] for r in low_ratio_results) / len(low_ratio_results)

        logger.info(f"\n--- Hypothesis Check ---")
        logger.info(f"Avg accuracy when spectral_ratio > 1: {avg_high:.1%}")
        logger.info(f"Avg accuracy when spectral_ratio <= 1: {avg_low:.1%}")

        if avg_high > avg_low + 0.1:
            logger.info(">>> HYPOTHESIS SUPPORTED: Higher entropy coverage = better generalization")
        else:
            logger.info(">>> HYPOTHESIS NOT SUPPORTED")


if __name__ == "__main__":
    run_experiment()
