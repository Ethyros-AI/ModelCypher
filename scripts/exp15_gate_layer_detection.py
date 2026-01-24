#!/usr/bin/env python3
"""Experiment 15: Gate Layer Auto-Detection.

Tests whether top-1 singular value energy > 50% reliably predicts gate layers.

Gate layers are layers where compression always fails regardless of:
- Calibration size
- Entropy coverage
- RMT filtering

Hypothesis: Layers with top-1 energy > 50% will have <50% accuracy after compression.

This would give us an automatic way to detect uncompressible layers without trial-and-error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_top1_energy(activations, backend):
    """Compute fraction of total variance in top singular value.

    Returns a value between 0 and 1:
    - 1.0 = ALL variance in one direction (extreme gate)
    - 0.0 = uniform spread across all directions
    """
    b = backend

    # Center activations
    mean = b.mean(activations, axis=0, keepdims=True)
    centered = activations - mean
    b.eval(centered)

    # SVD
    _, S, _ = b.svd(centered)
    b.eval(S)

    # Squared singular values = variance
    S_sq = S * S
    total_var = b.sum(S_sq)
    b.eval(total_var)

    total_val = float(b.to_scalar(total_var))
    if total_val < 1e-10:
        return 0.0

    # Top-1 energy
    top1_var = S_sq[0]
    b.eval(top1_var)
    top1_energy = float(b.to_scalar(top1_var)) / total_val

    return top1_energy


def run_experiment():
    """Profile all 36 layers and correlate top-1 energy with compression accuracy."""
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

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Calibration prompts - diverse set
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
        "Neurons transmit signals by",
        "Chemical bonds form when",
        "Democracy originated in",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    results = []

    for layer_idx in range(n_layers):
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect activations
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

        X_backend = backend.array(X_cal)
        Y_backend = backend.array(Y_cal)

        # Compute top-1 energy
        top1_energy = compute_top1_energy(X_backend, backend)
        logger.info(f"Top-1 energy: {top1_energy:.1%}")

        # Compress with RMT
        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        logger.info(f"RMT signal_rank: {rmt_result.signal_rank}/{rmt_result.total_rank}")

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

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Held-out accuracy: {accuracy:.1%}")

        # Classify as gate layer
        is_gate = top1_energy > 0.5
        logger.info(f"Gate layer (top1 > 50%): {is_gate}")

        results.append({
            "layer": layer_idx,
            "top1_energy": top1_energy,
            "accuracy": accuracy,
            "signal_rank": rmt_result.signal_rank,
            "is_gate": is_gate,
        })

    # Analysis
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS")
    logger.info(f"{'='*60}")

    # Summary table
    logger.info(f"\n{'Layer':<8} {'Top-1 Energy':<15} {'Accuracy':<12} {'Gate?':<8}")
    logger.info("-" * 50)

    for r in results:
        gate_marker = "GATE" if r["is_gate"] else ""
        logger.info(
            f"{r['layer']:<8} {r['top1_energy']*100:<15.1f} "
            f"{r['accuracy']*100:<12.1f} {gate_marker:<8}"
        )

    # Correlation analysis
    from statistics import correlation

    top1_energies = [r["top1_energy"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    try:
        corr = correlation(top1_energies, accuracies)
        logger.info(f"\nCorrelation(top1_energy, accuracy) = {corr:.4f}")
    except Exception as e:
        logger.warning(f"Correlation failed: {e}")

    # Validate hypothesis
    gate_layers = [r for r in results if r["is_gate"]]
    non_gate_layers = [r for r in results if not r["is_gate"]]

    if gate_layers:
        avg_gate_accuracy = sum(r["accuracy"] for r in gate_layers) / len(gate_layers)
        logger.info(f"\nGate layers ({len(gate_layers)}): avg accuracy = {avg_gate_accuracy:.1%}")

    if non_gate_layers:
        avg_non_gate_accuracy = sum(r["accuracy"] for r in non_gate_layers) / len(non_gate_layers)
        logger.info(f"Non-gate layers ({len(non_gate_layers)}): avg accuracy = {avg_non_gate_accuracy:.1%}")

    # Hypothesis check
    logger.info(f"\n--- Hypothesis Check ---")

    # Count gate layers with <50% accuracy
    gate_below_50 = [r for r in gate_layers if r["accuracy"] < 0.5]
    gate_above_50 = [r for r in gate_layers if r["accuracy"] >= 0.5]

    if gate_layers:
        logger.info(f"Gate layers with <50% accuracy: {len(gate_below_50)}/{len(gate_layers)}")

        if len(gate_below_50) / len(gate_layers) >= 0.8:
            logger.info(">>> HYPOTHESIS SUPPORTED: Top-1 energy > 50% predicts low accuracy")
        else:
            logger.info(">>> HYPOTHESIS NOT SUPPORTED")

    # Find optimal threshold
    logger.info(f"\n--- Threshold Search ---")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for thresh in thresholds:
        above = [r for r in results if r["top1_energy"] > thresh]
        below = [r for r in results if r["top1_energy"] <= thresh]

        if above and below:
            avg_above = sum(r["accuracy"] for r in above) / len(above)
            avg_below = sum(r["accuracy"] for r in below) / len(below)
            gap = avg_below - avg_above
            logger.info(f"Threshold {thresh:.0%}: above={avg_above:.1%} ({len(above)}), below={avg_below:.1%} ({len(below)}), gap={gap:.1%}")

    # Find 100% layers
    perfect_layers = [r["layer"] for r in results if r["accuracy"] >= 1.0 - 1e-6]
    logger.info(f"\n100% accuracy layers: {perfect_layers}")


if __name__ == "__main__":
    run_experiment()
