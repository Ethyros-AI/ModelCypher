#!/usr/bin/env python3
"""Experiment 26: Entropy Dynamics Under Compression.

User insight: "errors don't drop entropy. they don't reduce a model's confusion state."

Hypothesis: Compression doesn't add noise (which would increase entropy).
Instead, it rotates the output distribution, preserving or even reducing entropy.

If TRUE:
- Entropy should stay constant or decrease with compression
- The model stays confident, just about different tokens
- "Error" is a rotation, not diffusion

If FALSE:
- Entropy should increase with compression
- Model becomes less confident (more confused)
- Error is noise/diffusion

This changes everything about how we understand compression failure.
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

    # Softmax
    max_logit = mx.max(logits)
    shifted = logits - max_logit
    exp_logits = mx.exp(shifted)
    sum_exp = mx.sum(exp_logits)
    probs = exp_logits / sum_exp
    mx.eval(probs)

    # Entropy: -Σ p log p
    # Avoid log(0) by adding small epsilon
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    mx.eval(entropy)

    return float(entropy.item())


def compute_top_k_entropy(logits, k=10):
    """Compute entropy over just top-k tokens (where decisions happen)."""
    import mlx.core as mx

    # Get top-k
    sorted_indices = mx.argsort(logits)[::-1]
    top_k_logits = logits[sorted_indices[:k]]
    mx.eval(top_k_logits)

    # Softmax over top-k
    max_logit = mx.max(top_k_logits)
    shifted = top_k_logits - max_logit
    exp_logits = mx.exp(shifted)
    sum_exp = mx.sum(exp_logits)
    probs = exp_logits / sum_exp
    mx.eval(probs)

    # Entropy
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    mx.eval(entropy)

    return float(entropy.item())


def run_experiment():
    """Track entropy through sequential compression."""
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
    vocab_size = model.model.embed_tokens.weight.shape[0]
    logger.info(f"Model has {n_layers} layers, vocab size {vocab_size}")
    logger.info(f"Max possible entropy: ln({vocab_size}) = {math.log(vocab_size):.2f}")

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Water freezes at",
        "The largest planet is",
        "DNA stands for",
        "The speed of light is",
        "Photosynthesis occurs in",
        "The periodic table organizes",
        "Machine learning algorithms",
    ]

    # Calibration prompts
    calibration_prompts = [
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "Evolution by natural selection",
        "Neural networks are",
        "The derivative of a function",
        "Prime numbers are",
        "Shakespeare wrote",
        "The human brain contains",
        "Newton's laws state",
        "Climate change refers to",
        "The Amazon rainforest",
        "Gravity causes objects to",
        "Chemical bonds form when",
        "Cells divide through",
        "The internet was invented",
        "Artificial intelligence",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    test_tokens = [tokenizer.encode(p) for p in test_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Pre-compress transmission zone layers
    transmission_layers = list(range(8, 34))
    layer_T = {}

    logger.info(f"\n--- Pre-compressing layers ---")

    for layer_idx in transmission_layers:
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

        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T_mx = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T_mx)
        layer_T[layer_idx] = T_mx

        if layer_idx % 5 == 0:
            logger.info(f"Compressed layer {layer_idx}")

    def get_logits_with_compression(tokens, layer_indices):
        """Get final logits with specified layers compressed."""
        input_ids = mx.array([tokens])

        original_mlps = {}
        for idx in layer_indices:
            if idx in layer_T:
                layer = model.model.layers[idx]
                original_mlps[idx] = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T
                    def __call__(self, x):
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(layer_T[idx])

        try:
            logits = model(input_ids)
            mx.eval(logits)
            return logits[0, -1, :]
        finally:
            for idx in layer_indices:
                if idx in original_mlps:
                    model.model.layers[idx].mlp = original_mlps[idx]

    # Track entropy through compression
    logger.info(f"\n{'='*60}")
    logger.info("ENTROPY DYNAMICS THROUGH SEQUENTIAL COMPRESSION")
    logger.info(f"{'='*60}")

    start_layer = 15
    results = []

    for n_compressed in range(0, 13):
        layer_indices = list(range(start_layer, start_layer + n_compressed)) if n_compressed > 0 else []

        full_entropies = []
        top10_entropies = []
        flips = 0

        for tokens in test_tokens:
            # Original
            orig_logits = get_logits_with_compression(tokens, [])
            orig_full_H = compute_entropy(orig_logits)
            orig_top10_H = compute_top_k_entropy(orig_logits, k=10)
            orig_top = int(mx.argmax(orig_logits).item())

            # Compressed
            comp_logits = get_logits_with_compression(tokens, layer_indices)
            comp_full_H = compute_entropy(comp_logits)
            comp_top10_H = compute_top_k_entropy(comp_logits, k=10)
            comp_top = int(mx.argmax(comp_logits).item())

            # Entropy change
            full_entropies.append(comp_full_H - orig_full_H)
            top10_entropies.append(comp_top10_H - orig_top10_H)

            if comp_top != orig_top:
                flips += 1

        avg_full_delta = sum(full_entropies) / len(full_entropies)
        avg_top10_delta = sum(top10_entropies) / len(top10_entropies)
        accuracy = 1 - flips / len(test_tokens)

        results.append({
            "n": n_compressed,
            "full_H_delta": avg_full_delta,
            "top10_H_delta": avg_top10_delta,
            "accuracy": accuracy,
        })

        direction = "↑" if avg_full_delta > 0.01 else ("↓" if avg_full_delta < -0.01 else "→")

        logger.info(f"\nn={n_compressed} layers:")
        logger.info(f"  Full entropy Δ: {avg_full_delta:+.4f} {direction}")
        logger.info(f"  Top-10 entropy Δ: {avg_top10_delta:+.4f}")
        logger.info(f"  Accuracy: {accuracy:.1%}")

    # Analysis
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS: DOES COMPRESSION INCREASE ENTROPY?")
    logger.info(f"{'='*60}")

    # Check if entropy increases or decreases
    positive_deltas = [r for r in results if r["n"] > 0 and r["full_H_delta"] > 0.01]
    negative_deltas = [r for r in results if r["n"] > 0 and r["full_H_delta"] < -0.01]
    neutral_deltas = [r for r in results if r["n"] > 0 and abs(r["full_H_delta"]) <= 0.01]

    logger.info(f"\nEntropy INCREASED in {len(positive_deltas)}/12 cases")
    logger.info(f"Entropy DECREASED in {len(negative_deltas)}/12 cases")
    logger.info(f"Entropy UNCHANGED in {len(neutral_deltas)}/12 cases")

    if len(negative_deltas) >= len(positive_deltas):
        logger.info("\n>>> USER HYPOTHESIS SUPPORTED: Errors don't increase entropy!")
        logger.info(">>> Compression is ROTATION, not DIFFUSION")
    else:
        logger.info("\n>>> Errors DO increase entropy (diffusion model)")

    # Correlation between entropy change and accuracy
    logger.info(f"\n{'='*60}")
    logger.info("ENTROPY-ACCURACY CORRELATION")
    logger.info(f"{'='*60}")

    deltas = [r["full_H_delta"] for r in results if r["n"] > 0]
    accs = [r["accuracy"] for r in results if r["n"] > 0]

    if len(deltas) > 1:
        delta_mean = sum(deltas) / len(deltas)
        acc_mean = sum(accs) / len(accs)

        numerator = sum((deltas[i] - delta_mean) * (accs[i] - acc_mean) for i in range(len(deltas)))
        denom_delta = sum((d - delta_mean)**2 for d in deltas)
        denom_acc = sum((a - acc_mean)**2 for a in accs)

        if denom_delta > 0 and denom_acc > 0:
            corr = numerator / math.sqrt(denom_delta * denom_acc)
            logger.info(f"Correlation(entropy_change, accuracy) = {corr:.4f}")

            if corr < -0.3:
                logger.info(">>> Higher entropy → LOWER accuracy (noise interpretation)")
            elif corr > 0.3:
                logger.info(">>> Higher entropy → HIGHER accuracy (unusual!)")
            else:
                logger.info(">>> No strong correlation (entropy doesn't predict accuracy)")

    # The geometric interpretation
    logger.info(f"\n{'='*60}")
    logger.info("GEOMETRIC INTERPRETATION")
    logger.info(f"{'='*60}")

    logger.info("""
Two models of compression error:

MODEL 1: DIFFUSION (noise)
  - Error adds random perturbation to logits
  - Perturbation spreads probability mass
  - Entropy INCREASES
  - Model gets MORE confused
  - Eventually, top-1 and top-2 become indistinguishable

MODEL 2: ROTATION (bias)
  - Error rotates the logit vector
  - Probability mass shifts, doesn't spread
  - Entropy UNCHANGED or DECREASES
  - Model stays confident, just about different tokens
  - Ranking flips discretely when rotation crosses boundary

If the user's hypothesis is correct (Model 2):
  - Compression errors are SYSTEMATIC biases
  - They could potentially be CORRECTED
  - The phase transition is a BIFURCATION, not a noise threshold
  - Spread compression works by decorrelating biases, not dispersing noise
""")

    # Compare spread vs sequential entropy
    logger.info(f"\n{'='*60}")
    logger.info("SPREAD VS SEQUENTIAL ENTROPY")
    logger.info(f"{'='*60}")

    spread_5 = [10, 15, 20, 25, 30]
    seq_5 = list(range(15, 20))

    for pattern_name, layers in [("Spread", spread_5), ("Sequential", seq_5)]:
        entropies = []
        flips = 0

        for tokens in test_tokens:
            orig_logits = get_logits_with_compression(tokens, [])
            orig_H = compute_entropy(orig_logits)
            orig_top = int(mx.argmax(orig_logits).item())

            comp_logits = get_logits_with_compression(tokens, layers)
            comp_H = compute_entropy(comp_logits)
            comp_top = int(mx.argmax(comp_logits).item())

            entropies.append(comp_H - orig_H)
            if comp_top != orig_top:
                flips += 1

        avg_H_delta = sum(entropies) / len(entropies)
        accuracy = 1 - flips / len(test_tokens)

        logger.info(f"\n{pattern_name} {layers}:")
        logger.info(f"  Entropy Δ: {avg_H_delta:+.4f}")
        logger.info(f"  Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    run_experiment()
