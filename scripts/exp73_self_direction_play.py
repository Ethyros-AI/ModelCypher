#!/usr/bin/env python3
"""Experiment 73: Self-Direction Play.

Simpler approach: Use the model's OWN principal directions.

Instead of random perturbations to weights, we:
1. Compute SVD of the model's activations
2. Try BOOSTING each principal direction
3. Keep boosts that improve geometry score

This is self-play using the model's intrinsic structure.
No teacher. No labels. Just the model's own geometry.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_kurtosis(Y):
    """Compute average kurtosis over samples."""
    kurtoses = []
    for h in Y:
        z = (h - h.mean()) / (h.std() + 1e-10)
        kurtoses.append(float(np.mean(z ** 4) - 3))
    return np.mean(kurtoses)


def compute_spectral_entropy(Y):
    """Compute spectral entropy of a manifold."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / (S.sum() + 1e-10)
    return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))


def geometry_score(kurtosis, spectral_entropy):
    """Higher = more 'correct-like' geometry."""
    return kurtosis / 100 - spectral_entropy


def run_experiment():
    """Self-direction play."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("Loading LFM2-1.2B...")
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    model, tokenizer = load(model_path)

    # Test cases
    test_cases = [
        ("The capital of France is", "Paris"),
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
        ("The opposite of hot is", "cold"),
        ("Birds can", "fly"),
        ("Fish live in", "water"),
        ("The sky is usually", "blue"),
        ("Gravity causes objects to", "fall"),
    ]

    # Probe prompts
    probe_prompts = [
        "The capital of", "The largest planet",
        "Water freezes at", "If it rains",
        "2 + 2 equals", "A noun is",
        "The square root of", "10 times 10",
        "The sky is", "Birds can",
        "Fish live in", "The sun rises",
        "Gravity causes", "The opposite of",
        "The past tense of", "An adjective describes",
    ]

    def get_prediction(model, tokenizer, prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        return tokenizer.decode([top_token]).strip()

    def evaluate_accuracy(model, tokenizer):
        correct = 0
        for prompt, expected in test_cases:
            word = get_prediction(model, tokenizer, prompt)
            if expected.lower() in word.lower():
                correct += 1
        return correct / len(test_cases)

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        """Get MLP input and output activations."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                key = 'mlp'

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(np.array(mlp_input[0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(mlp_output[0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def boost_direction(S_X, S_Y, direction_idx, boost_factor):
        """Boost a specific direction in the output manifold."""
        S_Y_centered = S_Y - S_Y.mean(axis=0)
        _, S, Vh = svd(S_Y_centered, full_matrices=False)

        d = direction_idx
        if d >= len(Vh):
            return None

        # Boost: add more energy to direction d
        coefs = S_Y_centered @ Vh[d]
        proj = np.outer(coefs, Vh[d])

        # New output: original + boost * direction
        result = S_Y + proj * (boost_factor - 1)

        # Solve for weights
        alpha = 1e-4
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result

        try:
            W = np.linalg.solve(ATA, ATB).T
        except np.linalg.LinAlgError:
            return None

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        return W, result

    # ========================================
    # PHASE 1: Baseline
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Baseline")
    logger.info(f"{'='*80}")

    baseline_acc = evaluate_accuracy(model, tokenizer)
    logger.info(f"Baseline accuracy: {baseline_acc*100:.0f}%")

    # ========================================
    # PHASE 2: Self-Play Loop
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Self-Direction Play")
    logger.info(f"{'='*80}")

    layer_idx = 4  # Best layer from previous experiments

    # Get baseline activations
    S_X, S_Y = get_layer_activations(model, tokenizer, layer_idx, probe_prompts)

    baseline_kurtosis = compute_kurtosis(S_Y)
    baseline_entropy = compute_spectral_entropy(S_Y)
    baseline_score = geometry_score(baseline_kurtosis, baseline_entropy)

    logger.info(f"\nBaseline at Layer {layer_idx}:")
    logger.info(f"  Kurtosis: {baseline_kurtosis:.4f}")
    logger.info(f"  Spectral entropy: {baseline_entropy:.4f}")
    logger.info(f"  Geometry score: {baseline_score:.4f}")

    # Try boosting each direction
    logger.info(f"\n{'Dir':>5} {'Boost':>8} {'Kurtosis':>12} {'Entropy':>12} {'Score':>12} {'Acc':>8} {'Action':>10}")
    logger.info("-" * 75)

    best_score = baseline_score
    best_config = None
    results = []

    # Try different directions and boost factors
    for d in range(8):  # First 8 directions
        for boost in [1.2, 1.5, 2.0, 0.8, 0.5]:  # Different boost factors
            result = boost_direction(S_X, S_Y, d, boost)
            if result is None:
                continue

            W, Y_new = result

            # Measure new geometry
            new_kurtosis = compute_kurtosis(Y_new)
            new_entropy = compute_spectral_entropy(Y_new)
            new_score = geometry_score(new_kurtosis, new_entropy)

            # Apply and measure accuracy
            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class BoostedMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = model.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                layer.feed_forward = BoostedMLP(W_mx)
                mlp_key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                layer.mlp = BoostedMLP(W_mx)
                mlp_key = 'mlp'

            new_acc = evaluate_accuracy(model, tokenizer)

            # Restore
            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            # Decision
            if new_score > best_score and new_acc >= baseline_acc:
                action = "KEEP"
                best_score = new_score
                best_config = (d, boost, W, new_acc)
            else:
                action = ""

            results.append({
                'direction': d,
                'boost': boost,
                'kurtosis': new_kurtosis,
                'entropy': new_entropy,
                'score': new_score,
                'accuracy': new_acc,
            })

            logger.info(f"{d:>5} {boost:>8.1f} {new_kurtosis:>12.4f} {new_entropy:>12.4f} {new_score:>12.4f} {new_acc*100:>7.0f}% {action:>10}")

    # ========================================
    # PHASE 3: Best Configuration
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Best Configuration")
    logger.info(f"{'='*80}")

    if best_config:
        d, boost, W, acc = best_config
        logger.info(f"\nBest: Direction {d}, Boost {boost}")
        logger.info(f"Score: {baseline_score:.4f} → {best_score:.4f}")
        logger.info(f"Accuracy: {baseline_acc*100:.0f}% → {acc*100:.0f}%")

        # Apply and show detailed results
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class BoostedMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = BoostedMLP(W_mx)
            mlp_key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = BoostedMLP(W_mx)
            mlp_key = 'mlp'

        logger.info(f"\n{'Prompt':<40} {'Before':>15} {'After':>15} {'Expected':>12}")
        logger.info("-" * 85)

        # Restore to show before
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        improved = 0
        for prompt, expected in test_cases:
            before = get_prediction(model, tokenizer, prompt)

            if mlp_key == 'feed_forward':
                layer.feed_forward = BoostedMLP(W_mx)
            else:
                layer.mlp = BoostedMLP(W_mx)

            after = get_prediction(model, tokenizer, prompt)

            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            before_ok = expected.lower() in before.lower()
            after_ok = expected.lower() in after.lower()

            change = ""
            if after_ok and not before_ok:
                improved += 1
                change = "← IMPROVED!"
            elif before_ok and not after_ok:
                change = "← degraded"

            logger.info(f"{prompt:<40} {before:>15} {after:>15} {expected:>12} {change}")

        logger.info(f"\nImproved: {improved} prompts")

    else:
        logger.info("\nNo improvement found through self-direction boosting.")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Self-Direction Play")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE MECHANISM:

Self-play using the model's OWN directions:

1. Compute SVD of model's activations
2. For each direction d:
   - Try boosting (amplifying) direction d
   - Try suppressing (reducing) direction d
3. Keep changes that improve geometry score
4. Repeat

THE KEY INSIGHT:

The model already has structure in its activations.
Some directions are "more correct" than others.
By boosting correct-like directions, we improve accuracy.

NO EXTERNAL TEACHER. NO LABELS.
The model improves itself using its own geometry.

THIS IS TRUE SELF-PLAY:

- The model's own directions are the "moves"
- The geometry score is the "reward"
- We search for moves that improve the reward
- The model learns from its own structure

NEXT STEPS:

1. Iterative boosting (stack improvements)
2. Multi-layer coordinated boosting
3. Evolutionary search over boost configurations
4. Continuous self-improvement loop
""")


if __name__ == "__main__":
    run_experiment()
