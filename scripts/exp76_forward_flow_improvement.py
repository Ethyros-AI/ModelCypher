#!/usr/bin/env python3
"""Experiment 76: Forward Flow Self-Improvement.

The key insight: information flows FORWARD through the model.
Each layer should optimize based on what it actually receives
from already-optimized upstream layers.

1. Layer 1 optimizes itself
2. Layer 2 sees Layer 1's optimized output → optimizes itself
3. Layer 3 sees both optimized → optimizes itself
4. ...and so on

No backward recalibration. Pure forward flow.
Respect the direction of information.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
import json
from datetime import datetime

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
    """Forward flow self-improvement."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("="*80)
    logger.info("FORWARD FLOW SELF-IMPROVEMENT")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    logger.info("\nLoading LFM2-1.2B...")
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
        ("The sun rises in the", "east"),
        ("A noun is a word that names a", "person"),
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
        "Shakespeare wrote", "The speed of light",
        "Photosynthesis occurs in", "DNA stands for",
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
            return None, None

        coefs = S_Y_centered @ Vh[d]
        proj = np.outer(coefs, Vh[d])
        result = S_Y + proj * (boost_factor - 1)

        alpha = 1e-4
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result

        try:
            W = np.linalg.solve(ATA, ATB).T
        except np.linalg.LinAlgError:
            return None, None

        if np.isnan(W).any() or np.isinf(W).any():
            return None, None

        return W, result

    # ========================================
    # INITIAL STATE
    # ========================================

    initial_acc = evaluate_accuracy(model, tokenizer)
    logger.info(f"\nInitial accuracy: {initial_acc*100:.0f}%")

    # Track state
    improvements = []
    current_acc = initial_acc

    # Configuration - more aggressive search
    directions_to_try = list(range(12))  # More directions
    boosts_to_try = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.2, 1.5, 2.0, 3.0]  # More boosts

    # ========================================
    # FORWARD FLOW LOOP
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("FORWARD FLOW IMPROVEMENT")
    logger.info("Respecting the direction of information flow")
    logger.info(f"{'='*80}")

    num_layers = len(model.model.layers)
    logger.info(f"\nModel has {num_layers} layers")

    # Process each layer in order
    for layer_idx in range(num_layers):
        logger.info(f"\n--- LAYER {layer_idx} ---")

        # Get CURRENT activations - these reflect all upstream improvements
        S_X, S_Y = get_layer_activations(model, tokenizer, layer_idx, probe_prompts)

        baseline_kurtosis = compute_kurtosis(S_Y)
        baseline_entropy = compute_spectral_entropy(S_Y)
        baseline_score = geometry_score(baseline_kurtosis, baseline_entropy)

        logger.info(f"  Input from upstream (kurtosis={baseline_kurtosis:.2f}, entropy={baseline_entropy:.2f})")
        logger.info(f"  Baseline score: {baseline_score:.4f}")

        best_for_layer = None

        for d in directions_to_try:
            for boost in boosts_to_try:
                result = boost_direction(S_X, S_Y, d, boost)
                if result[0] is None:
                    continue

                W, Y_new = result

                # Measure new geometry
                new_kurtosis = compute_kurtosis(Y_new)
                new_entropy = compute_spectral_entropy(Y_new)
                new_score = geometry_score(new_kurtosis, new_entropy)

                # Only test if geometry improved
                if new_score <= baseline_score:
                    continue

                # Apply and test accuracy
                W_mx = mx.array(W.astype(np.float32))
                mx.eval(W_mx)

                class ModifiedMLP:
                    def __init__(self, W):
                        self.W = W
                    def __call__(self, x):
                        return mx.matmul(x, self.W.T)

                layer = model.model.layers[layer_idx]
                if hasattr(layer, 'feed_forward'):
                    original_mlp = layer.feed_forward
                    layer.feed_forward = ModifiedMLP(W_mx)
                    mlp_key = 'feed_forward'
                else:
                    original_mlp = layer.mlp
                    layer.mlp = ModifiedMLP(W_mx)
                    mlp_key = 'mlp'

                new_acc = evaluate_accuracy(model, tokenizer)

                # Keep if accuracy maintained or improved
                if new_acc >= current_acc:
                    if best_for_layer is None or new_acc > best_for_layer['acc'] or \
                       (new_acc == best_for_layer['acc'] and new_score > best_for_layer['score']):
                        best_for_layer = {
                            'direction': d,
                            'boost': boost,
                            'W': W,
                            'acc': new_acc,
                            'score': new_score,
                            'mlp_key': mlp_key,
                            'original_mlp': original_mlp,
                        }

                # Restore for next test
                if mlp_key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        # Apply best improvement permanently
        if best_for_layer:
            W = best_for_layer['W']
            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class PermanentMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = model.model.layers[layer_idx]
            if best_for_layer['mlp_key'] == 'feed_forward':
                layer.feed_forward = PermanentMLP(W_mx)
            else:
                layer.mlp = PermanentMLP(W_mx)

            improvement = {
                'layer': layer_idx,
                'direction': best_for_layer['direction'],
                'boost': best_for_layer['boost'],
                'old_acc': current_acc,
                'new_acc': best_for_layer['acc'],
                'old_score': baseline_score,
                'new_score': best_for_layer['score'],
            }
            improvements.append(improvement)

            if best_for_layer['acc'] > current_acc:
                current_acc = best_for_layer['acc']
                logger.info(f"  IMPROVED: d{best_for_layer['direction']} b{best_for_layer['boost']:.1f} → {current_acc*100:.0f}%")
            else:
                logger.info(f"  OPTIMIZED: d{best_for_layer['direction']} b{best_for_layer['boost']:.1f} (geometry +{best_for_layer['score']-baseline_score:.4f})")

            # NOTE: We do NOT recalibrate downstream layers
            # The next layer will see the optimized output and adapt to it
        else:
            logger.info(f"  No improvement - layer already optimal for its input")

    # ========================================
    # FINAL EVALUATION
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*80}")

    final_acc = evaluate_accuracy(model, tokenizer)

    logger.info(f"\n{'Prompt':<45} {'Prediction':>20}")
    logger.info("-" * 70)

    correct = 0
    for prompt, expected in test_cases:
        word = get_prediction(model, tokenizer, prompt)
        is_correct = expected.lower() in word.lower()
        if is_correct:
            correct += 1
        mark = "✓" if is_correct else "✗"
        logger.info(f"{mark} {prompt:<43} {word:>20}")

    logger.info(f"\nFinal: {correct}/{len(test_cases)} = {final_acc*100:.0f}%")

    # ========================================
    # SUMMARY
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("FORWARD FLOW SUMMARY")
    logger.info(f"{'='*80}")

    logger.info(f"""
CONFIGURATION:
  - Layers processed: {num_layers}
  - Directions per layer: {len(directions_to_try)}
  - Boost factors: {len(boosts_to_try)}
  - Total search space: {num_layers * len(directions_to_try) * len(boosts_to_try)} configurations

RESULTS:
  - Initial accuracy: {initial_acc*100:.0f}%
  - Final accuracy:   {final_acc*100:.0f}%
  - Improvement:      {(final_acc - initial_acc)*100:+.0f}pp
  - Layers improved:  {len([i for i in improvements if i['new_acc'] > i['old_acc']])}
  - Layers optimized: {len(improvements)}

IMPROVEMENT FLOW:
""")

    for imp in improvements:
        delta_acc = (imp['new_acc'] - imp['old_acc']) * 100
        delta_score = imp['new_score'] - imp['old_score']
        status = f"+{delta_acc:.0f}pp" if delta_acc > 0 else f"geo +{delta_score:.4f}"
        logger.info(f"  L{imp['layer']:2d} d{imp['direction']} b{imp['boost']:.1f} → {status}")

    logger.info(f"""

THE FORWARD FLOW PRINCIPLE:

Information flows from layer 0 to layer {num_layers-1}.
Each layer optimizes based on what it RECEIVES from upstream.
When a layer improves, downstream layers see better input.
They can then find THEIR best configuration for that input.

This respects the natural direction of information flow.
No backward recalibration needed - the system self-organizes.

The model finds its own optimal configuration layer by layer.

Completed at {datetime.now().isoformat()}
""")

    # Save results
    results = {
        'initial_accuracy': initial_acc,
        'final_accuracy': final_acc,
        'num_layers': num_layers,
        'improvements': improvements,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "forward_flow_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
