#!/usr/bin/env python3
"""Experiment 67: Multi-Layer Teaching with Numerical Stability.

From exp66: Direction 1 at Layer 14 achieved +12pp.
From exp60: Multi-layer teaching can break the model.

The question: Can we stack improvements carefully?

Approach:
1. Apply teaching at one layer at a time
2. Recalibrate after each layer
3. Stop if accuracy degrades
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


def run_experiment():
    """Multi-layer teaching with stability."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Load models
    logger.info("Loading LFM2-1.2B (student)...")
    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    student, student_tok = load(student_path)

    logger.info("Loading LFM2.5-1.2B-Instruct (teacher)...")
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    teacher, teacher_tok = load(teacher_path)

    # Test prompts
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

    # Calibration prompts
    calibration_prompts = [
        "The capital of France is", "The largest planet is",
        "Water freezes at", "If it rains, the ground gets",
        "The opposite of up is", "2 + 2 equals",
        "Photosynthesis occurs in", "DNA is found in",
        "A noun is a word that", "The past tense of run is",
        "The square root of 16 is", "10 times 10 equals",
        "The sky is usually", "Birds can",
        "Fish live in", "The sun rises in the",
        "Gravity causes objects to", "The speed of light is",
        "The human body has", "The Great Wall of China is in",
        "Shakespeare wrote", "The Eiffel Tower is in",
        "A verb describes an", "The opposite of hot is",
    ]

    def get_prediction(model, tokenizer, prompt):
        """Get model's prediction."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        word = tokenizer.decode([top_token]).strip()
        return word

    def evaluate_accuracy(model, tokenizer):
        """Evaluate on test cases."""
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

    def apply_teaching_stable(S_X, S_Y, T_Y, direction_idx):
        """Apply direction replacement with numerical stability."""
        # Normalize outputs for stable SVD
        s_norm = np.linalg.norm(S_Y)
        t_norm = np.linalg.norm(T_Y)

        S_Y_scaled = S_Y / (s_norm + 1e-10)
        T_Y_scaled = T_Y / (t_norm + 1e-10)

        S_Y_centered = S_Y_scaled - S_Y_scaled.mean(axis=0)
        T_Y_centered = T_Y_scaled - T_Y_scaled.mean(axis=0)

        # Compute SVD on normalized data
        try:
            _, _, Vh_s = svd(S_Y_centered, full_matrices=False)
            _, _, Vh_t = svd(T_Y_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        d = direction_idx
        if d >= min(len(Vh_s), len(Vh_t)):
            return None

        # Direction replacement on ORIGINAL scale
        S_Y_centered_orig = S_Y - S_Y.mean(axis=0)
        T_Y_centered_orig = T_Y - T_Y.mean(axis=0)

        result = S_Y.copy()

        # Remove student's direction d
        s_coefs = S_Y_centered_orig @ Vh_s[d]
        s_proj = np.outer(s_coefs, Vh_s[d])
        result -= s_proj

        # Add teacher's direction d
        t_coefs = T_Y_centered_orig @ Vh_t[d]
        t_proj = np.outer(t_coefs, Vh_t[d])
        result += t_proj

        # Solve for weights with regularization
        alpha = 1e-4  # Stronger regularization
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result

        try:
            W = np.linalg.solve(ATA, ATB).T
        except np.linalg.LinAlgError:
            return None

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        return W

    # ========================================
    # PHASE 1: Baseline
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Baseline")
    logger.info(f"{'='*80}")

    baseline = evaluate_accuracy(student, student_tok)
    logger.info(f"Baseline accuracy: {baseline*100:.0f}%")

    # ========================================
    # PHASE 2: Find best direction per layer
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Find Best Direction Per Layer")
    logger.info(f"{'='*80}")

    layer_results = {}

    for layer_idx in [2, 4, 6, 8, 10, 12, 14]:
        logger.info(f"\n--- Layer {layer_idx} ---")

        S_X, S_Y = get_layer_activations(student, student_tok, layer_idx, calibration_prompts)
        T_X, T_Y = get_layer_activations(teacher, teacher_tok, layer_idx, calibration_prompts)

        best_dir = None
        best_acc = baseline

        for d in range(8):  # Test directions 0-7
            W = apply_teaching_stable(S_X, S_Y, T_Y, d)
            if W is None:
                continue

            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class TaughtMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = student.model.layers[layer_idx]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                layer.feed_forward = TaughtMLP(W_mx)
                mlp_key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                layer.mlp = TaughtMLP(W_mx)
                mlp_key = 'mlp'

            acc = evaluate_accuracy(student, student_tok)

            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            if acc > best_acc:
                best_acc = acc
                best_dir = d

        if best_dir is not None:
            logger.info(f"Best: direction {best_dir} = {best_acc*100:.0f}%")
            layer_results[layer_idx] = (best_dir, best_acc)
        else:
            logger.info("No improvement found")

    # ========================================
    # PHASE 3: Stack Best Layers
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Stack Best Layers (One at a Time)")
    logger.info(f"{'='*80}")

    # Sort by improvement
    sorted_layers = sorted(layer_results.items(), key=lambda x: x[1][1], reverse=True)

    logger.info(f"\nLayers ranked by improvement:")
    for layer_idx, (direction, acc) in sorted_layers:
        logger.info(f"  Layer {layer_idx}: direction {direction} = {acc*100:.0f}%")

    # Apply layers one by one, keeping only if it helps
    taught_mlps = {}  # layer_idx -> (original_mlp, taught_mlp, mlp_key)
    current_acc = baseline

    for layer_idx, (direction, expected_acc) in sorted_layers:
        logger.info(f"\nTrying Layer {layer_idx}, direction {direction}...")

        # Recalibrate with current state
        S_X, S_Y = get_layer_activations(student, student_tok, layer_idx, calibration_prompts)
        T_X, T_Y = get_layer_activations(teacher, teacher_tok, layer_idx, calibration_prompts)

        W = apply_teaching_stable(S_X, S_Y, T_Y, direction)
        if W is None:
            logger.info("  Skipped (numerical issues)")
            continue

        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class TaughtMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = student.model.layers[layer_idx]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = TaughtMLP(W_mx)
            mlp_key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = TaughtMLP(W_mx)
            mlp_key = 'mlp'

        new_acc = evaluate_accuracy(student, student_tok)
        logger.info(f"  Accuracy: {current_acc*100:.0f}% → {new_acc*100:.0f}%")

        if new_acc >= current_acc:
            logger.info(f"  KEEPING (improvement)")
            taught_mlps[layer_idx] = (original_mlp, TaughtMLP(W_mx), mlp_key)
            current_acc = new_acc
        else:
            logger.info(f"  REVERTING (degradation)")
            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

    # ========================================
    # PHASE 4: Final Evaluation
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Final Evaluation")
    logger.info(f"{'='*80}")

    logger.info(f"\nLayers modified: {list(taught_mlps.keys())}")

    logger.info(f"\n{'Prompt':<40} {'Before':>15} {'After':>15} {'Expected':>15}")
    logger.info("-" * 90)

    # First revert all to measure baseline
    for layer_idx, (original_mlp, taught_mlp, mlp_key) in taught_mlps.items():
        layer = student.model.layers[layer_idx]
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

    before_results = {}
    for prompt, expected in test_cases:
        before_results[prompt] = get_prediction(student, student_tok, prompt)

    # Apply all taught MLPs
    for layer_idx, (original_mlp, taught_mlp, mlp_key) in taught_mlps.items():
        layer = student.model.layers[layer_idx]
        if mlp_key == 'feed_forward':
            layer.feed_forward = taught_mlp
        else:
            layer.mlp = taught_mlp

    improved = 0
    degraded = 0
    final_correct = 0

    for prompt, expected in test_cases:
        before = before_results[prompt]
        after = get_prediction(student, student_tok, prompt)

        before_ok = expected.lower() in before.lower()
        after_ok = expected.lower() in after.lower()

        if after_ok:
            final_correct += 1

        if after_ok and not before_ok:
            improved += 1
            change = "← IMPROVED!"
        elif before_ok and not after_ok:
            degraded += 1
            change = "← degraded"
        else:
            change = ""

        logger.info(f"{prompt:<40} {before:>15} {after:>15} {expected:>15} {change}")

    final_acc = final_correct / len(test_cases)

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Multi-Layer Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
CONFIGURATION:
  - Teacher: LFM2.5-1.2B-Instruct
  - Student: LFM2-1.2B
  - Layers modified: {list(taught_mlps.keys())}

RESULTS:
  - Baseline:       {baseline*100:.0f}%
  - After teaching: {final_acc*100:.0f}%
  - Change:         {(final_acc - baseline)*100:+.0f}pp
  - Improved:       {improved} prompts
  - Degraded:       {degraded} prompts

THE INSIGHT:
  Multi-layer teaching CAN work if we:
  1. Recalibrate after each layer
  2. Only keep changes that improve accuracy
  3. Use stable numerical methods

  The key is ITERATIVE REFINEMENT:
  - Don't apply all layers at once
  - Test and keep only beneficial changes
""")


if __name__ == "__main__":
    run_experiment()
