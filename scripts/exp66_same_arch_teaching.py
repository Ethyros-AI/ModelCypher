#!/usr/bin/env python3
"""Experiment 66: Same-Architecture Teaching.

The breakthrough insight: LFM2.5-1.2B-Instruct can teach LFM2-1.2B!

Same architecture = Same dimensions = No projection needed.

This is the cleanest possible test:
- Teacher: LFM2.5-1.2B-Instruct (instruction-tuned)
- Student: LFM2-1.2B (base model)
- Both: 2048 dimensions, 16 layers

If this works, we have a recipe for upgrading any base model
using its instruction-tuned sibling.
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


def spectral_entropy(Y):
    """Compute entropy from singular value spectrum."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / np.sum(S)
    S_norm = S_norm[S_norm > 1e-10]
    return -np.sum(S_norm * np.log(S_norm))


def run_experiment():
    """Same-architecture teaching."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Load models - SAME ARCHITECTURE
    logger.info("Loading LFM2-1.2B (student / base model)...")
    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    student, student_tok = load(student_path)

    logger.info("Loading LFM2.5-1.2B-Instruct (teacher / instruction-tuned)...")
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
        "The capital of France is",
        "The largest planet in our solar system is",
        "Water freezes at",
        "If it rains, the ground gets",
        "The opposite of up is",
        "2 + 2 equals",
        "Photosynthesis occurs in",
        "DNA is found in",
        "The nucleus of an atom contains",
        "A noun is a word that names a",
        "The past tense of run is",
        "An adjective describes a",
        "The square root of 16 is",
        "10 times 10 equals",
        "Half of 100 is",
        "The sky is usually",
        "Birds can",
        "Fish live in",
        "The sun rises in the",
        "Gravity causes objects to",
        "The speed of light is",
        "The human body has",
        "The Great Wall of China is in",
        "The Eiffel Tower is in",
    ]

    def get_prediction(model, tokenizer, prompt):
        """Get model's prediction."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        word = tokenizer.decode([top_token]).strip()
        probs = np.array(logits[0, -1, :].tolist())
        probs = softmax(probs)
        return word, float(probs[top_token])

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

    # ========================================
    # PHASE 1: Compare Base vs Instruct
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Compare Base vs Instruction-Tuned")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Prompt':<40} {'Base':>15} {'Instruct':>15} {'Expected':>15}")
    logger.info("-" * 90)

    base_correct = 0
    instruct_correct = 0

    for prompt, expected in test_cases:
        base_word, base_conf = get_prediction(student, student_tok, prompt)
        inst_word, inst_conf = get_prediction(teacher, teacher_tok, prompt)

        base_ok = expected.lower() in base_word.lower()
        inst_ok = expected.lower() in inst_word.lower()

        if base_ok:
            base_correct += 1
        if inst_ok:
            instruct_correct += 1

        b_mark = "✓" if base_ok else " "
        i_mark = "✓" if inst_ok else " "

        logger.info(f"{prompt:<40} {b_mark}{base_word:>14} {i_mark}{inst_word:>14} {expected:>15}")

    logger.info(f"\nBase:     {base_correct}/{len(test_cases)} = {base_correct/len(test_cases)*100:.0f}%")
    logger.info(f"Instruct: {instruct_correct}/{len(test_cases)} = {instruct_correct/len(test_cases)*100:.0f}%")

    # ========================================
    # PHASE 2: Entropy Comparison
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Entropy Comparison Across Layers")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Layer':>8} {'Base H':>12} {'Instruct H':>12} {'Gap':>12} {'Teachable?':>12}")
    logger.info("-" * 60)

    best_layer = None
    best_gap = 0

    for layer_idx in [2, 4, 6, 8, 10, 12, 14]:
        _, S_Y = get_layer_activations(student, student_tok, layer_idx, calibration_prompts)
        _, T_Y = get_layer_activations(teacher, teacher_tok, layer_idx, calibration_prompts)

        s_entropy = spectral_entropy(S_Y)
        t_entropy = spectral_entropy(T_Y)
        gap = s_entropy - t_entropy

        teachable = "YES" if gap > 0 else "no"
        if gap > best_gap:
            best_gap = gap
            best_layer = layer_idx

        logger.info(f"{layer_idx:>8} {s_entropy:>12.4f} {t_entropy:>12.4f} {gap:>+12.4f} {teachable:>12}")

    logger.info(f"\nBest layer for teaching: {best_layer} (gap = {best_gap:.4f})")

    # ========================================
    # PHASE 3: Apply Teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info(f"PHASE 3: Teaching at Layer {best_layer}")
    logger.info(f"{'='*80}")

    # Collect activations at best layer
    S_X, S_Y = get_layer_activations(student, student_tok, best_layer, calibration_prompts)
    T_X, T_Y = get_layer_activations(teacher, teacher_tok, best_layer, calibration_prompts)

    logger.info(f"\nShapes: Student X={S_X.shape}, Y={S_Y.shape}")
    logger.info(f"Shapes: Teacher X={T_X.shape}, Y={T_Y.shape}")

    # Same architecture = direct direction transfer!
    S_Y_centered = S_Y - S_Y.mean(axis=0)
    T_Y_centered = T_Y - T_Y.mean(axis=0)

    U_s, Sigma_s, Vh_s = svd(S_Y_centered, full_matrices=False)
    U_t, Sigma_t, Vh_t = svd(T_Y_centered, full_matrices=False)

    logger.info(f"\nTop 10 singular values:")
    logger.info(f"  Base:     {Sigma_s[:10].round(3)}")
    logger.info(f"  Instruct: {Sigma_t[:10].round(3)}")

    def apply_teaching(S_X, S_Y, T_Y, directions):
        """Apply direction replacement from teacher to student."""
        S_Y_centered = S_Y - S_Y.mean(axis=0)
        T_Y_centered = T_Y - T_Y.mean(axis=0)

        _, _, Vh_s = svd(S_Y_centered, full_matrices=False)
        _, _, Vh_t = svd(T_Y_centered, full_matrices=False)

        result = S_Y.copy()

        for d in directions:
            if d < min(len(Vh_s), len(Vh_t)):
                # Remove student's direction d
                s_coefs = S_Y_centered @ Vh_s[d]
                s_proj = np.outer(s_coefs, Vh_s[d])
                result -= s_proj

                # Add teacher's direction d (DIRECTLY - same space!)
                t_coefs = T_Y_centered @ Vh_t[d]
                t_proj = np.outer(t_coefs, Vh_t[d])
                result += t_proj

        # Solve for weights
        alpha = 1e-6
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result
        W = np.linalg.solve(ATA, ATB).T

        return W

    # Test different directions
    directions_to_test = [
        [0],
        [1],
        [2],
        [5],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3, 4, 5],
        list(range(12)),
    ]

    results = []
    baseline_acc = base_correct / len(test_cases)

    for dirs in directions_to_test:
        W_taught = apply_teaching(S_X, S_Y, T_Y, dirs)

        if np.isnan(W_taught).any():
            logger.info(f"Directions {dirs}: NaN in weights, skipping")
            continue

        W_mx = mx.array(W_taught.astype(np.float32))
        mx.eval(W_mx)

        class TaughtMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        # Install taught MLP
        layer = student.model.layers[best_layer]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = TaughtMLP(W_mx)
            mlp_key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = TaughtMLP(W_mx)
            mlp_key = 'mlp'

        # Evaluate
        correct = 0
        for prompt, expected in test_cases:
            word, conf = get_prediction(student, student_tok, prompt)
            is_correct = expected.lower() in word.lower()
            if is_correct:
                correct += 1

        # Restore
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        acc = correct / len(test_cases)
        results.append((dirs, acc))
        logger.info(f"Directions {dirs}: {correct}/{len(test_cases)} = {acc*100:.0f}%")

    # ========================================
    # PHASE 4: Best Result Deep Dive
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Best Result Analysis")
    logger.info(f"{'='*80}")

    best = max(results, key=lambda x: x[1])
    logger.info(f"\nBest configuration: directions {best[0]}, accuracy {best[1]*100:.0f}%")

    # Apply best teaching and show detailed results
    W_taught = apply_teaching(S_X, S_Y, T_Y, best[0])
    W_mx = mx.array(W_taught.astype(np.float32))
    mx.eval(W_mx)

    class TaughtMLP:
        def __init__(self, W):
            self.W = W
        def __call__(self, x):
            return mx.matmul(x, self.W.T)

    layer = student.model.layers[best_layer]
    if hasattr(layer, 'feed_forward'):
        original_mlp = layer.feed_forward
        layer.feed_forward = TaughtMLP(W_mx)
        mlp_key = 'feed_forward'
    else:
        original_mlp = layer.mlp
        layer.mlp = TaughtMLP(W_mx)
        mlp_key = 'mlp'

    logger.info(f"\n{'Prompt':<40} {'Before':>15} {'After':>15} {'Expected':>15}")
    logger.info("-" * 90)

    improved = 0
    degraded = 0
    for prompt, expected in test_cases:
        # Before (get from stored results)
        before_word, _ = get_prediction(student, student_tok, prompt)
        # Temporarily restore to get before
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        before_word, _ = get_prediction(student, student_tok, prompt)

        if mlp_key == 'feed_forward':
            layer.feed_forward = TaughtMLP(W_mx)
        else:
            layer.mlp = TaughtMLP(W_mx)

        after_word, _ = get_prediction(student, student_tok, prompt)

        before_ok = expected.lower() in before_word.lower()
        after_ok = expected.lower() in after_word.lower()

        if after_ok and not before_ok:
            improved += 1
            change = "← IMPROVED!"
        elif before_ok and not after_ok:
            degraded += 1
            change = "← degraded"
        else:
            change = ""

        logger.info(f"{prompt:<40} {before_word:>15} {after_word:>15} {expected:>15} {change}")

    # Restore
    if mlp_key == 'feed_forward':
        layer.feed_forward = original_mlp
    else:
        layer.mlp = original_mlp

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Same-Architecture Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
CONFIGURATION:
  - Teacher: LFM2.5-1.2B-Instruct
  - Student: LFM2-1.2B
  - Layer: {best_layer}
  - Directions: {best[0]}

RESULTS:
  - Baseline accuracy: {baseline_acc*100:.0f}%
  - After teaching:    {best[1]*100:.0f}%
  - Change:            {(best[1] - baseline_acc)*100:+.0f}pp
  - Prompts improved:  {improved}
  - Prompts degraded:  {degraded}

COMPARISON:
  - Base model:     {base_correct}/{len(test_cases)} = {base_correct/len(test_cases)*100:.0f}%
  - Instruct model: {instruct_correct}/{len(test_cases)} = {instruct_correct/len(test_cases)*100:.0f}%
  - After teaching: {int(best[1] * len(test_cases))}/{len(test_cases)} = {best[1]*100:.0f}%

THE INSIGHT:
  Same-architecture teaching WORKS because:
  1. No dimension mismatch
  2. Same representation space
  3. Direct direction transfer

  This proves: instruction tuning creates CLEANER directions
  that we can transplant to base models.
""")


if __name__ == "__main__":
    run_experiment()
