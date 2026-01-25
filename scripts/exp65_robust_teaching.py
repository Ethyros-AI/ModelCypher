#!/usr/bin/env python3
"""Experiment 65: Robust Cross-Architecture Teaching.

The problem from exp64: dimension mismatch (4096 → 2048) with insufficient samples
caused numerical instability.

The fix:
1. Work in the projected subspace (n samples × k directions)
2. Use SVD-based dimensionality reduction
3. Test different directions independently
4. Validate the math before applying

We're not giving up. We're getting smarter.
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
    """Robust cross-architecture teaching."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Load models
    logger.info("Loading LFM2-1.2B (student)...")
    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    student, student_tok = load(student_path)

    logger.info("Loading DeepSeek-R1-8B (teacher)...")
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    teacher, teacher_tok = load(teacher_path)

    # Test prompts for evaluation
    test_cases = [
        ("The capital of France is", "Paris"),
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
        ("The opposite of hot is", "cold"),
        ("Birds can", "fly"),
        ("Fish live in", "water"),
    ]

    # Calibration prompts (more than before)
    calibration_prompts = [
        "The capital of", "The largest", "Water freezes",
        "If A implies B", "The opposite of", "2 + 2",
        "Photosynthesis occurs", "DNA stands for", "The nucleus",
        "A noun is", "The past tense", "An adjective",
        "The square root", "10 times 10", "Half of 50",
        "The sky is", "Birds can", "Fish live",
        "The sun rises", "Gravity causes", "The speed of light",
        "A verb describes", "The human body", "The Great Wall",
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
    # PHASE 1: Baseline
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Baseline Performance")
    logger.info(f"{'='*80}")

    baseline_correct = 0
    for prompt, expected in test_cases:
        word, conf = get_prediction(student, student_tok, prompt)
        is_correct = expected.lower() in word.lower()
        if is_correct:
            baseline_correct += 1
        mark = "✓" if is_correct else "✗"
        logger.info(f"  {mark} '{prompt}' → '{word}' ({conf*100:.1f}%)")

    baseline_acc = baseline_correct / len(test_cases)
    logger.info(f"\nBaseline: {baseline_correct}/{len(test_cases)} = {baseline_acc*100:.0f}%")

    # ========================================
    # PHASE 2: Collect activations
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Collecting Activations")
    logger.info(f"{'='*80}")

    # Use layer 10 for student (16 total), layer 24 for teacher (36 total)
    student_layer = 10
    teacher_layer = 24

    logger.info(f"Collecting from {len(calibration_prompts)} prompts...")
    S_X, S_Y = get_layer_activations(student, student_tok, student_layer, calibration_prompts)
    T_X, T_Y = get_layer_activations(teacher, teacher_tok, teacher_layer, calibration_prompts)

    logger.info(f"Student: X={S_X.shape}, Y={S_Y.shape}")
    logger.info(f"Teacher: X={T_X.shape}, Y={T_Y.shape}")

    # Check for numerical issues
    logger.info(f"\nNumerical health check:")
    logger.info(f"  Student X: min={S_X.min():.2e}, max={S_X.max():.2e}, nan={np.isnan(S_X).sum()}")
    logger.info(f"  Student Y: min={S_Y.min():.2e}, max={S_Y.max():.2e}, nan={np.isnan(S_Y).sum()}")
    logger.info(f"  Teacher Y: min={T_Y.min():.2e}, max={T_Y.max():.2e}, nan={np.isnan(T_Y).sum()}")

    # ========================================
    # PHASE 3: Subspace Teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Subspace Teaching (Robust Approach)")
    logger.info(f"{'='*80}")

    # Compute principal directions in both spaces
    S_Y_centered = S_Y - S_Y.mean(axis=0)
    T_Y_centered = T_Y - T_Y.mean(axis=0)

    U_s, S_s, Vh_s = svd(S_Y_centered, full_matrices=False)
    U_t, S_t, Vh_t = svd(T_Y_centered, full_matrices=False)

    logger.info(f"\nStudent spectrum (top 10): {S_s[:10].round(2)}")
    logger.info(f"Teacher spectrum (top 10): {S_t[:10].round(2)}")

    # The key insight: work in the SAMPLE space (n × k), not feature space (d × d)
    # This avoids under-determined systems

    # Project both outputs to their top-k subspaces
    k = 8  # Number of directions to use
    logger.info(f"\nUsing k={k} principal directions")

    # Student in its own subspace: coefficients for each sample
    student_coefs = U_s[:, :k] * S_s[:k]  # (n, k)

    # Teacher in its own subspace
    teacher_coefs = U_t[:, :k] * S_t[:k]  # (n, k)

    # Now we can map teacher → student in the coefficient space
    # This is well-conditioned because k < n
    alpha = 1e-6
    coef_transfer = np.linalg.lstsq(teacher_coefs, student_coefs, rcond=None)[0]  # (k, k)

    logger.info(f"Coefficient transfer matrix condition: {np.linalg.cond(teacher_coefs):.2e}")

    # ========================================
    # PHASE 4: Direction Replacement in Subspace
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Direction Replacement (Subspace)")
    logger.info(f"{'='*80}")

    def apply_subspace_teaching(S_X, S_Y, T_Y, Vh_s, Vh_t, directions_to_replace):
        """Replace specific directions with teacher's (in subspace)."""
        S_Y_centered = S_Y - S_Y.mean(axis=0)
        T_Y_centered = T_Y - T_Y.mean(axis=0)

        result = S_Y.copy()

        for d in directions_to_replace:
            if d < min(len(Vh_s), len(Vh_t)):
                # Remove student's direction d
                s_coefs = S_Y_centered @ Vh_s[d]
                s_proj = np.outer(s_coefs, Vh_s[d])
                result -= s_proj

                # Add teacher's direction d (scaled to student space)
                t_coefs = T_Y_centered @ Vh_t[d]
                # Scale by ratio of singular values
                scale = S_s[d] / (T_t[d] + 1e-10) if d < len(S_s) and d < len(S_t) else 1.0
                t_proj = np.outer(t_coefs * scale, Vh_s[d])  # Use student's basis
                result += t_proj

        # Solve for weights
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result
        W = np.linalg.solve(ATA, ATB).T

        return W

    # Fix the variable reference issue
    S_t_global = S_t
    T_t = S_t  # Teacher singular values

    # Test different directions
    directions_to_test = [
        [0],      # Top direction only
        [1],      # Second direction
        [5],      # Direction 6 (worked before)
        [0, 1],   # Top 2
        [0, 1, 2],  # Top 3
    ]

    results = []

    for dirs in directions_to_test:
        logger.info(f"\n--- Testing directions: {dirs} ---")

        W_taught = apply_subspace_teaching(S_X, S_Y, T_Y, Vh_s, Vh_t, dirs)

        # Check for NaN
        if np.isnan(W_taught).any():
            logger.info("  WARNING: NaN in weights, skipping")
            results.append((dirs, -1))
            continue

        W_mx = mx.array(W_taught.astype(np.float32))
        mx.eval(W_mx)

        class TaughtMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        # Install taught MLP
        layer = student.model.layers[student_layer]
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
            mark = "✓" if is_correct else "✗"
            logger.info(f"    {mark} '{prompt}' → '{word}'")

        # Restore
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        acc = correct / len(test_cases)
        results.append((dirs, acc))
        logger.info(f"  Accuracy: {correct}/{len(test_cases)} = {acc*100:.0f}%")

    # ========================================
    # PHASE 5: Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Robust Teaching Results")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Directions':>20} {'Accuracy':>15} {'vs Baseline':>15}")
    logger.info("-" * 55)

    for dirs, acc in results:
        if acc >= 0:
            change = acc - baseline_acc
            arrow = "↑" if change > 0 else ("↓" if change < 0 else "=")
            logger.info(f"{str(dirs):>20} {acc*100:>14.0f}% {arrow} {change*100:>+13.0f}pp")

    best = max(results, key=lambda x: x[1])
    logger.info(f"\nBaseline: {baseline_acc*100:.0f}%")
    logger.info(f"Best: {best[0]} = {best[1]*100:.0f}%")

    if best[1] > baseline_acc:
        logger.info(f"IMPROVEMENT: +{(best[1] - baseline_acc)*100:.0f}pp!")
    else:
        logger.info("No improvement from teaching.")

    logger.info(f"""
ANALYSIS:

The issue is cross-architecture teaching is fundamentally harder than
same-architecture teaching:

1. DIMENSION MISMATCH: Teacher (4096) vs Student (2048)
   - Can't directly copy directions
   - Must project or interpolate

2. VOCABULARY MISMATCH: Different tokenizers
   - "Paris" might be token 1234 in teacher, 5678 in student
   - Output space semantics differ

3. STRUCTURAL DIFFERENCES: Feed-forward vs MLP architecture
   - LFM2 uses feed_forward (FFN)
   - DeepSeek uses standard MLP

WHAT MIGHT WORK BETTER:

1. SAME-ARCHITECTURE TEACHING
   - Teach LFM2-1.2B from LFM2-7B (if it existed)
   - Or from a fine-tuned LFM2-1.2B variant

2. OUTPUT-SPACE ALIGNMENT
   - Map teacher's logits to student's vocabulary
   - Then do behavioral cloning

3. REPRESENTATION SURGERY
   - Find shared subspaces between architectures
   - Transfer only in those subspaces

4. ENSEMBLE DISTILLATION
   - Use multiple teachers
   - Let student learn common patterns
""")


if __name__ == "__main__":
    run_experiment()
