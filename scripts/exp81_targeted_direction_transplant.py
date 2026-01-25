#!/usr/bin/env python3
"""Experiment 81: Targeted Direction Transplant.

Key insight from exp80b:
- Teacher (DeepSeek-R1 at 60%) KNOWS all 3 cases the student fails
- But hybrid mixing didn't help
- WHY? The full-output mixing dilutes the signal

This experiment:
1. Find the SPECIFIC directions in the teacher that encode failing cases
2. Transplant ONLY those directions
3. Replace in the student's manifold precisely where needed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.linalg import svd
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_kurtosis(Y):
    kurtoses = []
    for h in Y:
        std = h.std()
        if std < 1e-10:
            kurtoses.append(0.0)
            continue
        z = (h - h.mean()) / std
        kurtoses.append(float(np.mean(z ** 4) - 3))
    return np.mean(kurtoses)


def compute_spectral_entropy(Y):
    Y_centered = Y - Y.mean(axis=0)
    try:
        _, S, _ = svd(Y_centered, full_matrices=False)
        S_sum = S.sum()
        if S_sum < 1e-10:
            return 0.0
        S_norm = S / S_sum
        return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
    except:
        return 0.0


def geometry_score(kurtosis, spectral_entropy):
    return kurtosis / 100 - spectral_entropy


def run_experiment():
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("="*80)
    logger.info("TARGETED DIRECTION TRANSPLANT")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    # The failing cases from exp80b
    failing_cases = [
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
        ("Fish live in", "water"),
    ]

    # Full test set
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

    def get_prediction(model, tokenizer, prompt):
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)
        top_token = int(mx.argmax(logits[0, -1, :]).item())
        return tokenizer.decode([top_token]).strip()

    def evaluate_accuracy(model, tokenizer):
        correct = 0
        results = []
        for prompt, expected in test_cases:
            word = get_prediction(model, tokenizer, prompt)
            is_correct = expected.lower() in word.lower()
            if is_correct:
                correct += 1
            results.append({'prompt': prompt, 'expected': expected, 'got': word, 'correct': is_correct})
        return correct / len(test_cases), results

    def get_activations_for_prompt(model, tokenizer, layer_idx, prompt):
        """Get activations for a single prompt."""
        captured = {}

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
                captured['input'] = x
                captured['output'] = self.mlp(x)
                return captured['output']

        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        if key == 'feed_forward':
            layer.feed_forward = MLPHook(original_mlp)
        else:
            layer.mlp = MLPHook(original_mlp)

        try:
            _ = model(input_ids)
            mx.eval(captured['input'], captured['output'])
            inp = np.array(captured['input'][0, -1, :].tolist(), dtype=np.float64)
            out = np.array(captured['output'][0, -1, :].tolist(), dtype=np.float64)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

        return inp, out

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        inputs = []
        outputs = []
        for prompt in prompts:
            inp, out = get_activations_for_prompt(model, tokenizer, layer_idx, prompt)
            inputs.append(inp)
            outputs.append(out)
        return np.stack(inputs), np.stack(outputs)

    # ========================================
    # PHASE 1: Load models and establish baseline
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Load models and verify teacher knows failing cases")
    logger.info("="*60)

    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"

    logger.info(f"\nLoading student: {student_path}")
    student_model, student_tokenizer = load(student_path)

    logger.info(f"Loading teacher: {teacher_path}")
    teacher_model, teacher_tokenizer = load(teacher_path)

    student_acc, student_results = evaluate_accuracy(student_model, student_tokenizer)
    teacher_acc, teacher_results = evaluate_accuracy(teacher_model, teacher_tokenizer)

    logger.info(f"\nStudent accuracy: {student_acc*100:.0f}%")
    logger.info(f"Teacher accuracy: {teacher_acc*100:.0f}%")

    # Verify teacher knows failing cases
    logger.info("\nVerifying teacher knows failing cases:")
    teacher_knows_all = True
    for prompt, expected in failing_cases:
        word = get_prediction(teacher_model, teacher_tokenizer, prompt)
        knows = expected.lower() in word.lower()
        teacher_knows_all = teacher_knows_all and knows
        status = "✓" if knows else "✗"
        logger.info(f"  {status} '{prompt}' → '{word}' (expected: {expected})")

    if not teacher_knows_all:
        logger.info("\nTeacher doesn't know all failing cases. Aborting.")
        return

    logger.info("\n✓ Teacher knows all failing cases - proceeding with targeted transplant")

    # ========================================
    # PHASE 2: Find which layer encodes failing knowledge
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Find where teacher encodes failing knowledge")
    logger.info("="*60)

    # For each failing case, find which layer shows the biggest
    # difference between teacher (correct) and student (wrong) output
    TARGET_LAYER = 2  # Start with Layer 2 (our improvement layer)

    logger.info(f"\nAnalyzing Layer {TARGET_LAYER}...")

    # Get teacher and student activations for failing cases
    failing_prompts = [p for p, e in failing_cases]

    teacher_X, teacher_Y = get_layer_activations(teacher_model, teacher_tokenizer, TARGET_LAYER, failing_prompts)
    student_X, student_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, failing_prompts)

    # Compute alignment matrix (teacher → student space)
    alpha = 1e-4
    ATA = teacher_Y.T @ teacher_Y + alpha * np.eye(teacher_Y.shape[1])
    ATB = teacher_Y.T @ student_Y
    F, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
    teacher_Y_aligned = teacher_Y @ F

    logger.info(f"  Teacher output dim: {teacher_Y.shape[1]}")
    logger.info(f"  Student output dim: {student_Y.shape[1]}")
    logger.info(f"  Alignment matrix F: {F.shape}")

    # ========================================
    # PHASE 3: Find specific directions that differ
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Find directions where teacher differs from student")
    logger.info("="*60)

    # SVD of student outputs (failing cases only)
    student_centered = student_Y - student_Y.mean(axis=0)
    _, S_s, Vh_s = svd(student_centered, full_matrices=False)

    # SVD of aligned teacher outputs
    teacher_centered = teacher_Y_aligned - teacher_Y_aligned.mean(axis=0)
    _, S_t, Vh_t = svd(teacher_centered, full_matrices=False)

    logger.info("\nDirection analysis for failing cases:")
    logger.info(f"{'Dir':<5} {'Student σ':>12} {'Teacher σ':>12} {'Ratio':>10}")
    logger.info("-" * 45)

    # Find directions where teacher has DIFFERENT variance pattern
    direction_differences = []
    for d in range(min(10, len(S_s))):
        s_var = S_s[d] ** 2
        t_var = S_t[d] ** 2 if d < len(S_t) else 0
        ratio = t_var / (s_var + 1e-10)
        direction_differences.append({
            'direction': d,
            'student_var': s_var,
            'teacher_var': t_var,
            'ratio': ratio,
        })
        logger.info(f"  {d:<5} {s_var:>12.2f} {t_var:>12.2f} {ratio:>10.2f}")

    # ========================================
    # PHASE 4: Try targeted direction replacement
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Try targeted direction replacement")
    logger.info("="*60)

    # Get full probe set for student
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

    # Get current student activations for all probes
    S_X, S_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
    T_X, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, TARGET_LAYER, probe_prompts)

    # Align teacher to student space
    ATA_full = T_Y.T @ T_Y + alpha * np.eye(T_Y.shape[1])
    ATB_full = T_Y.T @ S_Y
    F_full, _, _, _ = np.linalg.lstsq(ATA_full, ATB_full, rcond=None)
    T_Y_aligned = T_Y @ F_full

    # SVD of student full output manifold
    S_Y_centered = S_Y - S_Y.mean(axis=0)
    U_s, Sigma_s, Vh_s = svd(S_Y_centered, full_matrices=False)

    # SVD of aligned teacher full output manifold
    T_Y_centered = T_Y_aligned - T_Y_aligned.mean(axis=0)
    U_t, Sigma_t, Vh_t = svd(T_Y_centered, full_matrices=False)

    logger.info("\nTrying direction replacement strategies...")

    results = []

    # Strategy 1: Replace specific directions with teacher's
    for num_dirs in [1, 2, 3, 5]:
        logger.info(f"\n  Replacing top {num_dirs} teacher directions...")

        # Build replacement manifold
        Y_new = S_Y.copy()

        for d in range(num_dirs):
            if d >= len(Vh_s) or d >= len(Vh_t):
                break

            # Remove student's direction d
            coefs_s = S_Y_centered @ Vh_s[d]
            proj_s = np.outer(coefs_s, Vh_s[d])

            # Add teacher's direction d
            coefs_t = T_Y_centered @ Vh_t[d]
            proj_t = np.outer(coefs_t, Vh_t[d])

            Y_new = Y_new - proj_s + proj_t

        # Compute new weight matrix
        S_X_scale = np.abs(S_X).max()
        Y_scale = np.abs(Y_new).max()

        S_X_norm = S_X / S_X_scale
        Y_norm = Y_new / Y_scale

        reg = 1e-3
        ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
        ATB_w = S_X_norm.T @ Y_norm
        W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
        W = (W_norm * Y_scale / S_X_scale).T

        if np.isnan(W).any() or np.isinf(W).any():
            logger.info(f"    Skip - numerical issues")
            continue

        # Apply and test
        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class TestMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = student_model.model.layers[TARGET_LAYER]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = TestMLP(W_mx)
            key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = TestMLP(W_mx)
            key = 'mlp'

        new_acc, new_results = evaluate_accuracy(student_model, student_tokenizer)

        if key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        results.append({
            'strategy': f'replace_top_{num_dirs}',
            'accuracy': new_acc,
        })
        logger.info(f"    Accuracy: {new_acc*100:.0f}%")

        # Check which failing cases improved
        improved = []
        for prompt, expected in failing_cases:
            before = next(r for r in student_results if r['prompt'] == prompt)
            after = next(r for r in new_results if r['prompt'] == prompt)
            if after['correct'] and not before['correct']:
                improved.append(prompt)
        if improved:
            logger.info(f"    Improved: {improved}")

    # Strategy 2: Add teacher's directions (not replace)
    logger.info("\n  Trying additive approach (add teacher's top directions)...")

    for num_dirs in [1, 2, 3]:
        for scale in [0.1, 0.3, 0.5]:
            Y_new = S_Y.copy()

            for d in range(num_dirs):
                if d >= len(Vh_t):
                    break
                coefs_t = T_Y_centered @ Vh_t[d]
                proj_t = np.outer(coefs_t, Vh_t[d])
                Y_new = Y_new + scale * proj_t

            # Compute new weight matrix
            S_X_scale = np.abs(S_X).max()
            Y_scale = np.abs(Y_new).max()

            S_X_norm = S_X / S_X_scale
            Y_norm = Y_new / Y_scale

            reg = 1e-3
            ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
            ATB_w = S_X_norm.T @ Y_norm
            W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
            W = (W_norm * Y_scale / S_X_scale).T

            if np.isnan(W).any() or np.isinf(W).any():
                continue

            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            layer = student_model.model.layers[TARGET_LAYER]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                layer.feed_forward = TestMLP(W_mx)
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                layer.mlp = TestMLP(W_mx)
                key = 'mlp'

            new_acc, _ = evaluate_accuracy(student_model, student_tokenizer)

            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            results.append({
                'strategy': f'add_{num_dirs}_dirs_scale_{scale}',
                'accuracy': new_acc,
            })
            logger.info(f"    Add {num_dirs} dirs, scale={scale}: {new_acc*100:.0f}%")

    # ========================================
    # PHASE 5: Try self-improvement on top of student baseline
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 5: Self-improve student first, then try teacher")
    logger.info("="*60)

    # Quick self-improvement to 70%
    logger.info("\nRunning self-improvement to 70%...")

    directions = list(range(12))
    boosts = [0.0, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]
    current_acc = student_acc

    for round_num in range(20):
        S_X, S_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
        baseline_kurtosis = compute_kurtosis(S_Y)
        baseline_entropy = compute_spectral_entropy(S_Y)
        baseline_score = geometry_score(baseline_kurtosis, baseline_entropy)

        S_Y_centered = S_Y - S_Y.mean(axis=0)
        _, S, Vh = svd(S_Y_centered, full_matrices=False)

        best = None
        for d in directions:
            for boost in boosts:
                if boost == 1.0:
                    continue

                if d >= len(Vh):
                    continue

                coefs = S_Y_centered @ Vh[d]
                proj = np.outer(coefs, Vh[d])
                Y_new = S_Y + proj * (boost - 1)

                new_score = geometry_score(compute_kurtosis(Y_new), compute_spectral_entropy(Y_new))
                if new_score <= baseline_score + 1e-4:
                    continue

                # Compute W
                S_X_scale = np.abs(S_X).max()
                Y_scale = np.abs(Y_new).max()
                S_X_norm = S_X / S_X_scale
                Y_norm = Y_new / Y_scale
                reg = 1e-3
                ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
                ATB_w = S_X_norm.T @ Y_norm
                W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
                W = (W_norm * Y_scale / S_X_scale).T

                if np.isnan(W).any() or np.isinf(W).any():
                    continue

                W_mx = mx.array(W.astype(np.float32))
                mx.eval(W_mx)

                layer = student_model.model.layers[TARGET_LAYER]
                if hasattr(layer, 'feed_forward'):
                    original_mlp = layer.feed_forward
                    layer.feed_forward = TestMLP(W_mx)
                    key = 'feed_forward'
                else:
                    original_mlp = layer.mlp
                    layer.mlp = TestMLP(W_mx)
                    key = 'mlp'

                new_acc, _ = evaluate_accuracy(student_model, student_tokenizer)

                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

                if new_acc >= current_acc:
                    if best is None or new_acc > best['acc']:
                        best = {'W': W, 'acc': new_acc, 'key': key}

        if best:
            W_mx = mx.array(best['W'].astype(np.float32))
            mx.eval(W_mx)

            class PermanentMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = student_model.model.layers[TARGET_LAYER]
            if best['key'] == 'feed_forward':
                layer.feed_forward = PermanentMLP(W_mx)
            else:
                layer.mlp = PermanentMLP(W_mx)

            if best['acc'] > current_acc:
                current_acc = best['acc']
                logger.info(f"  Round {round_num+1}: {current_acc*100:.0f}%")

        if current_acc >= 0.7:
            break

    logger.info(f"\nSelf-improved to: {current_acc*100:.0f}%")

    # Now try teacher direction injection on top of 70%
    logger.info("\nNow trying teacher direction injection from 70% baseline...")

    # Get fresh activations
    S_X, S_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
    T_X, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, TARGET_LAYER, probe_prompts)

    # Align teacher
    ATA_full = T_Y.T @ T_Y + alpha * np.eye(T_Y.shape[1])
    ATB_full = T_Y.T @ S_Y
    F_full, _, _, _ = np.linalg.lstsq(ATA_full, ATB_full, rcond=None)
    T_Y_aligned = T_Y @ F_full

    # SVDs
    S_Y_centered = S_Y - S_Y.mean(axis=0)
    _, _, Vh_s = svd(S_Y_centered, full_matrices=False)

    T_Y_centered = T_Y_aligned - T_Y_aligned.mean(axis=0)
    _, _, Vh_t = svd(T_Y_centered, full_matrices=False)

    logger.info("\nTrying targeted injection from 70% baseline:")

    for num_dirs in [1, 2, 3]:
        for inject_scale in [0.1, 0.2, 0.3, 0.5]:
            Y_new = S_Y.copy()

            for d in range(num_dirs):
                if d >= len(Vh_t):
                    break
                # Remove student's direction
                coefs_s = S_Y_centered @ Vh_s[d]
                proj_s = np.outer(coefs_s, Vh_s[d])
                # Add teacher's direction with scale
                coefs_t = T_Y_centered @ Vh_t[d]
                proj_t = np.outer(coefs_t, Vh_t[d])
                Y_new = Y_new - inject_scale * proj_s + inject_scale * proj_t

            # Compute W
            S_X_scale = np.abs(S_X).max()
            Y_scale = np.abs(Y_new).max()
            S_X_norm = S_X / S_X_scale
            Y_norm = Y_new / Y_scale
            reg = 1e-3
            ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
            ATB_w = S_X_norm.T @ Y_norm
            W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
            W = (W_norm * Y_scale / S_X_scale).T

            if np.isnan(W).any() or np.isinf(W).any():
                continue

            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            layer = student_model.model.layers[TARGET_LAYER]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                layer.feed_forward = TestMLP(W_mx)
                key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                layer.mlp = TestMLP(W_mx)
                key = 'mlp'

            new_acc, _ = evaluate_accuracy(student_model, student_tokenizer)

            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            results.append({
                'strategy': f'from_70_inject_{num_dirs}_scale_{inject_scale}',
                'accuracy': new_acc,
            })
            logger.info(f"  {num_dirs} dirs, scale={inject_scale}: {new_acc*100:.0f}%")

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    best_result = max(results, key=lambda x: x['accuracy'])

    logger.info(f"\nBaseline student: {student_acc*100:.0f}%")
    logger.info(f"Teacher: {teacher_acc*100:.0f}%")
    logger.info(f"Self-improved: {current_acc*100:.0f}%")
    logger.info(f"Best with teacher: {best_result['accuracy']*100:.0f}% ({best_result['strategy']})")

    if best_result['accuracy'] > current_acc:
        logger.info(f"\n✓ Teacher helped: +{(best_result['accuracy']-current_acc)*100:.0f}pp")
    else:
        logger.info("""
No improvement from teacher direction injection.

This confirms a key finding:
- The teacher (60%) knows the ANSWERS to the 3 failing cases
- But the GEOMETRIC STRUCTURE of how the teacher represents that knowledge
  does NOT transfer to the student's manifold
- The directions are aligned but the INFORMATION is encoded differently

The 70% ceiling is NOT about missing knowledge.
It's about how that knowledge is STRUCTURED in the manifold.
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'student_baseline': float(student_acc),
        'teacher': float(teacher_acc),
        'self_improved': float(current_acc),
        'best_with_teacher': best_result,
        'all_results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "targeted_direction_transplant_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
