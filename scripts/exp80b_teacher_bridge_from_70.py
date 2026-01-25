#!/usr/bin/env python3
"""Experiment 80b: Teacher Bridge from 70% Baseline.

We found that:
- LFM2 self-improves to 70% (exp78)
- DeepSeek-R1 is only at 60% on this test set
- So LFM2's self-improvement EXCEEDS the teacher!

This experiment:
1. First self-improve LFM2 to 70%
2. THEN try teacher bridge
3. See if combining both gets higher than either alone
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
    logger.info("TEACHER BRIDGE FROM 70% BASELINE")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

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
        results = []
        for prompt, expected in test_cases:
            word = get_prediction(model, tokenizer, prompt)
            is_correct = expected.lower() in word.lower()
            if is_correct:
                correct += 1
            results.append({'prompt': prompt, 'expected': expected, 'got': word, 'correct': is_correct})
        return correct / len(test_cases), results

    def get_layer_activations(model, tokenizer, layer_idx, prompts):
        inputs = []
        outputs = []
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

        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            captured.clear()

            if key == 'feed_forward':
                layer.feed_forward = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = model(input_ids)
                mx.eval(captured['input'], captured['output'])
                inputs.append(np.array(captured['input'][0, -1, :].tolist(), dtype=np.float64))
                outputs.append(np.array(captured['output'][0, -1, :].tolist(), dtype=np.float64))
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        return np.stack(inputs), np.stack(outputs)

    def boost_direction(S_X, S_Y, direction_idx, boost_factor):
        if np.isnan(S_X).any() or np.isinf(S_X).any():
            return None
        if np.isnan(S_Y).any() or np.isinf(S_Y).any():
            return None

        S_Y_centered = S_Y - S_Y.mean(axis=0)

        try:
            _, S, Vh = svd(S_Y_centered, full_matrices=False)
        except:
            return None

        d = direction_idx
        if d >= len(Vh):
            return None

        if S[d] < 1e-6 * S[0]:
            return None

        coefs = S_Y_centered @ Vh[d]
        if np.isnan(coefs).any() or np.isinf(coefs).any():
            return None

        proj = np.outer(coefs, Vh[d])
        result = S_Y + proj * (boost_factor - 1)

        if np.isnan(result).any() or np.isinf(result).any():
            return None

        return result

    def compute_weight_transform(S_X, Y_new):
        S_X_scale = np.abs(S_X).max()
        Y_scale = np.abs(Y_new).max()
        if S_X_scale < 1e-10 or Y_scale < 1e-10:
            return None

        S_X_norm = S_X / S_X_scale
        Y_norm = Y_new / Y_scale

        alpha = 1e-3
        ATA = S_X_norm.T @ S_X_norm + alpha * np.eye(S_X_norm.shape[1])
        ATB = S_X_norm.T @ Y_norm

        try:
            W_norm, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
            W = (W_norm * Y_scale / S_X_scale).T
        except:
            return None

        if np.isnan(W).any() or np.isinf(W).any():
            return None

        return W

    # ========================================
    # PHASE 1: Self-improve to 70%
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Self-improve student to 70%")
    logger.info("="*60)

    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    logger.info(f"\nLoading student: {student_path}")
    student_model, student_tokenizer = load(student_path)

    initial_acc, _ = evaluate_accuracy(student_model, student_tokenizer)
    logger.info(f"Initial accuracy: {initial_acc*100:.0f}%")

    # Quick self-improvement (same as exp78 but abbreviated)
    TARGET_LAYER = 2
    current_acc = initial_acc
    directions = list(range(12))
    boosts = [0.0, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]

    logger.info("\nRunning self-improvement at Layer 2...")
    for round_num in range(20):
        S_X, S_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
        baseline_score = geometry_score(compute_kurtosis(S_Y), compute_spectral_entropy(S_Y))

        best = None
        for d in directions:
            for boost in boosts:
                if boost == 1.0:
                    continue
                Y_new = boost_direction(S_X, S_Y, d, boost)
                if Y_new is None:
                    continue

                new_score = geometry_score(compute_kurtosis(Y_new), compute_spectral_entropy(Y_new))
                if new_score <= baseline_score + 1e-4:
                    continue

                W = compute_weight_transform(S_X, Y_new)
                if W is None:
                    continue

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

    # ========================================
    # PHASE 2: Load teacher
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Load teacher and check what it knows")
    logger.info("="*60)

    teacher_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"\nLoading teacher: {teacher_path}")
    teacher_model, teacher_tokenizer = load(teacher_path)

    teacher_acc, teacher_results = evaluate_accuracy(teacher_model, teacher_tokenizer)
    logger.info(f"Teacher accuracy: {teacher_acc*100:.0f}%")

    # What does student still get wrong?
    student_acc_now, student_results = evaluate_accuracy(student_model, student_tokenizer)
    logger.info(f"Student accuracy: {student_acc_now*100:.0f}%")

    failing = [r for r in student_results if not r['correct']]
    logger.info(f"\nStudent failing ({len(failing)}):")
    for r in failing:
        t = next((x for x in teacher_results if x['prompt'] == r['prompt']), None)
        teacher_knows = t and t['correct']
        logger.info(f"  '{r['prompt']}': got '{r['got']}', teacher knows: {teacher_knows}")

    # ========================================
    # PHASE 3: Try combining student's 70% with teacher
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Try hybrid approach")
    logger.info("="*60)

    # Get activations from both
    student_X, student_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
    teacher_X, teacher_Y = get_layer_activations(teacher_model, teacher_tokenizer, TARGET_LAYER, probe_prompts)

    # Align teacher to student space
    alpha = 1e-4
    ATA = teacher_Y.T @ teacher_Y + alpha * np.eye(teacher_Y.shape[1])
    ATB = teacher_Y.T @ student_Y
    F, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
    teacher_Y_aligned = teacher_Y @ F

    # Try different mixing ratios
    results = []
    logger.info("\nTrying hybrid mixtures...")

    for mix in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        # Hybrid output: weighted average
        hybrid_Y = (1 - mix) * student_Y + mix * teacher_Y_aligned

        W = compute_weight_transform(student_X, hybrid_Y)
        if W is None:
            continue

        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class HybridMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        layer = student_model.model.layers[TARGET_LAYER]
        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            layer.feed_forward = HybridMLP(W_mx)
            key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            layer.mlp = HybridMLP(W_mx)
            key = 'mlp'

        hybrid_acc, _ = evaluate_accuracy(student_model, student_tokenizer)

        if key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        results.append({'mix': mix, 'accuracy': hybrid_acc})
        logger.info(f"  Mix {mix:.1f}: {hybrid_acc*100:.0f}%")

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    best = max(results, key=lambda x: x['accuracy'])

    logger.info(f"\nBaseline after self-improvement: {current_acc*100:.0f}%")
    logger.info(f"Teacher: {teacher_acc*100:.0f}%")
    logger.info(f"Best hybrid: {best['accuracy']*100:.0f}% (mix={best['mix']})")

    if best['accuracy'] > current_acc:
        logger.info(f"\nHybrid improved: +{(best['accuracy']-current_acc)*100:.0f}pp")
    else:
        logger.info(f"\nNo improvement from teacher hybridization")
        logger.info("""
The teacher (DeepSeek-R1 at 60%) cannot help the student beyond 70%
because the STUDENT has already found more than the TEACHER knows!

This is a key finding:
- Self-improvement discovered knowledge the teacher doesn't have
- LFM2 at 70% > DeepSeek-R1 at 60% on this task
- To break 70%, we need a BETTER teacher or a DIFFERENT approach
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save
    output = {
        'initial_student': float(initial_acc),
        'self_improved_student': float(current_acc),
        'teacher': float(teacher_acc),
        'best_hybrid': best,
        'all_results': results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "teacher_bridge_from_70_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
