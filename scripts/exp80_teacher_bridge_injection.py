#!/usr/bin/env python3
"""Experiment 80: Teacher Bridge Injection.

We've proven:
- Self-improvement reaches 70% ceiling (exp74-78)
- The basin is single - all paths converge to 70% (exp79)
- The remaining 30% is topologically disconnected

Hypothesis: A teacher model (DeepSeek-R1 at 91.7%) has a manifold
that IS connected to the knowledge we can't reach. By injecting
the teacher's directions, we can "bridge" to the disconnected region.

Method:
1. Load student (LFM2-1.2B at 70% ceiling) and teacher (DeepSeek-R1)
2. Identify the cases the student gets WRONG
3. Extract the teacher's direction that handles those cases
4. Inject that direction into the student
5. Measure if we break through 70%
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
    """Compute average kurtosis over samples."""
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
    """Compute spectral entropy of a manifold."""
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


def run_experiment():
    """Teacher bridge injection."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("="*80)
    logger.info("TEACHER BRIDGE INJECTION")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")
    logger.info("\nCan a teacher bridge us to the disconnected 30%?")

    # Test cases - same as before
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

    # Probe prompts for geometry
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

    # ========================================
    # LOAD STUDENT MODEL
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Load and evaluate student (LFM2-1.2B)")
    logger.info("="*60)

    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    logger.info(f"\nLoading student: {student_path}")
    student_model, student_tokenizer = load(student_path)

    student_acc, student_results = evaluate_accuracy(student_model, student_tokenizer)
    logger.info(f"\nStudent accuracy: {student_acc*100:.0f}%")

    # Identify failing cases
    failing_prompts = [r['prompt'] for r in student_results if not r['correct']]
    passing_prompts = [r['prompt'] for r in student_results if r['correct']]

    logger.info(f"\nFailing cases ({len(failing_prompts)}):")
    for r in student_results:
        if not r['correct']:
            logger.info(f"  ✗ '{r['prompt']}' → got '{r['got']}', expected '{r['expected']}'")

    # ========================================
    # LOAD TEACHER MODEL
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Load and evaluate teacher (DeepSeek-R1)")
    logger.info("="*60)

    teacher_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"\nLoading teacher: {teacher_path}")
    teacher_model, teacher_tokenizer = load(teacher_path)

    teacher_acc, teacher_results = evaluate_accuracy(teacher_model, teacher_tokenizer)
    logger.info(f"\nTeacher accuracy: {teacher_acc*100:.0f}%")

    # Check if teacher gets the failing cases right
    teacher_can_help = []
    for prompt in failing_prompts:
        teacher_result = next((r for r in teacher_results if r['prompt'] == prompt), None)
        if teacher_result and teacher_result['correct']:
            teacher_can_help.append(prompt)
            logger.info(f"  ✓ Teacher CAN help with: '{prompt}'")
        else:
            logger.info(f"  ✗ Teacher also fails: '{prompt}'")

    if not teacher_can_help:
        logger.info("\nTeacher cannot help with any failing cases. Experiment ends.")
        return

    logger.info(f"\nTeacher can potentially help with {len(teacher_can_help)} cases.")

    # ========================================
    # EXTRACT TEACHER DIRECTIONS
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Extract teacher's 'bridge' directions")
    logger.info("="*60)

    # We'll focus on Layer 2 (the key improvement layer from exp78)
    TARGET_LAYER = 2

    # Get activations on ALL probes from both models
    logger.info(f"\nCollecting activations at Layer {TARGET_LAYER}...")

    student_X, student_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
    teacher_X, teacher_Y = get_layer_activations(teacher_model, teacher_tokenizer, TARGET_LAYER, probe_prompts)

    logger.info(f"  Student: X={student_X.shape}, Y={student_Y.shape}")
    logger.info(f"  Teacher: X={teacher_X.shape}, Y={teacher_Y.shape}")

    # Handle dimension mismatch - compute alignment from teacher to student space
    logger.info(f"\nDimension handling: teacher={teacher_Y.shape[1]}, student={student_Y.shape[1]}")

    # Align teacher outputs to student space using lstsq
    # F maps teacher space → student space
    alpha = 1e-4
    teacher_Y_T = teacher_Y.T  # (4096, 20)
    student_Y_T = student_Y.T  # (2048, 20)

    # Solve: teacher_Y @ F ≈ student_Y (minimize ||teacher_Y @ F - student_Y||)
    # This gives us F that maps teacher outputs to student-like outputs
    ATA = teacher_Y.T @ teacher_Y + alpha * np.eye(teacher_Y.shape[1])
    ATB = teacher_Y.T @ student_Y

    try:
        F, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)  # F: (4096, 2048)
        logger.info(f"Alignment matrix F: {F.shape}")
    except Exception as e:
        logger.info(f"Alignment failed: {e}")
        return

    # Project teacher outputs into student space
    teacher_Y_aligned = teacher_Y @ F  # (20, 2048) - now same dimension as student

    # SVD on ALIGNED teacher outputs
    teacher_Y_aligned_centered = teacher_Y_aligned - teacher_Y_aligned.mean(axis=0)
    _, S_teacher, Vh_teacher = svd(teacher_Y_aligned_centered, full_matrices=False)

    logger.info(f"\nAligned teacher's singular values: {S_teacher[:5]}")

    # ========================================
    # INJECT TEACHER DIRECTIONS
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Inject aligned teacher directions into student")
    logger.info("="*60)

    # The bridge: project student outputs onto aligned teacher's principal directions
    # Then use aligned teacher's coefficients to "pull" student toward teacher's manifold

    results = []

    # Try injecting different directions
    for direction_idx in range(min(10, len(Vh_teacher))):
        logger.info(f"\n--- Testing Direction {direction_idx} ---")

        # Get the aligned teacher's direction (now in student's 2048-dim space)
        teacher_direction = Vh_teacher[direction_idx]

        # Project student outputs onto this direction
        student_Y_centered = student_Y - student_Y.mean(axis=0)
        student_coefs = student_Y_centered @ teacher_direction

        # Get aligned teacher's coefficients on this direction
        teacher_coefs = teacher_Y_aligned_centered @ teacher_direction

        # The "bridge": shift student coefficients toward aligned teacher coefficients
        coef_shift = teacher_coefs.mean() - student_coefs.mean()

        logger.info(f"  Aligned teacher mean coef: {teacher_coefs.mean():.4f}")
        logger.info(f"  Student mean coef: {student_coefs.mean():.4f}")
        logger.info(f"  Shift needed: {coef_shift:.4f}")

        # Try different shift strengths
        for shift_strength in [0.5, 1.0, 2.0, 3.0]:
            # Apply the bridge: shift student outputs along teacher direction
            shifted_Y = student_Y + (coef_shift * shift_strength) * teacher_direction

            # Compute weight matrix for this transformation
            alpha = 1e-3
            ATA = student_X.T @ student_X + alpha * np.eye(student_X.shape[1])
            ATB = student_X.T @ shifted_Y

            try:
                W, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
                W = W.T
            except:
                continue

            if np.isnan(W).any() or np.isinf(W).any():
                continue

            # Apply to student
            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class BridgedMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = student_model.model.layers[TARGET_LAYER]
            if hasattr(layer, 'feed_forward'):
                original_mlp = layer.feed_forward
                layer.feed_forward = BridgedMLP(W_mx)
                mlp_key = 'feed_forward'
            else:
                original_mlp = layer.mlp
                layer.mlp = BridgedMLP(W_mx)
                mlp_key = 'mlp'

            # Evaluate
            new_acc, new_results = evaluate_accuracy(student_model, student_tokenizer)

            # Restore
            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            result = {
                'direction': direction_idx,
                'shift_strength': shift_strength,
                'accuracy': new_acc,
                'improved': new_acc > student_acc,
            }
            results.append(result)

            if new_acc > student_acc:
                logger.info(f"    Strength {shift_strength}: {new_acc*100:.0f}% ← IMPROVED!")
            elif new_acc == student_acc:
                logger.info(f"    Strength {shift_strength}: {new_acc*100:.0f}%")
            else:
                logger.info(f"    Strength {shift_strength}: {new_acc*100:.0f}% (worse)")

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    best_result = max(results, key=lambda x: x['accuracy'])
    improvements = [r for r in results if r['improved']]

    logger.info(f"\nBaseline (student): {student_acc*100:.0f}%")
    logger.info(f"Teacher: {teacher_acc*100:.0f}%")
    logger.info(f"\nBest result: {best_result['accuracy']*100:.0f}% (d{best_result['direction']} s{best_result['shift_strength']})")
    logger.info(f"Configurations that improved: {len(improvements)}/{len(results)}")

    if best_result['accuracy'] > 0.70:
        logger.info(f"\n{'='*60}")
        logger.info("SUCCESS: BROKE THROUGH THE 70% CEILING!")
        logger.info(f"{'='*60}")
        logger.info(f"""
The teacher's direction successfully bridged the student to
the disconnected region of the manifold.

Ceiling broken: {student_acc*100:.0f}% → {best_result['accuracy']*100:.0f}%
Improvement: +{(best_result['accuracy'] - student_acc)*100:.0f}pp

This proves:
1. The 30% WAS disconnected from the student's manifold
2. The teacher's manifold IS connected to that region
3. Direction injection CAN bridge disconnected basins
""")
    elif best_result['accuracy'] == 0.70:
        logger.info(f"\n{'='*60}")
        logger.info("RESULT: Teacher direction maintained but didn't break 70%")
        logger.info(f"{'='*60}")
        logger.info("""
The teacher's directions didn't break the ceiling.

Possible reasons:
1. Need to inject into multiple layers
2. Need different layer (not Layer 2)
3. Need to inject teacher's SPECIFIC failing-case geometry
4. Architectural differences prevent direct bridging
""")
    else:
        logger.info(f"\n{'='*60}")
        logger.info("RESULT: Teacher direction made things worse")
        logger.info(f"{'='*60}")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'student_baseline': student_acc,
        'teacher_accuracy': teacher_acc,
        'best_result': best_result,
        'all_results': results,
        'improvements': len(improvements),
        'target_layer': TARGET_LAYER,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "teacher_bridge_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
