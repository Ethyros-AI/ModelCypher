#!/usr/bin/env python3
"""Experiment 82: Multi-Layer Teacher Bridge.

Key insight from exp81:
- Teacher knows the failing cases
- Injecting at Layer 2 doesn't help
- Maybe the knowledge lives at a DIFFERENT layer
- Or needs to be injected at MULTIPLE layers

This experiment:
1. Find which layer(s) in the teacher best encode the failing cases
2. Inject at the corresponding layer(s) in the student
3. Try simultaneous multi-layer injection
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


def run_experiment():
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("="*80)
    logger.info("MULTI-LAYER TEACHER BRIDGE")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")

    # The failing cases
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
    # PHASE 1: Load models
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Load models")
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

    num_student_layers = len(student_model.model.layers)
    num_teacher_layers = len(teacher_model.model.layers)

    logger.info(f"Student layers: {num_student_layers}")
    logger.info(f"Teacher layers: {num_teacher_layers}")

    # ========================================
    # PHASE 2: Profile which teacher layers encode failing cases
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Profile teacher layers for failing cases")
    logger.info("="*60)

    failing_prompts = [p for p, e in failing_cases]

    # Get activations at each teacher layer for failing prompts
    logger.info("\nAnalyzing teacher layer activations for failing cases...")

    teacher_layer_profiles = []
    for layer_idx in range(num_teacher_layers):
        _, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, layer_idx, failing_prompts)

        # Measure how "separated" the 3 failing cases are in this layer
        T_Y_centered = T_Y - T_Y.mean(axis=0)
        try:
            _, S, _ = svd(T_Y_centered, full_matrices=False)
            effective_dim = (S**2).sum()**2 / ((S**4).sum() + 1e-10)
            variance_ratio = S[0]**2 / (S**2).sum() if len(S) > 0 else 0
        except:
            effective_dim = 0
            variance_ratio = 0

        teacher_layer_profiles.append({
            'layer': layer_idx,
            'effective_dim': effective_dim,
            'variance_ratio': variance_ratio,
            'output_norm': np.linalg.norm(T_Y)
        })

        if layer_idx % 4 == 0:
            logger.info(f"  Layer {layer_idx}: eff_dim={effective_dim:.2f}, var_ratio={variance_ratio:.2f}")

    # Find layers with highest separation (low effective dim, high variance ratio)
    best_teacher_layers = sorted(teacher_layer_profiles,
                                  key=lambda x: x['variance_ratio'],
                                  reverse=True)[:5]

    logger.info("\nBest teacher layers for failing cases:")
    for p in best_teacher_layers:
        logger.info(f"  Layer {p['layer']}: var_ratio={p['variance_ratio']:.3f}, eff_dim={p['effective_dim']:.2f}")

    # ========================================
    # PHASE 3: Try injection at each layer
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Try injection at different student layers")
    logger.info("="*60)

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

    results = []

    # Ratio to map teacher layers to student layers
    layer_ratio = num_student_layers / num_teacher_layers

    for student_layer in range(num_student_layers):
        # Map to corresponding teacher layer
        teacher_layer = int(student_layer / layer_ratio)
        teacher_layer = min(teacher_layer, num_teacher_layers - 1)

        logger.info(f"\nStudent Layer {student_layer} <- Teacher Layer {teacher_layer}:")

        # Get activations
        S_X, S_Y = get_layer_activations(student_model, student_tokenizer, student_layer, probe_prompts)
        T_X, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, teacher_layer, probe_prompts)

        # Align teacher to student
        alpha = 1e-4
        ATA = T_Y.T @ T_Y + alpha * np.eye(T_Y.shape[1])
        ATB = T_Y.T @ S_Y
        F, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
        T_Y_aligned = T_Y @ F

        # Mix student and teacher outputs
        for mix in [0.1, 0.3, 0.5]:
            hybrid_Y = (1 - mix) * S_Y + mix * T_Y_aligned

            # Compute W
            S_X_scale = np.abs(S_X).max()
            Y_scale = np.abs(hybrid_Y).max()
            if S_X_scale < 1e-10 or Y_scale < 1e-10:
                continue

            S_X_norm = S_X / S_X_scale
            Y_norm = hybrid_Y / Y_scale

            reg = 1e-3
            ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
            ATB_w = S_X_norm.T @ Y_norm
            W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
            W = (W_norm * Y_scale / S_X_scale).T

            if np.isnan(W).any() or np.isinf(W).any():
                continue

            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class TestMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = student_model.model.layers[student_layer]
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
                'student_layer': student_layer,
                'teacher_layer': teacher_layer,
                'mix': mix,
                'accuracy': new_acc,
            })
            logger.info(f"  mix={mix}: {new_acc*100:.0f}%")

            # Early exit if we find improvement
            if new_acc > 0.7:
                logger.info(f"\n*** FOUND IMPROVEMENT: {new_acc*100:.0f}% at Layer {student_layer} ***")

    # ========================================
    # PHASE 4: Try multi-layer simultaneous injection
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("PHASE 4: Multi-layer simultaneous injection")
    logger.info("="*60)

    # First, self-improve to 70%
    logger.info("\nSelf-improving to 70% baseline first...")

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

    TARGET_LAYER = 2
    directions = list(range(12))
    boosts = [0.0, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]
    current_acc = student_acc

    for round_num in range(20):
        S_X, S_Y = get_layer_activations(student_model, student_tokenizer, TARGET_LAYER, probe_prompts)
        baseline_score = geometry_score(compute_kurtosis(S_Y), compute_spectral_entropy(S_Y))

        S_Y_centered = S_Y - S_Y.mean(axis=0)
        _, S, Vh = svd(S_Y_centered, full_matrices=False)

        best = None
        for d in directions:
            for boost in boosts:
                if boost == 1.0 or d >= len(Vh):
                    continue

                coefs = S_Y_centered @ Vh[d]
                proj = np.outer(coefs, Vh[d])
                Y_new = S_Y + proj * (boost - 1)

                new_score = geometry_score(compute_kurtosis(Y_new), compute_spectral_entropy(Y_new))
                if new_score <= baseline_score + 1e-4:
                    continue

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

    # Now try multi-layer injection from this baseline
    logger.info("\nTrying multi-layer injection from 70% baseline...")

    # Collect teacher directions for all layers
    layers_to_try = [0, 2, 4, 8, 12]

    for num_layers in [2, 3, 4]:
        for layer_combo in [layers_to_try[:num_layers]]:
            logger.info(f"\n  Injecting at layers {layer_combo}:")

            # Store original MLPs
            original_mlps = {}
            for layer_idx in layer_combo:
                layer = student_model.model.layers[layer_idx]
                if hasattr(layer, 'feed_forward'):
                    original_mlps[layer_idx] = ('feed_forward', layer.feed_forward)
                else:
                    original_mlps[layer_idx] = ('mlp', layer.mlp)

            # Try different injection strengths
            for mix in [0.1, 0.2, 0.3]:
                # Apply injection at all layers
                for layer_idx in layer_combo:
                    teacher_layer = int(layer_idx / layer_ratio)
                    teacher_layer = min(teacher_layer, num_teacher_layers - 1)

                    S_X, S_Y = get_layer_activations(student_model, student_tokenizer, layer_idx, probe_prompts)
                    T_X, T_Y = get_layer_activations(teacher_model, teacher_tokenizer, teacher_layer, probe_prompts)

                    # Align
                    alpha = 1e-4
                    ATA = T_Y.T @ T_Y + alpha * np.eye(T_Y.shape[1])
                    ATB = T_Y.T @ S_Y
                    F, _, _, _ = np.linalg.lstsq(ATA, ATB, rcond=None)
                    T_Y_aligned = T_Y @ F

                    # Mix
                    hybrid_Y = (1 - mix) * S_Y + mix * T_Y_aligned

                    # Compute W
                    S_X_scale = np.abs(S_X).max()
                    Y_scale = np.abs(hybrid_Y).max()
                    if S_X_scale < 1e-10 or Y_scale < 1e-10:
                        continue

                    S_X_norm = S_X / S_X_scale
                    Y_norm = hybrid_Y / Y_scale

                    reg = 1e-3
                    ATA_w = S_X_norm.T @ S_X_norm + reg * np.eye(S_X_norm.shape[1])
                    ATB_w = S_X_norm.T @ Y_norm
                    W_norm, _, _, _ = np.linalg.lstsq(ATA_w, ATB_w, rcond=None)
                    W = (W_norm * Y_scale / S_X_scale).T

                    if np.isnan(W).any() or np.isinf(W).any():
                        continue

                    W_mx = mx.array(W.astype(np.float32))
                    mx.eval(W_mx)

                    class InjectMLP:
                        def __init__(self, W):
                            self.W = W
                        def __call__(self, x):
                            return mx.matmul(x, self.W.T)

                    layer = student_model.model.layers[layer_idx]
                    key, _ = original_mlps[layer_idx]
                    if key == 'feed_forward':
                        layer.feed_forward = InjectMLP(W_mx)
                    else:
                        layer.mlp = InjectMLP(W_mx)

                # Evaluate
                new_acc, _ = evaluate_accuracy(student_model, student_tokenizer)
                results.append({
                    'layers': layer_combo,
                    'mix': mix,
                    'accuracy': new_acc,
                })
                logger.info(f"    mix={mix}: {new_acc*100:.0f}%")

                # Restore
                for layer_idx in layer_combo:
                    layer = student_model.model.layers[layer_idx]
                    key, original = original_mlps[layer_idx]
                    if key == 'feed_forward':
                        layer.feed_forward = original
                    else:
                        layer.mlp = original

    # ========================================
    # ANALYSIS
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)

    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
    else:
        best_result = {'accuracy': 0.5, 'strategy': 'none'}

    logger.info(f"\nBaseline student: {student_acc*100:.0f}%")
    logger.info(f"Teacher: {teacher_acc*100:.0f}%")
    logger.info(f"Self-improved: {current_acc*100:.0f}%")
    logger.info(f"Best with multi-layer: {best_result['accuracy']*100:.0f}%")

    if best_result['accuracy'] > current_acc:
        logger.info(f"\n✓ Multi-layer injection helped: +{(best_result['accuracy']-current_acc)*100:.0f}pp")
    else:
        logger.info("""
No improvement from multi-layer teacher bridge.

This further confirms:
- The 70% ceiling is NOT about which layer we inject
- The teacher's geometric structure for this knowledge
  fundamentally differs from the student's
- We may need a COMPLETELY different approach:
  1. A teacher that exceeds 70% on this test set
  2. Or a different encoding/decoding of the knowledge
  3. Or the knowledge simply cannot be transferred geometrically
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'student_baseline': float(student_acc),
        'teacher': float(teacher_acc),
        'self_improved': float(current_acc),
        'best_result': best_result,
        'all_results': results,
        'teacher_layer_profiles': teacher_layer_profiles,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "multilayer_teacher_bridge_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x if isinstance(x, (int, float, str, bool, list, dict, type(None))) else str(x))

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
