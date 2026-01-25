#!/usr/bin/env python3
"""Experiment 71: Geometry-Guided Teaching.

From exp70 we learned:
- Correct answers have HIGH kurtosis (66 vs 38)
- Correct answers have LOW spectral entropy (0.68 vs 1.30)
- Correct answers have LOW effective rank (2 vs 4)

Now we use this to GUIDE teaching:
- Select directions that INCREASE kurtosis
- Select directions that DECREASE spectral entropy
- Push the model toward the "correct" manifold geometry

This is teaching with a geometric objective, not just mimicking the teacher.
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


def compute_kurtosis(h):
    """Compute kurtosis of hidden state."""
    z = (h - h.mean()) / (h.std() + 1e-10)
    return float(np.mean(z ** 4) - 3)


def compute_spectral_entropy(Y):
    """Compute spectral entropy of a manifold."""
    Y_centered = Y - Y.mean(axis=0)
    _, S, _ = svd(Y_centered, full_matrices=False)
    S_norm = S / (S.sum() + 1e-10)
    return -float(np.sum(S_norm * np.log(S_norm + 1e-10)))


def run_experiment():
    """Geometry-guided teaching."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("Loading LFM2-1.2B (student)...")
    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    student, student_tok = load(student_path)

    logger.info("Loading LFM2.5-1.2B-Instruct (teacher)...")
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    teacher, teacher_tok = load(teacher_path)

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

    # Calibration prompts
    calibration_prompts = [
        "The capital of", "The largest planet",
        "Water freezes at", "If it rains then",
        "The opposite of up is", "2 + 2 equals",
        "A noun is", "The past tense of run is",
        "The square root of", "10 times 10",
        "The sky is", "Birds can",
        "Fish live in", "The sun rises",
        "Gravity causes", "The speed of light",
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

    def apply_teaching_and_measure(S_X, S_Y, T_Y, direction_idx):
        """Apply direction replacement and measure geometry change."""
        s_norm = np.linalg.norm(S_Y)
        t_norm = np.linalg.norm(T_Y)

        S_Y_scaled = S_Y / (s_norm + 1e-10)
        T_Y_scaled = T_Y / (t_norm + 1e-10)

        S_Y_centered = S_Y_scaled - S_Y_scaled.mean(axis=0)
        T_Y_centered = T_Y_scaled - T_Y_scaled.mean(axis=0)

        try:
            _, _, Vh_s = svd(S_Y_centered, full_matrices=False)
            _, _, Vh_t = svd(T_Y_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None

        d = direction_idx
        if d >= min(len(Vh_s), len(Vh_t)):
            return None, None

        S_Y_centered_orig = S_Y - S_Y.mean(axis=0)
        T_Y_centered_orig = T_Y - T_Y.mean(axis=0)

        result = S_Y.copy()

        s_coefs = S_Y_centered_orig @ Vh_s[d]
        s_proj = np.outer(s_coefs, Vh_s[d])
        result -= s_proj

        t_coefs = T_Y_centered_orig @ Vh_t[d]
        t_proj = np.outer(t_coefs, Vh_t[d])
        result += t_proj

        alpha = 1e-4
        ATA = S_X.T @ S_X + alpha * np.eye(S_X.shape[1])
        ATB = S_X.T @ result

        try:
            W = np.linalg.solve(ATA, ATB).T
        except np.linalg.LinAlgError:
            return None, None

        if np.isnan(W).any() or np.isinf(W).any():
            return None, None

        # Measure geometry of result
        avg_kurtosis = np.mean([compute_kurtosis(r) for r in result])
        spectral_ent = compute_spectral_entropy(result)

        geometry = {
            'kurtosis': avg_kurtosis,
            'spectral_entropy': spectral_ent,
        }

        return W, geometry

    # ========================================
    # PHASE 1: Baseline
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Baseline Geometry")
    logger.info(f"{'='*80}")

    baseline_acc = evaluate_accuracy(student, student_tok)
    logger.info(f"Baseline accuracy: {baseline_acc*100:.0f}%")

    # Measure baseline geometry
    _, S_Y = get_layer_activations(student, student_tok, 10, calibration_prompts)
    baseline_kurtosis = np.mean([compute_kurtosis(y) for y in S_Y])
    baseline_spectral = compute_spectral_entropy(S_Y)

    logger.info(f"Baseline kurtosis: {baseline_kurtosis:.4f}")
    logger.info(f"Baseline spectral entropy: {baseline_spectral:.4f}")

    # ========================================
    # PHASE 2: Search for Best Geometry
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Search for Best Geometry-Improving Configuration")
    logger.info(f"{'='*80}")

    # Score function: higher kurtosis + lower spectral entropy
    def geometry_score(geo):
        # Normalize: kurtosis is good when high, spectral entropy when low
        # Higher score = better (more "correct-like")
        kurtosis_score = geo['kurtosis'] / 100  # Normalize
        entropy_score = -geo['spectral_entropy']  # Negative because lower is better
        return kurtosis_score + entropy_score

    best_config = None
    best_geo_score = geometry_score({'kurtosis': baseline_kurtosis, 'spectral_entropy': baseline_spectral})
    best_acc = baseline_acc

    results = []

    for layer_idx in [4, 6, 8, 10, 12, 14]:
        logger.info(f"\n--- Layer {layer_idx} ---")

        S_X, S_Y = get_layer_activations(student, student_tok, layer_idx, calibration_prompts)
        T_X, T_Y = get_layer_activations(teacher, teacher_tok, layer_idx, calibration_prompts)

        for d in range(8):
            W, geo = apply_teaching_and_measure(S_X, S_Y, T_Y, d)
            if W is None:
                continue

            geo_score = geometry_score(geo)

            # Apply and measure accuracy
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

            results.append({
                'layer': layer_idx,
                'direction': d,
                'accuracy': acc,
                'geo_score': geo_score,
                'kurtosis': geo['kurtosis'],
                'spectral_entropy': geo['spectral_entropy'],
            })

            if geo_score > best_geo_score and acc >= baseline_acc:
                best_geo_score = geo_score
                best_config = (layer_idx, d)
                best_acc = acc

        # Show best for this layer
        layer_results = [r for r in results if r['layer'] == layer_idx]
        if layer_results:
            best_for_layer = max(layer_results, key=lambda x: x['geo_score'])
            logger.info(f"  Best d{best_for_layer['direction']}: score={best_for_layer['geo_score']:.4f}, "
                       f"acc={best_for_layer['accuracy']*100:.0f}%")

    # ========================================
    # PHASE 3: Analyze Correlation
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Geometry-Accuracy Correlation")
    logger.info(f"{'='*80}")

    if len(results) > 2:
        accs = np.array([r['accuracy'] for r in results])
        geo_scores = np.array([r['geo_score'] for r in results])
        kurtoses = np.array([r['kurtosis'] for r in results])
        entropies = np.array([r['spectral_entropy'] for r in results])

        corr_geo = np.corrcoef(geo_scores, accs)[0, 1] if accs.std() > 0 else 0
        corr_kurtosis = np.corrcoef(kurtoses, accs)[0, 1] if accs.std() > 0 else 0
        corr_entropy = np.corrcoef(entropies, accs)[0, 1] if accs.std() > 0 else 0

        logger.info(f"\nCorrelation with accuracy:")
        logger.info(f"  Geometry score: {corr_geo:+.4f}")
        logger.info(f"  Kurtosis:       {corr_kurtosis:+.4f}")
        logger.info(f"  Spectral entropy: {corr_entropy:+.4f}")

    # ========================================
    # PHASE 4: Apply Best Configuration
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Apply Best Configuration")
    logger.info(f"{'='*80}")

    if best_config:
        layer_idx, d = best_config
        logger.info(f"\nBest config: Layer {layer_idx}, Direction {d}")
        logger.info(f"Geometry score: {best_geo_score:.4f} (baseline: {geometry_score({'kurtosis': baseline_kurtosis, 'spectral_entropy': baseline_spectral}):.4f})")

        # Apply and show results
        S_X, S_Y = get_layer_activations(student, student_tok, layer_idx, calibration_prompts)
        T_X, T_Y = get_layer_activations(teacher, teacher_tok, layer_idx, calibration_prompts)

        W, geo = apply_teaching_and_measure(S_X, S_Y, T_Y, d)
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

        logger.info(f"\n{'Prompt':<40} {'Before':>15} {'After':>15} {'Expected':>12}")
        logger.info("-" * 85)

        # First restore to show before
        if mlp_key == 'feed_forward':
            layer.feed_forward = original_mlp
        else:
            layer.mlp = original_mlp

        for prompt, expected in test_cases:
            before = get_prediction(student, student_tok, prompt)

            if mlp_key == 'feed_forward':
                layer.feed_forward = TaughtMLP(W_mx)
            else:
                layer.mlp = TaughtMLP(W_mx)

            after = get_prediction(student, student_tok, prompt)

            if mlp_key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

            before_ok = expected.lower() in before.lower()
            after_ok = expected.lower() in after.lower()

            change = ""
            if after_ok and not before_ok:
                change = "← IMPROVED!"
            elif before_ok and not after_ok:
                change = "← degraded"

            logger.info(f"{prompt:<40} {before:>15} {after:>15} {expected:>12} {change}")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Geometry-Guided Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
WHAT WE DID:

1. Defined a "correctness geometry score":
   score = kurtosis/100 - spectral_entropy

2. Searched for teaching configurations that:
   - INCREASE the geometry score
   - DON'T decrease accuracy

3. Measured correlation between geometry and accuracy

FINDINGS:

- Best baseline geometry score: {geometry_score({'kurtosis': baseline_kurtosis, 'spectral_entropy': baseline_spectral}):.4f}
- Best found geometry score: {best_geo_score:.4f}
- Best config: {best_config}

THE INSIGHT:

If geometry correlates with accuracy, then:
1. We can use geometry as a PROXY for correctness
2. Teaching toward "correct" geometry = teaching toward correct answers
3. No need for labeled data - just geometric optimization

THE VISION:

Instead of:
  "Make the student match the teacher's outputs"

We can:
  "Move the student toward higher-kurtosis, lower-entropy manifolds"

This is SELF-IMPROVING geometry - no teacher needed!
""")


if __name__ == "__main__":
    run_experiment()
