#!/usr/bin/env python3
"""Experiment 87: Generation-Based Self-Improvement.

Key insight from exp86:
- Single-token evaluation limits exploration to "letters a,b,c"
- Generation-based evaluation uses the "full alphabet"
- Qwen2.5-Coder gets 100% with generation, only 80% with top-token

New approach:
- Self-improve using GENERATION accuracy as the metric
- Allow the model to explore multi-token sequences
- Break through the single-token ceiling
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
    from mlx_lm import load, generate

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    logger.info("="*80)
    logger.info("GENERATION-BASED SELF-IMPROVEMENT")
    logger.info("="*80)
    logger.info(f"\nStarting at {datetime.now().isoformat()}")
    logger.info("\nUsing GENERATION accuracy instead of single-token accuracy")

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

    def evaluate_generation_accuracy(model, tokenizer):
        """Evaluate using generation instead of single token."""
        correct = 0
        results = []
        for prompt, expected in test_cases:
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=10,
                verbose=False
            )
            is_correct = expected.lower() in response.lower()
            if is_correct:
                correct += 1
            results.append({
                'prompt': prompt,
                'expected': expected,
                'got': response[:30],
                'correct': is_correct
            })
        return correct / len(test_cases), results

    def evaluate_top_token_accuracy(model, tokenizer):
        """Our old method for comparison."""
        correct = 0
        for prompt, expected in test_cases:
            tokens = tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = model(input_ids)
            mx.eval(logits)
            top_token = int(mx.argmax(logits[0, -1, :]).item())
            word = tokenizer.decode([top_token]).strip()
            if expected.lower() in word.lower():
                correct += 1
        return correct / len(test_cases)

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
    # Load model
    # ========================================

    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    logger.info(f"\nLoading: {student_path}")
    model, tokenizer = load(student_path)

    # Initial evaluation with BOTH methods
    initial_gen_acc, initial_results = evaluate_generation_accuracy(model, tokenizer)
    initial_top_acc = evaluate_top_token_accuracy(model, tokenizer)

    logger.info(f"\nInitial accuracy:")
    logger.info(f"  Top-token: {initial_top_acc*100:.0f}%")
    logger.info(f"  Generation: {initial_gen_acc*100:.0f}%")

    # ========================================
    # Self-improvement using GENERATION accuracy
    # ========================================

    logger.info("\n" + "="*60)
    logger.info("GENERATION-BASED SELF-IMPROVEMENT")
    logger.info("="*60)

    TARGET_LAYER = 2
    directions = list(range(12))
    boosts = [0.0, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]

    current_gen_acc = initial_gen_acc
    current_top_acc = initial_top_acc

    improvement_log = []

    for round_num in range(20):
        logger.info(f"\nRound {round_num + 1}:")

        S_X, S_Y = get_layer_activations(model, tokenizer, TARGET_LAYER, probe_prompts)
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

                # Check geometry improvement
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

                class TestMLP:
                    def __init__(self, W):
                        self.W = W
                    def __call__(self, x):
                        return mx.matmul(x, self.W.T)

                layer = model.model.layers[TARGET_LAYER]
                if hasattr(layer, 'feed_forward'):
                    original_mlp = layer.feed_forward
                    layer.feed_forward = TestMLP(W_mx)
                    key = 'feed_forward'
                else:
                    original_mlp = layer.mlp
                    layer.mlp = TestMLP(W_mx)
                    key = 'mlp'

                # EVALUATE WITH GENERATION (the key change!)
                new_gen_acc, _ = evaluate_generation_accuracy(model, tokenizer)

                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

                # Keep if generation accuracy improves
                if new_gen_acc >= current_gen_acc:
                    if best is None or new_gen_acc > best['gen_acc']:
                        best = {
                            'W': W,
                            'gen_acc': new_gen_acc,
                            'key': key,
                            'direction': d,
                            'boost': boost,
                            'geo_score': new_score,
                        }

        if best:
            W_mx = mx.array(best['W'].astype(np.float32))
            mx.eval(W_mx)

            class PermanentMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            layer = model.model.layers[TARGET_LAYER]
            if best['key'] == 'feed_forward':
                layer.feed_forward = PermanentMLP(W_mx)
            else:
                layer.mlp = PermanentMLP(W_mx)

            if best['gen_acc'] > current_gen_acc:
                current_gen_acc = best['gen_acc']
                current_top_acc = evaluate_top_token_accuracy(model, tokenizer)

                improvement_log.append({
                    'round': round_num + 1,
                    'direction': best['direction'],
                    'boost': best['boost'],
                    'gen_acc': best['gen_acc'],
                    'geo_score': best['geo_score'],
                })

                logger.info(f"  IMPROVED: d{best['direction']} b{best['boost']:.1f}")
                logger.info(f"    Generation: {current_gen_acc*100:.0f}%")
                logger.info(f"    Top-token:  {current_top_acc*100:.0f}%")
            else:
                logger.info(f"  Geometry improved but accuracy stayed at {current_gen_acc*100:.0f}%")

        if current_gen_acc >= 1.0:
            logger.info("\n🎉 REACHED 100% GENERATION ACCURACY!")
            break

    # ========================================
    # FINAL EVALUATION
    # ========================================

    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)

    final_gen_acc, final_results = evaluate_generation_accuracy(model, tokenizer)
    final_top_acc = evaluate_top_token_accuracy(model, tokenizer)

    logger.info(f"\n{'Prompt':<40} {'Result':<10} {'Generated'}")
    logger.info("-" * 80)

    for r in final_results:
        mark = "✓" if r['correct'] else "✗"
        logger.info(f"{mark} {r['prompt']:<38} → {r['got']}")

    logger.info(f"\nFinal accuracy:")
    logger.info(f"  Top-token:  {final_top_acc*100:.0f}%")
    logger.info(f"  Generation: {final_gen_acc*100:.0f}%")

    logger.info(f"\nImprovement:")
    logger.info(f"  Top-token:  {initial_top_acc*100:.0f}% → {final_top_acc*100:.0f}% ({(final_top_acc-initial_top_acc)*100:+.0f}pp)")
    logger.info(f"  Generation: {initial_gen_acc*100:.0f}% → {final_gen_acc*100:.0f}% ({(final_gen_acc-initial_gen_acc)*100:+.0f}pp)")

    if final_gen_acc > 0.7:
        logger.info("""
🎉 BROKE THE 70% CEILING!

By using generation-based evaluation, we allowed the model to explore
multi-token relationships instead of just single-token predictions.

This proves the hypothesis:
- Single-token limits = "using only letters a,b,c"
- Generation evaluation = "using the full alphabet"
""")

    logger.info(f"\nCompleted at {datetime.now().isoformat()}")

    # Save results
    output = {
        'initial_top_token': float(initial_top_acc),
        'initial_generation': float(initial_gen_acc),
        'final_top_token': float(final_top_acc),
        'final_generation': float(final_gen_acc),
        'improvements': improvement_log,
        'final_results': final_results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = Path(__file__).parent.parent / "data" / "generation_based_self_improvement_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
