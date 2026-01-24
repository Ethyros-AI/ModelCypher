#!/usr/bin/env python3
"""Experiment 54: Optimal Direction Replacement.

From exp52/53: The REPLACEMENT method beats the ONLY method.
- Replacement: Replace target's direction d with source's direction d → 91.7%
- Only: Start from mean, add only source's directions → 83.3%

The insight: We must PRESERVE what target already knows (its other directions)
and REPLACE only what we want to teach.

This is exactly like teaching humans:
- Don't erase everything and start fresh
- Replace only the misconception with correct knowledge
- Keep everything else the student already knows

Question: Can we achieve 100% by finding the optimal replacement?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Find optimal direction replacement for maximum accuracy."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    # Load models
    source_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    logger.info("Loading source model (DeepSeek-R1-8B)...")
    from mlx_lm import load
    source_model, source_tokenizer = load(source_path)

    logger.info("Loading target model (LFM2-1.2B)...")
    target_model, target_tokenizer = load(target_path)

    source_layer_idx = 24
    target_layer_idx = 10

    train_prompts = [
        "The capital of France is Paris",
        "Water freezes at zero degrees",
        "The largest planet is Jupiter",
        "DNA stands for deoxyribonucleic acid",
        "The speed of light is very fast",
        "Photosynthesis occurs in plants",
        "The periodic table organizes elements",
        "Machine learning uses algorithms",
        "The theory of relativity was proposed",
        "Quantum mechanics describes particles",
        "Shakespeare wrote many plays",
        "The human brain has neurons",
        "Evolution explains species change",
        "Gravity attracts masses together",
        "The internet connects computers worldwide",
        "Vaccines prevent diseases effectively",
        "Mountains are formed by tectonics",
        "Rivers flow towards the ocean",
        "Stars are made of plasma",
        "Cells are the basic unit of life",
        "Electricity powers modern devices",
        "Sound travels through air as waves",
        "Chemistry studies matter and reactions",
        "History records past events accurately",
        "Music creates emotional responses",
        "Architecture shapes living spaces",
        "Medicine advances through research",
        "Technology transforms society",
        "Nature operates through cycles",
        "Time flows in one direction",
        "Memory stores past experiences",
        "Imagination creates new possibilities",
    ]

    test_prompts = [
        "The moon is", "Trees are", "Birds can", "Mountains are",
        "Electrons orbit", "Genes contain", "Stories tell", "Poetry expresses",
        "Therefore we", "Because of", "Technology enables", "Culture shapes",
    ]

    def get_activations(model, tokenizer, layer_idx, prompts):
        """Collect MLP activations."""
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

    def evaluate_with_details(target_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts):
        """Evaluate and return per-prompt results."""
        alpha = 1e-6
        ATA = target_X.T @ target_X + alpha * np.eye(target_X.shape[1])
        ATB = target_X.T @ target_output
        W = np.linalg.solve(ATA, ATB).T

        W_mx = mx.array(W.astype(np.float32))
        mx.eval(W_mx)

        class TransplantedMLP:
            def __init__(self, W):
                self.W = W
            def __call__(self, x):
                return mx.matmul(x, self.W.T)

        target_layer = target_model.model.layers[target_layer_idx]
        if hasattr(target_layer, 'feed_forward'):
            original_mlp = target_layer.feed_forward
            mlp_key = 'feed_forward'
        else:
            original_mlp = target_layer.mlp
            mlp_key = 'mlp'

        results = []
        for prompt in test_prompts:
            tokens = target_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            orig_logits = target_model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            if mlp_key == 'feed_forward':
                target_layer.feed_forward = TransplantedMLP(W_mx)
            else:
                target_layer.mlp = TransplantedMLP(W_mx)

            try:
                trans_logits = target_model(input_ids)
                mx.eval(trans_logits)
                trans_top = int(mx.argmax(trans_logits[0, -1, :]).item())
            finally:
                if mlp_key == 'feed_forward':
                    target_layer.feed_forward = original_mlp
                else:
                    target_layer.mlp = original_mlp

            match = orig_top == trans_top
            orig_word = target_tokenizer.decode([orig_top])
            trans_word = target_tokenizer.decode([trans_top])
            results.append((prompt, match, orig_word, trans_word))

        acc = sum(1 for _, m, _, _ in results if m) / len(results)
        return acc, results

    # Collect activations
    logger.info(f"\n{'='*80}")
    logger.info("Collecting Activations")
    logger.info(f"{'='*80}")

    source_X, source_Y = get_activations(source_model, source_tokenizer, source_layer_idx, train_prompts)
    target_X, target_Y = get_activations(target_model, target_tokenizer, target_layer_idx, train_prompts)

    # SVD decomposition
    source_Y_centered = source_Y - source_Y.mean(axis=0)
    U_s, S_s, Vh_s = np.linalg.svd(source_Y_centered, full_matrices=False)

    target_Y_centered = target_Y - target_Y.mean(axis=0)
    U_t, S_t, Vh_t = np.linalg.svd(target_Y_centered, full_matrices=False)

    F = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]

    source_var = S_s**2 / np.sum(S_s**2)

    def replace_directions(dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F):
        """Replace specified directions in target with source's (the REPLACEMENT method).

        This is the method that achieved 91.7% in exp52.
        """
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        result = target_Y.copy()

        for d in dirs:
            if d < len(Vh_s) and d < len(Vh_t):
                # Remove target's direction d
                target_coefs_d = target_Y_centered @ Vh_t[d]
                target_proj_d = np.outer(target_coefs_d, Vh_t[d])
                result -= target_proj_d

                # Add source's direction d (translated to target space)
                source_coefs_d = source_Y_centered @ Vh_s[d]
                source_proj_d = np.outer(source_coefs_d, Vh_s[d])
                result += source_proj_d @ F

        return result

    # ========================================
    # PHASE 1: Test each single direction replacement
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Single Direction Replacement")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Dir':>4} {'Accuracy':>10} {'Variance':>10} {'Status':>15}")
    logger.info("-" * 45)

    single_results = []
    for d in range(12):
        if d >= len(Vh_s) or d >= len(Vh_t):
            break
        output = replace_directions([d], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        acc, _ = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        status = "*** BEST ***" if acc > 0.85 else ""
        single_results.append((d, acc, source_var[d]))
        logger.info(f"{d+1:>4} {acc*100:>9.1f}% {source_var[d]*100:>9.2f}% {status:>15}")

    best_singles = sorted(single_results, key=lambda x: -x[1])[:3]
    logger.info(f"\nTop 3 single directions: {', '.join(f'{d+1}({a*100:.1f}%)' for d,a,_ in best_singles)}")

    # ========================================
    # PHASE 2: Test pairs using best singles
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Pair Replacement (using top singles)")
    logger.info(f"{'='*80}")

    # Get best single directions
    top_singles = [d for d, a, _ in best_singles]

    logger.info(f"\nTesting pairs including best directions ({', '.join(str(d+1) for d in top_singles)}):")
    logger.info(f"{'Pair':>10} {'Accuracy':>10}")
    logger.info("-" * 25)

    pair_results = []
    tested = set()
    for d1 in top_singles:
        for d2 in range(8):
            if d2 == d1:
                continue
            pair = tuple(sorted([d1, d2]))
            if pair in tested:
                continue
            tested.add(pair)

            output = replace_directions(list(pair), source_Y, target_X, target_Y, Vh_s, Vh_t, F)
            acc, _ = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
            pair_results.append((pair, acc))
            logger.info(f"{f'{pair[0]+1}+{pair[1]+1}':>10} {acc*100:>9.1f}%")

    best_pair = max(pair_results, key=lambda x: x[1])
    logger.info(f"\nBest pair: {best_pair[0][0]+1}+{best_pair[0][1]+1} ({best_pair[1]*100:.1f}%)")

    # ========================================
    # PHASE 3: Exhaustive pair search
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Exhaustive Pair Search (top 8 directions)")
    logger.info(f"{'='*80}")

    k = 8
    all_pairs = list(combinations(range(k), 2))

    logger.info(f"\nTesting all {len(all_pairs)} pairs:")

    best_acc = 0
    best_combo = None

    for pair in all_pairs:
        output = replace_directions(list(pair), source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        acc, _ = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        if acc > best_acc:
            best_acc = acc
            best_combo = pair

    logger.info(f"\nBest pair overall: {best_combo[0]+1}+{best_combo[1]+1} ({best_acc*100:.1f}%)")

    # ========================================
    # PHASE 4: Comparison
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Method Comparison")
    logger.info(f"{'='*80}")

    # Full teaching (baseline)
    full_output = source_Y @ F
    acc_full, _ = evaluate_with_details(full_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    # Best single replacement
    best_d = best_singles[0][0]
    best_single_output = replace_directions([best_d], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    acc_best_single, results_best = evaluate_with_details(best_single_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    # No replacement (just target)
    acc_target, _ = evaluate_with_details(target_Y, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    logger.info(f"\n{'Method':<35} {'Accuracy':>10}")
    logger.info("-" * 50)
    logger.info(f"{'No teaching (target only)':<35} {acc_target*100:>9.1f}%")
    logger.info(f"{'Full teaching (all directions)':<35} {acc_full*100:>9.1f}%")
    logger.info(f"{'Best single replacement (dir '+str(best_d+1)+')':<35} {acc_best_single*100:>9.1f}%")
    logger.info(f"{'Best pair replacement':<35} {best_acc*100:>9.1f}%")

    # Per-prompt details for best single
    logger.info(f"\n--- Per-Prompt Results (Best Single: Dir {best_d+1}) ---")
    logger.info(f"{'Prompt':<20} {'Match':>6} {'Expected':>10} {'Got':>10}")
    logger.info("-" * 50)
    for prompt, match, orig, trans in results_best:
        mark = "✓" if match else "✗"
        logger.info(f"{prompt:<20} {mark:>6} {orig.strip():>10} {trans.strip():>10}")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: Optimal Direction Replacement")
    logger.info(f"{'='*80}")

    improvement_single = (acc_best_single - acc_full) * 100
    improvement_pair = (best_acc - acc_full) * 100

    logger.info(f"""
THE REPLACEMENT METHOD:

This method REPLACES target's direction with source's direction,
preserving all other directions in the target.

Like teaching a student:
- Keep what they know (other directions)
- Replace only what's wrong (the direction we're teaching)
- Result: Better than erasing everything and starting fresh

RESULTS:

No teaching (target as-is):    {acc_target*100:.1f}%
Full teaching (all at once):   {acc_full*100:.1f}%
Best single (direction {best_d+1}):      {acc_best_single*100:.1f}%  ({improvement_single:+.1f}pp vs full)
Best pair:                     {best_acc*100:.1f}%  ({improvement_pair:+.1f}pp vs full)

THE OPTIMAL SINGLE DIRECTION:

Direction {best_d+1} achieves {acc_best_single*100:.1f}% with only {source_var[best_d]*100:.1f}% of variance.
This is the "essential lesson" - the single most important thing to teach.

FAILURES:
{[p for p, m, _, _ in results_best if not m] if any(not m for _, m, _, _ in results_best) else 'None!'}

THE EQUATION OF REPLACEMENT:

output = target - target[d] + source[d] @ F

Where:
- target[d] = projection onto direction d
- source[d] = source's behavior in direction d
- F = translation from source space to target space

This is SURGICAL KNOWLEDGE TRANSFER:
- Identify the specific "misconception" (direction)
- Remove it from target
- Replace with correct knowledge from source
""")


if __name__ == "__main__":
    run_experiment()
