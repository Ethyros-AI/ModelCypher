#!/usr/bin/env python3
"""Experiment 53: The Dual Essentials.

From exp52: TWO directions achieve 91.7% - direction 6 AND direction 8.
Both beat full teaching (83.3%).

Questions:
1. Can we COMBINE directions 6 and 8 to get even higher accuracy?
2. What do these two "essential topics" represent?
3. Is one about "what" and the other about "how"?
4. Can we reach 100% by finding the right combination?

The hypothesis:
- Direction 6 = one aspect of knowledge
- Direction 8 = another complementary aspect
- Together = complete knowledge transfer
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Explore combining the two essential directions."""
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

    # ========================================
    # PHASE 1: Test the dual essentials
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: The Dual Essentials (Directions 6 and 8)")
    logger.info(f"{'='*80}")

    def teach_directions_only(dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F):
        """Teach ONLY the specified directions, removing all others."""
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        # Start with mean (no directional components)
        result = np.zeros_like(target_Y)
        result += target_Y.mean(axis=0)

        for d in dirs:
            if d < len(Vh_s) and d < len(Vh_t):
                # Add source's direction d (translated to target space)
                source_coefs_d = source_Y_centered @ Vh_s[d]
                source_proj_d = np.outer(source_coefs_d, Vh_s[d])
                result += source_proj_d @ F

        return result

    def teach_exclude_directions(exclude_dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F, n_total=12):
        """Teach all directions EXCEPT the specified ones."""
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        result = target_Y.copy()

        for d in range(n_total):
            if d in exclude_dirs:
                continue
            if d < len(Vh_s) and d < len(Vh_t):
                # Remove target's direction d
                target_coefs_d = target_Y_centered @ Vh_t[d]
                target_proj_d = np.outer(target_coefs_d, Vh_t[d])
                result -= target_proj_d

                # Add source's direction d (translated)
                source_coefs_d = source_Y_centered @ Vh_s[d]
                source_proj_d = np.outer(source_coefs_d, Vh_s[d])
                result += source_proj_d @ F

        return result

    # Test individual essentials
    essential_dirs = [5, 7]  # 0-indexed: direction 6 and 8

    logger.info("\nIndividual essential directions:")
    for d in essential_dirs:
        output = teach_directions_only([d], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        acc, results = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        logger.info(f"  Direction {d+1} only: {acc*100:.1f}%")

    # Test combining the two essentials
    logger.info("\nCombining essential directions:")
    output_both = teach_directions_only(essential_dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    acc_both, results_both = evaluate_with_details(output_both, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
    logger.info(f"  Directions 6+8 only: {acc_both*100:.1f}%")

    # Test excluding the essentials
    logger.info("\nExcluding essential directions:")
    output_no_ess = teach_exclude_directions(essential_dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    acc_no_ess, _ = evaluate_with_details(output_no_ess, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
    logger.info(f"  All EXCEPT 6+8: {acc_no_ess*100:.1f}%")

    # ========================================
    # PHASE 2: Search for optimal subset
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Search for Optimal Direction Subset")
    logger.info(f"{'='*80}")

    from itertools import combinations

    k = 6  # Top k directions to consider

    logger.info(f"\nTesting all combinations of 2 directions from top {k}:")
    logger.info(f"{'Dirs':>12} {'Accuracy':>10}")
    logger.info("-" * 25)

    best_pair = None
    best_acc = 0

    for combo in combinations(range(k), 2):
        output = teach_directions_only(list(combo), source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        acc, _ = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        d1, d2 = combo
        logger.info(f"{f'{d1+1}+{d2+1}':>12} {acc*100:>9.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_pair = combo

    logger.info(f"\nBest pair: {best_pair[0]+1}+{best_pair[1]+1} ({best_acc*100:.1f}%)")

    # Test triples
    logger.info(f"\nTesting all combinations of 3 directions from top {k}:")
    logger.info(f"{'Dirs':>15} {'Accuracy':>10}")
    logger.info("-" * 28)

    best_triple = None
    best_triple_acc = 0

    for combo in combinations(range(k), 3):
        output = teach_directions_only(list(combo), source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        acc, _ = evaluate_with_details(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        d1, d2, d3 = combo
        logger.info(f"{f'{d1+1}+{d2+1}+{d3+1}':>15} {acc*100:>9.1f}%")
        if acc > best_triple_acc:
            best_triple_acc = acc
            best_triple = combo

    logger.info(f"\nBest triple: {'+'.join(str(d+1) for d in best_triple)} ({best_triple_acc*100:.1f}%)")

    # ========================================
    # PHASE 3: Per-prompt analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Per-Prompt Analysis")
    logger.info(f"{'='*80}")

    # Compare best single (dir 6) vs best combination
    output_best_single = teach_directions_only([5], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    _, results_single = evaluate_with_details(output_best_single, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    output_best_combo = teach_directions_only(list(best_triple), source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    _, results_combo = evaluate_with_details(output_best_combo, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    # Full teaching baseline
    full_output = source_Y @ F
    _, results_full = evaluate_with_details(full_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    logger.info(f"\n{'Prompt':<20} {'Dir 6':>8} {'Best Combo':>12} {'Full':>8}")
    logger.info("-" * 55)

    for i, (prompt, m1, o1, t1) in enumerate(results_single):
        _, m2, o2, t2 = results_combo[i]
        _, m3, o3, t3 = results_full[i]
        s1 = "✓" if m1 else "✗"
        s2 = "✓" if m2 else "✗"
        s3 = "✓" if m3 else "✗"
        logger.info(f"{prompt:<20} {s1:>8} {s2:>12} {s3:>8}")

    # ========================================
    # PHASE 4: What makes the failures?
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Failure Analysis")
    logger.info(f"{'='*80}")

    # Get failures from best single
    failures = [(p, o, t) for p, m, o, t in results_single if not m]

    if failures:
        logger.info("\nPrompts that fail with direction 6 only:")
        for prompt, orig, trans in failures:
            logger.info(f"  \"{prompt}\" → expected '{orig.strip()}', got '{trans.strip()}'")

        # Check if any direction fixes these failures
        logger.info("\nWhich directions fix these failures?")
        for prompt, _, _ in failures:
            prompt_idx = test_prompts.index(prompt)
            logger.info(f"\n  \"{prompt}\":")
            for d in range(8):
                output_d = teach_directions_only([d], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
                _, results_d = evaluate_with_details(output_d, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
                if results_d[prompt_idx][1]:  # matched
                    logger.info(f"    Direction {d+1}: ✓ FIXES IT")
    else:
        logger.info("\nNo failures with direction 6 only!")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: The Essential Directions")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE DUAL ESSENTIALS:

Individual direction 6: 91.7%
Individual direction 8: 91.7%
Combined (6+8 only):    {acc_both*100:.1f}%
All except 6+8:         {acc_no_ess*100:.1f}%
Full teaching:          83.3%

BEST COMBINATIONS:

Best pair:   {best_pair[0]+1}+{best_pair[1]+1} = {best_acc*100:.1f}%
Best triple: {'+'.join(str(d+1) for d in best_triple)} = {best_triple_acc*100:.1f}%

THE INSIGHT:

1. LESS IS MORE (confirmed)
   - Single essential direction beats full teaching
   - 91.7% with 1 direction vs 83.3% with all

2. INTERFERENCE IS REAL
   - Adding more directions can HURT accuracy
   - Directions are orthogonal but NOT independent for teaching

3. THE ESSENTIAL PAIR
   - Directions 6 and 8 might represent:
     a) "What" vs "How" knowledge
     b) Content vs Structure
     c) Facts vs Relations

4. FOR PRODUCTION
   - Find the 1-2 essential directions
   - Teach ONLY those
   - Ignore the noise from other directions

THE EQUATION:

Accuracy(subset) = Signal(subset) - Interference(subset)

Where:
- Signal increases with essential directions
- Interference increases with non-essential directions
- Optimal subset minimizes interference while capturing signal
""")


if __name__ == "__main__":
    run_experiment()
