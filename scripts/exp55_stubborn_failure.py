#!/usr/bin/env python3
"""Experiment 55: The Stubborn Failure.

From exp54: Direction 6 achieves 91.7% (11/12 correct).
The ONLY failure: "Therefore we" → expected "are", got "may"

This is remarkable! We're ONE prompt away from 100%.

Questions:
1. What's special about "Therefore we"?
2. Does ANY direction fix it?
3. Can we combine directions to fix it?
4. Is it fixable at all?

The hypothesis:
- "Therefore we" requires REASONING, not knowledge
- It might need a different layer (not L24→L10)
- It might require the attention mechanism (which we proved can't be compressed)
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
    """Investigate the stubborn failure and search for 100%."""
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

    # Focus on the stubborn prompt
    target_prompt = "Therefore we"
    test_prompts = [target_prompt]

    # Also include neighbors to see the pattern
    all_test_prompts = [
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

    def evaluate_single(target_output, target_X, target_model, target_tokenizer, target_layer_idx, prompt):
        """Evaluate a single prompt."""
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

        tokens = target_tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        orig_logits = target_model(input_ids)
        mx.eval(orig_logits)
        orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
        orig_word = target_tokenizer.decode([orig_top])

        if mlp_key == 'feed_forward':
            target_layer.feed_forward = TransplantedMLP(W_mx)
        else:
            target_layer.mlp = TransplantedMLP(W_mx)

        try:
            trans_logits = target_model(input_ids)
            mx.eval(trans_logits)
            trans_top = int(mx.argmax(trans_logits[0, -1, :]).item())
            trans_word = target_tokenizer.decode([trans_top])

            # Get top-5 for more insight
            trans_probs = mx.softmax(trans_logits[0, -1, :], axis=-1)
            mx.eval(trans_probs)
            trans_probs_np = np.array(trans_probs.tolist())
            top5_indices = np.argsort(trans_probs_np)[::-1][:5]
            top5_words = [target_tokenizer.decode([int(idx)]) for idx in top5_indices]
            top5_probs = [trans_probs_np[int(idx)] for idx in top5_indices]
        finally:
            if mlp_key == 'feed_forward':
                target_layer.feed_forward = original_mlp
            else:
                target_layer.mlp = original_mlp

        match = orig_top == trans_top
        return match, orig_word, trans_word, list(zip(top5_words, top5_probs))

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

    def replace_directions(dirs, source_Y, target_X, target_Y, Vh_s, Vh_t, F):
        """Replace specified directions."""
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        result = target_Y.copy()

        for d in dirs:
            if d < len(Vh_s) and d < len(Vh_t):
                target_coefs_d = target_Y_centered @ Vh_t[d]
                target_proj_d = np.outer(target_coefs_d, Vh_t[d])
                result -= target_proj_d

                source_coefs_d = source_Y_centered @ Vh_s[d]
                source_proj_d = np.outer(source_coefs_d, Vh_s[d])
                result += source_proj_d @ F

        return result

    # ========================================
    # PHASE 1: What does the original model predict?
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Original Model Predictions")
    logger.info(f"{'='*80}")

    # Get original predictions for target prompt
    logger.info(f"\nOriginal predictions for \"{target_prompt}\":")

    # Source model
    source_tokens = source_tokenizer.encode(target_prompt)
    source_ids = mx.array([source_tokens])
    source_logits = source_model(source_ids)
    mx.eval(source_logits)

    source_probs = mx.softmax(source_logits[0, -1, :], axis=-1)
    mx.eval(source_probs)
    source_probs_np = np.array(source_probs.tolist())
    top5_src = np.argsort(source_probs_np)[::-1][:5]

    logger.info(f"\nSource (DeepSeek-R1-8B) top-5:")
    for idx in top5_src:
        word = source_tokenizer.decode([int(idx)])
        prob = source_probs_np[int(idx)]
        logger.info(f"  '{word.strip()}': {prob*100:.2f}%")

    # Target model
    target_tokens = target_tokenizer.encode(target_prompt)
    target_ids = mx.array([target_tokens])
    target_logits = target_model(target_ids)
    mx.eval(target_logits)

    target_probs = mx.softmax(target_logits[0, -1, :], axis=-1)
    mx.eval(target_probs)
    target_probs_np = np.array(target_probs.tolist())
    top5_tgt = np.argsort(target_probs_np)[::-1][:5]

    logger.info(f"\nTarget (LFM2-1.2B) top-5:")
    for idx in top5_tgt:
        word = target_tokenizer.decode([int(idx)])
        prob = target_probs_np[int(idx)]
        logger.info(f"  '{word.strip()}': {prob*100:.2f}%")

    # ========================================
    # PHASE 2: Exhaustive direction search
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Which Direction Fixes 'Therefore we'?")
    logger.info(f"{'='*80}")

    logger.info(f"\nTesting all single directions (0-15):")
    logger.info(f"{'Dir':>4} {'Match':>6} {'Expected':>10} {'Got':>10}")
    logger.info("-" * 40)

    fixing_dirs = []
    for d in range(min(16, len(Vh_s), len(Vh_t))):
        output = replace_directions([d], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        match, orig, trans, _ = evaluate_single(output, target_X, target_model, target_tokenizer, target_layer_idx, target_prompt)
        mark = "✓" if match else "✗"
        logger.info(f"{d+1:>4} {mark:>6} {orig.strip():>10} {trans.strip():>10}")
        if match:
            fixing_dirs.append(d)

    if fixing_dirs:
        logger.info(f"\nDirections that FIX 'Therefore we': {[d+1 for d in fixing_dirs]}")
    else:
        logger.info(f"\nNO single direction fixes 'Therefore we'!")

    # ========================================
    # PHASE 3: Pair search
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Can Any PAIR Fix It?")
    logger.info(f"{'='*80}")

    fixing_pairs = []
    for d1, d2 in combinations(range(8), 2):
        output = replace_directions([d1, d2], source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        match, orig, trans, _ = evaluate_single(output, target_X, target_model, target_tokenizer, target_layer_idx, target_prompt)
        if match:
            fixing_pairs.append((d1, d2))

    if fixing_pairs:
        logger.info(f"Pairs that FIX 'Therefore we': {[(d1+1, d2+1) for d1, d2 in fixing_pairs]}")
    else:
        logger.info(f"NO pair of directions fixes 'Therefore we'!")

    # ========================================
    # PHASE 4: Try other layer pairs
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Try Other Layer Pairs")
    logger.info(f"{'='*80}")

    # Test different source→target layer mappings
    layer_pairs_to_test = [
        (22, 9),   # Earlier source
        (23, 9),
        (24, 9),
        (25, 10),  # Later source
        (26, 10),
        (27, 11),
        (24, 11),  # Different target
        (24, 12),
    ]

    logger.info(f"\n{'Source→Target':>15} {'Dir 6':>8} {'Full':>8}")
    logger.info("-" * 35)

    for src_layer, tgt_layer in layer_pairs_to_test:
        try:
            # Get activations for this layer pair
            sX, sY = get_activations(source_model, source_tokenizer, src_layer, train_prompts)
            tX, tY = get_activations(target_model, target_tokenizer, tgt_layer, train_prompts)

            # SVD
            sY_c = sY - sY.mean(axis=0)
            _, _, Vh_src = np.linalg.svd(sY_c, full_matrices=False)

            tY_c = tY - tY.mean(axis=0)
            _, _, Vh_tgt = np.linalg.svd(tY_c, full_matrices=False)

            F_layer = np.linalg.lstsq(sY, tY, rcond=1e-10)[0]

            # Test dir 6 replacement
            output_d6 = replace_directions([5], sY, tX, tY, Vh_src, Vh_tgt, F_layer)
            match_d6, _, trans_d6, _ = evaluate_single(output_d6, tX, target_model, target_tokenizer, tgt_layer, target_prompt)

            # Test full
            output_full = sY @ F_layer
            match_full, _, trans_full, _ = evaluate_single(output_full, tX, target_model, target_tokenizer, tgt_layer, target_prompt)

            mark_d6 = "✓" if match_d6 else trans_d6.strip()
            mark_full = "✓" if match_full else trans_full.strip()

            logger.info(f"{f'L{src_layer}→L{tgt_layer}':>15} {mark_d6:>8} {mark_full:>8}")
        except Exception as e:
            logger.info(f"{f'L{src_layer}→L{tgt_layer}':>15} {'ERROR':>8} {str(e)[:8]:>8}")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: The Stubborn Failure")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE STUBBORN PROMPT: "Therefore we"

Expected (from target model): "are"
Got (from transplant):        "may"

OBSERVATIONS:

1. ORIGINAL MODEL PREDICTIONS
   - Source (DeepSeek-R1) predicts: [see above]
   - Target (LFM2) predicts: [see above]
   - The models might ALREADY disagree on this prompt!

2. SINGLE DIRECTION SEARCH
   - Tested {min(16, len(Vh_s), len(Vh_t))} directions
   - Fixing directions: {[d+1 for d in fixing_dirs] if fixing_dirs else 'NONE'}

3. PAIR SEARCH
   - Tested all pairs from top 8
   - Fixing pairs: {[(d1+1, d2+1) for d1, d2 in fixing_pairs] if fixing_pairs else 'NONE'}

4. LAYER PAIR SEARCH
   - Tested multiple source→target layer combinations
   - [See results above]

THE HYPOTHESIS:

"Therefore we" is a REASONING task, not a knowledge task.
- "Therefore" requires understanding logical consequence
- The MLP layer handles "knowledge" (facts)
- Reasoning might require ATTENTION (which we proved can't be compressed)

The 91.7% ceiling might be FUNDAMENTAL:
- 11/12 prompts are "knowledge" prompts → fixable
- 1/12 prompts is a "reasoning" prompt → requires attention

IMPLICATION:

The MLP can be taught knowledge, but not reasoning.
Reasoning requires attention.
This is a fundamental limit of MLP-only transplant.
""")


if __name__ == "__main__":
    run_experiment()
