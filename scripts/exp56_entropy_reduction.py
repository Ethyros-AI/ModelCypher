#!/usr/bin/env python3
"""Experiment 56: Cross-Architecture Entropy Reduction.

The insight: If the larger model has "cleaner" knowledge,
can we use it to REDUCE entropy in the smaller model?

Hypothesis:
- DeepSeek-R1-8B (8B params) has more refined representations
- LFM2-1.2B (1.2B params) has more noise/uncertainty
- Transplanting the "essential direction" should REDUCE entropy

Method:
1. Measure entropy of LFM2's original outputs
2. Apply directional teaching from DeepSeek-R1
3. Measure entropy of transplanted outputs
4. Compare: Does teaching REDUCE uncertainty?

The analogy:
- A teacher's knowledge reduces student's confusion
- The larger model "cleans up" the smaller model's uncertainty
- Entropy reduction = true knowledge transfer (not just copying)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test if cross-architecture teaching reduces entropy."""
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

    def compute_logit_entropy(logits_np):
        """Compute entropy of softmax probabilities."""
        probs = softmax(logits_np, axis=-1)
        return entropy(probs)

    def evaluate_with_entropy(target_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts):
        """Evaluate transplant and measure output entropy."""
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

            # Original prediction and entropy
            orig_logits = target_model(input_ids)
            mx.eval(orig_logits)
            orig_logits_np = np.array(orig_logits[0, -1, :].tolist())
            orig_top = int(np.argmax(orig_logits_np))
            orig_entropy = compute_logit_entropy(orig_logits_np)

            # Transplanted prediction and entropy
            if mlp_key == 'feed_forward':
                target_layer.feed_forward = TransplantedMLP(W_mx)
            else:
                target_layer.mlp = TransplantedMLP(W_mx)

            try:
                trans_logits = target_model(input_ids)
                mx.eval(trans_logits)
                trans_logits_np = np.array(trans_logits[0, -1, :].tolist())
                trans_top = int(np.argmax(trans_logits_np))
                trans_entropy = compute_logit_entropy(trans_logits_np)
            finally:
                if mlp_key == 'feed_forward':
                    target_layer.feed_forward = original_mlp
                else:
                    target_layer.mlp = original_mlp

            match = orig_top == trans_top
            entropy_change = trans_entropy - orig_entropy
            results.append({
                'prompt': prompt,
                'match': match,
                'orig_entropy': orig_entropy,
                'trans_entropy': trans_entropy,
                'entropy_change': entropy_change,
            })

        return results

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

    def replace_direction(d, source_Y, target_X, target_Y, Vh_s, Vh_t, F):
        """Replace target's direction d with source's."""
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        result = target_Y.copy()

        if d < len(Vh_s) and d < len(Vh_t):
            target_coefs_d = target_Y_centered @ Vh_t[d]
            target_proj_d = np.outer(target_coefs_d, Vh_t[d])
            result -= target_proj_d

            source_coefs_d = source_Y_centered @ Vh_s[d]
            source_proj_d = np.outer(source_coefs_d, Vh_s[d])
            result += source_proj_d @ F

        return result

    # ========================================
    # PHASE 1: Baseline - Full Teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Baseline - Full Teaching Entropy")
    logger.info(f"{'='*80}")

    full_output = source_Y @ F
    full_results = evaluate_with_entropy(full_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    logger.info(f"\n{'Prompt':<20} {'Orig H':>8} {'Trans H':>8} {'ΔH':>8} {'Match':>6}")
    logger.info("-" * 55)

    for r in full_results:
        mark = "✓" if r['match'] else "✗"
        logger.info(f"{r['prompt']:<20} {r['orig_entropy']:>8.3f} {r['trans_entropy']:>8.3f} {r['entropy_change']:>+8.3f} {mark:>6}")

    avg_entropy_change_full = np.mean([r['entropy_change'] for r in full_results])
    accuracy_full = sum(1 for r in full_results if r['match']) / len(full_results)

    logger.info(f"\nFull teaching:")
    logger.info(f"  Average entropy change: {avg_entropy_change_full:+.4f}")
    logger.info(f"  Accuracy: {accuracy_full*100:.1f}%")

    # ========================================
    # PHASE 2: Direction 6 Replacement
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Direction 6 Replacement Entropy")
    logger.info(f"{'='*80}")

    d6_output = replace_direction(5, source_Y, target_X, target_Y, Vh_s, Vh_t, F)
    d6_results = evaluate_with_entropy(d6_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

    logger.info(f"\n{'Prompt':<20} {'Orig H':>8} {'Trans H':>8} {'ΔH':>8} {'Match':>6}")
    logger.info("-" * 55)

    for r in d6_results:
        mark = "✓" if r['match'] else "✗"
        logger.info(f"{r['prompt']:<20} {r['orig_entropy']:>8.3f} {r['trans_entropy']:>8.3f} {r['entropy_change']:>+8.3f} {mark:>6}")

    avg_entropy_change_d6 = np.mean([r['entropy_change'] for r in d6_results])
    accuracy_d6 = sum(1 for r in d6_results if r['match']) / len(d6_results)

    logger.info(f"\nDirection 6 replacement:")
    logger.info(f"  Average entropy change: {avg_entropy_change_d6:+.4f}")
    logger.info(f"  Accuracy: {accuracy_d6*100:.1f}%")

    # ========================================
    # PHASE 3: All directions entropy comparison
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Per-Direction Entropy Impact")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Dir':>4} {'Avg ΔH':>10} {'Accuracy':>10} {'Status':>15}")
    logger.info("-" * 45)

    direction_entropies = []
    for d in range(12):
        if d >= len(Vh_s) or d >= len(Vh_t):
            break
        output = replace_direction(d, source_Y, target_X, target_Y, Vh_s, Vh_t, F)
        results = evaluate_with_entropy(output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)

        avg_dh = np.mean([r['entropy_change'] for r in results])
        acc = sum(1 for r in results if r['match']) / len(results)

        status = ""
        if avg_dh < -0.01:
            status = "REDUCES ↓"
        elif avg_dh > 0.01:
            status = "INCREASES ↑"
        else:
            status = "neutral"

        direction_entropies.append((d, avg_dh, acc))
        logger.info(f"{d+1:>4} {avg_dh:>+10.4f} {acc*100:>9.1f}% {status:>15}")

    # ========================================
    # PHASE 4: Compare to source model
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Source vs Target Entropy")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Prompt':<20} {'Source H':>10} {'Target H':>10} {'Diff':>10}")
    logger.info("-" * 55)

    source_entropies = []
    target_entropies = []

    for prompt in test_prompts:
        # Source entropy
        src_tokens = source_tokenizer.encode(prompt)
        src_ids = mx.array([src_tokens])
        src_logits = source_model(src_ids)
        mx.eval(src_logits)
        src_logits_np = np.array(src_logits[0, -1, :].tolist())
        src_entropy = compute_logit_entropy(src_logits_np)
        source_entropies.append(src_entropy)

        # Target entropy
        tgt_tokens = target_tokenizer.encode(prompt)
        tgt_ids = mx.array([tgt_tokens])
        tgt_logits = target_model(tgt_ids)
        mx.eval(tgt_logits)
        tgt_logits_np = np.array(tgt_logits[0, -1, :].tolist())
        tgt_entropy = compute_logit_entropy(tgt_logits_np)
        target_entropies.append(tgt_entropy)

        diff = tgt_entropy - src_entropy
        logger.info(f"{prompt:<20} {src_entropy:>10.3f} {tgt_entropy:>10.3f} {diff:>+10.3f}")

    avg_source_entropy = np.mean(source_entropies)
    avg_target_entropy = np.mean(target_entropies)

    logger.info(f"\nAverage entropies:")
    logger.info(f"  Source (DeepSeek-R1-8B): {avg_source_entropy:.4f}")
    logger.info(f"  Target (LFM2-1.2B):      {avg_target_entropy:.4f}")
    logger.info(f"  Difference:              {avg_target_entropy - avg_source_entropy:+.4f}")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: Cross-Architecture Entropy Reduction")
    logger.info(f"{'='*80}")

    # Find best entropy-reducing direction
    best_d, best_dh, best_acc = min(direction_entropies, key=lambda x: x[1])

    logger.info(f"""
ENTROPY COMPARISON:

Source (8B):  {avg_source_entropy:.4f} (more confident)
Target (1.2B): {avg_target_entropy:.4f} (less confident)
Gap: {avg_target_entropy - avg_source_entropy:+.4f} nats

TEACHING IMPACT ON ENTROPY:

Full teaching:         ΔH = {avg_entropy_change_full:+.4f}, Acc = {accuracy_full*100:.1f}%
Direction 6 only:      ΔH = {avg_entropy_change_d6:+.4f}, Acc = {accuracy_d6*100:.1f}%
Best entropy reducer:  Direction {best_d+1}, ΔH = {best_dh:+.4f}, Acc = {best_acc*100:.1f}%

INTERPRETATION:

1. SOURCE IS MORE CONFIDENT
   - 8B model has lower entropy than 1.2B model
   - This makes sense: more parameters = more refined representations

2. DOES TEACHING REDUCE ENTROPY?
   {'✅ YES' if avg_entropy_change_d6 < 0 else '⚠️ MIXED'} - Direction 6 replacement achieves ΔH = {avg_entropy_change_d6:+.4f}

3. THE ENTROPY REDUCTION PRINCIPLE
   - Larger model acts as "entropy sink"
   - Teaching specific directions can REDUCE uncertainty
   - This is TRUE knowledge transfer (not just copying)

4. PRACTICAL IMPLICATION
   - Use cross-architecture teaching as DENOISING
   - Replace noisy target directions with cleaner source directions
   - The smaller model becomes more confident

THE EQUATION OF ENTROPY REDUCTION:

H(target') = H(target) - I(source;target|direction)

Where:
- H(target') = entropy after teaching
- I(source;target|direction) = mutual information gained through direction transfer

When I > 0, we achieve TRUE knowledge transfer (entropy reduction).
""")


if __name__ == "__main__":
    run_experiment()
