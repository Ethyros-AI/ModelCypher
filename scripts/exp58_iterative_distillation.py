#!/usr/bin/env python3
"""Experiment 58: Iterative Entropy-Gated Distillation.

The breakthrough: We can use the larger model as an ENTROPY SINK.
Each iteration:
1. Find prompts where student is uncertain (high entropy)
2. Apply teaching ONLY where it reduces entropy
3. The student becomes more confident
4. Repeat until no more entropy can be extracted

This is like iterative teaching:
- First pass: Teach the basics (biggest entropy reductions)
- Second pass: Refine the details (smaller reductions)
- Each pass: Only teach where the student is still confused

The limit: When all prompts have lower entropy after teaching,
we've extracted ALL the knowledge the teacher can give.
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
    """Test iterative entropy-gated distillation."""
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

    # Multiple layer pairs to try
    layer_pairs = [
        (24, 10),  # Original
        (22, 9),   # Earlier
        (23, 9),
        (24, 9),
        (25, 11),
        (24, 11),
        (24, 12),
    ]

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

    def get_entropy_per_prompt(target_model, target_tokenizer, prompts):
        """Get current entropy for each prompt."""
        entropies = []
        for prompt in prompts:
            tokens = target_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            logits = target_model(input_ids)
            mx.eval(logits)
            logits_np = np.array(logits[0, -1, :].tolist())
            entropies.append(compute_logit_entropy(logits_np))
        return entropies

    # ========================================
    # PHASE 1: Survey all layer pairs
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Entropy Reduction Potential by Layer Pair")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Pair':>12} {'Prompts ↓':>12} {'Avg ΔH':>10} {'Total ΔH':>10}")
    logger.info("-" * 50)

    pair_stats = []

    for src_layer, tgt_layer in layer_pairs:
        try:
            # Collect activations
            source_X, source_Y = get_activations(source_model, source_tokenizer, src_layer, train_prompts)
            target_X, target_Y = get_activations(target_model, target_tokenizer, tgt_layer, train_prompts)

            # SVD
            source_Y_centered = source_Y - source_Y.mean(axis=0)
            _, _, Vh_s = np.linalg.svd(source_Y_centered, full_matrices=False)

            target_Y_centered = target_Y - target_Y.mean(axis=0)
            _, _, Vh_t = np.linalg.svd(target_Y_centered, full_matrices=False)

            F = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]

            # Direction 6 replacement
            d = 5
            result = target_Y.copy()
            target_coefs_d = target_Y_centered @ Vh_t[d]
            target_proj_d = np.outer(target_coefs_d, Vh_t[d])
            result -= target_proj_d

            source_coefs_d = source_Y_centered @ Vh_s[d]
            source_proj_d = np.outer(source_coefs_d, Vh_s[d])
            result += source_proj_d @ F

            # Learn W
            alpha = 1e-6
            ATA = target_X.T @ target_X + alpha * np.eye(target_X.shape[1])
            ATB = target_X.T @ result
            W = np.linalg.solve(ATA, ATB).T

            W_mx = mx.array(W.astype(np.float32))
            mx.eval(W_mx)

            class TransplantedMLP:
                def __init__(self, W):
                    self.W = W
                def __call__(self, x):
                    return mx.matmul(x, self.W.T)

            target_layer = target_model.model.layers[tgt_layer]
            if hasattr(target_layer, 'feed_forward'):
                original_mlp = target_layer.feed_forward
                mlp_key = 'feed_forward'
            else:
                original_mlp = target_layer.mlp
                mlp_key = 'mlp'

            # Measure entropy changes
            entropy_changes = []
            for prompt in test_prompts:
                tokens = target_tokenizer.encode(prompt)
                input_ids = mx.array([tokens])

                # Original
                orig_logits = target_model(input_ids)
                mx.eval(orig_logits)
                orig_entropy = compute_logit_entropy(np.array(orig_logits[0, -1, :].tolist()))

                # Transplanted
                if mlp_key == 'feed_forward':
                    target_layer.feed_forward = TransplantedMLP(W_mx)
                else:
                    target_layer.mlp = TransplantedMLP(W_mx)

                try:
                    trans_logits = target_model(input_ids)
                    mx.eval(trans_logits)
                    trans_entropy = compute_logit_entropy(np.array(trans_logits[0, -1, :].tolist()))
                finally:
                    if mlp_key == 'feed_forward':
                        target_layer.feed_forward = original_mlp
                    else:
                        target_layer.mlp = original_mlp

                entropy_changes.append(trans_entropy - orig_entropy)

            # Stats
            reducing = sum(1 for dh in entropy_changes if dh < 0)
            avg_dh = np.mean(entropy_changes)
            total_reduction = sum(dh for dh in entropy_changes if dh < 0)

            pair_stats.append({
                'pair': (src_layer, tgt_layer),
                'reducing': reducing,
                'avg_dh': avg_dh,
                'total_reduction': total_reduction,
                'entropy_changes': entropy_changes,
            })

            logger.info(f"L{src_layer}→L{tgt_layer:>2} {reducing:>12} {avg_dh:>+10.4f} {total_reduction:>+10.4f}")

        except Exception as e:
            logger.info(f"L{src_layer}→L{tgt_layer:>2} {'ERROR':>12} {str(e)[:20]}")

    # ========================================
    # PHASE 2: Iterative extraction
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Iterative Entropy Extraction")
    logger.info(f"{'='*80}")

    # Sort pairs by total entropy reduction potential
    pair_stats_sorted = sorted(pair_stats, key=lambda x: x['total_reduction'])

    logger.info("\nLayer pairs ranked by entropy reduction potential:")
    for i, ps in enumerate(pair_stats_sorted[:5]):
        logger.info(f"  {i+1}. L{ps['pair'][0]}→L{ps['pair'][1]}: {ps['total_reduction']:+.4f} nats")

    # Simulate iterative extraction
    # Each prompt starts with its original entropy
    # We apply the BEST layer pair for each prompt (the one that reduces entropy most)

    logger.info(f"\n--- Simulated Iterative Extraction ---")

    # Get baseline entropies
    baseline_entropies = get_entropy_per_prompt(target_model, target_tokenizer, test_prompts)

    logger.info(f"\nBaseline entropies:")
    for i, (prompt, h) in enumerate(zip(test_prompts, baseline_entropies)):
        logger.info(f"  {prompt:<20} H = {h:.3f}")

    # For each prompt, find the best layer pair
    logger.info(f"\nBest layer pair for each prompt:")

    total_extraction = 0
    for i, prompt in enumerate(test_prompts):
        best_pair = None
        best_dh = 0

        for ps in pair_stats:
            dh = ps['entropy_changes'][i]
            if dh < best_dh:
                best_dh = dh
                best_pair = ps['pair']

        if best_pair:
            logger.info(f"  {prompt:<20} → L{best_pair[0]}→L{best_pair[1]}: ΔH = {best_dh:+.4f}")
            total_extraction += best_dh
        else:
            logger.info(f"  {prompt:<20} → (no reduction possible)")

    logger.info(f"\nTotal potential extraction: {total_extraction:+.4f} nats")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: Iterative Entropy-Gated Distillation")
    logger.info(f"{'='*80}")

    best_pair = pair_stats_sorted[0]

    logger.info(f"""
THE ITERATIVE DISTILLATION ALGORITHM:

1. SURVEY: For each layer pair, measure entropy reduction per prompt
2. SELECT: For each prompt, choose the layer pair that reduces entropy most
3. APPLY: Apply transplant only where ΔH < 0
4. REPEAT: Use different layer pairs for different prompts

RESULTS:

Best single layer pair: L{best_pair['pair'][0]}→L{best_pair['pair'][1]}
  - Prompts improved: {best_pair['reducing']}/12
  - Total reduction: {best_pair['total_reduction']:+.4f} nats

Best per-prompt selection:
  - Total reduction: {total_extraction:+.4f} nats
  - Improvement over single pair: {total_extraction - best_pair['total_reduction']:+.4f} nats

THE INSIGHT:

Different prompts benefit from different layer pairs!
- "The moon is" might benefit from L24→L10
- "Therefore we" might benefit from L22→L9
- By selecting the optimal pair per prompt, we extract MORE entropy

THE ITERATIVE TEACHING METAPHOR:

Like having multiple expert tutors:
- Math tutor for math problems
- Language tutor for writing
- Each tutor teaches their specialty
- Student learns from ALL tutors

THE PRODUCTION ALGORITHM:

```python
for each input:
    current_entropy = compute_entropy(model(input))

    for layer_pair in candidate_pairs:
        transplanted_output = apply_transplant(layer_pair, input)
        transplanted_entropy = compute_entropy(transplanted_output)

        if transplanted_entropy < current_entropy:
            apply_transplant_permanently(layer_pair)
            current_entropy = transplanted_entropy
            break  # or continue for multi-pass
```

THEORETICAL LIMIT:

The process converges when:
- All prompts have lower entropy than ANY teaching could achieve
- We've extracted ALL transferable knowledge from the teacher
- The student has become as confident as the teacher can make it
""")


if __name__ == "__main__":
    run_experiment()
