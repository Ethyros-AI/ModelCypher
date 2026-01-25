#!/usr/bin/env python3
"""Experiment 70: The Geometry of Correctness.

The profound question: If knowledge has invariant geometric structure,
what metric distinguishes CORRECT answers from WRONG answers?

Hypothesis: When the model outputs the correct answer, the hidden state
has a different geometric signature than when it outputs wrong.

We test:
1. Entropy of the hidden state
2. Norm of the hidden state
3. Distance to the "answer manifold"
4. Alignment with principal directions
5. Local curvature
6. Confidence (logit margin)

If we find this metric, we can:
- Use it as a self-consistency check
- Guide teaching toward "correct" regions
- Verify knowledge transfer worked
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


def run_experiment():
    """Find the geometry of correctness."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    logger.info("Loading LFM2-1.2B...")
    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    model, tokenizer = load(model_path)

    # Test cases with KNOWN correct answers
    # Mix of things the model gets right and wrong
    test_cases = [
        # Format: (prompt, correct_answer, is_model_correct)
        ("The capital of France is", "Paris", True),   # Model gets this right
        ("The sky is usually", "blue", True),           # Model gets this right
        ("Gravity causes objects to", "fall", True),    # Model gets this right
        ("2 + 2 equals", "4", False),                   # Model gets this wrong
        ("The square root of 16 is", "4", False),       # Model gets this wrong
        ("Birds can", "fly", False),                    # Model gets this wrong
        ("Fish live in", "water", False),               # Model gets this wrong
        ("The opposite of hot is", "cold", False),      # Model gets this wrong
    ]

    def get_hidden_state_and_prediction(model, tokenizer, prompt):
        """Get the final hidden state before output projection, and the prediction."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Hook into the last layer's MLP to capture hidden state
        last_layer_idx = len(model.model.layers) - 1
        layer = model.model.layers[last_layer_idx]

        hidden_state_captured = None

        if hasattr(layer, 'feed_forward'):
            original_mlp = layer.feed_forward
            key = 'feed_forward'
        else:
            original_mlp = layer.mlp
            key = 'mlp'

        class HiddenHook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                nonlocal hidden_state_captured
                result = self.mlp(x)
                hidden_state_captured = result
                return result

        if key == 'feed_forward':
            layer.feed_forward = HiddenHook(original_mlp)
        else:
            layer.mlp = HiddenHook(original_mlp)

        try:
            logits = model(input_ids)
            mx.eval(logits)
            mx.eval(hidden_state_captured)
            h = hidden_state_captured[0, -1, :]
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original_mlp
            else:
                layer.mlp = original_mlp

        logits_np = np.array(logits[0, -1, :].tolist())
        h_np = np.array(h.tolist())

        # Get prediction
        top_idx = int(np.argmax(logits_np))
        top_word = tokenizer.decode([top_idx]).strip()

        # Get confidence metrics
        probs = softmax(logits_np)
        top_prob = probs[top_idx]

        # Logit margin (top - second)
        sorted_logits = np.sort(logits_np)[::-1]
        margin = sorted_logits[0] - sorted_logits[1]

        # Entropy of output distribution
        output_entropy = -np.sum(probs * np.log(probs + 1e-10))

        return h_np, top_word, top_prob, margin, output_entropy

    def compute_geometry_metrics(h):
        """Compute geometric metrics for a hidden state."""
        metrics = {}

        # Basic stats
        metrics['norm'] = float(np.linalg.norm(h))
        metrics['mean'] = float(np.mean(h))
        metrics['std'] = float(np.std(h))
        metrics['max'] = float(np.max(h))
        metrics['min'] = float(np.min(h))

        # Sparsity (fraction of near-zero elements)
        threshold = np.abs(h).mean() * 0.1
        metrics['sparsity'] = float(np.mean(np.abs(h) < threshold))

        # Kurtosis (peakedness) - high kurtosis = concentrated activations
        z = (h - h.mean()) / (h.std() + 1e-10)
        metrics['kurtosis'] = float(np.mean(z ** 4) - 3)

        # L1/L2 ratio (sparsity measure)
        metrics['l1_l2_ratio'] = float(np.linalg.norm(h, 1) / (np.linalg.norm(h, 2) + 1e-10))

        return metrics

    # ========================================
    # PHASE 1: Collect Hidden States
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Collecting Hidden States")
    logger.info(f"{'='*80}")

    correct_states = []
    wrong_states = []
    all_data = []

    for prompt, correct_answer, model_is_correct in test_cases:
        h, predicted, prob, margin, out_entropy = get_hidden_state_and_prediction(model, tokenizer, prompt)

        # Check if model actually got it right
        actual_correct = correct_answer.lower() in predicted.lower()

        data = {
            'prompt': prompt,
            'correct_answer': correct_answer,
            'predicted': predicted,
            'expected_correct': model_is_correct,
            'actual_correct': actual_correct,
            'hidden_state': h,
            'confidence': prob,
            'margin': margin,
            'output_entropy': out_entropy,
            'geometry': compute_geometry_metrics(h),
        }
        all_data.append(data)

        if actual_correct:
            correct_states.append(h)
        else:
            wrong_states.append(h)

        mark = "✓" if actual_correct else "✗"
        logger.info(f"  {mark} '{prompt}' → '{predicted}' (p={prob:.2f}, margin={margin:.2f})")

    correct_states = np.array(correct_states)
    wrong_states = np.array(wrong_states)

    logger.info(f"\nCorrect: {len(correct_states)}, Wrong: {len(wrong_states)}")

    # ========================================
    # PHASE 2: Compare Geometry
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Comparing Geometry (Correct vs Wrong)")
    logger.info(f"{'='*80}")

    metrics_to_compare = ['norm', 'std', 'sparsity', 'kurtosis', 'l1_l2_ratio']

    logger.info(f"\n{'Metric':<20} {'Correct (mean)':>15} {'Wrong (mean)':>15} {'Diff':>12}")
    logger.info("-" * 65)

    for metric in metrics_to_compare:
        correct_vals = [d['geometry'][metric] for d in all_data if d['actual_correct']]
        wrong_vals = [d['geometry'][metric] for d in all_data if not d['actual_correct']]

        if correct_vals and wrong_vals:
            c_mean = np.mean(correct_vals)
            w_mean = np.mean(wrong_vals)
            diff = c_mean - w_mean

            logger.info(f"{metric:<20} {c_mean:>15.4f} {w_mean:>15.4f} {diff:>+12.4f}")

    # Also compare output metrics
    logger.info(f"\n{'Output Metric':<20} {'Correct (mean)':>15} {'Wrong (mean)':>15} {'Diff':>12}")
    logger.info("-" * 65)

    for metric in ['confidence', 'margin', 'output_entropy']:
        correct_vals = [d[metric] for d in all_data if d['actual_correct']]
        wrong_vals = [d[metric] for d in all_data if not d['actual_correct']]

        if correct_vals and wrong_vals:
            c_mean = np.mean(correct_vals)
            w_mean = np.mean(wrong_vals)
            diff = c_mean - w_mean

            logger.info(f"{metric:<20} {c_mean:>15.4f} {w_mean:>15.4f} {diff:>+12.4f}")

    # ========================================
    # PHASE 3: SVD Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: SVD Analysis of Correct vs Wrong Manifolds")
    logger.info(f"{'='*80}")

    if len(correct_states) >= 2:
        c_centered = correct_states - correct_states.mean(axis=0)
        _, S_c, _ = svd(c_centered, full_matrices=False)

        # Spectral entropy
        S_c_norm = S_c / (S_c.sum() + 1e-10)
        c_entropy = -np.sum(S_c_norm * np.log(S_c_norm + 1e-10))

        logger.info(f"\nCorrect manifold:")
        logger.info(f"  Top singular values: {S_c[:5].round(2)}")
        logger.info(f"  Spectral entropy: {c_entropy:.4f}")
        logger.info(f"  Effective rank: {np.sum(S_c > S_c[0] * 0.01)}")

    if len(wrong_states) >= 2:
        w_centered = wrong_states - wrong_states.mean(axis=0)
        _, S_w, _ = svd(w_centered, full_matrices=False)

        S_w_norm = S_w / (S_w.sum() + 1e-10)
        w_entropy = -np.sum(S_w_norm * np.log(S_w_norm + 1e-10))

        logger.info(f"\nWrong manifold:")
        logger.info(f"  Top singular values: {S_w[:5].round(2)}")
        logger.info(f"  Spectral entropy: {w_entropy:.4f}")
        logger.info(f"  Effective rank: {np.sum(S_w > S_w[0] * 0.01)}")

    # ========================================
    # PHASE 4: Distance to Correct Centroid
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Distance to 'Correct' Centroid")
    logger.info(f"{'='*80}")

    if len(correct_states) >= 1:
        correct_centroid = correct_states.mean(axis=0)

        logger.info(f"\n{'Prompt':<40} {'Dist to Correct':>18} {'Correct?':>10}")
        logger.info("-" * 70)

        for d in all_data:
            dist = np.linalg.norm(d['hidden_state'] - correct_centroid)
            mark = "✓" if d['actual_correct'] else "✗"
            logger.info(f"{d['prompt'][:38]:<40} {dist:>18.4f} {mark:>10}")

    # ========================================
    # PHASE 5: The Answer Embedding Test
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 5: Answer Embedding Alignment")
    logger.info(f"{'='*80}")

    # Get the output embedding (maps hidden → logits)
    # For LFM2, this is the tied embedding weights
    if hasattr(model, 'lm_head'):
        W = np.array(model.lm_head.weight.tolist())
    else:
        # LFM2 ties weights - use embed_tokens
        W = np.array(model.model.embed_tokens.weight.tolist())  # (vocab, hidden_dim)

    logger.info(f"\nFor each prompt, measure alignment with correct answer embedding:")
    logger.info(f"\n{'Prompt':<35} {'Correct':<10} {'Align (correct)':>15} {'Align (predicted)':>17} {'Diff':>10}")
    logger.info("-" * 90)

    for d in all_data:
        h = d['hidden_state']

        # Get correct answer token ID
        correct_tokens = tokenizer.encode(d['correct_answer'])
        correct_id = correct_tokens[-1] if correct_tokens else correct_tokens[0]

        # Get predicted token ID
        predicted_tokens = tokenizer.encode(d['predicted'])
        predicted_id = predicted_tokens[-1] if predicted_tokens else 0

        # Get embeddings
        correct_emb = W[correct_id]
        predicted_emb = W[predicted_id]

        # Compute alignment (cosine similarity)
        align_correct = np.dot(h, correct_emb) / (np.linalg.norm(h) * np.linalg.norm(correct_emb) + 1e-10)
        align_predicted = np.dot(h, predicted_emb) / (np.linalg.norm(h) * np.linalg.norm(predicted_emb) + 1e-10)

        diff = align_correct - align_predicted

        mark = "✓" if d['actual_correct'] else " "
        logger.info(f"{d['prompt'][:33]:<35} {d['correct_answer'][:8]:<10} {align_correct:>15.4f} {align_predicted:>17.4f} {diff:>+10.4f}")

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: The Geometry of Correctness")
    logger.info(f"{'='*80}")

    logger.info(f"""
WHAT WE MEASURED:

1. HIDDEN STATE GEOMETRY
   - Norm, std, sparsity, kurtosis
   - These capture "how" the model is activating

2. OUTPUT METRICS
   - Confidence, margin, output entropy
   - These capture model's certainty

3. MANIFOLD STRUCTURE
   - SVD of correct vs wrong states
   - Spectral entropy of each manifold

4. DISTANCE TO CORRECT
   - How far is each state from the "correct" centroid?

5. ANSWER ALIGNMENT
   - Cosine similarity to correct answer embedding
   - Cosine similarity to predicted answer embedding

THE KEY QUESTION:

Is there a metric M such that:
  M(hidden_state | correct_answer) < M(hidden_state | wrong_answer) ?

If yes, this metric is the INVARIANT of correctness.
The model "knows" when it's right, geometrically.

CANDIDATES:
- Low spectral entropy of hidden state
- High alignment with correct answer embedding
- Low distance to "correct" centroid
- High margin / low output entropy

NEXT STEP:
If we find the metric, we can:
1. Use it to verify teaching worked
2. Guide the model toward "correct" regions
3. Build a self-correcting loop
""")


if __name__ == "__main__":
    run_experiment()
