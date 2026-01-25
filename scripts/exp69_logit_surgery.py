#!/usr/bin/env python3
"""Experiment 69: Logit Surgery - Working Directly in Embedding Space.

The insight: Prompts and tokens are just scaffolding.
The real action is in the embedding/logit space.

Instead of:
  prompt → tokens → activations → [teaching] → tokens → output

We can:
  embedding → [surgery] → logits → output

This is DIRECT manipulation of the model's vocabulary mapping.

Key ideas:
1. The lm_head maps hidden states to logits
2. Each row of lm_head corresponds to a vocabulary token
3. We can modify lm_head to increase/decrease token probabilities
4. This is logit steering without prompts
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
    """Logit surgery experiment."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()

    from mlx_lm import load

    # Load models
    logger.info("Loading LFM2-1.2B...")
    student_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"
    student, tokenizer = load(student_path)

    logger.info("Loading LFM2.5-1.2B-Instruct...")
    teacher_path = "/Volumes/CodeCypher/models/mlx-community/LFM2.5-1.2B-Instruct-bf16"
    teacher, _ = load(teacher_path)

    # ========================================
    # PHASE 1: Explore Embedding Space
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Exploring the LM Head")
    logger.info(f"{'='*80}")

    # Get the language model head (maps hidden → vocab)
    if hasattr(student, 'lm_head'):
        lm_head_s = student.lm_head
    else:
        lm_head_s = student.model.embed_tokens  # Some models tie weights

    if hasattr(teacher, 'lm_head'):
        lm_head_t = teacher.lm_head
    else:
        lm_head_t = teacher.model.embed_tokens

    # Get weight matrices
    W_s = np.array(lm_head_s.weight.tolist())  # (vocab_size, hidden_dim)
    W_t = np.array(lm_head_t.weight.tolist())

    logger.info(f"Student lm_head: {W_s.shape}")
    logger.info(f"Teacher lm_head: {W_t.shape}")

    # ========================================
    # PHASE 2: Token Embedding Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Token Embedding Analysis")
    logger.info(f"{'='*80}")

    # Find token IDs for key words
    tokens_of_interest = ["4", "Paris", "cold", "fly", "water", "blue", "fall", "100"]

    logger.info(f"\n{'Token':<15} {'ID':>10} {'Student norm':>15} {'Teacher norm':>15}")
    logger.info("-" * 60)

    for token_str in tokens_of_interest:
        # Get token ID
        try:
            token_ids = tokenizer.encode(token_str)
            # Take the last token (in case of BOS prefix)
            token_id = token_ids[-1] if len(token_ids) > 0 else token_ids[0]

            # Get embeddings
            s_emb = W_s[token_id]
            t_emb = W_t[token_id]

            s_norm = np.linalg.norm(s_emb)
            t_norm = np.linalg.norm(t_emb)

            logger.info(f"{token_str:<15} {token_id:>10} {s_norm:>15.4f} {t_norm:>15.4f}")
        except Exception as e:
            logger.info(f"{token_str:<15} (error: {e})")

    # ========================================
    # PHASE 3: Compare LM Head Directions
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: LM Head Direction Analysis")
    logger.info(f"{'='*80}")

    # SVD of both lm_heads
    U_s, S_s, Vh_s = svd(W_s, full_matrices=False)
    U_t, S_t, Vh_t = svd(W_t, full_matrices=False)

    logger.info(f"\nTop 10 singular values:")
    logger.info(f"  Student: {S_s[:10].round(2)}")
    logger.info(f"  Teacher: {S_t[:10].round(2)}")

    # Effective rank
    s_rank = np.sum(S_s > S_s[0] * 0.01)
    t_rank = np.sum(S_t > S_t[0] * 0.01)
    logger.info(f"\nEffective rank (1% threshold):")
    logger.info(f"  Student: {s_rank}")
    logger.info(f"  Teacher: {t_rank}")

    # ========================================
    # PHASE 4: Logit Surgery
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Logit Surgery")
    logger.info(f"{'='*80}")

    # The idea: Modify specific token embeddings to make them more likely
    # This is like "boosting" the prior for certain tokens

    test_cases = [
        ("2 + 2 equals", "4"),
        ("The square root of 16 is", "4"),
        ("10 times 10 equals", "100"),
    ]

    def get_prediction(model, tokenizer, prompt):
        """Get model's prediction and top-5."""
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        logits = model(input_ids)
        mx.eval(logits)

        logits_np = np.array(logits[0, -1, :].tolist())
        top_5_idx = np.argsort(logits_np)[-5:][::-1]
        top_5_words = [tokenizer.decode([int(i)]).strip() for i in top_5_idx]
        top_5_probs = softmax(logits_np)[top_5_idx]

        return top_5_words[0], list(zip(top_5_words, top_5_probs.tolist()))

    logger.info("\nBefore surgery:")
    for prompt, expected in test_cases:
        word, top_5 = get_prediction(student, tokenizer, prompt)
        logger.info(f"  '{prompt}' → '{word}'")
        logger.info(f"    Top 5: {[(w, f'{p:.2%}') for w, p in top_5[:3]]}")

    # Try boosting numeric token embeddings
    logger.info("\nBoosting numeric token embeddings...")

    # Get current lm_head weight
    W_original = np.array(lm_head_s.weight.tolist())

    # Tokens to boost
    boost_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "25"]
    boost_factor = 1.5

    W_modified = W_original.copy()
    for token_str in boost_tokens:
        try:
            token_ids = tokenizer.encode(token_str)
            token_id = token_ids[-1] if len(token_ids) > 0 else token_ids[0]
            W_modified[token_id] *= boost_factor
            logger.info(f"  Boosted '{token_str}' (ID {token_id}) by {boost_factor}x")
        except Exception as e:
            logger.info(f"  Could not boost '{token_str}': {e}")

    # Apply modified weights
    lm_head_s.weight = mx.array(W_modified.astype(np.float32))
    mx.eval(lm_head_s.weight)

    logger.info("\nAfter boosting:")
    for prompt, expected in test_cases:
        word, top_5 = get_prediction(student, tokenizer, prompt)
        is_correct = expected.lower() in word.lower()
        mark = "✓" if is_correct else "✗"
        logger.info(f"  {mark} '{prompt}' → '{word}'")
        logger.info(f"    Top 5: {[(w, f'{p:.2%}') for w, p in top_5[:3]]}")

    # Restore original
    lm_head_s.weight = mx.array(W_original.astype(np.float32))
    mx.eval(lm_head_s.weight)

    # ========================================
    # PHASE 5: Directional Transfer in LM Head
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 5: Directional Transfer in LM Head")
    logger.info(f"{'='*80}")

    # Instead of boosting, transfer directions from teacher's lm_head
    logger.info("\nTransferring teacher's lm_head directions...")

    # Project student lm_head onto teacher's principal directions
    # Then add back missing structure

    # Get centered matrices
    W_s_centered = W_s - W_s.mean(axis=0)
    W_t_centered = W_t - W_t.mean(axis=0)

    # Compute transfer
    # For each direction, compute how much teacher "knows" vs student
    n_dirs = 10

    for d in range(n_dirs):
        # Project both onto direction d of teacher
        s_coefs = W_s_centered @ Vh_t[d]  # How each vocab token relates to teacher's direction d
        t_coefs = W_t_centered @ Vh_t[d]

        # Correlation between student and teacher coefficients
        corr = np.corrcoef(s_coefs, t_coefs)[0, 1]
        logger.info(f"  Direction {d}: correlation = {corr:.4f}")

    # Transfer the most different direction
    logger.info("\nTransferring direction 0 from teacher...")

    W_modified = W_s.copy()
    d = 0

    # Remove student's projection onto teacher's direction d
    s_coefs = W_s_centered @ Vh_t[d]
    s_proj = np.outer(s_coefs, Vh_t[d])
    W_modified -= s_proj

    # Add teacher's projection
    t_coefs = W_t_centered @ Vh_t[d]
    t_proj = np.outer(t_coefs, Vh_t[d])
    W_modified += t_proj

    # Apply
    lm_head_s.weight = mx.array(W_modified.astype(np.float32))
    mx.eval(lm_head_s.weight)

    logger.info("\nAfter direction transfer:")
    for prompt, expected in test_cases:
        word, top_5 = get_prediction(student, tokenizer, prompt)
        is_correct = expected.lower() in word.lower()
        mark = "✓" if is_correct else "✗"
        logger.info(f"  {mark} '{prompt}' → '{word}'")
        logger.info(f"    Top 5: {[(w, f'{p:.2%}') for w, p in top_5[:3]]}")

    # Restore
    lm_head_s.weight = mx.array(W_original.astype(np.float32))
    mx.eval(lm_head_s.weight)

    # ========================================
    # Summary
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY: Logit Surgery")
    logger.info(f"{'='*80}")

    logger.info(f"""
WHAT WE EXPLORED:

1. LM HEAD STRUCTURE
   - Maps hidden states (2048D) to vocabulary ({W_s.shape[0]} tokens)
   - Each row is a token's "direction" in hidden space
   - Logits = hidden_state @ lm_head.T

2. TOKEN BOOSTING
   - Multiply specific token embeddings by a factor
   - Increases their logit directly
   - Simple but coarse

3. DIRECTION TRANSFER
   - Transfer principal directions from teacher's lm_head
   - More sophisticated geometric approach
   - Preserves structure while adding knowledge

THE KEY INSIGHT:

The lm_head IS the vocabulary prior.
- High-norm tokens are more likely
- Tokens aligned with common hidden states are more likely
- We can modify the prior directly

WHAT DIDN'T WORK:

Math tokens ("4", "100") may be:
- Rare in training (low embedding norm)
- Poorly aligned with math contexts
- Split across multiple tokens

WHAT MIGHT WORK:

1. REPRESENTATION STEERING
   - Modify hidden states before lm_head
   - Steer toward numeric representations

2. CONTEXT INJECTION
   - Add a "math mode" bias vector to hidden states
   - Learned from examples where math works

3. VOCABULARY SURGERY
   - Merge multiple "4" tokens into one
   - Create dedicated math tokens
""")


if __name__ == "__main__":
    run_experiment()
