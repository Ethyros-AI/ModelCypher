#!/usr/bin/env python3
"""Experiment 51: Directional Teaching - Teaching Topics, Not Subjects.

The insight: We've been teaching whole layers (subjects).
But good teachers break subjects into topics.

A layer has k=6 essential dimensions. Each dimension is a "topic."
What if we could teach specific topics without disturbing others?

Method:
1. Decompose the layer into k directions (SVD)
2. Teach ONE direction at a time
3. Measure: does teaching direction i affect direction j?
4. Find: which directions are independent? Which interfere?

The physics analogy:
- Orthogonal directions = independent topics (can teach separately)
- Coupled directions = related topics (must teach together)

This is like eigenvalue decomposition of knowledge.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Explore directional teaching - topics within a layer."""
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

    # Training prompts
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

    # Collect activations
    logger.info(f"\n{'='*80}")
    logger.info("Collecting Activations")
    logger.info(f"{'='*80}")

    source_X, source_Y = get_activations(source_model, source_tokenizer, source_layer_idx, train_prompts)
    target_X, target_Y = get_activations(target_model, target_tokenizer, target_layer_idx, train_prompts)

    logger.info(f"Source: {source_X.shape} → {source_Y.shape}")
    logger.info(f"Target: {target_X.shape} → {target_Y.shape}")

    # ========================================
    # PHASE 1: Decompose into directions
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Decomposing into Directions (Topics)")
    logger.info(f"{'='*80}")

    # SVD of source outputs (the "knowledge" to transfer)
    source_Y_centered = source_Y - source_Y.mean(axis=0)
    U_s, S_s, Vh_s = np.linalg.svd(source_Y_centered, full_matrices=False)

    # SVD of target outputs (the "current understanding")
    target_Y_centered = target_Y - target_Y.mean(axis=0)
    U_t, S_t, Vh_t = np.linalg.svd(target_Y_centered, full_matrices=False)

    logger.info(f"Source singular values (top 10): {S_s[:10]}")
    logger.info(f"Target singular values (top 10): {S_t[:10]}")

    # Variance explained by each direction
    source_var = S_s**2 / np.sum(S_s**2)
    target_var = S_t**2 / np.sum(S_t**2)

    logger.info(f"\nVariance by direction (topics):")
    logger.info(f"{'Dir':>4} {'Source':>10} {'Target':>10} {'Cumulative':>12}")
    logger.info("-" * 40)
    for i in range(10):
        cum_s = np.sum(source_var[:i+1])
        logger.info(f"{i+1:>4} {source_var[i]*100:>9.2f}% {target_var[i]*100:>9.2f}% {cum_s*100:>11.1f}%")

    # ========================================
    # PHASE 2: Alignment between directions
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Alignment Between Source and Target Directions")
    logger.info(f"{'='*80}")

    # How well do source directions align with target directions?
    # Compute overlap matrix
    k = 6  # Number of "topics" to consider

    # Project source outputs onto source's top-k directions
    source_Y_k = source_Y_centered @ Vh_s[:k].T  # (n, k)

    # Project target outputs onto target's top-k directions
    target_Y_k = target_Y_centered @ Vh_t[:k].T  # (n, k)

    # Correlation between source direction i and target direction j
    alignment = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            corr = np.corrcoef(source_Y_k[:, i], target_Y_k[:, j])[0, 1]
            alignment[i, j] = corr if not np.isnan(corr) else 0

    logger.info(f"\nDirection alignment matrix (|correlation|):")
    logger.info(f"Source → Target directions")
    header = "     " + "".join(f"  T{j+1:>2}" for j in range(k))
    logger.info(header)
    for i in range(k):
        row = f"S{i+1:>2}  " + "".join(f"{abs(alignment[i,j]):>5.2f}" for j in range(k))
        logger.info(row)

    # Find best matching pairs
    logger.info(f"\nBest matching direction pairs:")
    for i in range(k):
        best_j = np.argmax(np.abs(alignment[i, :]))
        logger.info(f"  Source {i+1} → Target {best_j+1} (r={alignment[i, best_j]:.3f})")

    # ========================================
    # PHASE 3: Teach individual directions
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Teaching Individual Directions")
    logger.info(f"{'='*80}")

    def teach_direction(d, source_Y, target_X, target_Y, Vh_s, Vh_t):
        """Teach only direction d from source to target.

        Returns a modified target output that has direction d from source
        but keeps other directions from target.
        """
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        # Get source's contribution in direction d
        # source_Y_centered @ Vh_s[d] gives (n,) - the projection coefficients
        # Then outer product with Vh_s[d] to get back to (n, d_hidden)
        source_coefs_d = source_Y_centered @ Vh_s[d]  # (n,)
        source_proj_d = np.outer(source_coefs_d, Vh_s[d])  # (n, d_hidden)

        # Get target's contribution WITHOUT direction d
        target_coefs_d = target_Y_centered @ Vh_t[d]  # (n,)
        target_proj_d = np.outer(target_coefs_d, Vh_t[d])  # (n, d_hidden)
        target_proj_other = target_Y_centered - target_proj_d

        # Map source's direction d to target's space using F
        F = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]
        source_d_in_target = source_proj_d @ F

        combined = target_proj_other + source_d_in_target + target_Y.mean(axis=0)

        return combined

    def evaluate_teaching(target_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts):
        """Evaluate a teaching by creating W from the modified outputs."""
        # Learn W from target_X → modified target_output
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

        correct = 0
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

            if orig_top == trans_top:
                correct += 1

        return correct / len(test_prompts)

    # Test teaching each direction individually
    logger.info(f"\nTeaching individual directions (topics):")
    logger.info(f"{'Dir':>4} {'Accuracy':>10} {'Variance':>10}")
    logger.info("-" * 30)

    direction_results = []
    for d in range(k):
        modified_output = teach_direction(d, source_Y, target_X, target_Y, Vh_s, Vh_t)
        acc = evaluate_teaching(modified_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        direction_results.append((d, acc, source_var[d]))
        logger.info(f"{d+1:>4} {acc*100:>9.1f}% {source_var[d]*100:>9.2f}%")

    # ========================================
    # PHASE 4: Cumulative teaching
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Cumulative Teaching (Adding Topics)")
    logger.info(f"{'='*80}")

    def teach_directions(dirs, source_Y, target_X, target_Y, Vh_s, Vh_t):
        """Teach multiple directions from source."""
        source_Y_centered = source_Y - source_Y.mean(axis=0)
        target_Y_centered = target_Y - target_Y.mean(axis=0)

        F = np.linalg.lstsq(source_Y, target_Y, rcond=1e-10)[0]

        # Start with target, replace specified directions with source
        result = target_Y.copy()

        for d in dirs:
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

    logger.info(f"\nCumulative teaching (adding directions 1→k):")
    logger.info(f"{'Dirs':>10} {'Accuracy':>10} {'Var Covered':>12}")
    logger.info("-" * 35)

    for n_dirs in range(1, k+1):
        dirs = list(range(n_dirs))
        modified_output = teach_directions(dirs, source_Y, target_X, target_Y, Vh_s, Vh_t)
        acc = evaluate_teaching(modified_output, target_X, target_model, target_tokenizer, target_layer_idx, test_prompts)
        var_covered = np.sum(source_var[:n_dirs])
        logger.info(f"{'1-'+str(n_dirs):>10} {acc*100:>9.1f}% {var_covered*100:>11.1f}%")

    # ========================================
    # Analysis
    # ========================================

    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS: Directional Teaching")
    logger.info(f"{'='*80}")

    logger.info(f"""
THE TOPIC DECOMPOSITION:

Each layer's behavior can be decomposed into k=6 "topics" (principal directions).
Each topic captures a different aspect of what the layer "knows."

INDIVIDUAL TOPIC TEACHING:

When we teach just ONE topic at a time:
- Some topics transfer well (high accuracy when taught alone)
- Some topics interfere (lower accuracy)
- This reveals which "topics" are compatible

THE ALIGNMENT MATRIX:

Source and target directions don't align perfectly.
The correlation matrix shows:
- Diagonal dominance = similar topic structure
- Off-diagonal = topics are "translated" differently

CUMULATIVE TEACHING:

Adding topics one by one shows:
- First few topics = foundation (most variance)
- Later topics = refinement (less variance, may help or hurt)

THE PEDAGOGICAL INSIGHT:

Like teaching a subject:
- Some topics must be taught first (foundational)
- Some topics build on others (cumulative)
- Some topics are independent (can be taught in any order)

NEXT QUESTIONS:

1. Can we teach topics ACROSS layers?
   (Learn topic 1 from L24, topic 2 from L25, etc.)

2. Are there "universal topics" across architectures?
   (Topics that mean the same thing in DeepSeek and LFM2)

3. Can we create a "topic dictionary"?
   (Map each topic to human-interpretable concepts)
""")


if __name__ == "__main__":
    run_experiment()
