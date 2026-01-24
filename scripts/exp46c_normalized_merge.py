#!/usr/bin/env python3
"""Experiment 46c: Normalized Cross-Architecture Merge.

Problem from 46b: Source activations are 30x larger than target.
This causes numerical overflow during alignment.

Solution:
1. Normalize activations to unit variance before alignment
2. Use small k (like the k=6 from compression experiments)
3. Apply proper scaling after transplant
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Cross-architecture merge with proper normalization."""
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

    # Calibration prompts
    cal_prompts = [
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
    ]

    test_prompts = [
        "The moon orbits",
        "Birds can fly",
        "Music has",
        "Plants need",
        "Fire requires",
        "Ice is frozen",
        "Math uses",
        "Art expresses",
        "Clouds contain",
        "Books store",
        "Trees produce",
        "Oceans cover",
    ]

    def get_activations(model, tokenizer, layer_idx, prompts, mlp_key='mlp'):
        """Get MLP activations with proper extraction."""
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
                inputs.append(mlp_input[0, -1, :])
                outputs.append(mlp_output[0, -1, :])
            finally:
                if key == 'feed_forward':
                    layer.feed_forward = original_mlp
                else:
                    layer.mlp = original_mlp

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return np.array(X.tolist()).astype(np.float64), np.array(Y.tolist()).astype(np.float64)

    # Collect activations
    logger.info(f"\n{'='*80}")
    logger.info("Collecting Activations (float64 for stability)")
    logger.info(f"{'='*80}")

    source_X, source_Y = get_activations(source_model, source_tokenizer, source_layer_idx, cal_prompts)
    target_X, target_Y = get_activations(target_model, target_tokenizer, target_layer_idx, cal_prompts)

    logger.info(f"Source: X={source_X.shape}, Y={source_Y.shape}")
    logger.info(f"Target: X={target_X.shape}, Y={target_Y.shape}")

    # Check activation magnitudes
    logger.info(f"\nActivation magnitudes:")
    logger.info(f"  Source Y: mean={np.mean(np.abs(source_Y)):.2f}, max={np.max(np.abs(source_Y)):.2f}")
    logger.info(f"  Target Y: mean={np.mean(np.abs(target_Y)):.2f}, max={np.max(np.abs(target_Y)):.2f}")

    # Normalize to unit variance
    logger.info(f"\n{'='*80}")
    logger.info("Normalizing Activations")
    logger.info(f"{'='*80}")

    # Center and scale
    source_Y_mean = source_Y.mean(axis=0)
    source_Y_std = source_Y.std(axis=0) + 1e-8
    source_Y_norm = (source_Y - source_Y_mean) / source_Y_std

    target_Y_mean = target_Y.mean(axis=0)
    target_Y_std = target_Y.std(axis=0) + 1e-8
    target_Y_norm = (target_Y - target_Y_mean) / target_Y_std

    logger.info(f"After normalization:")
    logger.info(f"  Source Y: mean={np.mean(source_Y_norm):.4f}, std={np.std(source_Y_norm):.4f}")
    logger.info(f"  Target Y: mean={np.mean(target_Y_norm):.4f}, std={np.std(target_Y_norm):.4f}")

    # Low-rank projection with k=6 (golden k from compression)
    logger.info(f"\n{'='*80}")
    logger.info("Low-Rank Projection (k=6, the golden k)")
    logger.info(f"{'='*80}")

    k = 6

    U_s, S_s, Vh_s = np.linalg.svd(source_Y_norm, full_matrices=False)
    U_t, S_t, Vh_t = np.linalg.svd(target_Y_norm, full_matrices=False)

    logger.info(f"Source singular values: {S_s[:10]}")
    logger.info(f"Target singular values: {S_t[:10]}")

    # Project to k dimensions
    source_Y_k = source_Y_norm @ Vh_s[:k].T  # (n, k)
    target_Y_k = target_Y_norm @ Vh_t[:k].T  # (n, k)

    logger.info(f"Projected shapes: source={source_Y_k.shape}, target={target_Y_k.shape}")

    # CKA in low-rank space
    def compute_cka(X, Y):
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)
        K = X_c @ X_c.T
        L = Y_c @ Y_c.T
        hsic = np.sum(K * L)
        norm_x = np.sqrt(np.sum(K * K))
        norm_y = np.sqrt(np.sum(L * L))
        return hsic / (norm_x * norm_y) if norm_x > 1e-10 and norm_y > 1e-10 else 0.0

    cka_k = compute_cka(source_Y_k, target_Y_k)
    logger.info(f"CKA in k={k} space: {cka_k:.4f}")

    # Procrustes alignment in low-rank space
    U, S, Vh = np.linalg.svd(target_Y_k.T @ source_Y_k)
    R = Vh.T @ U.T  # Orthogonal rotation

    source_Y_aligned = source_Y_k @ R
    alignment_error = np.linalg.norm(source_Y_aligned - target_Y_k, 'fro') / np.linalg.norm(target_Y_k, 'fro')
    logger.info(f"Procrustes alignment error: {alignment_error:.4f}")

    cka_aligned = compute_cka(source_Y_aligned, target_Y_k)
    logger.info(f"CKA after Procrustes: {cka_aligned:.4f}")

    # Strategy: Learn W that maps target_X → source behavior in target space
    logger.info(f"\n{'='*80}")
    logger.info("Building Transplant Transform")
    logger.info(f"{'='*80}")

    # Lift aligned source back to target's full space
    # source_Y_aligned is in k-dim target PCA space
    # Lift to full target space: source_Y_aligned @ Vh_t[:k]
    source_Y_in_target_norm = source_Y_aligned @ Vh_t[:k]
    source_Y_in_target = source_Y_in_target_norm * target_Y_std + target_Y_mean

    logger.info(f"Source behavior in target space: {source_Y_in_target.shape}")

    # Now learn W: target_X @ W.T ≈ source_Y_in_target
    # Use regularized least squares
    alpha = 1e-4
    W = np.linalg.lstsq(target_X.T @ target_X + alpha * np.eye(target_X.shape[1]),
                         target_X.T @ source_Y_in_target, rcond=None)[0].T

    # Reconstruction check
    pred = target_X @ W.T
    recon_error = np.linalg.norm(pred - source_Y_in_target, 'fro') / np.linalg.norm(source_Y_in_target, 'fro')
    logger.info(f"Transplant W shape: {W.shape}")
    logger.info(f"Reconstruction error: {recon_error:.4f}")

    # Phase: Inference test
    logger.info(f"\n{'='*80}")
    logger.info("Inference Test")
    logger.info(f"{'='*80}")

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

    logger.info("\n--- Original vs Transplanted ---")

    for prompt in test_prompts[:6]:
        tokens = target_tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Original
        orig_logits = target_model(input_ids)
        mx.eval(orig_logits)
        orig_tokens = []
        for _ in range(5):
            next_token = int(mx.argmax(orig_logits[0, -1, :]).item())
            orig_tokens.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
            orig_logits = target_model(input_ids)
            mx.eval(orig_logits)
        orig_cont = target_tokenizer.decode(orig_tokens)

        # Transplant
        if mlp_key == 'feed_forward':
            target_layer.feed_forward = TransplantedMLP(W_mx)
        else:
            target_layer.mlp = TransplantedMLP(W_mx)

        try:
            tokens = target_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            trans_logits = target_model(input_ids)
            mx.eval(trans_logits)
            trans_tokens = []
            for _ in range(5):
                next_token = int(mx.argmax(trans_logits[0, -1, :]).item())
                trans_tokens.append(next_token)
                input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
                trans_logits = target_model(input_ids)
                mx.eval(trans_logits)
            trans_cont = target_tokenizer.decode(trans_tokens)
        finally:
            if mlp_key == 'feed_forward':
                target_layer.feed_forward = original_mlp
            else:
                target_layer.mlp = original_mlp

        logger.info(f"\nPrompt: \"{prompt}\"")
        logger.info(f"  Original:   ...{orig_cont}")
        logger.info(f"  Transplant: ...{trans_cont}")

    # Quantitative
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

    accuracy = correct / len(test_prompts)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS")
    logger.info(f"{'='*80}")

    logger.info(f"""
1. NORMALIZATION
   - Source Y magnitude: {np.mean(np.abs(source_Y)):.2f}
   - Target Y magnitude: {np.mean(np.abs(target_Y)):.2f}
   - Ratio: {np.mean(np.abs(source_Y)) / np.mean(np.abs(target_Y)):.1f}x
   - After normalization: unit variance for both

2. LOW-RANK ALIGNMENT (k={k})
   - CKA in k-space: {cka_k:.4f}
   - CKA after Procrustes: {cka_aligned:.4f}
   - Alignment error: {alignment_error:.4f}

3. TRANSPLANT
   - W shape: {W.shape}
   - Reconstruction error: {recon_error:.4f}

4. INFERENCE
   - Top-1 token agreement: {accuracy*100:.1f}% ({correct}/{len(test_prompts)})
   - Exp46 baseline: 62.5%

5. INTERPRETATION
   The k=6 space captures the "essential behavior" found in compression.
   Procrustes finds the optimal rotation between these spaces.
   The transplant applies source's essential behavior through target's structure.
""")


if __name__ == "__main__":
    run_experiment()
