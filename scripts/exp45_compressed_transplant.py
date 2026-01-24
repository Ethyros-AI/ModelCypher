#!/usr/bin/env python3
"""Experiment 45: Compressed Source Transplant.

This connects compression research back to the original goal:
Cross-architecture merging (DeepSeek-R1 → LFM2).

Question: Does compressing source layers help cross-architecture merging?

Method:
1. Compress DeepSeek-R1 Layer 24 using our golden layer technique
2. Attempt to align compressed activations with LFM2 target layer
3. Compare alignment quality: compressed vs uncompressed source
4. Measure: CKA, reconstruction error, semantic preservation

Hypothesis: Compressed layers transfer better because:
- Noise is removed (low-rank projection)
- Essential behavior is preserved (100% accuracy)
- Smaller effective dimension may align more cleanly

The physics insight:
Compression finds the "essential coordinates" of the layer's behavior.
These essential coordinates may be more universal across architectures
than the full high-dimensional representation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_cka(X, Y):
    """Compute Centered Kernel Alignment between two activation matrices.

    CKA measures similarity of representations independent of rotation.
    CKA = 1 means identical representation structure.
    """
    # Center the matrices
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)

    # Compute Gram matrices
    K = X_centered @ X_centered.T
    L = Y_centered @ Y_centered.T

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.sum(K * L)

    # Normalize
    norm_x = np.sqrt(np.sum(K * K))
    norm_y = np.sqrt(np.sum(L * L))

    if norm_x < 1e-10 or norm_y < 1e-10:
        return 0.0

    return hsic / (norm_x * norm_y)


def compute_procrustes_alignment(X, Y):
    """Find optimal orthogonal alignment from X to Y.

    Returns the transformation matrix and alignment error.
    """
    # X @ R ≈ Y (find R)
    U, S, Vh = np.linalg.svd(Y.T @ X)
    R = Vh.T @ U.T
    error = np.linalg.norm(X @ R - Y, 'fro') / np.linalg.norm(Y, 'fro')
    return R, error


def run_experiment():
    """Test compressed transplant across architectures."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()

    # Load both models
    source_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    target_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16"

    logger.info("Loading source model (DeepSeek-R1-8B)...")
    from mlx_lm import load
    source_model, source_tokenizer = load(source_path)

    logger.info("Loading target model (LFM2-1.2B)...")
    target_model, target_tokenizer = load(target_path)

    source_n_layers = len(source_model.model.layers)
    target_n_layers = len(target_model.model.layers)

    logger.info(f"\nSource: {source_n_layers} layers, hidden_dim={source_model.model.layers[0].mlp.gate_proj.weight.shape[1]}")
    logger.info(f"Target: {target_n_layers} layers")

    # Check target structure
    target_layer = target_model.model.layers[0]
    if 'feed_forward' in target_layer:
        ff = target_layer['feed_forward']
        # LFM2 uses w1/w2/w3 naming
        if hasattr(ff, 'w1'):
            target_hidden = ff.w1.weight.shape[1]
        elif hasattr(ff, 'gate_proj'):
            target_hidden = ff.gate_proj.weight.shape[1]
        else:
            # Get from first weight found
            for name in ['gate_proj', 'w1', 'fc1']:
                if hasattr(ff, name):
                    target_hidden = getattr(ff, name).weight.shape[1]
                    break
            else:
                target_hidden = 2048  # Default for 1.2B
        logger.info(f"Target hidden_dim={target_hidden}")
        mlp_key = 'feed_forward'
    else:
        target_hidden = target_layer.mlp.gate_proj.weight.shape[1]
        mlp_key = 'mlp'

    compressor = RMTAwareCompressor(backend=backend)

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
    ]

    def get_source_activations(layer_idx, prompts):
        """Get MLP activations from source model."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = source_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = source_model.model.layers[layer_idx]
            original_mlp = layer.mlp

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            layer.mlp = MLPHook(original_mlp)

            try:
                _ = source_model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(mlp_input[0, -1, :])
                outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return np.array(X.tolist()), np.array(Y.tolist())

    def get_target_activations(layer_idx, prompts):
        """Get MLP activations from target model."""
        inputs = []
        outputs = []

        for prompt in prompts:
            tokens = target_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])
            mlp_input = None
            mlp_output = None

            layer = target_model.model.layers[layer_idx]
            if mlp_key == 'feed_forward':
                original_mlp = layer['feed_forward']
            else:
                original_mlp = layer.mlp

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            if mlp_key == 'feed_forward':
                layer['feed_forward'] = MLPHook(original_mlp)
            else:
                layer.mlp = MLPHook(original_mlp)

            try:
                _ = target_model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(mlp_input[0, -1, :])
                outputs.append(mlp_output[0, -1, :])
            finally:
                if mlp_key == 'feed_forward':
                    layer['feed_forward'] = original_mlp
                else:
                    layer.mlp = original_mlp

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return np.array(X.tolist()), np.array(Y.tolist())

    # Phase 1: Get source activations (Layer 24 - Golden Layer)
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Source Activations (DeepSeek-R1 Layer 24)")
    logger.info(f"{'='*80}")

    source_X, source_Y = get_source_activations(24, cal_prompts)
    logger.info(f"Source input shape: {source_X.shape}")
    logger.info(f"Source output shape: {source_Y.shape}")

    # Apply low-rank compression
    Y_mean = source_Y.mean(axis=0)
    U, S, Vh = np.linalg.svd(source_Y - Y_mean, full_matrices=False)

    k = 6  # Golden k
    compressed_Y = (source_Y - Y_mean) @ Vh[:k].T @ Vh[:k] + Y_mean

    logger.info(f"\nCompression (k={k}):")
    logger.info(f"  Original dim: {source_Y.shape[1]}")
    logger.info(f"  Effective dim: {k}")
    logger.info(f"  Variance preserved: {np.sum(S[:k]**2) / np.sum(S**2) * 100:.1f}%")

    # Phase 2: Get target activations at equivalent depth
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Target Activations (LFM2-1.2B)")
    logger.info(f"{'='*80}")

    # Map source layer 24/36 = 67% → target layer at same relative depth
    source_depth = 24 / source_n_layers
    target_layer_idx = int(source_depth * target_n_layers)
    logger.info(f"Source depth: {source_depth:.1%} → Target layer: {target_layer_idx}")

    target_X, target_Y = get_target_activations(target_layer_idx, cal_prompts)
    logger.info(f"Target input shape: {target_X.shape}")
    logger.info(f"Target output shape: {target_Y.shape}")

    # Phase 3: Compare alignment quality
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Alignment Comparison")
    logger.info(f"{'='*80}")

    # CKA between source and target outputs
    cka_original = compute_cka(source_Y, target_Y)
    cka_compressed = compute_cka(compressed_Y, target_Y)

    logger.info(f"\nCKA (representational similarity):")
    logger.info(f"  Original source vs target: {cka_original:.4f}")
    logger.info(f"  Compressed source vs target: {cka_compressed:.4f}")
    logger.info(f"  Change: {(cka_compressed - cka_original):+.4f}")

    # Procrustes alignment
    logger.info(f"\nProcrustes alignment (needs matching dims):")

    if source_Y.shape[1] != target_Y.shape[1]:
        # Project both to common space using PCA
        common_dim = min(source_Y.shape[1], target_Y.shape[1], k * 2)

        # PCA on source
        source_centered = source_Y - source_Y.mean(axis=0)
        U_s, S_s, Vh_s = np.linalg.svd(source_centered, full_matrices=False)
        source_proj = source_centered @ Vh_s[:common_dim].T

        # PCA on compressed source
        comp_centered = compressed_Y - compressed_Y.mean(axis=0)
        source_comp_proj = comp_centered @ Vh_s[:common_dim].T

        # PCA on target
        target_centered = target_Y - target_Y.mean(axis=0)
        U_t, S_t, Vh_t = np.linalg.svd(target_centered, full_matrices=False)
        target_proj = target_centered @ Vh_t[:common_dim].T

        logger.info(f"  Projecting to common {common_dim}-dim space for comparison")

        R_orig, error_orig = compute_procrustes_alignment(source_proj, target_proj)
        R_comp, error_comp = compute_procrustes_alignment(source_comp_proj, target_proj)

        logger.info(f"  Original alignment error: {error_orig:.4f}")
        logger.info(f"  Compressed alignment error: {error_comp:.4f}")
        logger.info(f"  Change: {(error_comp - error_orig):+.4f}")
    else:
        R_orig, error_orig = compute_procrustes_alignment(source_Y, target_Y)
        R_comp, error_comp = compute_procrustes_alignment(compressed_Y, target_Y)

        logger.info(f"  Original alignment error: {error_orig:.4f}")
        logger.info(f"  Compressed alignment error: {error_comp:.4f}")

    # Phase 4: Singular value comparison
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Spectral Analysis")
    logger.info(f"{'='*80}")

    logger.info("\nSingular value profiles (top 10):")
    logger.info(f"  Source: {S[:10]}")

    U_t, S_t, Vh_t = np.linalg.svd(target_Y - target_Y.mean(axis=0), full_matrices=False)
    logger.info(f"  Target: {S_t[:10]}")

    # Effective rank comparison
    def effective_rank(S):
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]
        return np.exp(-np.sum(S_norm * np.log(S_norm)))

    source_eff_rank = effective_rank(S)
    target_eff_rank = effective_rank(S_t)

    logger.info(f"\nEffective rank:")
    logger.info(f"  Source: {source_eff_rank:.2f}")
    logger.info(f"  Target: {target_eff_rank:.2f}")
    logger.info(f"  Ratio: {source_eff_rank / target_eff_rank:.2f}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("FINDINGS: Compressed Transplant Analysis")
    logger.info(f"{'='*80}")

    logger.info(f"""
RESULTS:

1. REPRESENTATIONAL SIMILARITY (CKA)
   - Original source → target: {cka_original:.4f}
   - Compressed source → target: {cka_compressed:.4f}
   - Compression {'improves' if cka_compressed > cka_original else 'reduces'} alignment

2. DIMENSION MISMATCH
   - Source dim: {source_Y.shape[1]}
   - Target dim: {target_Y.shape[1]}
   - This {source_Y.shape[1] / target_Y.shape[1]:.1f}x mismatch is the cross-arch challenge

3. EFFECTIVE DIMENSIONALITY
   - Source effective rank: {source_eff_rank:.1f}
   - Target effective rank: {target_eff_rank:.1f}
   - Compressed dim (k={k}) may be closer to target's effective dim

4. THE COMPRESSION-TRANSFER HYPOTHESIS
   By compressing to k={k} dimensions:
   - We remove noise (variance not in top k)
   - We find "essential coordinates"
   - These may be more universal across architectures

5. NEXT STEPS
   - Build actual transplant: project compressed source to target dim
   - Test inference quality on transplanted layer
   - Compare to uncompressed transplant

THE MODEL PLANCK CONSTANT:
If compression finds universal coordinates, then ℏ_model defines
the scale at which these coordinates become meaningful:
  ℏ_source ≈ 1/k_source (for DeepSeek-R1, k=6 → ℏ ≈ 0.17)
  ℏ_target ≈ 1/k_target (for LFM2, need to measure)

Cross-architecture transfer requires matching ℏ values - finding
the "quantum of action" that's compatible between models.
""")


if __name__ == "__main__":
    run_experiment()
