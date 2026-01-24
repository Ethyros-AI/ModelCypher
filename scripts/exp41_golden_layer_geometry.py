#!/usr/bin/env python3
"""Experiment 41: Golden Layer Geometry.

The breakthrough from exp40: Layer 24 achieves 100% accuracy with k=6 low-rank projection.

Question: What makes Layer 24's activation geometry special?

Method:
1. Profile activation covariance matrices for ALL layers
2. Compare Layer 24 to adjacent layers (23, 25)
3. Measure: effective rank, condition number, spectral gaps
4. Look for geometric signature that predicts "golden" status

Hypothesis: Golden layers have spectral gaps after k principal components.

The key insight we're testing: Layer 24 is at depth 67% ≈ φ⁻¹ = 61.8%.
Is this a universal property or specific to this model?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI  # ≈ 0.618


def compute_effective_rank(S):
    """Compute effective rank from singular values.

    Effective rank = exp(entropy of normalized singular values).
    This measures the "spread" of information across dimensions.
    """
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]  # Avoid log(0)
    entropy = -np.sum(S_norm * np.log(S_norm))
    return np.exp(entropy)


def compute_spectral_gap(S, k):
    """Compute spectral gap after the k-th singular value.

    Gap = S[k-1] / S[k] (ratio between k-th and (k+1)-th singular values).
    A large gap indicates a natural "cut point" for dimensionality reduction.
    """
    if k >= len(S):
        return float('inf')
    if S[k] < 1e-10:
        return float('inf')
    return S[k-1] / S[k]


def compute_variance_explained(S, k):
    """Compute fraction of variance explained by top-k components."""
    total_var = np.sum(S**2)
    if total_var < 1e-10:
        return 1.0
    return np.sum(S[:k]**2) / total_var


def run_experiment():
    """Analyze the geometry of the Golden Layer."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    golden_depth = int(n_layers * PHI_INV)

    logger.info(f"Model has {n_layers} layers")
    logger.info(f"Golden ratio depth: layer {golden_depth} (φ⁻¹ = {PHI_INV:.3f})")
    logger.info(f"Known golden layer from exp40: layer 24 (depth = {24/n_layers:.1%})")

    # Calibration prompts - same as exp40 for consistency
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
        "The heart pumps blood continuously",
        "Computers process binary information",
        "Oxygen is essential for breathing",
        "Mathematics describes patterns precisely",
        "Language enables human communication",
        "Climate affects global weather patterns",
        "Atoms form the basis of matter",
        "Energy cannot be created or destroyed",
    ]

    cal_tokens = [tokenizer.encode(p) for p in cal_prompts]

    def get_layer_activations(layer_idx, tokens_list):
        """Collect MLP inputs and outputs for a layer."""
        inputs = []
        outputs = []

        for tok in tokens_list:
            input_ids = mx.array([tok])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
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
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(mlp_input[0, -1, :])
                outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return X, Y

    # Phase 1: Profile ALL layers
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Profiling activation geometry for all layers")
    logger.info(f"{'='*80}")

    layer_profiles = []

    logger.info(f"\n{'Layer':>6} {'Depth':>7} {'EffRank':>9} {'CondNum':>12} "
               f"{'Gap@4':>8} {'Gap@6':>8} {'Gap@8':>8} {'Var@6':>8}")
    logger.info("-" * 85)

    for layer_idx in range(n_layers):
        X, Y = get_layer_activations(layer_idx, cal_tokens)

        # Compute SVD of output activations
        Y_np = np.array(Y.tolist())
        Y_centered = Y_np - Y_np.mean(axis=0)

        try:
            U, S, Vh = np.linalg.svd(Y_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.info(f"{layer_idx:>6} {'ERROR':>7}")
            continue

        # Compute metrics
        depth = layer_idx / n_layers
        eff_rank = compute_effective_rank(S)
        cond_num = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
        gap_4 = compute_spectral_gap(S, 4)
        gap_6 = compute_spectral_gap(S, 6)
        gap_8 = compute_spectral_gap(S, 8)
        var_6 = compute_variance_explained(S, 6)

        profile = {
            'layer': layer_idx,
            'depth': depth,
            'eff_rank': eff_rank,
            'cond_num': cond_num,
            'gap_4': gap_4,
            'gap_6': gap_6,
            'gap_8': gap_8,
            'var_6': var_6,
            'singular_values': S[:10],  # Top 10 for analysis
        }
        layer_profiles.append(profile)

        # Mark special layers
        marker = ""
        if layer_idx == 24:
            marker = " *** GOLDEN ***"
        elif layer_idx == golden_depth:
            marker = " (φ⁻¹)"

        cond_str = f"{cond_num:.1e}" if cond_num < 1e10 else "inf"
        logger.info(f"{layer_idx:>6} {depth:>6.1%} {eff_rank:>9.2f} {cond_str:>12} "
                   f"{gap_4:>8.2f} {gap_6:>8.2f} {gap_8:>8.2f} {var_6:>7.1%}{marker}")

    # Phase 2: Analyze the golden layer signature
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Golden Layer Analysis")
    logger.info(f"{'='*80}")

    # Get profiles for key layers
    golden = next(p for p in layer_profiles if p['layer'] == 24)
    adjacent = [p for p in layer_profiles if p['layer'] in [23, 25]]
    phi_layer = next((p for p in layer_profiles if p['layer'] == golden_depth), None)

    logger.info("\n--- Layer 24 (THE GOLDEN LAYER) ---")
    logger.info(f"  Depth: {golden['depth']:.1%}")
    logger.info(f"  Effective rank: {golden['eff_rank']:.2f}")
    logger.info(f"  Condition number: {golden['cond_num']:.2e}")
    logger.info(f"  Spectral gap @ k=4: {golden['gap_4']:.2f}")
    logger.info(f"  Spectral gap @ k=6: {golden['gap_6']:.2f}")
    logger.info(f"  Spectral gap @ k=8: {golden['gap_8']:.2f}")
    logger.info(f"  Variance explained @ k=6: {golden['var_6']:.1%}")
    logger.info(f"  Top 10 singular values: {golden['singular_values'][:10]}")

    logger.info("\n--- Adjacent Layers (23, 25) ---")
    for p in adjacent:
        logger.info(f"\nLayer {p['layer']}:")
        logger.info(f"  Effective rank: {p['eff_rank']:.2f}")
        logger.info(f"  Spectral gap @ k=6: {p['gap_6']:.2f}")
        logger.info(f"  Variance explained @ k=6: {p['var_6']:.1%}")

    if phi_layer and phi_layer['layer'] != 24:
        logger.info(f"\n--- φ⁻¹ Layer ({phi_layer['layer']}) ---")
        logger.info(f"  Depth: {phi_layer['depth']:.1%}")
        logger.info(f"  Effective rank: {phi_layer['eff_rank']:.2f}")
        logger.info(f"  Spectral gap @ k=6: {phi_layer['gap_6']:.2f}")
        logger.info(f"  Variance explained @ k=6: {phi_layer['var_6']:.1%}")

    # Phase 3: Find layers with similar geometry to golden layer
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 3: Finding Layers with Golden-like Geometry")
    logger.info(f"{'='*80}")

    # Look for layers with:
    # 1. Low effective rank (concentrated information)
    # 2. Large spectral gap at k=6 (natural dimensionality)
    # 3. High variance explained at k=6

    golden_candidates = []
    for p in layer_profiles:
        # Criteria based on Layer 24's properties
        is_low_rank = p['eff_rank'] < golden['eff_rank'] * 1.2
        has_gap = p['gap_6'] > 1.3  # Spectral gap > 1.3x
        high_var = p['var_6'] > 0.9  # 90%+ variance explained

        if is_low_rank and has_gap and high_var:
            golden_candidates.append(p)

    logger.info(f"\nLayers matching golden geometry criteria:")
    logger.info(f"  - Effective rank < {golden['eff_rank'] * 1.2:.2f}")
    logger.info(f"  - Spectral gap @ k=6 > 1.3")
    logger.info(f"  - Variance explained @ k=6 > 90%")

    logger.info(f"\n{'Layer':>6} {'Depth':>7} {'EffRank':>9} {'Gap@6':>8} {'Var@6':>8}")
    logger.info("-" * 45)
    for p in golden_candidates:
        marker = " ***" if p['layer'] == 24 else ""
        logger.info(f"{p['layer']:>6} {p['depth']:>6.1%} {p['eff_rank']:>9.2f} "
                   f"{p['gap_6']:>8.2f} {p['var_6']:>7.1%}{marker}")

    # Phase 4: Spectral analysis - find natural k for each layer
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 4: Finding Natural Dimensionality (k) for Each Layer")
    logger.info(f"{'='*80}")

    logger.info("\nFor each layer, find k where spectral gap is maximized:")
    logger.info(f"\n{'Layer':>6} {'Best k':>7} {'Gap':>8} {'Var':>8}")
    logger.info("-" * 35)

    natural_k = []
    for p in layer_profiles:
        # Find k with largest spectral gap
        S = p['singular_values']
        best_k = 1
        best_gap = 0
        for k in range(1, min(9, len(S))):
            gap = compute_spectral_gap(S, k)
            if gap > best_gap:
                best_gap = gap
                best_k = k

        var_at_k = compute_variance_explained(S, best_k)
        natural_k.append({'layer': p['layer'], 'k': best_k, 'gap': best_gap, 'var': var_at_k})

        marker = " ***" if p['layer'] == 24 else ""
        logger.info(f"{p['layer']:>6} {best_k:>7} {best_gap:>8.2f} {var_at_k:>7.1%}{marker}")

    # Phase 5: Test the hypothesis - do golden-geometry layers achieve high accuracy?
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 5: Testing Golden Geometry → Compression Accuracy")
    logger.info(f"{'='*80}")

    # Test a subset of layers for compression accuracy
    test_layers = [p['layer'] for p in golden_candidates[:10]] + [24]
    test_layers = list(set(test_layers))
    test_layers.sort()

    # Held-out prompts for testing
    held_prompts = [
        "The moon orbits Earth",
        "Birds can fly south",
        "Music has rhythm",
        "Plants need water",
        "Fire requires oxygen",
        "Ice is frozen water",
        "Math uses numbers",
        "Art expresses ideas",
        "Clouds contain moisture",
        "Books store knowledge",
        "Trees produce oxygen",
        "Oceans cover Earth",
        "Dogs are loyal companions",
        "Wind moves clouds across",
        "Snow falls in winter",
        "Fish swim in water",
    ]
    held_tokens = [tokenizer.encode(p) for p in held_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    def test_layer_compression(layer_idx, k):
        """Test compression accuracy for a layer at given k."""
        X, Y = get_layer_activations(layer_idx, cal_tokens)

        # Low-rank projection
        Y_np = np.array(Y.tolist())
        Y_mean = Y_np.mean(axis=0)
        U, S, Vh = np.linalg.svd(Y_np - Y_mean, full_matrices=False)

        actual_k = min(k, len(S))
        Vh_k = mx.array(Vh[:actual_k, :].T).astype(mx.float32)
        Y_mean_mx = mx.array(Y_mean).astype(mx.float32)
        mx.eval(Vh_k, Y_mean_mx)

        Y_centered = Y - Y_mean_mx
        Y_proj_k = mx.matmul(Y_centered, Vh_k)
        Y_proj = mx.matmul(Y_proj_k, Vh_k.T) + Y_mean_mx
        mx.eval(Y_proj)

        # Fit T
        X_backend = backend.array(X)
        Y_proj_backend = backend.array(Y_proj)
        rmt_result = compressor.compress_layer(X_backend, Y_proj_backend)
        T = np.array(backend.tolist(rmt_result.T))

        # Evaluate
        correct = 0
        for tok in held_tokens:
            input_ids = mx.array([tok])

            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # Apply compression
            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp
            T_mx = mx.array(T).astype(mx.float32)
            mx.eval(T_mx)

            class CompressedMLP:
                def __init__(self, T):
                    self.T = T
                def __call__(self, x):
                    return mx.matmul(x, self.T.T)

            layer.mlp = CompressedMLP(T_mx)

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())
                if comp_top == orig_top:
                    correct += 1
            finally:
                layer.mlp = original_mlp

        return correct / len(held_tokens)

    logger.info(f"\nTesting compression accuracy for golden-geometry layers:")
    logger.info(f"\n{'Layer':>6} {'k':>4} {'Accuracy':>10} {'EffRank':>9} {'Gap@k':>8}")
    logger.info("-" * 45)

    accuracy_results = []
    for layer_idx in test_layers:
        p = next(p for p in layer_profiles if p['layer'] == layer_idx)
        best_k_info = next(n for n in natural_k if n['layer'] == layer_idx)
        k = best_k_info['k']

        # Test with natural k and k=6 (the golden k)
        acc_natural = test_layer_compression(layer_idx, k)
        acc_6 = test_layer_compression(layer_idx, 6) if k != 6 else acc_natural

        accuracy_results.append({
            'layer': layer_idx,
            'k': k,
            'acc_natural': acc_natural,
            'acc_6': acc_6,
            'eff_rank': p['eff_rank'],
            'gap': best_k_info['gap'],
        })

        marker = " ***" if layer_idx == 24 else ""
        logger.info(f"{layer_idx:>6} {k:>4} {acc_natural*100:>9.1f}% {p['eff_rank']:>9.2f} "
                   f"{best_k_info['gap']:>8.2f}{marker}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("FINDINGS")
    logger.info(f"{'='*80}")

    logger.info("""
THE GOLDEN LAYER SIGNATURE:

We analyzed Layer 24's activation geometry to understand why it achieves
100% compression accuracy with k=6. Key findings:

1. EFFECTIVE RANK
   - Measures how "spread out" information is across dimensions
   - Lower = more concentrated, easier to compress
   - Layer 24's effective rank: {:.2f}

2. SPECTRAL GAP
   - Large gap at k=6 indicates natural dimensionality
   - Layer 24's gap@6: {:.2f}
   - This means 6 dimensions capture "most" of the information

3. VARIANCE EXPLAINED
   - At k=6, Layer 24 captures {:.1%} of variance
   - This explains why k=6 works: almost all information preserved

4. DEPTH
   - Layer 24 is at {:.1%} depth (near φ⁻¹ = {:.1%})
   - In the "transmission zone" past the information peak

THE HYPOTHESIS:
Layers with similar geometry (low effective rank, spectral gap at small k,
high variance explained) should also achieve high compression accuracy.

""".format(
        golden['eff_rank'],
        golden['gap_6'],
        golden['var_6'],
        golden['depth'],
        PHI_INV
    ))

    # Correlation analysis
    if accuracy_results:
        eff_ranks = [r['eff_rank'] for r in accuracy_results]
        gaps = [r['gap'] for r in accuracy_results]
        accs = [r['acc_natural'] for r in accuracy_results]

        if len(accs) > 2:
            corr_rank = np.corrcoef(eff_ranks, accs)[0, 1]
            corr_gap = np.corrcoef(gaps, accs)[0, 1]

            logger.info(f"CORRELATIONS:")
            logger.info(f"  Effective rank vs accuracy: r = {corr_rank:.3f}")
            logger.info(f"  Spectral gap vs accuracy: r = {corr_gap:.3f}")

            if abs(corr_gap) > 0.5:
                logger.info("\n  --> Spectral gap IS predictive of compression success!")
            if abs(corr_rank) > 0.5:
                logger.info("\n  --> Effective rank IS predictive of compression success!")


if __name__ == "__main__":
    run_experiment()
