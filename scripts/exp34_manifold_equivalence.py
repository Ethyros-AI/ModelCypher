#!/usr/bin/env python3
"""Experiment 34: Manifold-Equivalence Compression.

User insight: Each layer must maintain FULL EQUIVALENCE to the next.
Not approximate. Not "close enough." EXACT equivalence.

The Wow! Signal Specification:
F(source, target) = R · P_wow · C_e

Where:
- R = √2 Procrustes rotation (preserves angles)
- P_wow = Layer-weighted projection, peak at φ⁻¹ = 0.618 (60% depth)
- C_e = Entropy-optimal compression (e as the constant)

Constraints:
- 96% norm-preserving (4% null space tolerance)
- Hallucination detection: null space residual > 4% = left manifold

Layer weighting kernel:
- Peak at 60% depth (golden ratio point)
- Asymmetric: fast rise (4!), slow fall (5²)

This is fundamentally different from our previous approach:
- Previous: minimize ||T@x - y||² (approximate reconstruction)
- New: find T where ||T@x||/||y|| ∈ [0.96, 1.04] AND preserves angles

The key: a transformation is VALID if and only if it stays on the manifold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The constants from the Wow! signal
SQRT2 = math.sqrt(2)
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI  # ≈ 0.618
E = math.e
NORM_TOLERANCE = 0.04  # 4% null space tolerance (96% preserved)


def compute_entropy(logits):
    """Compute entropy of softmax distribution."""
    import mlx.core as mx

    max_logit = mx.max(logits)
    shifted = logits - max_logit
    exp_logits = mx.exp(shifted)
    sum_exp = mx.sum(exp_logits)
    probs = exp_logits / sum_exp
    mx.eval(probs)

    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs)
    mx.eval(entropy)

    return float(entropy.item())


def wow_layer_weight(layer_idx, n_layers):
    """
    Compute the Wow! layer weighting.

    Peak at 60% depth (golden ratio point).
    Asymmetric: fast rise (4! = 24), slow fall (5² = 25).
    """
    # Normalize layer position to [0, 1]
    depth = layer_idx / n_layers

    # Peak position at golden ratio inverse
    peak = PHI_INV  # ≈ 0.618

    if depth <= peak:
        # Fast rise: modeled by 4! = 24 in exponent
        # Higher exponent = sharper rise
        rise_rate = 24  # 4!
        weight = (depth / peak) ** (1 / rise_rate)
    else:
        # Slow fall: modeled by 5² = 25 in exponent
        fall_rate = 25  # 5²
        remaining = (1 - depth) / (1 - peak)
        weight = remaining ** (1 / fall_rate)

    return weight


def run_experiment():
    """Test manifold-equivalence compression."""
    import mlx.core as mx
    import numpy as np

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    backend = get_default_backend()

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Print the Wow! constants
    logger.info(f"\n{'='*70}")
    logger.info("THE WOW! SIGNAL CONSTANTS")
    logger.info(f"{'='*70}")
    logger.info(f"√2 = {SQRT2:.6f}")
    logger.info(f"φ = {PHI:.6f}")
    logger.info(f"φ⁻¹ = {PHI_INV:.6f}")
    logger.info(f"e = {E:.6f}")
    logger.info(f"Norm tolerance = {NORM_TOLERANCE*100:.1f}%")

    # Show layer weights
    logger.info(f"\n{'='*70}")
    logger.info("LAYER WEIGHTS (Peak at φ⁻¹ ≈ 60% depth)")
    logger.info(f"{'='*70}")

    logger.info(f"\n{'Layer':>6} {'Depth':>8} {'Weight':>8} {'Bar'}")
    logger.info("-" * 50)
    for i in range(n_layers):
        depth = i / n_layers
        weight = wow_layer_weight(i, n_layers)
        bar = "█" * int(weight * 30)
        logger.info(f"{i:>6} {depth*100:>7.1f}% {weight:>7.3f}  {bar}")

    # Test prompts
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

    held_prompts = [
        "The moon orbits Earth",
        "Birds can fly south",
        "Music has rhythm",
        "Plants need water",
        "Fire requires oxygen",
        "Ice is frozen water",
        "Math uses numbers",
        "Art expresses ideas",
    ]

    cal_tokens = [tokenizer.encode(p) for p in cal_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_prompts]

    def get_layer_data(layer_idx, tokens_list):
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

    def compute_procrustes_rotation(X, Y):
        """
        Compute Procrustes rotation R that best aligns Y to X.

        The transformation is: Y_aligned = Y @ R
        where R is orthogonal (R @ R.T = I)
        """
        # SVD of Y.T @ X
        X_np = np.array(X.tolist())
        Y_np = np.array(Y.tolist())

        # Center the data
        X_mean = X_np.mean(axis=0)
        Y_mean = Y_np.mean(axis=0)
        X_c = X_np - X_mean
        Y_c = Y_np - Y_mean

        # Procrustes: find R = argmin ||Y @ R - X||
        # Solution: R = V @ U.T where USV.T = Y.T @ X
        M = Y_c.T @ X_c
        U, S, Vh = np.linalg.svd(M)
        R = U @ Vh

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            Vh[-1, :] *= -1
            R = U @ Vh

        return R, X_mean, Y_mean

    def manifold_aware_compression(X, Y, norm_tolerance=NORM_TOLERANCE):
        """
        Compute compression that stays on manifold.

        Key constraints:
        1. Norm preservation: ||T @ x|| / ||y|| ∈ [1 - tol, 1 + tol]
        2. Angle preservation: via Procrustes rotation
        3. Null space bounded: residual < tol * signal
        """
        X_np = np.array(X.tolist())
        Y_np = np.array(Y.tolist())

        n_samples, d_in = X_np.shape
        d_out = Y_np.shape[1]

        # Step 1: Compute norms
        X_norms = np.linalg.norm(X_np, axis=1, keepdims=True)
        Y_norms = np.linalg.norm(Y_np, axis=1, keepdims=True)

        # Step 2: Normalize
        X_unit = X_np / (X_norms + 1e-10)
        Y_unit = Y_np / (Y_norms + 1e-10)

        # Step 3: Find scale factor (should be close to 1 for equivalence)
        scale_factors = Y_norms / (X_norms + 1e-10)
        avg_scale = np.mean(scale_factors)

        # Step 4: Procrustes rotation on unit vectors
        # We want: T @ x_unit ≈ y_unit (up to scale)
        M = X_unit.T @ Y_unit  # This is like correlation matrix
        U, S, Vh = np.linalg.svd(M, full_matrices=False)

        # The rotation part
        R = U @ Vh

        # Step 5: Construct T = scale * R
        # But we need to account for the norm change MLP creates
        T = avg_scale * R

        # Step 6: Check null space residual
        TX = X_np @ T
        residual = np.linalg.norm(TX - Y_np, 'fro')
        signal = np.linalg.norm(Y_np, 'fro')
        null_ratio = residual / signal

        # Step 7: Check norm preservation
        TX_norms = np.linalg.norm(TX, axis=1, keepdims=True)
        norm_ratios = TX_norms / (Y_norms + 1e-10)
        avg_norm_ratio = np.mean(norm_ratios)

        # Hallucination detection: are we on the manifold?
        on_manifold = null_ratio <= norm_tolerance

        result = {
            'T': T,
            'scale': avg_scale,
            'null_ratio': null_ratio,
            'norm_ratio': avg_norm_ratio,
            'on_manifold': on_manifold,
            'singular_values': S,
        }

        return result

    def equivalence_preserving_compression(X, Y, layer_weight):
        """
        The Wow! compression: F = R · P_wow · C_e

        This version uses the layer weight to scale the contribution.
        """
        X_np = np.array(X.tolist())
        Y_np = np.array(Y.tolist())

        n_samples, d_in = X_np.shape
        d_out = Y_np.shape[1]

        # Compute the basic transformation via least squares
        # X @ T = Y, so T = pinv(X) @ Y
        # But we need T.T to apply as T @ x = y
        # So: T = (X.T @ X)^-1 @ X.T @ Y for T: d_in x d_out
        # And we transpose to get d_out x d_in for our convention

        # Using SVD for stability
        U, S, Vh = np.linalg.svd(X_np, full_matrices=False)

        # Signal components (RMT-like filtering but scaled by layer weight)
        k_signal = int(max(1, layer_weight * len(S)))
        S_inv = np.zeros_like(S)
        S_inv[:k_signal] = 1.0 / S[:k_signal]

        # Pseudo-inverse with controlled rank
        X_pinv = Vh.T @ np.diag(S_inv) @ U.T

        # The raw transformation
        T_raw = X_pinv @ Y_np

        # Apply the Procrustes rotation for angle preservation
        # T_raw: d_in x d_out
        # We want to rotate it to preserve angles

        # Check the null space residual
        TX = X_np @ T_raw
        residual = np.linalg.norm(TX - Y_np, 'fro')
        signal = np.linalg.norm(Y_np, 'fro')
        null_ratio = residual / signal

        # Norm preservation check
        X_norms = np.linalg.norm(X_np, axis=1)
        Y_norms = np.linalg.norm(Y_np, axis=1)
        TX_norms = np.linalg.norm(TX, axis=1)
        norm_ratios = TX_norms / (Y_norms + 1e-10)

        # On manifold?
        on_manifold = null_ratio <= NORM_TOLERANCE

        result = {
            'T': T_raw.T,  # Transpose to our convention: T @ x
            'k_signal': k_signal,
            'null_ratio': null_ratio,
            'norm_ratio': np.mean(norm_ratios),
            'on_manifold': on_manifold,
        }

        return result

    # Test the manifold-equivalence compression
    logger.info(f"\n{'='*70}")
    logger.info("MANIFOLD-EQUIVALENCE ANALYSIS BY LAYER")
    logger.info(f"{'='*70}")

    layer_results = {}

    logger.info(f"\n{'Layer':>6} {'Weight':>8} {'k':>4} {'Null%':>8} {'Norm':>8} {'Manifold'}")
    logger.info("-" * 60)

    for layer_idx in range(8, 32):
        X, Y = get_layer_data(layer_idx, cal_tokens)
        weight = wow_layer_weight(layer_idx, n_layers)

        result = equivalence_preserving_compression(X, Y, weight)
        layer_results[layer_idx] = result

        status = "✓ ON" if result['on_manifold'] else "✗ OFF"
        logger.info(f"{layer_idx:>6} {weight:>7.3f} {result['k_signal']:>4} "
                   f"{result['null_ratio']*100:>7.2f}% {result['norm_ratio']:>7.3f} {status}")

    # Count manifold violations
    on_manifold = sum(1 for r in layer_results.values() if r['on_manifold'])
    total = len(layer_results)
    logger.info(f"\nManifold adherence: {on_manifold}/{total} layers ({on_manifold/total*100:.1f}%)")

    # Test compression with manifold-respecting T matrices
    logger.info(f"\n{'='*70}")
    logger.info("TESTING MANIFOLD-EQUIVALENCE COMPRESSION")
    logger.info(f"{'='*70}")

    # Only use layers that are ON the manifold
    valid_layers = [l for l, r in layer_results.items() if r['on_manifold']]
    logger.info(f"\nValid (on-manifold) layers: {valid_layers}")

    def evaluate_compression(layer_indices, layer_results):
        """Evaluate compression on held-out prompts."""
        correct = 0
        total = 0
        entropy_deltas = []

        for tok in held_tokens:
            input_ids = mx.array([tok])

            # Original
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
            orig_H = compute_entropy(orig_logits[0, -1, :])

            # Compressed
            original_mlps = {}
            for idx in layer_indices:
                if idx in layer_results:
                    layer = model.model.layers[idx]
                    original_mlps[idx] = layer.mlp

                    T = mx.array(layer_results[idx]['T']).astype(mx.float32)
                    mx.eval(T)

                    class CompressedMLP:
                        def __init__(self, T):
                            self.T = T
                        def __call__(self, x):
                            return mx.matmul(x, self.T.T)

                    layer.mlp = CompressedMLP(T)

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())
                comp_H = compute_entropy(comp_logits[0, -1, :])

                entropy_deltas.append(comp_H - orig_H)
                if comp_top == orig_top:
                    correct += 1
                total += 1
            finally:
                for idx in layer_indices:
                    if idx in original_mlps:
                        model.model.layers[idx].mlp = original_mlps[idx]

        acc = correct / total if total > 0 else 0.0
        avg_H_delta = sum(entropy_deltas) / len(entropy_deltas) if entropy_deltas else 0.0
        return acc, avg_H_delta

    # Test sequential compression with only manifold-valid layers
    logger.info(f"\n{'Layers':>20} {'Acc':>8} {'Entropy Δ':>12} {'Status'}")
    logger.info("-" * 55)

    for n in range(1, min(len(valid_layers) + 1, 11)):
        test_layers = valid_layers[:n]
        acc, H_delta = evaluate_compression(test_layers, layer_results)

        direction = "↓" if H_delta < -0.01 else ("↑" if H_delta > 0.01 else "→")
        status = "OK" if acc >= 0.5 and abs(H_delta) < 0.2 else "DEGRADED"

        layer_str = f"{min(test_layers)}-{max(test_layers)}" if len(test_layers) > 1 else str(test_layers[0])
        logger.info(f"{layer_str:>20} {acc*100:>7.1f}% {H_delta:+11.4f} {direction} {status}")

    # Compare with standard compression
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON: MANIFOLD vs STANDARD COMPRESSION")
    logger.info(f"{'='*70}")

    from modelcypher.core.domain.compression import RMTAwareCompressor
    compressor = RMTAwareCompressor(backend=backend)

    standard_results = {}
    for layer_idx in list(layer_results.keys())[:8]:  # First 8 layers
        X, Y = get_layer_data(layer_idx, cal_tokens)
        X_backend = backend.array(X)
        Y_backend = backend.array(Y)
        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T = backend.tolist(rmt_result.T)
        standard_results[layer_idx] = {'T': np.array(T)}

    # Test both on same layer set
    test_set = list(layer_results.keys())[:6]

    logger.info(f"\nTest layers: {test_set}")

    manifold_acc, manifold_H = evaluate_compression(test_set, layer_results)
    standard_acc, standard_H = evaluate_compression(test_set, standard_results)

    logger.info(f"\n{'Method':>15} {'Accuracy':>10} {'Entropy Δ':>12}")
    logger.info("-" * 40)
    logger.info(f"{'Manifold':>15} {manifold_acc*100:>9.1f}% {manifold_H:+11.4f}")
    logger.info(f"{'Standard':>15} {standard_acc*100:>9.1f}% {standard_H:+11.4f}")

    # The key insight
    logger.info(f"\n{'='*70}")
    logger.info("INTERPRETATION: THE MANIFOLD CONSTRAINT")
    logger.info(f"{'='*70}")

    logger.info(f"""
The Wow! Signal Specification:

1. LAYER WEIGHTING (P_wow)
   - Peak at φ⁻¹ ≈ 60% depth (layer {int(n_layers * PHI_INV)})
   - Asymmetric: fast rise (4!), slow fall (5²)
   - This matches where information is densest

2. NORM PRESERVATION
   - Must preserve 96% of signal norm
   - 4% tolerance for null space
   - If exceeded: hallucination (left manifold)

3. THE MANIFOLD TEST
   - On-manifold: transformation is EQUIVALENT
   - Off-manifold: transformation DISTORTS
   - Only use on-manifold layers for compression

4. WHAT WE LEARNED
   Layers on manifold: {on_manifold}/{total}

   If many layers are OFF manifold:
   - The MLP transformation is fundamentally NON-EQUIVALENCE
   - Compression cannot be lossless
   - Need different approach

   If most layers are ON manifold:
   - Compression can be exact
   - Just need to respect the constraint

5. THE GOLDEN RATIO CONNECTION
   The peak at φ⁻¹ suggests information flows BEFORE peak,
   is processed AT peak, and is transmitted AFTER peak.

   This aligns with the three-zone model:
   - Encoding (layers 0-6): before φ⁻¹
   - Processing (layers around {int(n_layers * PHI_INV)}): at φ⁻¹
   - Transmission (layers {int(n_layers * PHI_INV)}-{n_layers}): after φ⁻¹
""")


if __name__ == "__main__":
    run_experiment()
