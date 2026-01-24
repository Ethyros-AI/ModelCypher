#!/usr/bin/env python3
"""Experiment 32: Low-Rank Multi-Layer Compression.

Exp31 showed: projecting to k=1-4 components BEFORE fitting T gives:
- LOWER entropy (true compression!)
- HIGHER accuracy

Question: Does this prevent error compounding across multiple layers?

If the minimal subspace approach works:
- Each layer removes noise → errors don't accumulate
- Entropy should stay low even after many layers
- We might achieve TRUE lossless compression of 10+ layers

This is the key test for the compression breakthrough.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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


def run_experiment():
    """Test low-rank compression across multiple layers."""
    import mlx.core as mx
    import numpy as np

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

    compressor = RMTAwareCompressor(backend=backend)

    # Test layers in transmission zone
    test_layers = list(range(10, 28))

    # Find optimal k for each layer (from exp31 insights)
    # Use small k values since that worked best
    default_k = 4  # A reasonable starting point

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

    def compress_layer_lowrank(X, Y, k):
        """Compress with low-rank projection of outputs."""
        Y_np = np.array(Y.tolist())
        Y_mean = Y_np.mean(axis=0)
        U, S, Vh = np.linalg.svd(Y_np - Y_mean, full_matrices=False)

        # Project to top-k components
        actual_k = min(k, len(S))
        Vh_k = mx.array(Vh[:actual_k, :].T).astype(mx.float32)
        Y_mean_mx = mx.array(Y_mean).astype(mx.float32)
        mx.eval(Vh_k, Y_mean_mx)

        Y_centered = Y - Y_mean_mx
        Y_proj_k = mx.matmul(Y_centered, Vh_k)
        Y_proj = mx.matmul(Y_proj_k, Vh_k.T) + Y_mean_mx
        mx.eval(Y_proj)

        # Fit T to projected outputs
        X_backend = backend.array(X)
        Y_proj_backend = backend.array(Y_proj)
        rmt_result = compressor.compress_layer(X_backend, Y_proj_backend)
        T = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T)

        # Also store projection info for runtime
        return T, Vh_k, Y_mean_mx

    def compress_layer_standard(X, Y):
        """Standard full-rank compression."""
        X_backend = backend.array(X)
        Y_backend = backend.array(Y)
        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T)
        return T

    logger.info(f"\n{'='*70}")
    logger.info("MULTI-LAYER COMPRESSION: LOW-RANK vs STANDARD")
    logger.info(f"{'='*70}")

    # Pre-compress all layers with both methods
    lowrank_T = {}
    standard_T = {}

    logger.info("\nPre-compressing layers...")
    for layer_idx in test_layers:
        X, Y = get_layer_data(layer_idx, cal_tokens)

        # Low-rank compression (k=4)
        T_lr, Vh_k, Y_mean = compress_layer_lowrank(X, Y, k=default_k)
        lowrank_T[layer_idx] = T_lr

        # Standard compression
        T_std = compress_layer_standard(X, Y)
        standard_T[layer_idx] = T_std

        if layer_idx % 5 == 0:
            logger.info(f"  Compressed layer {layer_idx}")

    def evaluate_compression(layer_indices, T_dict, method_name):
        """Evaluate compression on held-out prompts."""
        correct = 0
        total = 0
        entropy_deltas = []

        for tok in held_tokens:
            input_ids = mx.array([tok])

            # Original model output
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())
            orig_H = compute_entropy(orig_logits[0, -1, :])

            # Compressed model output
            original_mlps = {}
            for idx in layer_indices:
                if idx in T_dict:
                    layer = model.model.layers[idx]
                    original_mlps[idx] = layer.mlp

                    class CompressedMLP:
                        def __init__(self, T):
                            self.T = T
                        def __call__(self, x):
                            return mx.matmul(x, self.T.T)

                    layer.mlp = CompressedMLP(T_dict[idx])

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
        avg_entropy_delta = sum(entropy_deltas) / len(entropy_deltas) if entropy_deltas else 0.0

        return acc, avg_entropy_delta

    # Test increasing numbers of sequential layers
    logger.info(f"\n{'='*70}")
    logger.info("SEQUENTIAL COMPRESSION: Error Accumulation Test")
    logger.info(f"{'='*70}")

    start_layer = 15

    logger.info(f"\n{'Layers':>10} | {'Low-Rank Acc':>12} {'LR Entropy Δ':>14} | {'Standard Acc':>12} {'Std Entropy Δ':>14}")
    logger.info("-" * 80)

    for n_layers in range(1, 11):
        layer_indices = list(range(start_layer, start_layer + n_layers))

        # Low-rank compression
        lr_acc, lr_H_delta = evaluate_compression(layer_indices, lowrank_T, "LowRank")

        # Standard compression
        std_acc, std_H_delta = evaluate_compression(layer_indices, standard_T, "Standard")

        lr_dir = "↓" if lr_H_delta < -0.01 else ("↑" if lr_H_delta > 0.01 else "→")
        std_dir = "↓" if std_H_delta < -0.01 else ("↑" if std_H_delta > 0.01 else "→")

        layer_str = f"{start_layer}-{start_layer + n_layers - 1}"
        logger.info(f"{layer_str:>10} | {lr_acc*100:10.1f}% {lr_H_delta:+12.4f} {lr_dir} | {std_acc*100:10.1f}% {std_H_delta:+12.4f} {std_dir}")

    # Test spread compression
    logger.info(f"\n{'='*70}")
    logger.info("SPREAD COMPRESSION: Low-Rank vs Standard")
    logger.info(f"{'='*70}")

    spread_patterns = [
        [10, 15, 20],
        [10, 15, 20, 25],
        [10, 13, 16, 19, 22, 25],
        [10, 12, 14, 16, 18, 20, 22, 24, 26],
    ]

    logger.info(f"\n{'Spread Pattern':>30} | {'LR Acc':>8} {'LR Δ':>10} | {'Std Acc':>8} {'Std Δ':>10}")
    logger.info("-" * 80)

    for pattern in spread_patterns:
        lr_acc, lr_H_delta = evaluate_compression(pattern, lowrank_T, "LowRank")
        std_acc, std_H_delta = evaluate_compression(pattern, standard_T, "Standard")

        pattern_str = str(pattern)[:30]
        lr_dir = "↓" if lr_H_delta < -0.01 else ("↑" if lr_H_delta > 0.01 else "→")
        std_dir = "↓" if std_H_delta < -0.01 else ("↑" if std_H_delta > 0.01 else "→")

        logger.info(f"{pattern_str:>30} | {lr_acc*100:6.1f}% {lr_H_delta:+8.4f}{lr_dir} | {std_acc*100:6.1f}% {std_H_delta:+8.4f}{std_dir}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*70}")

    logger.info("""
The key comparison:

1. SEQUENTIAL COMPRESSION
   - Standard: entropy increases, accuracy drops with more layers
   - Low-rank: entropy should stay lower, accuracy should drop slower

2. THE HYPOTHESIS
   If low-rank projection removes noise (not signal):
   - Errors don't compound because noise is filtered out
   - Each layer's essential subspace is preserved
   - TRUE compression = fewer bits with same semantics

3. THE MECHANISM
   - Full-rank T preserves noise → noise accumulates → entropy rises
   - Low-rank T removes noise → only signal propagates → entropy stable

4. IMPLICATIONS
   If low-rank wins:
   - We've found the "real" dimensionality of MLP computation
   - Most MLP output is noise/redundancy
   - Compression should TARGET the essential subspace

5. NEXT STEPS
   If this works:
   - Find optimal k per layer automatically
   - Test if k correlates with layer function (encoding/transmission/decoding)
   - Verify on more diverse prompts
""")


if __name__ == "__main__":
    run_experiment()
