#!/usr/bin/env python3
"""Experiment 33: Adaptive Rank Selection Per Layer.

Exp32 showed: fixed k=4 doesn't prevent error compounding.
Exp31 showed: optimal k varies by layer (1-11 depending on layer).

Question: Can we find optimal k per layer and use it for multi-layer compression?

The hypothesis: Each layer has a specific "essential dimensionality" k*.
Using k* for each layer should:
1. Remove noise (entropy decreases)
2. Preserve signal (accuracy maintained)
3. Prevent error compounding (errors don't accumulate)
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
    """Find optimal k per layer and test multi-layer compression."""
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

    # Test layers
    test_layers = list(range(10, 26))

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

    def compress_layer_lowrank(X, Y, k, backend, compressor):
        """Compress with low-rank projection of outputs."""
        Y_np = np.array(Y.tolist())
        Y_mean = Y_np.mean(axis=0)
        U, S, Vh = np.linalg.svd(Y_np - Y_mean, full_matrices=False)

        # Variance explained
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var

        # Project to top-k components
        actual_k = min(k, len(S))
        var_explained = cumvar[actual_k - 1] if actual_k > 0 else 0

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

        return T, var_explained

    def evaluate_single_layer(layer_idx, T):
        """Evaluate single layer compression."""
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
            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp

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
                layer.mlp = original_mlp

        acc = correct / total if total > 0 else 0.0
        avg_H_delta = sum(entropy_deltas) / len(entropy_deltas) if entropy_deltas else 0.0
        return acc, avg_H_delta

    # PHASE 1: Find optimal k for each layer
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: FINDING OPTIMAL k PER LAYER")
    logger.info(f"{'='*70}")

    optimal_k = {}
    layer_T = {}

    for layer_idx in test_layers:
        logger.info(f"\n--- Layer {layer_idx} ---")
        X, Y = get_layer_data(layer_idx, cal_tokens)

        best_k = None
        best_score = -float('inf')
        best_T = None
        best_result = None

        logger.info(f"{'k':>3} {'Var%':>8} {'Acc':>8} {'Entropy Δ':>12}")
        logger.info("-" * 40)

        for k in range(1, 16):
            T, var_explained = compress_layer_lowrank(X, Y, k, backend, compressor)
            acc, H_delta = evaluate_single_layer(layer_idx, T)

            # Score: prioritize accuracy, then entropy reduction
            score = acc - abs(H_delta) * 0.1

            if score > best_score:
                best_score = score
                best_k = k
                best_T = T
                best_result = (acc, H_delta, var_explained)

            direction = "↓" if H_delta < -0.01 else ("↑" if H_delta > 0.01 else "→")
            note = "← BEST" if k == best_k and score == best_score else ""

            logger.info(f"{k:3d} {var_explained*100:7.1f}% {acc*100:7.1f}% {H_delta:+11.4f} {direction} {note}")

        optimal_k[layer_idx] = best_k
        layer_T[layer_idx] = best_T

        acc, H_delta, var_explained = best_result
        logger.info(f"\n>>> Layer {layer_idx}: optimal k = {best_k} "
                   f"(acc={acc*100:.1f}%, ΔH={H_delta:+.4f}, var={var_explained*100:.1f}%)")

    # Summary of optimal k values
    logger.info(f"\n{'='*70}")
    logger.info("OPTIMAL k VALUES BY LAYER")
    logger.info(f"{'='*70}")

    logger.info(f"\n{'Layer':>6} {'k*':>4} {'Description'}")
    logger.info("-" * 30)
    for layer_idx in test_layers:
        k = optimal_k[layer_idx]
        desc = ""
        if k <= 2:
            desc = "Very low-dim (gate-like?)"
        elif k <= 5:
            desc = "Low-dim essential"
        elif k <= 10:
            desc = "Medium-dim"
        else:
            desc = "High-dim (needs more)"
        logger.info(f"{layer_idx:>6} {k:>4}  {desc}")

    # PHASE 2: Test multi-layer compression with optimal k values
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: MULTI-LAYER WITH ADAPTIVE k")
    logger.info(f"{'='*70}")

    def evaluate_multilayer(layer_indices, T_dict):
        """Evaluate multi-layer compression."""
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
        avg_H_delta = sum(entropy_deltas) / len(entropy_deltas) if entropy_deltas else 0.0
        return acc, avg_H_delta

    # Test sequential compression with adaptive k
    start_layer = 12

    logger.info(f"\n{'Layers':>12} {'Acc':>8} {'Entropy Δ':>12} {'Avg k':>8}")
    logger.info("-" * 50)

    for n_layers in range(1, 11):
        layer_indices = list(range(start_layer, min(start_layer + n_layers, 26)))
        layer_indices = [l for l in layer_indices if l in layer_T]

        if not layer_indices:
            continue

        acc, H_delta = evaluate_multilayer(layer_indices, layer_T)
        avg_k = sum(optimal_k.get(l, 4) for l in layer_indices) / len(layer_indices)

        direction = "↓" if H_delta < -0.01 else ("↑" if H_delta > 0.01 else "→")
        layer_str = f"{min(layer_indices)}-{max(layer_indices)}"

        logger.info(f"{layer_str:>12} {acc*100:7.1f}% {H_delta:+11.4f} {direction} {avg_k:7.1f}")

    # Test spread compression with adaptive k
    logger.info(f"\n{'='*70}")
    logger.info("SPREAD COMPRESSION WITH ADAPTIVE k")
    logger.info(f"{'='*70}")

    spread_patterns = [
        [10, 15, 20, 25],
        [10, 13, 16, 19, 22, 25],
        [10, 12, 14, 16, 18, 20, 22, 24],
    ]

    logger.info(f"\n{'Pattern':>35} {'Acc':>8} {'Entropy Δ':>12}")
    logger.info("-" * 60)

    for pattern in spread_patterns:
        valid_layers = [l for l in pattern if l in layer_T]
        if not valid_layers:
            continue

        acc, H_delta = evaluate_multilayer(valid_layers, layer_T)
        direction = "↓" if H_delta < -0.01 else ("↑" if H_delta > 0.01 else "→")

        pattern_str = str(valid_layers)[:35]
        logger.info(f"{pattern_str:>35} {acc*100:7.1f}% {H_delta:+11.4f} {direction}")

    # Analysis
    logger.info(f"\n{'='*70}")
    logger.info("ANALYSIS: DOES ADAPTIVE k PREVENT ERROR COMPOUNDING?")
    logger.info(f"{'='*70}")

    logger.info("""
Key observations:

1. OPTIMAL k DISTRIBUTION
   - Gate layers should have k ≈ 1 (one essential direction)
   - Transmission layers should have k ≈ 2-5 (a few directions)
   - Encoding layers should have k > 5 (spreading information)

2. MULTI-LAYER BEHAVIOR
   - If adaptive k prevents compounding: entropy stays stable
   - If it doesn't: entropy still grows with layers

3. THE FUNDAMENTAL QUESTION
   Is compression failure due to:
   a) Keeping too much noise (k too high)?
   b) Losing essential signal (k too low)?
   c) Something else entirely (alignment, rotation)?

4. NEXT STEP
   If adaptive k STILL compounds errors:
   - The problem isn't noise removal
   - It's ALIGNMENT between compressed layers
   - Each T needs to match the next layer's input distribution
""")


if __name__ == "__main__":
    run_experiment()
