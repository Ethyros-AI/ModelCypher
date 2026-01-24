#!/usr/bin/env python3
"""Experiment 42: Cross-Architecture Golden Layers.

Finding from exp41: The geometry is uniform across layers.
The "golden layer" property is about POSITION, not geometry.

Question: Does every architecture have a golden layer at ~67% depth?

Method:
1. Test on LFM2-1.2B (different architecture)
2. Test on LFM2-700M (smaller scale)
3. For each, find the layer with max accuracy at optimal k
4. Check if depth ratio ≈ φ⁻¹ ≈ 0.618

Hypothesis: Golden layer depth is universal at ~67% (φ⁻¹).

If true, this would suggest the golden ratio reflects a fundamental
property of how transformers process information.
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


def test_model_golden_layer(model_path, model_name):
    """Find the golden layer for a given model."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {model_name}")
    logger.info(f"Path: {model_path}")
    logger.info(f"{'='*80}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    golden_depth = int(n_layers * PHI_INV)

    logger.info(f"Model has {n_layers} layers")
    logger.info(f"φ⁻¹ depth: layer {golden_depth} ({PHI_INV:.1%})")
    logger.info(f"2/3 depth: layer {int(n_layers * 0.67)} (67%)")

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
        "Clouds contain moisture",
        "Books store knowledge",
        "Trees produce oxygen",
        "Oceans cover Earth",
    ]

    cal_tokens = [tokenizer.encode(p) for p in cal_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    def get_mlp_attr(layer):
        """Get the MLP/feed_forward attribute name for this architecture."""
        if hasattr(layer, 'mlp'):
            return 'mlp', layer.mlp
        elif hasattr(layer, 'feed_forward'):
            return 'feed_forward', layer.feed_forward
        elif 'mlp' in layer:
            return 'mlp', layer['mlp']
        elif 'feed_forward' in layer:
            return 'feed_forward', layer['feed_forward']
        else:
            raise AttributeError(f"Layer has no MLP: {list(layer.keys()) if hasattr(layer, 'keys') else dir(layer)}")

    def set_mlp_attr(layer, attr_name, value):
        """Set the MLP attribute."""
        if hasattr(layer, attr_name):
            setattr(layer, attr_name, value)
        else:
            layer[attr_name] = value

    def get_layer_data(layer_idx, tokens_list):
        """Collect MLP inputs and outputs."""
        inputs = []
        outputs = []

        for tok in tokens_list:
            input_ids = mx.array([tok])
            mlp_input = None
            mlp_output = None

            layer = model.model.layers[layer_idx]
            mlp_attr, original_mlp = get_mlp_attr(layer)

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input, mlp_output
                    mlp_input = x
                    mlp_output = self.mlp(x)
                    return mlp_output

            set_mlp_attr(layer, mlp_attr, MLPHook(original_mlp))

            try:
                _ = model(input_ids)
                mx.eval(mlp_input, mlp_output)
                inputs.append(mlp_input[0, -1, :])
                outputs.append(mlp_output[0, -1, :])
            finally:
                set_mlp_attr(layer, mlp_attr, original_mlp)

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return X, Y

    def compress_and_test(layer_idx, k):
        """Compress a layer and test accuracy."""
        X, Y = get_layer_data(layer_idx, cal_tokens)

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
            mlp_attr, original_mlp = get_mlp_attr(layer)
            T_mx = mx.array(T).astype(mx.float32)
            mx.eval(T_mx)

            class CompressedMLP:
                def __init__(self, T):
                    self.T = T
                def __call__(self, x):
                    return mx.matmul(x, self.T.T)

            set_mlp_attr(layer, mlp_attr, CompressedMLP(T_mx))

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())
                if comp_top == orig_top:
                    correct += 1
            finally:
                set_mlp_attr(layer, mlp_attr, original_mlp)

        return correct / len(held_tokens)

    # Test ALL layers with k values 1-8
    logger.info(f"\n--- Testing all {n_layers} layers ---")
    logger.info(f"{'Layer':>6} {'Depth':>7} {'Best k':>7} {'Accuracy':>10}")
    logger.info("-" * 40)

    layer_results = []

    for layer_idx in range(n_layers):
        best_acc = 0
        best_k = 1

        for k in [2, 4, 6, 8]:  # Test a few k values
            try:
                acc = compress_and_test(layer_idx, k)
                if acc > best_acc:
                    best_acc = acc
                    best_k = k
            except Exception as e:
                logger.warning(f"  Layer {layer_idx} k={k}: {e}")
                continue

        depth = layer_idx / n_layers
        layer_results.append({
            'layer': layer_idx,
            'depth': depth,
            'best_k': best_k,
            'best_acc': best_acc,
        })

        marker = ""
        if layer_idx == golden_depth:
            marker = " (φ⁻¹)"
        if layer_idx == int(n_layers * 0.67):
            marker = " (2/3)"

        logger.info(f"{layer_idx:>6} {depth:>6.1%} {best_k:>7} {best_acc*100:>9.1f}%{marker}")

    # Find the best layer
    best_layer = max(layer_results, key=lambda r: r['best_acc'])
    logger.info(f"\n--- BEST LAYER ---")
    logger.info(f"  Layer: {best_layer['layer']}")
    logger.info(f"  Depth: {best_layer['depth']:.1%}")
    logger.info(f"  k: {best_layer['best_k']}")
    logger.info(f"  Accuracy: {best_layer['best_acc']*100:.1f}%")
    logger.info(f"  φ⁻¹ = {PHI_INV:.1%}, 2/3 = 66.7%")

    # Find all 100% layers
    perfect_layers = [r for r in layer_results if r['best_acc'] >= 0.99]
    high_acc_layers = [r for r in layer_results if r['best_acc'] >= 0.9]

    logger.info(f"\n  100% accuracy layers: {[r['layer'] for r in perfect_layers]}")
    logger.info(f"  90%+ accuracy layers: {[r['layer'] for r in high_acc_layers]}")

    return {
        'model_name': model_name,
        'n_layers': n_layers,
        'golden_depth': golden_depth,
        'best_layer': best_layer,
        'perfect_layers': perfect_layers,
        'high_acc_layers': high_acc_layers,
        'all_results': layer_results,
    }


def run_experiment():
    """Test golden layer hypothesis on multiple architectures."""

    models = [
        ("/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16", "LFM2-1.2B"),
        ("/Volumes/CodeCypher/models/mlx-community/LFM2-700M-bf16", "LFM2-700M"),
    ]

    results = []
    for model_path, model_name in models:
        try:
            result = test_model_golden_layer(model_path, model_name)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-ARCHITECTURE SUMMARY")
    logger.info(f"{'='*80}")

    logger.info(f"\n{'Model':<15} {'Layers':>7} {'Best':>6} {'Depth':>7} {'Acc':>8} {'φ⁻¹':>6}")
    logger.info("-" * 55)

    for r in results:
        best = r['best_layer']
        logger.info(f"{r['model_name']:<15} {r['n_layers']:>7} {best['layer']:>6} "
                   f"{best['depth']:>6.1%} {best['best_acc']*100:>7.1f}% "
                   f"{r['golden_depth']:>6}")

    # Compare to DeepSeek-R1 (from exp40)
    logger.info(f"\n{'DeepSeek-R1-8B':<15} {36:>7} {24:>6} {'66.7%':>7} {'100%':>8} {22:>6}")

    logger.info("""
FINDINGS:

1. DEPTH HYPOTHESIS
   If all models have their golden layer at ~67% depth, this suggests
   the golden ratio is a fundamental property of transformer processing.

2. ARCHITECTURE DEPENDENCE
   Different architectures may have different optimal depths.
   LFM2 is a different architecture than Qwen3.

3. SCALE DEPENDENCE
   Smaller models (700M vs 1.2B) may have different dynamics.

4. THE KEY QUESTION
   Is the golden layer property:
   a) Universal (always ~67% depth)
   b) Architecture-specific
   c) Scale-dependent
   d) Training-dependent
""")


if __name__ == "__main__":
    run_experiment()
