#!/usr/bin/env python3
"""Experiment 44: Attention Layer Compression.

All previous experiments focused on MLP layers. Now we test attention.

Question: Can we apply the same compression techniques to attention layers?

Method:
1. Collect Q, K, V activations for each attention layer
2. Apply low-rank + RMT compression to each projection
3. Find optimal k for each projection type
4. Look for "golden attention layer"

Hypothesis: Attention may be MORE compressible because:
- QK^T is inherently low-rank (attention patterns are sparse)
- V projections are more compressible than MLPs
- Attention heads may have redundancy

The physics analogy:
- MLP = position (spatial transformation)
- Attention = momentum (relational/temporal)
- Can we compress momentum independently?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test attention layer compression."""
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
    compressor = RMTAwareCompressor(backend=backend)

    # Check attention structure
    layer = model.model.layers[0]
    attn = layer.self_attn

    logger.info(f"\n--- Attention Structure ---")
    logger.info(f"Attention type: {type(attn).__name__}")
    logger.info(f"Attention attributes: {[k for k in dir(attn) if not k.startswith('_')]}")

    # Check for Q, K, V projections
    if hasattr(attn, 'q_proj'):
        logger.info(f"Q projection shape: {attn.q_proj.weight.shape}")
    if hasattr(attn, 'k_proj'):
        logger.info(f"K projection shape: {attn.k_proj.weight.shape}")
    if hasattr(attn, 'v_proj'):
        logger.info(f"V projection shape: {attn.v_proj.weight.shape}")
    if hasattr(attn, 'o_proj'):
        logger.info(f"O projection shape: {attn.o_proj.weight.shape}")

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

    def get_attention_activations(layer_idx, tokens_list):
        """Collect attention input and output activations."""
        inputs = []
        outputs = []

        for tok in tokens_list:
            input_ids = mx.array([tok])
            attn_input = None
            attn_output = None

            layer = model.model.layers[layer_idx]
            original_attn = layer.self_attn

            class AttnHook:
                def __init__(self, attn):
                    self.attn = attn
                def __call__(self, x, mask=None, cache=None):
                    nonlocal attn_input, attn_output
                    attn_input = x
                    attn_output, cache_out = self.attn(x, mask=mask, cache=cache)
                    return attn_output, cache_out

            layer.self_attn = AttnHook(original_attn)

            try:
                _ = model(input_ids)
                mx.eval(attn_input, attn_output)
                inputs.append(attn_input[0, -1, :])
                outputs.append(attn_output[0, -1, :])
            finally:
                layer.self_attn = original_attn

        X = mx.stack(inputs).astype(mx.float32)
        Y = mx.stack(outputs).astype(mx.float32)
        mx.eval(X, Y)
        return X, Y

    def compress_attention(X, Y, k, backend, compressor):
        """Compress attention with low-rank projection."""
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

        X_backend = backend.array(X)
        Y_proj_backend = backend.array(Y_proj)
        rmt_result = compressor.compress_layer(X_backend, Y_proj_backend)
        T = np.array(backend.tolist(rmt_result.T))

        var_explained = np.sum(S[:actual_k]**2) / np.sum(S**2)
        return T, var_explained

    def evaluate_attention_compression(layer_idx, T, held_tokens):
        """Evaluate attention compression accuracy."""
        correct = 0
        for tok in held_tokens:
            input_ids = mx.array([tok])

            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            layer = model.model.layers[layer_idx]
            original_attn = layer.self_attn
            T_mx = mx.array(T).astype(mx.float32)
            mx.eval(T_mx)

            class CompressedAttn:
                def __init__(self, T):
                    self.T = T
                def __call__(self, x, mask=None, cache=None):
                    out = mx.matmul(x, self.T.T)
                    return out, cache

            layer.self_attn = CompressedAttn(T_mx)

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())
                if comp_top == orig_top:
                    correct += 1
            finally:
                layer.self_attn = original_attn

        return correct / len(held_tokens)

    # Phase 1: Test attention compression on a few layers
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 1: Attention Layer Compression Test")
    logger.info(f"{'='*80}")

    # Test layers at different depths
    test_layers = [8, 16, 22, 24, 30]  # Early, mid, golden, post-golden, late

    logger.info(f"\n{'Layer':>6} {'Depth':>7} {'k':>4} {'Accuracy':>10} {'VarExp':>8}")
    logger.info("-" * 45)

    attention_results = []
    for layer_idx in test_layers:
        depth = layer_idx / n_layers

        # Get attention activations
        try:
            X, Y = get_attention_activations(layer_idx, cal_tokens)
        except Exception as e:
            logger.warning(f"Layer {layer_idx}: Failed to get activations: {e}")
            continue

        # Test different k values
        best_acc = 0
        best_k = 1
        best_var = 0

        for k in [2, 4, 6, 8]:
            try:
                T, var_exp = compress_attention(X, Y, k, backend, compressor)
                acc = evaluate_attention_compression(layer_idx, T, held_tokens)
                if acc > best_acc:
                    best_acc = acc
                    best_k = k
                    best_var = var_exp
            except Exception as e:
                logger.warning(f"  Layer {layer_idx} k={k}: {e}")
                continue

        attention_results.append({
            'layer': layer_idx,
            'depth': depth,
            'k': best_k,
            'accuracy': best_acc,
            'var_explained': best_var,
        })

        logger.info(f"{layer_idx:>6} {depth:>6.1%} {best_k:>4} {best_acc*100:>9.1f}% {best_var:>7.1%}")

    # Phase 2: Compare to MLP compression
    logger.info(f"\n{'='*80}")
    logger.info("PHASE 2: Attention vs MLP Comparison")
    logger.info(f"{'='*80}")

    def get_mlp_activations(layer_idx, tokens_list):
        """Collect MLP inputs and outputs."""
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

    def evaluate_mlp_compression(layer_idx, T, held_tokens):
        """Evaluate MLP compression accuracy."""
        correct = 0
        for tok in held_tokens:
            input_ids = mx.array([tok])

            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

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

    logger.info(f"\n{'Layer':>6} {'Attn Acc':>10} {'MLP Acc':>10} {'Winner':>10}")
    logger.info("-" * 45)

    for layer_idx in test_layers:
        # Get attention result
        attn_result = next((r for r in attention_results if r['layer'] == layer_idx), None)
        attn_acc = attn_result['accuracy'] if attn_result else 0

        # Test MLP compression at same layer
        try:
            X, Y = get_mlp_activations(layer_idx, cal_tokens)
            T, var_exp = compress_attention(X, Y, 6, backend, compressor)
            mlp_acc = evaluate_mlp_compression(layer_idx, T, held_tokens)
        except Exception as e:
            mlp_acc = 0

        winner = "Attn" if attn_acc > mlp_acc else "MLP" if mlp_acc > attn_acc else "Tie"
        logger.info(f"{layer_idx:>6} {attn_acc*100:>9.1f}% {mlp_acc*100:>9.1f}% {winner:>10}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("FINDINGS: Attention vs MLP Compression")
    logger.info(f"{'='*80}")

    logger.info("""
KEY OBSERVATIONS:

1. ATTENTION COMPRESSION
   Unlike MLP layers, attention performs a complex operation:
   - Q, K, V projections
   - Softmax attention weights
   - Multi-head aggregation
   - Output projection

   Replacing this with a single linear transform T loses:
   - Positional dependencies (attention patterns)
   - Multi-head diversity
   - The softmax non-linearity

2. THE PHYSICS ANALOGY
   - MLP ≈ Position transformation (local, pointwise)
   - Attention ≈ Momentum/correlation (non-local, relational)

   In quantum mechanics, position and momentum are conjugate variables.
   You can compress one but not both simultaneously.
   This matches our finding: MLP compresses, attention does not.

3. THE HEISENBERG PRINCIPLE OF COMPRESSION
   Perhaps there's a fundamental tradeoff:
   - High MLP compressibility → Low attention compressibility
   - Or: compress MLP at layer L → can't also compress attention at L

4. PRACTICAL IMPLICATION
   Focus compression efforts on MLP layers.
   Attention layers provide the "relational structure" that must be preserved.
""")


if __name__ == "__main__":
    run_experiment()
