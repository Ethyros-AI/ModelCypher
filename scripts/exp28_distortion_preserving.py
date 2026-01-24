#!/usr/bin/env python3
"""Experiment 28: Distortion-Preserving Compression.

The problem: Standard compression (T = Y @ pinv(X)) smooths out the
intentional non-isometric distortion that MLPs perform.

New approach: Instead of minimizing ||Y - TX||, we should preserve
the PATTERN of distortion - which pairs get closer, which get farther.

Method:
1. Compute distance matrix D_in for inputs X
2. Compute distance matrix D_out for outputs Y
3. Compute distortion matrix: R = D_out / D_in (ratio of distances)
4. Find T that makes T@X have the SAME distortion pattern as Y

This is like Procrustes but for the metric tensor, not the points.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import math

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_distance_matrix(points):
    """Compute matrix of pairwise Euclidean distances."""
    import mlx.core as mx

    n = points.shape[0]
    # Efficient computation using broadcasting
    diff = points[:, None, :] - points[None, :, :]  # (n, n, d)
    distances = mx.sqrt(mx.sum(diff * diff, axis=2))
    mx.eval(distances)
    return distances


def compute_distortion_matrix(D_in, D_out):
    """Compute ratio of output to input distances."""
    import mlx.core as mx

    # Avoid division by zero
    R = D_out / (D_in + 1e-10)
    mx.eval(R)
    return R


def distortion_loss(R_pred, R_target):
    """Compute how well predicted distortion matches target distortion."""
    import mlx.core as mx

    # We want the RATIOS to be the same, not the absolute values
    # Use log ratio to make it symmetric: log(R_pred/R_target) should be 0
    log_ratio = mx.log(R_pred + 1e-10) - mx.log(R_target + 1e-10)

    # Exclude diagonal (self-distance = 0)
    n = R_pred.shape[0]
    mask = 1 - mx.eye(n)

    loss = mx.mean((log_ratio * mask) ** 2)
    mx.eval(loss)
    return float(loss.item())


def run_experiment():
    """Test distortion-preserving compression."""
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

    n_layers = len(model.model.layers)

    # Calibration prompts
    prompts = [
        "The capital of France is",
        "Water freezes at zero degrees",
        "The largest planet is Jupiter",
        "DNA stands for deoxyribonucleic acid",
        "The speed of light is fast",
        "Photosynthesis occurs in plants",
        "The periodic table organizes elements",
        "Machine learning uses algorithms",
        "The theory of relativity was proposed",
        "Quantum mechanics describes particles",
        "Shakespeare wrote many plays",
        "The human brain has neurons",
        "Evolution explains species change",
        "Gravity attracts masses together",
        "The internet connects computers",
        "Vaccines prevent diseases",
    ]

    tokens = [tokenizer.encode(p) for p in prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Test on layer 15
    layer_idx = 15

    logger.info(f"\n{'='*60}")
    logger.info(f"LAYER {layer_idx}: DISTORTION ANALYSIS")
    logger.info(f"{'='*60}")

    # Collect activations
    inputs = []
    outputs = []

    for tok in tokens:
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

    # Compute distance matrices
    D_in = compute_distance_matrix(X)
    D_out = compute_distance_matrix(Y)

    # Compute true distortion pattern
    R_true = compute_distortion_matrix(D_in, D_out)

    logger.info(f"\n--- Original MLP Distortion Pattern ---")

    # Analyze distortion
    R_np = np.array(R_true.tolist())
    n = R_np.shape[0]

    # Get upper triangle (excluding diagonal)
    ratios = []
    for i in range(n):
        for j in range(i+1, n):
            ratios.append(R_np[i, j])

    logger.info(f"Distortion ratio range: [{min(ratios):.3f}, {max(ratios):.3f}]")
    logger.info(f"Mean distortion: {sum(ratios)/len(ratios):.3f}")
    logger.info(f"Std distortion: {np.std(ratios):.3f}")

    # Count compression vs expansion
    compressions = sum(1 for r in ratios if r < 1)
    expansions = sum(1 for r in ratios if r > 1)
    logger.info(f"Compressions (r<1): {compressions}/{len(ratios)}")
    logger.info(f"Expansions (r>1): {expansions}/{len(ratios)}")

    # Standard compression: T = Y @ pinv(X)
    logger.info(f"\n--- Standard Compression (MSE) ---")

    X_backend = backend.array(X)
    Y_backend = backend.array(Y)

    rmt_result = compressor.compress_layer(X_backend, Y_backend)
    T_std = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
    mx.eval(T_std)

    Y_std = mx.matmul(X, T_std.T)
    mx.eval(Y_std)

    D_std = compute_distance_matrix(Y_std)
    R_std = compute_distortion_matrix(D_in, D_std)

    std_loss = distortion_loss(R_std, R_true)
    logger.info(f"Distortion preservation loss: {std_loss:.6f}")

    # How well does standard compression preserve the pattern?
    R_std_np = np.array(R_std.tolist())
    std_ratios = []
    for i in range(n):
        for j in range(i+1, n):
            std_ratios.append(R_std_np[i, j])

    logger.info(f"Compressed ratio range: [{min(std_ratios):.3f}, {max(std_ratios):.3f}]")
    logger.info(f"Mean: {sum(std_ratios)/len(std_ratios):.3f}")

    # Correlation of distortion patterns
    corr_num = sum((ratios[k] - np.mean(ratios)) * (std_ratios[k] - np.mean(std_ratios))
                   for k in range(len(ratios)))
    corr_den = math.sqrt(sum((r - np.mean(ratios))**2 for r in ratios) *
                         sum((r - np.mean(std_ratios))**2 for r in std_ratios))
    corr = corr_num / corr_den if corr_den > 0 else 0

    logger.info(f"Distortion pattern correlation: {corr:.4f}")

    # Try to find a BETTER T that preserves distortion
    logger.info(f"\n--- Distortion-Preserving Optimization ---")

    # Idea: Start from standard T, then adjust to match distortion pattern
    # Use gradient descent on distortion loss

    T = T_std * 1.0  # Copy
    mx.eval(T)

    lr = 0.001
    best_loss = std_loss
    best_T = T * 1.0

    for step in range(100):
        # Compute current distortion
        Y_curr = mx.matmul(X, T.T)
        D_curr = compute_distance_matrix(Y_curr)
        R_curr = compute_distortion_matrix(D_in, D_curr)

        curr_loss = distortion_loss(R_curr, R_true)

        if curr_loss < best_loss:
            best_loss = curr_loss
            best_T = T * 1.0
            mx.eval(best_T)

        # Numerical gradient (very slow but simple)
        if step % 20 == 0:
            logger.info(f"Step {step}: distortion_loss = {curr_loss:.6f}")

        if step < 50:  # Only do gradient steps for first 50
            # Perturb T in a random direction and see if it helps
            noise = mx.random.normal(T.shape) * 0.01
            T_try = T + noise
            mx.eval(T_try)

            Y_try = mx.matmul(X, T_try.T)
            D_try = compute_distance_matrix(Y_try)
            R_try = compute_distortion_matrix(D_in, D_try)
            try_loss = distortion_loss(R_try, R_true)

            if try_loss < curr_loss:
                T = T_try
                mx.eval(T)

    logger.info(f"\nBest distortion loss: {best_loss:.6f} (started at {std_loss:.6f})")
    improvement = (std_loss - best_loss) / std_loss * 100
    logger.info(f"Improvement: {improvement:.1f}%")

    # Check if improved T gives better accuracy
    logger.info(f"\n--- Accuracy Comparison ---")

    # Held-out prompts
    held_out = [
        "The moon orbits Earth",
        "Birds can fly south",
        "Chemistry studies matter",
        "Music has rhythm",
    ]
    held_tokens = [tokenizer.encode(p) for p in held_out]

    def test_accuracy(T_test, layer_idx):
        correct = 0
        total = 0

        for tok in held_tokens:
            input_ids = mx.array([tok])

            # Original
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # With compressed MLP
            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp

            class CompressedMLP:
                def __init__(self, T):
                    self.T = T
                def __call__(self, x):
                    return mx.matmul(x, self.T.T)

            layer.mlp = CompressedMLP(T_test)
            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())

                if comp_top == orig_top:
                    correct += 1
                total += 1
            finally:
                layer.mlp = original_mlp

        return correct / total if total > 0 else 0.0

    std_acc = test_accuracy(T_std, layer_idx)
    best_acc = test_accuracy(best_T, layer_idx)

    logger.info(f"Standard T accuracy: {std_acc:.1%}")
    logger.info(f"Distortion-preserving T accuracy: {best_acc:.1%}")

    # The key question
    logger.info(f"\n{'='*60}")
    logger.info("KEY INSIGHT")
    logger.info(f"{'='*60}")

    logger.info("""
The MLP performs SELECTIVE distortion:
- Some concept pairs get CLOSER (compression)
- Some concept pairs get FARTHER (expansion)

Standard compression (MSE) finds the average linear fit,
which SMOOTHS OUT this selective distortion.

The distortion pattern IS the computation.

To truly compress without losing information:
1. Identify WHICH pairs should compress vs expand
2. Preserve those RELATIONSHIPS, not absolute positions
3. This is a TOPOLOGICAL constraint, not Euclidean

The MLP is like a lens that focuses some things and blurs others.
Our compression is grinding the lens flat.
""")

    # Visualize distortion pattern
    logger.info(f"\n--- Sample Distortions ---")
    logger.info(f"{'Pair':<20} {'True R':<10} {'Comp R':<10} {'Error':<10}")
    logger.info("-" * 50)

    for i in range(min(5, n)):
        for j in range(i+1, min(6, n)):
            true_r = R_np[i, j]
            comp_r = R_std_np[i, j]
            err = abs(true_r - comp_r)
            logger.info(f"({i},{j}){'':<14} {true_r:<10.3f} {comp_r:<10.3f} {err:<10.3f}")


if __name__ == "__main__":
    run_experiment()
