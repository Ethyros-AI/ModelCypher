#!/usr/bin/env python3
"""Experiment 11: Test Gromov-Wasserstein distance as predictor for layer combinations.

Hypothesis: Layers with similar metric structure (low GW distance) can be
combined safely. GW distance might predict which combinations preserve ranking.

Tests:
1. Compute GW distance between layer activation patterns
2. Correlate GW distance with combination success
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
from dataclasses import dataclass
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LayerPairResult:
    layer_i: int
    layer_j: int
    gw_distance: float
    combined_accuracy: float


def run_experiment():
    """Test GW distance as predictor for layer combination success."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.gromov_wasserstein import (
        GromovWassersteinDistance,
    )
    from modelcypher.core.domain.geometry.rmt_signal_separation import (
        compute_signal_rank_from_singular_values,
    )

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Prompts
    calibration_prompts = [
        "The capital of France is",
        "In mathematics, the derivative of",
        "The largest planet in our solar system is",
        "Water freezes at",
        "The speed of light is approximately",
        "Photosynthesis is the process by which",
        "The human heart has",
        "DNA stands for",
        "The chemical symbol for gold is",
        "Shakespeare wrote",
        "The Great Wall of China was built",
        "E = mc² was discovered by",
        "The mitochondria is",
        "Python is a programming language that",
        "Machine learning algorithms",
        "The stock market",
        "Climate change refers to",
        "Quantum mechanics describes",
        "The Renaissance was a period",
        "Artificial intelligence",
    ]

    held_out_prompts = [
        "The theory of relativity states",
        "Neurons in the brain",
        "The periodic table",
        "Evolution by natural selection",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    # Test layers - use a subset for speed
    test_layers = [1, 2, 5, 6, 7, 10]

    # Step 1: Collect activations for all layers
    logger.info("\nCollecting activations for all layers...")
    layer_activations = {}

    for layer_idx in test_layers:
        cal_inputs = []

        for tokens in cal_tokens:
            input_ids = mx.array([tokens])
            mlp_input = None

            layer = model.model.layers[layer_idx]
            original_mlp = layer.mlp

            class MLPHook:
                def __init__(self, mlp):
                    self.mlp = mlp
                def __call__(self, x):
                    nonlocal mlp_input
                    mlp_input = x
                    return self.mlp(x)

            layer.mlp = MLPHook(original_mlp)
            try:
                _ = model(input_ids)
                mx.eval(mlp_input)
                cal_inputs.append(mlp_input[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X = mx.stack(cal_inputs).astype(mx.float32)
        mx.eval(X)
        layer_activations[layer_idx] = backend.array(X)
        logger.info(f"  Layer {layer_idx}: {X.shape}")

    # Step 2: Compute T matrices for each layer (RMT-aware)
    logger.info("\nComputing T matrices...")
    layer_T = {}

    for layer_idx in test_layers:
        cal_inputs = []
        cal_outputs = []

        for tokens in cal_tokens:
            input_ids = mx.array([tokens])
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
                cal_inputs.append(mlp_input[0, -1, :])
                cal_outputs.append(mlp_output[0, -1, :])
            finally:
                layer.mlp = original_mlp

        X_cal = mx.stack(cal_inputs).astype(mx.float32)
        Y_cal = mx.stack(cal_outputs).astype(mx.float32)
        mx.eval(X_cal, Y_cal)

        n_samples, d_in = X_cal.shape

        # RMT-aware pinv
        U, S, Vt = mx.linalg.svd(X_cal, stream=mx.cpu)
        mx.eval(U, S, Vt)

        S_backend = backend.array(S)
        mp_result = compute_signal_rank_from_singular_values(
            S_backend, n_samples=n_samples, n_features=d_in, backend=backend
        )
        signal_rank = max(1, min(int(mp_result.signal_rank), int(S.shape[0])))

        eps = 1e-6
        k = int(S.shape[0])
        Vt_k = Vt[:k, :]

        # Use signal_rank for truncation
        U_sr = U[:, :signal_rank]
        S_sr = S[:signal_rank]
        Vt_sr = Vt[:signal_rank, :]
        S_inv = 1.0 / (S_sr + eps)
        V_sr = Vt_sr.T
        VS_sr = V_sr * S_inv
        pinv_rmt = mx.matmul(VS_sr, U_sr.T)
        T = mx.matmul(pinv_rmt, Y_cal).T
        mx.eval(T)

        layer_T[layer_idx] = T
        logger.info(f"  Layer {layer_idx}: T shape={T.shape}, signal_rank={signal_rank}")

    # Step 3: Compute GW distance for each pair
    logger.info("\nComputing GW distances...")
    gw_distances = {}

    gw_solver = GromovWassersteinDistance(backend=backend)

    def compute_pairwise_distances(X):
        """Compute pairwise Euclidean distance matrix."""
        # X is [n, d]
        # D[i,j] = ||X[i] - X[j]||
        n = X.shape[0]
        X_sq = backend.sum(X * X, axis=1)  # [n]
        # D^2 = X_sq + X_sq.T - 2 * X @ X.T
        XXT = backend.matmul(X, backend.transpose(X))
        D_sq = X_sq[:, None] + X_sq[None, :] - 2 * XXT
        # Clamp to avoid negative due to numerical errors
        D_sq = backend.maximum(D_sq, backend.zeros_like(D_sq))
        D = backend.sqrt(D_sq)
        backend.eval(D)
        return D

    for i, j in combinations(test_layers, 2):
        X_i = layer_activations[i]
        X_j = layer_activations[j]

        # Compute pairwise distance matrices for each layer
        # GW compares the "shape" of the point clouds
        try:
            D_i = compute_pairwise_distances(X_i)
            D_j = compute_pairwise_distances(X_j)
            result = gw_solver.compute(D_i, D_j)
            gw_distances[(i, j)] = result.distance
            logger.info(f"  GW({i}, {j}) = {result.distance:.4f} (converged={result.converged})")
        except Exception as e:
            logger.warning(f"  GW({i}, {j}) failed: {e}")
            gw_distances[(i, j)] = float('inf')

    # Step 4: Test combinations and correlate with GW
    logger.info("\nTesting layer combinations...")
    results = []

    for i, j in combinations(test_layers, 2):
        # Evaluate combined compression
        correct = 0
        total = 0

        for tokens in held_tokens:
            input_ids = mx.array([tokens])

            # Original logits
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # Apply compression to both layers i and j
            layer_i = model.model.layers[i]
            layer_j = model.model.layers[j]
            original_mlp_i = layer_i.mlp
            original_mlp_j = layer_j.mlp

            class CompressedMLP:
                def __init__(self, T):
                    self.T = T
                def __call__(self, x):
                    return mx.matmul(x, self.T.T)

            layer_i.mlp = CompressedMLP(layer_T[i])
            layer_j.mlp = CompressedMLP(layer_T[j])

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())

                if comp_top == orig_top:
                    correct += 1
                total += 1
            finally:
                layer_i.mlp = original_mlp_i
                layer_j.mlp = original_mlp_j

        accuracy = correct / total if total > 0 else 0.0
        gw_dist = gw_distances.get((i, j), float('inf'))

        results.append(LayerPairResult(
            layer_i=i,
            layer_j=j,
            gw_distance=gw_dist,
            combined_accuracy=accuracy,
        ))

        logger.info(f"  ({i}, {j}): GW={gw_dist:.4f}, accuracy={accuracy:.1%}")

    # Step 5: Correlation analysis
    logger.info(f"\n{'='*60}")
    logger.info("CORRELATION ANALYSIS")
    logger.info(f"{'='*60}")

    valid_results = [r for r in results if r.gw_distance != float('inf')]

    if len(valid_results) >= 3:
        gw_values = [r.gw_distance for r in valid_results]
        acc_values = [r.combined_accuracy for r in valid_results]

        # Simple correlation
        n = len(gw_values)
        mean_gw = sum(gw_values) / n
        mean_acc = sum(acc_values) / n

        cov = sum((g - mean_gw) * (a - mean_acc) for g, a in zip(gw_values, acc_values)) / n
        std_gw = (sum((g - mean_gw) ** 2 for g in gw_values) / n) ** 0.5
        std_acc = (sum((a - mean_acc) ** 2 for a in acc_values) / n) ** 0.5

        if std_gw > 0 and std_acc > 0:
            correlation = cov / (std_gw * std_acc)
            logger.info(f"\nCorrelation(GW, Accuracy) = {correlation:.4f}")

            if correlation < -0.5:
                logger.info(">>> HIGH NEGATIVE CORRELATION: Lower GW = Better accuracy")
                logger.info(">>> GW DISTANCE PREDICTS COMBINATION SUCCESS")
            elif correlation > 0.5:
                logger.info(">>> HIGH POSITIVE CORRELATION: Higher GW = Better accuracy (unexpected)")
            else:
                logger.info(">>> WEAK CORRELATION: GW does not predict success well")
        else:
            logger.info("Cannot compute correlation (zero variance)")

    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Layers':<12} {'GW Distance':<15} {'Accuracy':<10}")
    logger.info("-" * 40)

    for r in sorted(results, key=lambda x: x.gw_distance):
        logger.info(f"({r.layer_i}, {r.layer_j})    {r.gw_distance:>10.4f}     {r.combined_accuracy:>6.1%}")


if __name__ == "__main__":
    run_experiment()
