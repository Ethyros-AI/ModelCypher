#!/usr/bin/env python3
"""Experiment 18: Contiguous Range Analysis.

Tests why contiguous layers combine better than non-contiguous.

From Exp 1: Contiguous 7-12 = 100%, but non-contiguous {1,2,7,10,13,14} = 83.3%

Hypotheses:
1. Contiguous layers share activation subspaces → errors don't cascade
2. Adjacent layers have correlated error patterns → errors cancel
3. Non-contiguous skips "critical" layers that amplify errors

Method:
1. Test all contiguous ranges of size 2, 3, 4
2. Measure accuracy for each
3. Compare to randomly selected non-contiguous sets of same size
4. Analyze what makes successful ranges work
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test contiguous vs non-contiguous layer combinations."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.compression import RMTAwareCompressor

    initialize_default_backend()
    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    # Load model
    model_path = "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16"
    logger.info(f"Loading model: {model_path}")

    from mlx_lm import load
    model, tokenizer = load(model_path)

    n_layers = len(model.model.layers)
    logger.info(f"Model has {n_layers} layers")

    # Calibration prompts
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
        "The theory of evolution states",
        "Neurons in the brain",
        "The periodic table organizes",
        "Gravity is a force that",
    ]

    held_out_prompts = [
        "The theory of relativity states",
        "Neurons transmit signals by",
        "Chemical bonds form when",
        "Democracy originated in",
        "The internet was invented",
        "Vaccines work by",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Pre-compress all layers individually
    logger.info("\n--- Pre-compressing all layers ---")

    layer_T = {}  # layer_idx -> T matrix (MLX)

    for layer_idx in range(n_layers):
        logger.info(f"Compressing layer {layer_idx}...")

        # Collect activations
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

        X_backend = backend.array(X_cal)
        Y_backend = backend.array(Y_cal)

        rmt_result = compressor.compress_layer(X_backend, Y_backend)
        T_mx = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T_mx)
        layer_T[layer_idx] = T_mx

    def evaluate_layer_set(layer_indices):
        """Evaluate accuracy with a set of compressed layers."""
        correct = 0
        total = 0

        for tokens in held_tokens:
            input_ids = mx.array([tokens])

            # Get original prediction
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # Apply compression to specified layers
            original_mlps = {}
            for idx in layer_indices:
                layer = model.model.layers[idx]
                original_mlps[idx] = layer.mlp

                class CompressedMLP:
                    def __init__(self, T):
                        self.T = T
                    def __call__(self, x):
                        return mx.matmul(x, self.T.T)

                layer.mlp = CompressedMLP(layer_T[idx])

            try:
                comp_logits = model(input_ids)
                mx.eval(comp_logits)
                comp_top = int(mx.argmax(comp_logits[0, -1, :]).item())

                if comp_top == orig_top:
                    correct += 1
                total += 1
            finally:
                # Restore original MLPs
                for idx in layer_indices:
                    model.model.layers[idx].mlp = original_mlps[idx]

        return correct / total if total > 0 else 0.0

    # Test contiguous ranges of various sizes
    results = []

    # Test all contiguous ranges of size 2
    logger.info("\n--- Testing Contiguous Ranges (size 2) ---")
    for start in range(n_layers - 1):
        layer_set = [start, start + 1]
        accuracy = evaluate_layer_set(layer_set)
        results.append({
            "type": "contiguous",
            "size": 2,
            "layers": layer_set,
            "accuracy": accuracy,
        })
        logger.info(f"Layers {layer_set}: {accuracy:.1%}")

    # Test all contiguous ranges of size 3
    logger.info("\n--- Testing Contiguous Ranges (size 3) ---")
    for start in range(n_layers - 2):
        layer_set = [start, start + 1, start + 2]
        accuracy = evaluate_layer_set(layer_set)
        results.append({
            "type": "contiguous",
            "size": 3,
            "layers": layer_set,
            "accuracy": accuracy,
        })
        logger.info(f"Layers {layer_set}: {accuracy:.1%}")

    # Test all contiguous ranges of size 4
    logger.info("\n--- Testing Contiguous Ranges (size 4) ---")
    for start in range(n_layers - 3):
        layer_set = [start, start + 1, start + 2, start + 3]
        accuracy = evaluate_layer_set(layer_set)
        results.append({
            "type": "contiguous",
            "size": 4,
            "layers": layer_set,
            "accuracy": accuracy,
        })
        logger.info(f"Layers {layer_set}: {accuracy:.1%}")

    # Test some non-contiguous combinations
    logger.info("\n--- Testing Non-Contiguous Combinations ---")

    # Evenly spaced
    non_contiguous_sets = [
        [0, 12, 24],  # Evenly spaced
        [1, 10, 20],  # Spread out
        [5, 15, 25],  # Spread out
        [0, 5, 10, 15],  # 4 layers evenly spaced
        [0, 10, 20, 30],  # 4 layers far apart
        [1, 2, 10, 20],  # 2 contiguous + 2 far
        [5, 6, 25, 26],  # 2 pairs far apart
    ]

    for layer_set in non_contiguous_sets:
        if max(layer_set) < n_layers:
            accuracy = evaluate_layer_set(layer_set)
            results.append({
                "type": "non-contiguous",
                "size": len(layer_set),
                "layers": layer_set,
                "accuracy": accuracy,
            })
            logger.info(f"Layers {layer_set}: {accuracy:.1%}")

    # Analysis
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS")
    logger.info(f"{'='*60}")

    # Compare contiguous vs non-contiguous by size
    for size in [2, 3, 4]:
        contiguous = [r for r in results if r["type"] == "contiguous" and r["size"] == size]
        non_contiguous = [r for r in results if r["type"] == "non-contiguous" and r["size"] == size]

        if contiguous:
            avg_cont = sum(r["accuracy"] for r in contiguous) / len(contiguous)
            perfect_cont = len([r for r in contiguous if r["accuracy"] >= 1.0 - 1e-6])
            logger.info(f"\nSize {size} contiguous: avg={avg_cont:.1%}, 100%={perfect_cont}/{len(contiguous)}")

        if non_contiguous:
            avg_non = sum(r["accuracy"] for r in non_contiguous) / len(non_contiguous)
            perfect_non = len([r for r in non_contiguous if r["accuracy"] >= 1.0 - 1e-6])
            logger.info(f"Size {size} non-contiguous: avg={avg_non:.1%}, 100%={perfect_non}/{len(non_contiguous)}")

    # Find best contiguous ranges
    logger.info(f"\n--- Best Contiguous Ranges ---")
    perfect_ranges = [r for r in results if r["type"] == "contiguous" and r["accuracy"] >= 1.0 - 1e-6]
    perfect_ranges.sort(key=lambda r: r["size"], reverse=True)

    for r in perfect_ranges[:10]:
        logger.info(f"Layers {r['layers']}: 100%")

    # Find worst contiguous ranges (to identify problem layers)
    logger.info(f"\n--- Worst Contiguous Ranges ---")
    worst_ranges = [r for r in results if r["type"] == "contiguous"]
    worst_ranges.sort(key=lambda r: r["accuracy"])

    for r in worst_ranges[:10]:
        logger.info(f"Layers {r['layers']}: {r['accuracy']:.1%}")

    # Identify layers that appear in failing ranges
    logger.info(f"\n--- Problem Layer Analysis ---")

    layer_failures = {i: 0 for i in range(n_layers)}
    layer_appearances = {i: 0 for i in range(n_layers)}

    for r in results:
        if r["type"] == "contiguous":
            for layer in r["layers"]:
                layer_appearances[layer] += 1
                if r["accuracy"] < 1.0 - 1e-6:
                    layer_failures[layer] += 1

    failure_rates = {}
    for i in range(n_layers):
        if layer_appearances[i] > 0:
            failure_rates[i] = layer_failures[i] / layer_appearances[i]

    # Sort by failure rate
    sorted_layers = sorted(failure_rates.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\nLayers most associated with failures:")
    for layer, rate in sorted_layers[:10]:
        logger.info(f"  Layer {layer}: {rate:.1%} failure rate ({layer_failures[layer]}/{layer_appearances[layer]})")

    # Find the "safe zone"
    logger.info(f"\n--- Safe Zone Discovery ---")

    # Look for longest contiguous range with 100%
    longest_perfect = None
    for r in perfect_ranges:
        if longest_perfect is None or r["size"] > longest_perfect["size"]:
            longest_perfect = r

    if longest_perfect:
        logger.info(f"Longest 100% contiguous range: layers {longest_perfect['layers']}")

    # Check if there's a pattern in starting position
    logger.info(f"\n--- Starting Position Analysis ---")

    size3_results = [r for r in results if r["type"] == "contiguous" and r["size"] == 3]
    for r in size3_results:
        start = r["layers"][0]
        status = "100%" if r["accuracy"] >= 1.0 - 1e-6 else f"{r['accuracy']:.0%}"
        marker = "*" if r["accuracy"] >= 1.0 - 1e-6 else ""
        logger.info(f"  Start={start}: {status} {marker}")


if __name__ == "__main__":
    run_experiment()
