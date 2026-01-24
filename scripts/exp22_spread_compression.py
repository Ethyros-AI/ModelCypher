#!/usr/bin/env python3
"""Experiment 22: Spread Compression.

Hypothesis: Error compounds through ADJACENT layers.
If we compress spread-out layers, errors don't amplify each other.

Compare:
- Sequential: layers 20, 21, 22, 23, 24 (error compounds)
- Spread: layers 10, 15, 20, 25, 30 (errors isolated)

If spread compression works better, it means:
- The architecture's sequential nature causes compounding
- A parallel or sparse architecture would be better
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test spread vs sequential compression."""
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

    # Calibration
    calibration_prompts = [
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "DNA replication occurs when",
        "The periodic table organizes",
        "Photosynthesis is the process by which",
        "Evolution by natural selection",
        "The speed of light is approximately",
        "Machine learning algorithms",
        "The internet was invented",
        "Artificial intelligence",
        "Computer programming involves",
        "Neural networks are",
        "The derivative of a function",
        "Prime numbers are",
        "Calculus was invented by",
        "Shakespeare wrote",
        "Cells divide through",
        "The human brain contains",
        "Newton's laws state",
        "Gravity causes objects to",
        "The Amazon rainforest",
        "Climate change refers to",
        "Supply and demand determines",
        "The stock market",
    ]

    held_out_prompts = [
        "The capital of Japan is",
        "Water boils at",
        "The largest ocean is",
        "Neurons transmit signals by",
        "Chemical bonds form when",
        "The Eiffel Tower was built",
        "Oxygen is essential for",
        "The human genome contains",
    ]

    cal_tokens = [tokenizer.encode(p) for p in calibration_prompts]
    held_tokens = [tokenizer.encode(p) for p in held_out_prompts]

    compressor = RMTAwareCompressor(backend=backend)

    # Pre-compress all needed layers
    needed_layers = set(range(8, 34))  # All transmission zone layers
    layer_T = {}

    logger.info(f"\n--- Pre-compressing layers ---")

    for layer_idx in sorted(needed_layers):
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

        if layer_idx % 5 == 0:
            logger.info(f"Compressed layer {layer_idx}")

    def evaluate_compressed_model(layer_indices):
        """Evaluate with compressed MLPs in specified layers."""
        correct = 0
        total = 0

        for tokens in held_tokens:
            input_ids = mx.array([tokens])

            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            original_mlps = {}
            for idx in layer_indices:
                if idx in layer_T:
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
                for idx in layer_indices:
                    if idx in original_mlps:
                        model.model.layers[idx].mlp = original_mlps[idx]

        return correct / total if total > 0 else 0.0

    # Compare sequential vs spread
    logger.info(f"\n{'='*60}")
    logger.info("SEQUENTIAL VS SPREAD COMPARISON")
    logger.info(f"{'='*60}")

    comparisons = [
        # (name, sequential_layers, spread_layers)
        ("2 layers", [20, 21], [15, 25]),
        ("3 layers", [20, 21, 22], [12, 20, 28]),
        ("4 layers", [20, 21, 22, 23], [10, 16, 24, 30]),
        ("5 layers", [20, 21, 22, 23, 24], [10, 15, 20, 25, 30]),
        ("6 layers", [20, 21, 22, 23, 24, 25], [10, 14, 18, 22, 26, 30]),
        ("8 layers", [16, 17, 18, 19, 20, 21, 22, 23], [10, 13, 16, 19, 22, 25, 28, 31]),
    ]

    results = []

    for name, seq_layers, spread_layers in comparisons:
        seq_acc = evaluate_compressed_model(seq_layers)
        spread_acc = evaluate_compressed_model(spread_layers)

        results.append({
            "name": name,
            "seq_layers": seq_layers,
            "spread_layers": spread_layers,
            "seq_acc": seq_acc,
            "spread_acc": spread_acc,
            "diff": spread_acc - seq_acc,
        })

        logger.info(f"\n{name}:")
        logger.info(f"  Sequential {seq_layers}: {seq_acc:.1%}")
        logger.info(f"  Spread {spread_layers}: {spread_acc:.1%}")
        if spread_acc > seq_acc:
            logger.info(f"  >>> SPREAD WINS by {(spread_acc - seq_acc)*100:.1f}pp")
        elif seq_acc > spread_acc:
            logger.info(f"  >>> SEQUENTIAL WINS by {(seq_acc - spread_acc)*100:.1f}pp")
        else:
            logger.info(f"  >>> TIE")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Layers':<12} {'Sequential':<15} {'Spread':<15} {'Winner':<10}")
    logger.info("-" * 55)

    spread_wins = 0
    seq_wins = 0

    for r in results:
        if r["spread_acc"] > r["seq_acc"]:
            winner = "SPREAD"
            spread_wins += 1
        elif r["seq_acc"] > r["spread_acc"]:
            winner = "SEQ"
            seq_wins += 1
        else:
            winner = "TIE"

        logger.info(
            f"{r['name']:<12} {r['seq_acc']*100:<15.1f} {r['spread_acc']*100:<15.1f} {winner:<10}"
        )

    logger.info(f"\nSpread wins: {spread_wins}")
    logger.info(f"Sequential wins: {seq_wins}")

    # Test maximum spread compression
    logger.info(f"\n{'='*60}")
    logger.info("MAXIMUM SPREAD COMPRESSION")
    logger.info(f"{'='*60}")

    # Spread every 3rd layer
    spread_3 = list(range(9, 34, 3))  # [9, 12, 15, 18, 21, 24, 27, 30, 33]
    spread_3_acc = evaluate_compressed_model(spread_3)
    logger.info(f"Every 3rd layer ({len(spread_3)} layers): {spread_3_acc:.1%}")

    # Spread every 4th layer
    spread_4 = list(range(8, 34, 4))  # [8, 12, 16, 20, 24, 28, 32]
    spread_4_acc = evaluate_compressed_model(spread_4)
    logger.info(f"Every 4th layer ({len(spread_4)} layers): {spread_4_acc:.1%}")

    # Spread every 5th layer
    spread_5 = list(range(10, 34, 5))  # [10, 15, 20, 25, 30]
    spread_5_acc = evaluate_compressed_model(spread_5)
    logger.info(f"Every 5th layer ({len(spread_5)} layers): {spread_5_acc:.1%}")

    # Find optimal spread
    logger.info(f"\n--- Optimal Spread Search ---")

    best_config = None
    best_accuracy = 0

    for step in range(2, 8):
        for start in range(8, 12):
            layers = list(range(start, 34, step))
            if len(layers) >= 3:
                acc = evaluate_compressed_model(layers)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_config = {"start": start, "step": step, "layers": layers, "acc": acc}

    if best_config:
        logger.info(f"Best spread config: start={best_config['start']}, step={best_config['step']}")
        logger.info(f"Layers: {best_config['layers']}")
        logger.info(f"Count: {len(best_config['layers'])}")
        logger.info(f"Accuracy: {best_config['acc']:.1%}")

    # Conclusion
    logger.info(f"\n{'='*60}")
    logger.info("CONCLUSION")
    logger.info(f"{'='*60}")

    if spread_wins > seq_wins:
        logger.info(">>> SPREAD COMPRESSION IS BETTER")
        logger.info(">>> Errors compound through adjacent layers")
        logger.info(">>> Spacing layers prevents error amplification")
    elif seq_wins > spread_wins:
        logger.info(">>> SEQUENTIAL COMPRESSION IS BETTER")
        logger.info(">>> Spacing doesn't help - errors propagate regardless")
    else:
        logger.info(">>> MIXED RESULTS")
        logger.info(">>> Spacing helps sometimes but not consistently")


if __name__ == "__main__":
    run_experiment()
