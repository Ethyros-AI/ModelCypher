#!/usr/bin/env python3
"""Experiment 21: Attention-Only Model.

The mega-skip failed because attention is non-linear and context-dependent.

New hypothesis: Keep all attention layers, but collapse the MLPs.

Each transformer layer = Attention + MLP
- Attention: KEEP (non-linearizable, context-dependent)
- MLP: COMPRESS (linearizable, proven in exp 1-19)

If we compress ALL MLPs (layers 7-33) to linear transforms,
we get a model that's:
- Attention-rich (preserves context understanding)
- Compute-cheap (MLPs are just matmuls)

This might work because:
1. Individual MLPs CAN be linearized (100% on many layers)
2. Attention is preserved, so context understanding is intact
3. Error compounding is the only issue - but maybe less severe?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test attention-only model: compress all MLPs, keep all attention."""
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

    # Large calibration set
    calibration_prompts = [
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "DNA replication occurs when",
        "The periodic table organizes",
        "Photosynthesis is the process by which",
        "Evolution by natural selection",
        "The speed of light is approximately",
        "Atoms are composed of",
        "The Roman Empire fell because",
        "World War II began when",
        "The Renaissance was a period of",
        "Ancient Egypt developed",
        "The Industrial Revolution started",
        "Democracy originated in",
        "Machine learning algorithms",
        "The internet was invented",
        "Artificial intelligence",
        "Computer programming involves",
        "Quantum computing uses",
        "Neural networks are",
        "Plato's theory of forms",
        "Kant's categorical imperative",
        "Existentialism emphasizes",
        "The problem of consciousness",
        "The derivative of a function",
        "Prime numbers are",
        "Calculus was invented by",
        "Statistical inference",
        "Linear algebra deals with",
        "Probability theory describes",
        "Shakespeare wrote",
        "The novel as a form",
        "Poetry differs from prose",
        "Narrative structure",
        "Cells divide through",
        "The human brain contains",
        "Genetics is the study of",
        "Ecosystems are composed of",
        "Newton's laws state",
        "Thermodynamics describes",
        "Electromagnetic waves",
        "Gravity causes objects to",
        "The Amazon rainforest",
        "Tectonic plates move",
        "Climate change refers to",
        "The ocean covers",
        "Supply and demand determines",
        "Inflation occurs when",
        "The stock market",
        "GDP measures",
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

    # Compress MLPs for layers in transmission zone
    # Skip encoding zone (0-6) and decoding zone (34-35)
    transmission_layers = list(range(7, 34))  # 27 layers

    logger.info(f"\n{'='*60}")
    logger.info(f"Compressing {len(transmission_layers)} MLPs (layers 7-33)")
    logger.info(f"{'='*60}")

    layer_T = {}

    for layer_idx in transmission_layers:
        logger.info(f"Compressing layer {layer_idx} MLP...")

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

    def evaluate_compressed_model(layer_indices):
        """Evaluate with compressed MLPs in specified layers."""
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

    # Test progressively larger compression
    logger.info(f"\n{'='*60}")
    logger.info("TESTING PROGRESSIVE MLP COMPRESSION")
    logger.info(f"{'='*60}")

    results = []

    # Test single layers first
    logger.info("\n--- Single Layer Compression ---")
    for layer_idx in [8, 15, 20, 25, 30, 33]:
        if layer_idx in layer_T:
            acc = evaluate_compressed_model([layer_idx])
            results.append({"layers": [layer_idx], "count": 1, "accuracy": acc})
            logger.info(f"Layer {layer_idx}: {acc:.1%}")

    # Test growing ranges from a safe starting point
    logger.info("\n--- Growing Ranges from Layer 20 ---")
    for end in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        layers = list(range(20, end + 1))
        if all(l in layer_T for l in layers):
            acc = evaluate_compressed_model(layers)
            results.append({"layers": layers, "count": len(layers), "accuracy": acc})
            logger.info(f"Layers 20-{end} ({len(layers)} MLPs): {acc:.1%}")
            if acc < 0.25:
                logger.info("  >>> Accuracy dropped below 25%, stopping expansion")
                break

    # Test full transmission zone
    logger.info("\n--- Full Transmission Zone (7-33) ---")
    full_acc = evaluate_compressed_model(transmission_layers)
    results.append({"layers": transmission_layers, "count": len(transmission_layers), "accuracy": full_acc})
    logger.info(f"All 27 MLPs compressed: {full_acc:.1%}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Layers':<20} {'Count':<8} {'Accuracy':<10}")
    logger.info("-" * 40)

    for r in results:
        if len(r["layers"]) <= 5:
            layer_str = str(r["layers"])
        else:
            layer_str = f"[{r['layers'][0]}...{r['layers'][-1]}]"
        logger.info(f"{layer_str:<20} {r['count']:<8} {r['accuracy']*100:<10.1f}")

    # Find crossover point
    logger.info("\n--- Crossover Analysis ---")
    perfect = [r for r in results if r["accuracy"] >= 1.0 - 1e-6]
    good = [r for r in results if r["accuracy"] >= 0.75]
    acceptable = [r for r in results if r["accuracy"] >= 0.5]

    if perfect:
        best_perfect = max(perfect, key=lambda r: r["count"])
        logger.info(f"Max 100% accuracy: {best_perfect['count']} layers")

    if good:
        best_good = max(good, key=lambda r: r["count"])
        logger.info(f"Max ≥75% accuracy: {best_good['count']} layers")

    if acceptable:
        best_acc = max(acceptable, key=lambda r: r["count"])
        logger.info(f"Max ≥50% accuracy: {best_acc['count']} layers")

    # The key question: how many MLPs can we compress?
    logger.info(f"\n{'='*60}")
    logger.info("KEY INSIGHT")
    logger.info(f"{'='*60}")

    if full_acc >= 0.5:
        logger.info(f">>> ALL 27 MLPs can be compressed at {full_acc:.1%} accuracy!")
        logger.info(f">>> This represents 75% of MLP compute (27/36 layers)")
    else:
        # Find the cliff
        for r in sorted(results, key=lambda x: x["count"]):
            if r["accuracy"] < 0.5:
                logger.info(f">>> Accuracy cliff at {r['count']} compressed MLPs")
                break


if __name__ == "__main__":
    run_experiment()
