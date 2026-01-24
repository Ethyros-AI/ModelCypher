#!/usr/bin/env python3
"""Experiment 20: Mega-Skip Hypothesis.

What if we don't need 36 layers? What if the model is really:
1. Encoding gates (0-6): Lock onto the manifold
2. One linear transform: Skip layers 7-33 entirely
3. Decoding (34-35): Project to vocabulary

This experiment tests: can we learn a SINGLE linear transform T that
replaces the entire transmission zone (27 layers)?

T: X_7 -> X_34  (skip 27 layers with one matrix multiply!)

If this works, we've reduced a 36-layer model to ~9 layers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_experiment():
    """Test mega-skip: one transform to replace 27 layers."""
    import mlx.core as mx

    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

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

    # Large calibration set for learning the mega-transform
    calibration_prompts = [
        # Science
        "The theory of relativity states that",
        "Quantum mechanics describes",
        "DNA replication occurs when",
        "The periodic table organizes",
        "Photosynthesis is the process by which",
        "Evolution by natural selection",
        "The speed of light is approximately",
        "Atoms are composed of",
        # History
        "The Roman Empire fell because",
        "World War II began when",
        "The Renaissance was a period of",
        "Ancient Egypt developed",
        "The Industrial Revolution started",
        "Democracy originated in",
        # Technology
        "Machine learning algorithms",
        "The internet was invented",
        "Artificial intelligence",
        "Computer programming involves",
        "Quantum computing uses",
        "Neural networks are",
        # Philosophy
        "Plato's theory of forms",
        "Kant's categorical imperative",
        "Existentialism emphasizes",
        "The problem of consciousness",
        # Math
        "The derivative of a function",
        "Prime numbers are",
        "Calculus was invented by",
        "Statistical inference",
        "Linear algebra deals with",
        "Probability theory describes",
        # Literature
        "Shakespeare wrote",
        "The novel as a form",
        "Poetry differs from prose",
        "Narrative structure",
        # Biology
        "Cells divide through",
        "The human brain contains",
        "Genetics is the study of",
        "Ecosystems are composed of",
        # Physics
        "Newton's laws state",
        "Thermodynamics describes",
        "Electromagnetic waves",
        "Gravity causes objects to",
        # Geography
        "The Amazon rainforest",
        "Tectonic plates move",
        "Climate change refers to",
        "The ocean covers",
        # Economics
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

    # Test different skip configurations
    skip_configs = [
        {"name": "skip_7_to_34", "start": 7, "end": 34},   # Skip transmission zone
        {"name": "skip_7_to_30", "start": 7, "end": 30},   # Smaller skip
        {"name": "skip_10_to_30", "start": 10, "end": 30}, # Middle skip
        {"name": "skip_7_to_20", "start": 7, "end": 20},   # Early skip
        {"name": "skip_20_to_34", "start": 20, "end": 34}, # Late skip
    ]

    results = []

    for config in skip_configs:
        start_layer = config["start"]
        end_layer = config["end"]
        name = config["name"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {name}: layer {start_layer} -> layer {end_layer}")
        logger.info(f"Skipping {end_layer - start_layer} layers")
        logger.info(f"{'='*60}")

        # Collect activation pairs (X_start, X_end) from calibration
        X_starts = []
        X_ends = []

        for tokens in cal_tokens:
            input_ids = mx.array([tokens])

            # Hook to capture activations after start_layer and after end_layer
            activations = {}

            def make_hook(layer_idx):
                def hook(module, args, output):
                    # Transformer layers return (hidden_state, ...) or just hidden_state
                    if isinstance(output, tuple):
                        activations[layer_idx] = output[0]
                    else:
                        activations[layer_idx] = output
                    return output
                return hook

            # We need to capture the hidden state AFTER the layer's forward pass
            # In MLX-LM, layers are called directly. We'll use a different approach:
            # Run partial forward passes.

            # Get embedding
            x = model.model.embed_tokens(input_ids)
            mx.eval(x)

            # Run through layers 0 to start_layer
            for i in range(start_layer):
                layer = model.model.layers[i]
                x = layer(x, mask=None, cache=None)
                if isinstance(x, tuple):
                    x = x[0]
                mx.eval(x)

            # Capture X_start (after layer start_layer-1, before layer start_layer)
            X_start = x * 1.0  # Copy via multiplication
            mx.eval(X_start)

            # Run through layers start_layer to end_layer
            for i in range(start_layer, end_layer):
                layer = model.model.layers[i]
                x = layer(x, mask=None, cache=None)
                if isinstance(x, tuple):
                    x = x[0]
                mx.eval(x)

            # Capture X_end (after layer end_layer-1)
            X_end = x * 1.0  # Copy via multiplication
            mx.eval(X_end)

            # Take last token position
            X_starts.append(X_start[0, -1, :])
            X_ends.append(X_end[0, -1, :])

        X_start_mat = mx.stack(X_starts).astype(mx.float32)
        X_end_mat = mx.stack(X_ends).astype(mx.float32)
        mx.eval(X_start_mat, X_end_mat)

        logger.info(f"Calibration shapes: X_start={X_start_mat.shape}, X_end={X_end_mat.shape}")

        # Learn mega-transform T: X_end ≈ X_start @ T.T
        # Using RMT-aware compression
        X_backend = backend.array(X_start_mat)
        Y_backend = backend.array(X_end_mat)

        from modelcypher.core.domain.compression import RMTAwareCompressor
        compressor = RMTAwareCompressor(backend=backend)
        rmt_result = compressor.compress_layer(X_backend, Y_backend)

        logger.info(f"RMT signal_rank: {rmt_result.signal_rank}/{rmt_result.total_rank}")
        logger.info(f"Reconstruction error: {rmt_result.reconstruction_error:.4f}")

        T_mx = mx.array(backend.tolist(rmt_result.T)).astype(mx.float32)
        mx.eval(T_mx)

        # Evaluate on held-out
        correct = 0
        total = 0

        for tokens in held_tokens:
            input_ids = mx.array([tokens])

            # Original forward pass
            orig_logits = model(input_ids)
            mx.eval(orig_logits)
            orig_top = int(mx.argmax(orig_logits[0, -1, :]).item())

            # Modified forward pass with mega-skip
            x = model.model.embed_tokens(input_ids)
            mx.eval(x)

            # Run through layers 0 to start_layer
            for i in range(start_layer):
                layer = model.model.layers[i]
                x = layer(x, mask=None, cache=None)
                if isinstance(x, tuple):
                    x = x[0]
                mx.eval(x)

            # MEGA-SKIP: Apply T instead of layers start_layer to end_layer
            x = mx.matmul(x, T_mx.T)
            mx.eval(x)

            # Run through remaining layers end_layer to n_layers
            for i in range(end_layer, n_layers):
                layer = model.model.layers[i]
                x = layer(x, mask=None, cache=None)
                if isinstance(x, tuple):
                    x = x[0]
                mx.eval(x)

            # Final norm and output
            x = model.model.norm(x)
            skip_logits = model.lm_head(x)
            mx.eval(skip_logits)
            skip_top = int(mx.argmax(skip_logits[0, -1, :]).item())

            if skip_top == orig_top:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        layers_skipped = end_layer - start_layer
        remaining_layers = start_layer + (n_layers - end_layer)

        logger.info(f"Held-out accuracy: {accuracy:.1%}")
        logger.info(f"Layers skipped: {layers_skipped}")
        logger.info(f"Effective model: {remaining_layers} layers + 1 linear transform")

        results.append({
            "name": name,
            "start": start_layer,
            "end": end_layer,
            "skipped": layers_skipped,
            "remaining": remaining_layers,
            "accuracy": accuracy,
            "signal_rank": rmt_result.signal_rank,
            "recon_error": rmt_result.reconstruction_error,
        })

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: MEGA-SKIP RESULTS")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Config':<20} {'Skipped':<10} {'Remaining':<12} {'Accuracy':<10} {'Signal Rank':<12}")
    logger.info("-" * 70)

    for r in results:
        logger.info(
            f"{r['name']:<20} {r['skipped']:<10} {r['remaining']:<12} "
            f"{r['accuracy']*100:<10.1f} {r['signal_rank']:<12}"
        )

    # Find best configuration
    best = max(results, key=lambda r: r["accuracy"])
    logger.info(f"\n>>> BEST CONFIG: {best['name']}")
    logger.info(f">>> Accuracy: {best['accuracy']:.1%}")
    logger.info(f">>> Effective layers: {best['remaining']} + 1 transform = {best['remaining'] + 1}")

    if best["accuracy"] >= 0.75:
        logger.info(f"\n>>> HYPOTHESIS SUPPORTED: Can skip {best['skipped']} layers with one transform!")
        logger.info(f">>> Model reduced from {n_layers} to {best['remaining'] + 1} effective layers")
    else:
        logger.info(f"\n>>> HYPOTHESIS NOT SUPPORTED at 75% threshold")

    # Test the most aggressive skip: 7 -> 34
    aggressive = next((r for r in results if r["name"] == "skip_7_to_34"), None)
    if aggressive:
        if aggressive["accuracy"] >= 0.5:
            logger.info(f"\n>>> AGGRESSIVE SKIP (7->34) achieves {aggressive['accuracy']:.1%}")
            logger.info(f">>> This means: 7 encoding layers + 1 transform + 2 decoding = 10 effective layers")


if __name__ == "__main__":
    run_experiment()
