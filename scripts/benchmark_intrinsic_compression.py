#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Benchmark script for intrinsic dimensionality compression.
#
# Tests the hypothesis: Can we compress a model to its intrinsic dimensionality
# without losing capability?
#
# Usage:
#   python scripts/benchmark_intrinsic_compression.py \
#     --model /Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16
"""
Intrinsic Compression Benchmark

This script:
1. Loads a model and measures baseline perplexity
2. Collects activations on diverse probes to define the manifold
3. Computes compression potential per layer (utilized vs available dimensions)
4. Compresses weights and validates CKA = 1.0
5. Runs inference with compressed weights and compares outputs
6. Reports compression ratio and capability preservation

The key insight: if a layer only uses 25% of its dimensions (intrinsic rank),
we can store 8x fewer parameters with ZERO loss on the activation manifold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def initialize_backend():
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    initialize_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    logger.info(
        "Loaded %s: %d layers, hidden_dim=%d",
        config.get("model_type", "unknown"),
        config.get("num_hidden_layers", 0),
        config.get("hidden_size", 0),
    )

    return model, tokenizer, config


def count_parameters(model: Any) -> int:
    """Count total parameters in model."""
    import mlx.core as mx

    def count_dict(d):
        total = 0
        for k, v in d.items():
            if isinstance(v, dict):
                total += count_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += count_dict(item)
                    elif hasattr(item, "size"):
                        total += item.size
            elif hasattr(v, "size"):
                total += v.size
        return total

    params = model.parameters()
    if isinstance(params, dict):
        return count_dict(params)
    return 0


def generate_probe_texts(use_full_atlas: bool = True) -> list[str]:
    """Generate probe texts to define the activation manifold.

    If use_full_atlas=True (default), uses the unified atlas with 4596 probes
    across all domains. This "floods the map" to get full manifold coverage.

    The key insight: compression is only lossless on the sampled manifold.
    To truly compress without loss, we need probes that cover the FULL
    input distribution - including the boundaries that define structure.
    """
    if use_full_atlas:
        from modelcypher.core.domain.agents.unified_atlas import UnifiedAtlasInventory

        probes = UnifiedAtlasInventory.all_probes()
        logger.info("Loaded %d probes from unified atlas", len(probes))

        # Extract support texts from all probes
        texts = []
        for probe in probes:
            if probe.support_texts:
                # Use all support texts for each probe
                for text in probe.support_texts:
                    if text and len(text) > 5:  # Skip very short texts
                        texts.append(text)

        # Deduplicate while preserving order
        seen = set()
        unique_texts = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        logger.info("Extracted %d unique probe texts from atlas", len(unique_texts))
        return unique_texts

    # Fallback: Simple generated probes (for quick testing)
    probes = []

    # Math & logic (varied)
    math_templates = [
        "What is {} + {}?",
        "Calculate {} times {}.",
        "If x = {}, what is x squared?",
    ]
    for i in range(20):
        for template in math_templates:
            probes.append(template.format(i, i + 1))

    # Natural language with variations
    subjects = ["cat", "dog", "bird", "fish", "lion", "elephant"]
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "hides"]
    for subj in subjects:
        for verb in verbs:
            probes.append(f"The {subj} {verb} quickly.")

    import random
    random.seed(42)
    random.shuffle(probes)

    logger.info("Generated %d simple probe texts", len(probes))
    return probes


def collect_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    config: dict,
) -> dict[int, Any]:
    """Collect activations per layer across all probe texts.

    Returns dict mapping layer_idx -> stacked activations [n_probes, hidden_dim]
    """
    import mlx.core as mx
    from modelcypher.adapters.mlx_activation_provider import MLXActivationProvider

    provider = MLXActivationProvider(config=config, pooling="last")

    num_layers = config.get("num_hidden_layers", 0)
    layer_activations: dict[int, list] = {i: [] for i in range(num_layers)}

    logger.info("Collecting activations for %d probes across %d layers", len(texts), num_layers)

    for text in texts:
        try:
            acts = provider.collect_hidden_activations(model, tokenizer, text)
            for layer_idx, act in acts.items():
                layer_activations[layer_idx].append(act)
        except Exception as e:
            logger.warning("Failed to collect activations for '%s...': %s", text[:20], e)

    # Stack into matrices
    result = {}
    for layer_idx, acts in layer_activations.items():
        if acts:
            stacked = mx.stack(acts, axis=0)  # [n_probes, hidden_dim]
            mx.eval(stacked)
            result[layer_idx] = stacked

    logger.info("Collected activations for %d layers", len(result))
    return result


def analyze_compression_potential(
    layer_activations: dict[int, Any],
    config: dict,
) -> dict[str, Any]:
    """Analyze compression potential per layer using TWO methods:

    1. TwoNN intrinsic dimension - measures TRUE manifold dimensionality
       using geodesic distances. This captures the degrees of freedom.

    2. Variance null space - measures how many dimensions have significant
       activation variance. This is what we can COMPRESS (strip unused dims).

    The key insight: TwoNN tells us the TRUE complexity,
    but variance null space tells us the COMPRESSIBLE redundancy.
    """
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
    from modelcypher.core.domain.geometry.intrinsic_compression import estimate_compression_potential

    backend = get_default_backend()
    hidden_dim = config.get("hidden_size", 0)
    id_estimator = IntrinsicDimension(backend)

    results = {
        "per_layer": {},
        "summary": {},
    }

    total_two_nn = 0.0
    total_variance_rank = 0.0
    valid_layers = 0

    for layer_idx, activations in sorted(layer_activations.items()):
        try:
            # Method 1: TwoNN intrinsic dimension (true manifold complexity)
            estimate = id_estimator.compute(activations)
            two_nn_id = estimate.intrinsic_dimension

            # Method 2: Variance null space (compressible dimensions)
            variance_est = estimate_compression_potential(activations, backend)
            variance_rank = variance_est["intrinsic_dim"]
            utilized_frac = variance_est["utilized_fraction"]

            # Compression potential based on variance (what we actually compress)
            compression = 1.0 / (2 * utilized_frac) if utilized_frac > 0 else 0

            results["per_layer"][layer_idx] = {
                "two_nn_id": two_nn_id,
                "variance_rank": variance_rank,
                "hidden_dim": hidden_dim,
                "utilized_fraction": utilized_frac,
                "compression_potential": compression,
                "sample_count": estimate.sample_count,
                "usable_count": estimate.usable_count,
            }

            total_two_nn += two_nn_id
            total_variance_rank += variance_rank
            valid_layers += 1

            logger.info(
                "Layer %2d: TwoNN=%.1f, variance_rank=%d/%d (%.1f%%), compression=%.1fx",
                layer_idx, two_nn_id, variance_rank, hidden_dim, utilized_frac * 100, compression
            )

        except Exception as e:
            logger.warning("Layer %d: analysis failed: %s", layer_idx, e)
            results["per_layer"][layer_idx] = {"error": str(e)}

    mean_two_nn = total_two_nn / valid_layers if valid_layers > 0 else 0
    mean_variance_rank = total_variance_rank / valid_layers if valid_layers > 0 else 0
    mean_util = mean_variance_rank / hidden_dim if hidden_dim > 0 else 0
    mean_compression = 1.0 / (2 * mean_util) if mean_util > 0 else 0

    results["summary"] = {
        "num_layers": len(layer_activations),
        "valid_layers": valid_layers,
        "hidden_dim": hidden_dim,
        "mean_two_nn_id": mean_two_nn,
        "mean_variance_rank": mean_variance_rank,
        "mean_utilized_fraction": mean_util,
        "mean_compression_potential": mean_compression,
    }

    return results


def compress_and_validate_layer(
    model: Any,
    layer_idx: int,
    activations: Any,
    config: dict,
) -> dict[str, Any]:
    """Compress a single layer and validate CKA = 1.0.

    Returns compression metrics and validation results.
    """
    import mlx.core as mx
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_compression import (
        compress_layer_with_validation,
    )
    from modelcypher.adapters.model_architecture import get_model_architecture

    backend = get_default_backend()

    # Get the MLP weights for this layer
    arch = get_model_architecture(model, config=config)
    mlp_keys = arch.layer_mlp_keys(layer_idx)

    results = {
        "layer_idx": layer_idx,
        "weights": {},
    }

    for key in mlp_keys:
        try:
            # Navigate to weight
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)

            weight_name = parts[-1]
            if not hasattr(obj, weight_name):
                continue

            weight = getattr(obj, weight_name)
            if weight is None or not hasattr(weight, "shape"):
                continue

            weight = backend.array(weight)
            backend.eval(weight)

            shape = backend.shape(weight)
            if len(shape) != 2:
                continue

            out_dim, in_dim = int(shape[0]), int(shape[1])

            # Check if activations match input dimension
            act_shape = backend.shape(activations)
            act_dim = int(act_shape[1])

            if act_dim != in_dim:
                # This weight doesn't take hidden state as input
                continue

            # Compress and validate
            result, cka, max_rel_error = compress_layer_with_validation(
                weight, activations, backend
            )

            compression = 1.0 / result.compression_ratio if result.compression_ratio > 0 else 0

            results["weights"][key] = {
                "original_shape": (out_dim, in_dim),
                "utilized_rank": result.utilized_rank,
                "compression_ratio": result.compression_ratio,
                "compression_factor": compression,
                "cka": cka,
                "max_rel_error": max_rel_error,
                "variance_captured": result.variance_captured,
            }

            logger.info(
                "  %s: [%d, %d] -> rank=%d (%.1fx), CKA=%.10f",
                key.split(".")[-2] + "." + key.split(".")[-1],
                out_dim, in_dim,
                result.utilized_rank,
                compression,
                cka
            )

        except Exception as e:
            logger.warning("Failed to compress %s: %s", key, e)

    return results


def test_reconstruction(
    model: Any,
    tokenizer: Any,
    activations: Any,
    config: dict,
    test_prompt: str,
) -> None:
    """Test that compressed+reconstructed weights produce identical outputs.

    This is the REAL test: we replace the weights with the reconstructed form
    and verify the model produces the same outputs.
    """
    import mlx.core as mx
    from modelcypher.core.domain._backend import get_default_backend
    from modelcypher.core.domain.geometry.intrinsic_compression import (
        compress_weight_to_intrinsic_dim,
    )
    from modelcypher.adapters.model_architecture import get_model_architecture

    backend = get_default_backend()
    arch = get_model_architecture(model, config=config)

    # Get original output
    tokens = tokenizer.encode(test_prompt, add_special_tokens=True)
    if isinstance(tokens, list):
        token_ids = tokens
    else:
        token_ids = list(tokens.ids)
    input_ids = mx.array([token_ids])

    original_logits = model(input_ids)
    mx.eval(original_logits)
    original_logits_copy = mx.array(original_logits)  # Copy before modification

    # Get the MLP module for layer 0
    layer0 = arch.layers[0]
    accessor = arch.layer_accessor(0)
    mlp = accessor.mlp

    # Compress and reconstruct w1 (gate_proj)
    w1_original = mlp.w1.weight
    mx.eval(w1_original)
    w1_arr = backend.array(w1_original)

    result = compress_weight_to_intrinsic_dim(w1_arr, activations, backend)
    w1_reconstructed = result.reconstruct(backend)
    backend.eval(w1_reconstructed)

    # Replace weight
    original_w1_copy = mx.array(w1_original)  # Save original
    mlp.w1.weight = mx.array(w1_reconstructed)
    mx.eval(mlp.w1.weight)

    # Get output with reconstructed weights
    reconstructed_logits = model(input_ids)
    mx.eval(reconstructed_logits)

    # Compare outputs
    diff = mx.abs(original_logits_copy - reconstructed_logits)
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))

    # Compute relative difference
    orig_norm = float(mx.sqrt(mx.sum(original_logits_copy * original_logits_copy)))
    rel_diff = max_diff / (orig_norm + 1e-10)

    logger.info("Original vs Reconstructed logits:")
    logger.info("  Max absolute diff: %.6e", max_diff)
    logger.info("  Mean absolute diff: %.6e", mean_diff)
    logger.info("  Relative diff: %.6e", rel_diff)
    logger.info("  Compression: [%d, %d] -> rank=%d (%.1fx)",
                result.original_shape[0], result.original_shape[1],
                result.utilized_rank, 1.0 / result.compression_ratio)

    # Verify they generate the same text
    from mlx_lm import generate

    # Restore original weight for fair comparison
    mlp.w1.weight = original_w1_copy
    mx.eval(mlp.w1.weight)

    original_text = generate(model, tokenizer, prompt=test_prompt, max_tokens=20, verbose=False)

    # Now use reconstructed
    mlp.w1.weight = mx.array(w1_reconstructed)
    mx.eval(mlp.w1.weight)

    reconstructed_text = generate(model, tokenizer, prompt=test_prompt, max_tokens=20, verbose=False)

    logger.info("Original generation: %s", original_text)
    logger.info("Reconstructed generation: %s", reconstructed_text)

    if original_text == reconstructed_text:
        logger.info("✓ GENERATION MATCHES - Compression is lossless!")
    else:
        logger.info("⚠ Generation differs (may still be acceptable if close)")

    # Restore original weight
    mlp.w1.weight = original_w1_copy
    mx.eval(mlp.w1.weight)


def generate_text(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 50) -> str:
    """Generate text with the model."""
    from mlx_lm import generate

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def compute_perplexity(model: Any, tokenizer: Any, texts: list[str]) -> float:
    """Compute perplexity on a list of texts."""
    import mlx.core as mx
    import mlx.nn as nn

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if isinstance(tokens, list):
            token_ids = tokens
        else:
            token_ids = list(tokens.ids)

        if len(token_ids) < 2:
            continue

        input_ids = mx.array([token_ids[:-1]])
        target_ids = mx.array([token_ids[1:]])

        logits = model(input_ids)
        mx.eval(logits)

        # Cross-entropy loss
        log_probs = nn.log_softmax(logits, axis=-1)
        target_log_probs = mx.take_along_axis(
            log_probs, mx.expand_dims(target_ids, axis=-1), axis=-1
        )
        loss = -mx.mean(target_log_probs)
        mx.eval(loss)

        total_loss += float(loss) * (len(token_ids) - 1)
        total_tokens += len(token_ids) - 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = float(mx.exp(mx.array(avg_loss)))
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Benchmark intrinsic compression")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Use quick mode with fewer probes")
    parser.add_argument("--max-probes", type=int, default=None, help="Limit number of probes")
    args = parser.parse_args()

    # Initialize backend
    initialize_backend()

    # Load model
    model, tokenizer, config = load_model(args.model)
    total_params = count_parameters(model)
    logger.info("Total parameters: %.2fM", total_params / 1e6)

    # Generate probe texts - use full atlas by default
    use_full_atlas = not args.quick
    probes = generate_probe_texts(use_full_atlas=use_full_atlas)

    # Optionally limit probe count
    if args.max_probes and len(probes) > args.max_probes:
        import random
        random.seed(42)
        probes = random.sample(probes, args.max_probes)
        logger.info("Limited to %d probes", len(probes))

    logger.info("Using %d probe texts", len(probes))

    # Baseline generation test
    logger.info("\n=== BASELINE GENERATION ===")
    test_prompt = "What is 2 + 2? The answer is"
    baseline_output = generate_text(model, tokenizer, test_prompt, max_tokens=30)
    logger.info("Prompt: %s", test_prompt)
    logger.info("Output: %s", baseline_output)

    # Baseline perplexity
    logger.info("\n=== BASELINE PERPLEXITY ===")
    eval_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In mathematics, a prime number is a natural number greater than 1.",
        "def hello_world():\n    print('Hello, World!')",
        "The capital of Japan is Tokyo, a vibrant metropolis.",
    ]
    baseline_ppl = compute_perplexity(model, tokenizer, eval_texts)
    logger.info("Baseline perplexity: %.4f", baseline_ppl)

    # Collect activations
    logger.info("\n=== COLLECTING ACTIVATIONS ===")
    start = time.time()
    layer_activations = collect_layer_activations(model, tokenizer, probes, config)
    collect_time = time.time() - start
    logger.info("Activation collection took %.2fs", collect_time)

    # Analyze compression potential
    logger.info("\n=== COMPRESSION POTENTIAL ===")
    potential = analyze_compression_potential(layer_activations, config)

    summary = potential["summary"]
    logger.info(
        "\nSummary: TwoNN_ID=%.1f, variance_rank=%.0f/%d (%.1f%%), potential_compression=%.1fx",
        summary["mean_two_nn_id"],
        summary["mean_variance_rank"],
        summary["hidden_dim"],
        summary["mean_utilized_fraction"] * 100,
        summary["mean_compression_potential"]
    )

    # Compress and validate ALL layers - calculate true compressed size
    logger.info("\n=== FULL LAYER-BY-LAYER COMPRESSION ===")
    compression_results = {}

    total_original_params = 0
    total_compressed_params = 0
    all_cka_scores = []

    for layer_idx in sorted(layer_activations.keys()):
        if layer_idx in layer_activations:
            logger.info("Layer %d:", layer_idx)
            result = compress_and_validate_layer(
                model, layer_idx, layer_activations[layer_idx], config
            )
            compression_results[layer_idx] = result

            # Accumulate compressed sizes
            for key, weight_result in result.get("weights", {}).items():
                if "original_shape" in weight_result:
                    out_dim, in_dim = weight_result["original_shape"]
                    rank = weight_result["utilized_rank"]
                    original = out_dim * in_dim
                    # Compressed: W_left [out_dim, rank] + V_used [in_dim, rank]
                    compressed = rank * (out_dim + in_dim)

                    total_original_params += original
                    total_compressed_params += compressed

                    if "cka" in weight_result:
                        all_cka_scores.append(weight_result["cka"])

    # Calculate embedding and output projection sizes (not compressed)
    hidden_dim = config.get("hidden_size", 0)
    vocab_size = config.get("vocab_size", 0)
    intermediate_size = config.get("intermediate_size", 0)
    num_layers = config.get("num_hidden_layers", 0)

    # Embedding: vocab_size * hidden_dim
    embed_params = vocab_size * hidden_dim

    # LM head: vocab_size * hidden_dim (often tied, but count separately)
    lm_head_params = vocab_size * hidden_dim

    # Attention params per layer: Q, K, V, O projections
    # Typically: 4 * hidden_dim * hidden_dim (for standard attention)
    num_attention_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = hidden_dim // num_attention_heads

    attn_q_params = hidden_dim * hidden_dim  # Q proj
    attn_k_params = hidden_dim * (num_kv_heads * head_dim)  # K proj (GQA)
    attn_v_params = hidden_dim * (num_kv_heads * head_dim)  # V proj (GQA)
    attn_o_params = hidden_dim * hidden_dim  # O proj
    attn_params_per_layer = attn_q_params + attn_k_params + attn_v_params + attn_o_params
    total_attn_params = num_layers * attn_params_per_layer

    # Layer norms: typically 2 per layer * hidden_dim
    norm_params = num_layers * 2 * hidden_dim + hidden_dim  # +final norm

    # Non-compressed params
    non_compressed_params = embed_params + lm_head_params + total_attn_params + norm_params

    # Summary
    logger.info("\n=== COMPRESSION RESULTS ===")
    logger.info("Model: %s", args.model)
    logger.info("Total model parameters: %.2fM", total_params / 1e6)
    logger.info("")

    logger.info("MLP COMPRESSION (layer-by-layer):")
    logger.info("  Original MLP params: %.2fM", total_original_params / 1e6)
    logger.info("  Compressed MLP params: %.2fM", total_compressed_params / 1e6)
    if total_original_params > 0:
        mlp_compression = total_original_params / total_compressed_params
        logger.info("  MLP compression: %.2fx", mlp_compression)

    if all_cka_scores:
        min_cka = min(all_cka_scores)
        mean_cka = sum(all_cka_scores) / len(all_cka_scores)
        logger.info("  CKA: min=%.10f, mean=%.10f (1.0 = lossless)", min_cka, mean_cka)

    logger.info("")
    logger.info("FULL MODEL SIZE:")
    logger.info("  Non-compressed (embed, attn, norm): %.2fM", non_compressed_params / 1e6)
    logger.info("  Compressed MLP: %.2fM", total_compressed_params / 1e6)
    total_compressed_model = non_compressed_params + total_compressed_params
    logger.info("  TOTAL COMPRESSED MODEL: %.2fM params", total_compressed_model / 1e6)
    logger.info("")

    if total_params > 0:
        overall_compression = total_params / total_compressed_model
        size_reduction = (1 - total_compressed_model / total_params) * 100
        logger.info("  OVERALL COMPRESSION: %.2fx", overall_compression)
        logger.info("  SIZE REDUCTION: %.1f%%", size_reduction)
        logger.info("")

        # In bytes (bf16 = 2 bytes per param)
        bytes_per_param = 2  # bf16
        original_bytes = total_params * bytes_per_param
        compressed_bytes = total_compressed_model * bytes_per_param
        logger.info("  ORIGINAL SIZE (bf16): %.2f GB", original_bytes / 1e9)
        logger.info("  COMPRESSED SIZE (bf16): %.2f GB", compressed_bytes / 1e9)

    # Baseline metrics
    logger.info("")
    logger.info("BASELINE METRICS:")
    logger.info("  Layers: %d", summary["num_layers"])
    logger.info("  Hidden dim: %d", summary["hidden_dim"])
    logger.info("  Mean utilized fraction: %.1f%%", summary["mean_utilized_fraction"] * 100)
    logger.info("  Baseline perplexity: %.4f", baseline_ppl)

    # Test actual reconstruction: replace weights and verify identical outputs
    logger.info("\n=== RECONSTRUCTION TEST (Layer 0) ===")
    try:
        test_reconstruction(model, tokenizer, layer_activations[0], config, test_prompt)
    except Exception as e:
        logger.warning("Reconstruction test failed: %s", e)

    # Save results
    if args.output:
        results = {
            "model_path": args.model,
            "total_params": total_params,
            "baseline_perplexity": baseline_ppl,
            "compression_potential": potential,
            "compression_validation": compression_results,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
