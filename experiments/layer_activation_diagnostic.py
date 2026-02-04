#!/usr/bin/env python3
"""Layer-by-layer activation diagnostic for merged models.

Identifies where activations diverge between target and merged models.
Key metrics:
- Variance ratio: merged_var / target_var (should be ~1.0)
- Magnitude ratio: merged_norm / target_norm (should be ~1.0)
- Correlation: how similar are the activation patterns
- CKA: representation similarity

If variance ratio << 1.0, variance collapse is occurring (REPAIR paper).
"""

import json
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelcypher.adapters.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_layer_activations(model, input_ids: mx.array) -> dict[int, mx.array]:
    """Collect hidden state after each transformer layer.

    Returns dict mapping layer_idx -> activations [seq_len, hidden_dim].
    Does NOT pool - preserves full sequence structure.
    """
    activations = {}

    # Get embedding
    if hasattr(model, "model"):
        # Huggingface-style model
        base = model.model
    else:
        base = model

    # Add batch dimension if needed: [seq] -> [1, seq]
    if len(input_ids.shape) == 1:
        input_ids = input_ids.reshape(1, -1)

    # Embed tokens
    if hasattr(base, "embed_tokens"):
        h = base.embed_tokens(input_ids)
    elif hasattr(base, "wte"):
        h = base.wte(input_ids)
    else:
        raise ValueError("Cannot find embedding layer")

    mx.eval(h)
    # h is now [batch, seq, hidden] - keep batch dim for layer forward pass
    activations[-1] = h[0]  # Store without batch: [seq, hidden]

    # Get layers
    if hasattr(base, "layers"):
        layers = base.layers
    elif hasattr(base, "h"):
        layers = base.h
    else:
        raise ValueError("Cannot find transformer layers")

    # Create attention mask [seq, seq]
    seq_len = h.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Run through each layer
    for i, layer in enumerate(layers):
        try:
            # Most transformer layers take (hidden_states, mask, cache)
            # Try different signatures
            try:
                result = layer(h, mask=mask, cache=None)
            except TypeError:
                try:
                    result = layer(h, mask, None)
                except TypeError:
                    result = layer(h, mask)

            if isinstance(result, tuple):
                h = result[0]  # (hidden, cache) or (hidden, cache, ...)
            else:
                h = result

            mx.eval(h)
            activations[i] = h[0]  # Store without batch: [seq, hidden]
        except Exception as e:
            logger.warning(f"Layer {i} failed: {e}")
            break

    return activations


def compute_activation_stats(acts: mx.array) -> dict:
    """Compute statistics for an activation tensor [seq_len, hidden_dim]."""
    # Flatten for overall stats
    flat = acts.reshape(-1)

    # Per-neuron variance (across sequence)
    var_per_neuron = mx.var(acts, axis=0)  # [hidden_dim]
    mean_var = float(mx.mean(var_per_neuron))

    # Overall variance
    overall_var = float(mx.var(flat))

    # Magnitude (Frobenius norm)
    magnitude = float(mx.sqrt(mx.sum(flat * flat)))

    # Mean absolute value
    mean_abs = float(mx.mean(mx.abs(flat)))

    # Max absolute value
    max_abs = float(mx.max(mx.abs(flat)))

    # Min/max values
    min_val = float(mx.min(flat))
    max_val = float(mx.max(flat))

    return {
        "mean_neuron_var": mean_var,
        "overall_var": overall_var,
        "magnitude": magnitude,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "min": min_val,
        "max": max_val,
        "shape": list(acts.shape),
    }


def compute_correlation(acts1: mx.array, acts2: mx.array) -> float:
    """Compute Pearson correlation between two activation tensors."""
    flat1 = acts1.reshape(-1)
    flat2 = acts2.reshape(-1)

    mean1 = mx.mean(flat1)
    mean2 = mx.mean(flat2)

    centered1 = flat1 - mean1
    centered2 = flat2 - mean2

    cov = mx.sum(centered1 * centered2)
    std1 = mx.sqrt(mx.sum(centered1 * centered1))
    std2 = mx.sqrt(mx.sum(centered2 * centered2))

    corr = cov / (std1 * std2 + 1e-8)
    return float(corr)


def compute_cka(acts1: mx.array, acts2: mx.array) -> float:
    """Compute linear CKA between two activation tensors [seq_len, hidden_dim]."""
    # Center the activations
    acts1 = acts1 - mx.mean(acts1, axis=0, keepdims=True)
    acts2 = acts2 - mx.mean(acts2, axis=0, keepdims=True)

    # Compute Gram matrices
    K1 = mx.matmul(acts1, mx.transpose(acts1))  # [seq_len, seq_len]
    K2 = mx.matmul(acts2, mx.transpose(acts2))

    # CKA = <K1, K2>_F / (||K1||_F * ||K2||_F)
    hsic = mx.sum(K1 * K2)
    norm1 = mx.sqrt(mx.sum(K1 * K1))
    norm2 = mx.sqrt(mx.sum(K2 * K2))

    cka = hsic / (norm1 * norm2 + 1e-8)
    return float(cka)


def run_diagnostic(
    target_model_path: str,
    merged_model_path: str,
    test_prompt: str = "The capital of France is Paris, which is known for",
) -> dict:
    """Run layer-by-layer diagnostic comparing target and merged models."""

    logger.info("=" * 80)
    logger.info("LAYER-BY-LAYER ACTIVATION DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info(f"Target: {target_model_path}")
    logger.info(f"Merged: {merged_model_path}")
    logger.info(f"Prompt: {test_prompt!r}")
    logger.info("=" * 80)

    # Load models
    loader = ModelLoader()

    logger.info("Loading target model...")
    target_model, target_tokenizer = loader.load_model_for_training(target_model_path)

    logger.info("Loading merged model...")
    merged_model, merged_tokenizer = loader.load_model_for_training(merged_model_path)

    # Tokenize
    input_ids = mx.array(target_tokenizer.encode(test_prompt))
    logger.info(f"Input tokens: {input_ids.shape[0]}")

    # Collect activations
    logger.info("Collecting target activations...")
    target_acts = collect_layer_activations(target_model, input_ids)

    logger.info("Collecting merged activations...")
    merged_acts = collect_layer_activations(merged_model, input_ids)

    # Compare layer by layer
    results = {
        "prompt": test_prompt,
        "num_tokens": int(input_ids.shape[0]),
        "layers": {},
    }

    common_layers = sorted(set(target_acts.keys()) & set(merged_acts.keys()))

    logger.info("\n" + "=" * 80)
    logger.info("LAYER-BY-LAYER COMPARISON")
    logger.info("=" * 80)
    logger.info(
        f"{'Layer':>6} | {'VarRatio':>10} | {'MagRatio':>10} | "
        f"{'Corr':>8} | {'CKA':>8} | {'TgtVar':>12} | {'MrgVar':>12}"
    )
    logger.info("-" * 80)

    cumulative_var_ratio = 1.0

    for layer_idx in common_layers:
        tgt = target_acts[layer_idx]
        mrg = merged_acts[layer_idx]

        # Handle shape mismatches
        if tgt.shape != mrg.shape:
            logger.warning(
                f"Layer {layer_idx}: Shape mismatch target={tgt.shape} merged={mrg.shape}"
            )
            results["layers"][layer_idx] = {
                "error": f"Shape mismatch: {tgt.shape} vs {mrg.shape}"
            }
            continue

        tgt_stats = compute_activation_stats(tgt)
        mrg_stats = compute_activation_stats(mrg)

        # Compute ratios
        var_ratio = mrg_stats["mean_neuron_var"] / (tgt_stats["mean_neuron_var"] + 1e-12)
        mag_ratio = mrg_stats["magnitude"] / (tgt_stats["magnitude"] + 1e-12)

        # Compute similarity metrics
        corr = compute_correlation(tgt, mrg)
        cka = compute_cka(tgt, mrg)

        # Track cumulative variance loss
        if layer_idx >= 0:  # Skip embedding
            cumulative_var_ratio *= var_ratio

        layer_name = "Emb" if layer_idx == -1 else f"L{layer_idx:02d}"
        logger.info(
            f"{layer_name:>6} | {var_ratio:>10.4f} | {mag_ratio:>10.4f} | "
            f"{corr:>8.4f} | {cka:>8.4f} | {tgt_stats['mean_neuron_var']:>12.4e} | "
            f"{mrg_stats['mean_neuron_var']:>12.4e}"
        )

        results["layers"][layer_idx] = {
            "target_stats": tgt_stats,
            "merged_stats": mrg_stats,
            "var_ratio": var_ratio,
            "mag_ratio": mag_ratio,
            "correlation": corr,
            "cka": cka,
        }

    logger.info("-" * 80)

    # Summary statistics
    var_ratios = [r["var_ratio"] for r in results["layers"].values() if "var_ratio" in r]
    mag_ratios = [r["mag_ratio"] for r in results["layers"].values() if "mag_ratio" in r]
    correlations = [r["correlation"] for r in results["layers"].values() if "correlation" in r]
    ckas = [r["cka"] for r in results["layers"].values() if "cka" in r]

    results["summary"] = {
        "mean_var_ratio": sum(var_ratios) / len(var_ratios) if var_ratios else 0,
        "min_var_ratio": min(var_ratios) if var_ratios else 0,
        "cumulative_var_ratio": cumulative_var_ratio,
        "mean_mag_ratio": sum(mag_ratios) / len(mag_ratios) if mag_ratios else 0,
        "mean_correlation": sum(correlations) / len(correlations) if correlations else 0,
        "mean_cka": sum(ckas) / len(ckas) if ckas else 0,
        "min_cka": min(ckas) if ckas else 0,
    }

    logger.info("\nSUMMARY:")
    logger.info(f"  Mean variance ratio: {results['summary']['mean_var_ratio']:.4f}")
    logger.info(f"  Min variance ratio:  {results['summary']['min_var_ratio']:.4f}")
    logger.info(f"  Cumulative var loss: {results['summary']['cumulative_var_ratio']:.6f}")
    logger.info(f"  Mean magnitude ratio: {results['summary']['mean_mag_ratio']:.4f}")
    logger.info(f"  Mean correlation:    {results['summary']['mean_correlation']:.4f}")
    logger.info(f"  Mean CKA:            {results['summary']['mean_cka']:.4f}")
    logger.info(f"  Min CKA:             {results['summary']['min_cka']:.4f}")

    # Diagnosis
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS")
    logger.info("=" * 80)

    if results["summary"]["cumulative_var_ratio"] < 0.01:
        logger.warning(
            "VARIANCE COLLAPSE DETECTED: Cumulative variance dropped to "
            f"{results['summary']['cumulative_var_ratio']*100:.2f}% of target. "
            "This causes all outputs to converge to same value. "
            "Solution: Apply REPAIR-style variance rescaling."
        )
    elif results["summary"]["mean_var_ratio"] < 0.5:
        logger.warning(
            f"VARIANCE REDUCTION: Mean layer variance is {results['summary']['mean_var_ratio']*100:.1f}% "
            "of target. This compounds through layers and may cause issues."
        )

    if results["summary"]["min_cka"] < 0.5:
        # Find the problematic layers
        bad_layers = [
            l for l, r in results["layers"].items()
            if "cka" in r and r["cka"] < 0.5
        ]
        logger.warning(
            f"LOW CKA at layers {bad_layers}: Merged representations diverge significantly. "
            "Check weight stitching at these layers."
        )

    if results["summary"]["mean_correlation"] < 0.3:
        logger.warning(
            "LOW CORRELATION: Merged activations don't correlate with target. "
            "Weight alignment may be fundamentally wrong."
        )

    if (results["summary"]["mean_var_ratio"] > 0.8 and
        results["summary"]["mean_cka"] > 0.8 and
        results["summary"]["mean_correlation"] > 0.7):
        logger.info(
            "ACTIVATIONS LOOK REASONABLE. If output is still garbage, "
            "check LM head or final LayerNorm."
        )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer activation diagnostic")
    parser.add_argument("--target", required=True, help="Path to target model")
    parser.add_argument("--merged", required=True, help="Path to merged model")
    parser.add_argument("--prompt", default="The capital of France is Paris, which is known for")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    results = run_diagnostic(args.target, args.merged, args.prompt)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
