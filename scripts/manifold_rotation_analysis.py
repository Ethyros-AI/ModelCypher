#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Manifold Rotation Analysis
"""
Manifold Rotation Analysis

Measures both hypotheses for why weight projection fails:
1. Delta hypothesis: The layer CONTRIBUTION (delta = h_out - h_in) is low-dimensional
2. Rotation hypothesis: The input manifold rotates to a different output manifold

Key insight: We were measuring the CUMULATIVE residual stream, not the layer contribution.
The residual stream accumulates: h_L = h_0 + Σ delta_l
Even if each delta is 3D, the sum spans many more dimensions.

Usage:
    python manifold_rotation_analysis.py \
        --model /path/to/model \
        --all-layers
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Reuse semantic primes from the analysis script
SEMANTIC_PRIMES = {
    "substantives": ["I", "you", "someone", "something", "people", "body"],
    "determiners": ["this", "the same", "other", "else"],
    "quantifiers": ["one", "two", "some", "all", "much", "many", "little", "few"],
    "evaluators": ["good", "bad"],
    "descriptors": ["big", "small"],
    "mental": ["think", "know", "want", "feel", "see", "hear"],
    "speech": ["say", "words", "true"],
    "actions": ["do", "happen", "move"],
    "existence": ["there is", "be", "live", "die"],
    "possession": ["have", "part"],
    "logical": ["not", "maybe", "can", "because", "if"],
    "time": ["when", "now", "before", "after", "a long time", "a short time", "moment"],
    "space": ["where", "here", "above", "below", "far", "near", "side", "inside", "touch"],
    "taxonomy": ["kind of", "like"],
}


@dataclass
class LayerAnalysis:
    """Analysis results for one layer."""
    layer_idx: int
    input_dim: int      # Effective dimension of h_in (99% variance)
    output_dim: int     # Effective dimension of h_out (99% variance)
    delta_dim: int      # Effective dimension of delta (99% variance)
    cka_in_out: float   # CKA between input and output manifolds
    input_var: float    # Variance captured at input_dim
    output_var: float   # Variance captured at output_dim
    delta_var: float    # Variance captured at delta_dim


def initialize_backend() -> "Backend":
    """Initialize the MLX backend."""
    from modelcypher.backends import initialize_default_backend
    from modelcypher.core.domain._backend import get_default_backend

    initialize_default_backend()
    return get_default_backend()


def load_model(model_path: str) -> tuple[Any, Any, dict]:
    """Load MLX model and tokenizer."""
    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", model_path)
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

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


def get_prime_contexts() -> list[tuple[str, str, str]]:
    """Get semantic primes with minimal contexts for activation.

    Returns list of (prime, context, category) tuples.
    Using minimal context to get pure semantic activation.
    (Copied from semantic_prime_manifold.py for consistency)
    """
    contexts = []

    for category, primes in SEMANTIC_PRIMES.items():
        for prime in primes:
            # Use prime in minimal sentence context
            # Goal: activate the semantic concept, not syntax
            if prime in ["I", "you", "someone", "something", "people", "body"]:
                context = prime  # Bare noun/pronoun
            elif prime in ["this", "the same", "other", "else"]:
                context = f"{prime} thing"
            elif prime in ["one", "two", "some", "all", "many", "much", "little", "few"]:
                context = f"{prime} things"
            elif prime in ["good", "bad", "big", "small", "true"]:
                context = f"It is {prime}"
            elif prime in ["more", "very"]:
                context = f"{prime} good"
            elif prime in ["think", "know", "want", "feel", "see", "hear"]:
                context = f"I {prime}"
            elif prime in ["say"]:
                context = "I say"
            elif prime in ["words"]:
                context = "words"
            elif prime in ["do", "happen", "move"]:
                context = f"Things {prime}"
            elif prime in ["there is"]:
                context = "There is something"
            elif prime in ["be"]:
                context = "I am"
            elif prime in ["live", "die"]:
                context = f"People {prime}"
            elif prime in ["have", "part"]:
                context = f"I have"
            elif prime in ["not"]:
                context = "not this"
            elif prime in ["maybe"]:
                context = "maybe"
            elif prime in ["can"]:
                context = "I can"
            elif prime in ["because"]:
                context = "because"
            elif prime in ["if"]:
                context = "if"
            elif prime in ["when", "now", "before", "after"]:
                context = prime
            elif prime in ["a long time", "a short time", "moment"]:
                context = prime
            elif prime in ["where", "here"]:
                context = prime
            elif prime in ["above", "below", "far", "near", "inside"]:
                context = prime
            elif prime in ["side"]:
                context = "side"
            elif prime in ["touch"]:
                context = "touch"
            elif prime in ["kind of", "part of", "like"]:
                context = prime
            else:
                context = prime

            contexts.append((prime, context, category))

    return contexts


def collect_layer_io_activations(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
) -> tuple["Array", "Array", "Array"]:
    """Collect input, output, and delta activations for a layer.

    Returns:
        Tuple of (h_in, h_out, delta) arrays, each [n_samples, hidden_dim]
    """
    import mlx.core as mx
    from modelcypher.adapters.model_architecture import get_model_architecture

    arch = get_model_architecture(model, config=config)
    contexts = get_prime_contexts()

    h_ins = []
    h_outs = []
    deltas = []

    for prime, context, category in contexts:
        try:
            # Tokenize
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            # Get embeddings
            embed = arch.embed_module
            h = embed(input_ids)
            mx.eval(h)

            # Process layers up to target
            for idx, layer in enumerate(arch.layers):
                if idx == layer_idx:
                    # Save input to this layer (mx.array creates a copy)
                    h_in = mx.array(h)
                    mx.eval(h_in)

                # Forward through layer
                result = layer(h)
                if isinstance(result, tuple):
                    h = result[0]
                else:
                    h = result
                mx.eval(h)

                if idx == layer_idx:
                    # Save output from this layer
                    h_out = mx.array(h)
                    mx.eval(h_out)
                    break

            # Pool to get single vector (last token for causal)
            h_in_pooled = h_in[0, -1, :]
            h_out_pooled = h_out[0, -1, :]
            delta_pooled = h_out_pooled - h_in_pooled
            mx.eval(h_in_pooled, h_out_pooled, delta_pooled)

            h_ins.append(h_in_pooled)
            h_outs.append(h_out_pooled)
            deltas.append(delta_pooled)

        except Exception as e:
            logger.debug("Failed on '%s': %s", prime, e)
            continue

    if not h_ins:
        raise ValueError(f"No activations collected for layer {layer_idx}")

    # Stack into arrays
    H_in = mx.stack(h_ins, axis=0)
    H_out = mx.stack(h_outs, axis=0)
    Delta = mx.stack(deltas, axis=0)
    mx.eval(H_in, H_out, Delta)

    return H_in, H_out, Delta


def collect_layer_component_deltas(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
) -> tuple["Array", "Array", "Array"]:
    """Collect attention and MLP deltas separately for a layer.

    This decomposes delta = attn_delta + mlp_delta to understand
    where the low-rank structure comes from.

    Returns:
        Tuple of (attn_delta, mlp_delta, total_delta) arrays, each [n_samples, hidden_dim]
    """
    import mlx.core as mx
    from modelcypher.adapters.model_architecture import get_model_architecture

    arch = get_model_architecture(model, config=config)
    contexts = get_prime_contexts()

    attn_deltas = []
    mlp_deltas = []
    total_deltas = []

    for prime, context, category in contexts:
        try:
            # Tokenize
            tokens = tokenizer.encode(context)
            input_ids = mx.array([tokens])
            mx.eval(input_ids)

            # Get embeddings
            embed = arch.embed_module
            h = embed(input_ids)
            mx.eval(h)

            # Process layers up to target
            for idx, layer_obj in enumerate(arch.layers):
                if idx < layer_idx:
                    # Just forward
                    result = layer_obj(h)
                    if isinstance(result, tuple):
                        h = result[0]
                    else:
                        h = result
                    mx.eval(h)
                elif idx == layer_idx:
                    # Decompose this layer's computation
                    h_in = mx.array(h)
                    mx.eval(h_in)

                    # For LFM2 architecture, layer structure is:
                    # h_attn = h + attn(norm1(h))
                    # h_out = h_attn + mlp(norm2(h_attn))

                    # Access layer components
                    # Check architecture by available keys (LFM2 uses dict-like access)
                    layer_keys = list(layer_obj.keys()) if hasattr(layer_obj, 'keys') else []

                    if 'operator_norm' in layer_keys:
                        # LFM2 architecture (may have conv OR self_attn)
                        norm1 = layer_obj['operator_norm']
                        norm2 = layer_obj['ffn_norm']
                        mlp = layer_obj['feed_forward']
                        # Some layers use conv, others use self_attn
                        if 'conv' in layer_keys:
                            self_attn = layer_obj['conv']
                        else:
                            self_attn = layer_obj['self_attn']
                    elif hasattr(layer_obj, 'input_layernorm'):
                        # Standard Llama-like architecture
                        norm1 = layer_obj.input_layernorm
                        norm2 = layer_obj.post_attention_layernorm
                        self_attn = layer_obj.self_attn
                        mlp = layer_obj.mlp
                    elif hasattr(layer_obj, 'ln_1'):
                        # GPT-2 style architecture
                        norm1 = layer_obj.ln_1
                        norm2 = layer_obj.ln_2
                        self_attn = layer_obj.attn
                        mlp = layer_obj.mlp
                    else:
                        raise ValueError(f"Unknown layer architecture: {type(layer_obj)}, keys={layer_keys}")

                    # Attention path
                    h_normed = norm1(h)
                    mx.eval(h_normed)

                    attn_out = self_attn(h_normed)
                    if isinstance(attn_out, tuple):
                        attn_out = attn_out[0]
                    mx.eval(attn_out)

                    # After attention residual
                    h_attn = h + attn_out
                    mx.eval(h_attn)

                    # MLP path
                    h_attn_normed = norm2(h_attn)
                    mx.eval(h_attn_normed)

                    mlp_out = mlp(h_attn_normed)
                    mx.eval(mlp_out)

                    # Final output
                    h_out = h_attn + mlp_out
                    mx.eval(h_out)

                    # Pool to single vector (last token)
                    attn_delta = attn_out[0, -1, :]
                    mlp_delta = mlp_out[0, -1, :]
                    total_delta = h_out[0, -1, :] - h_in[0, -1, :]
                    mx.eval(attn_delta, mlp_delta, total_delta)

                    attn_deltas.append(attn_delta)
                    mlp_deltas.append(mlp_delta)
                    total_deltas.append(total_delta)
                    break

        except Exception as e:
            logger.debug("Failed component decomposition on '%s': %s", prime, e)
            continue

    if not attn_deltas:
        raise ValueError(f"No component deltas collected for layer {layer_idx}")

    # Stack into arrays
    Attn_delta = mx.stack(attn_deltas, axis=0)
    Mlp_delta = mx.stack(mlp_deltas, axis=0)
    Total_delta = mx.stack(total_deltas, axis=0)
    mx.eval(Attn_delta, Mlp_delta, Total_delta)

    return Attn_delta, Mlp_delta, Total_delta


def compute_effective_dim(
    X: "Array",
    backend: "Backend",
    target_variance: float = 0.99,
) -> tuple[int, float]:
    """Compute effective dimensionality via PCA.

    Returns:
        Tuple of (effective_dim, variance_captured)
    """
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend
    X = _promote_precision_float32(b.array(X), b)
    b.eval(X)

    shape = b.shape(X)
    n_samples = int(shape[0])

    # Center
    mean = b.mean(X, axis=0)
    b.eval(mean)
    X_centered = X - mean
    b.eval(X_centered)

    # Gram matrix
    G = b.matmul(X_centered, b.transpose(X_centered))
    b.eval(G)

    # Eigendecomposition
    eigenvalues, _ = b.eigh(G)
    b.eval(eigenvalues)

    # Sort descending
    indices = b.argsort(-eigenvalues)
    b.eval(indices)
    eigenvalues = eigenvalues[indices]
    b.eval(eigenvalues)

    # Find dimension for target variance
    eigenvalues_list = [max(0, e) for e in eigenvalues.tolist()]
    total_var = sum(eigenvalues_list)

    if total_var == 0:
        return n_samples, 1.0

    cumvar = 0.0
    effective_dim = 0
    for i, eig in enumerate(eigenvalues_list):
        cumvar += eig
        effective_dim = i + 1
        if cumvar / total_var >= target_variance:
            break

    variance_captured = cumvar / total_var
    return max(effective_dim, 1), variance_captured


def compute_cka(
    X: "Array",
    Y: "Array",
    backend: "Backend",
) -> float:
    """Compute Centered Kernel Alignment between two activation sets."""
    from modelcypher.core.domain.geometry.precision_utils import (
        _promote_precision_float32,
    )

    b = backend

    X = _promote_precision_float32(b.array(X), b)
    Y = _promote_precision_float32(b.array(Y), b)
    b.eval(X, Y)

    # Center
    X_mean = b.mean(X, axis=0)
    Y_mean = b.mean(Y, axis=0)
    b.eval(X_mean, Y_mean)
    X_c = X - X_mean
    Y_c = Y - Y_mean
    b.eval(X_c, Y_c)

    # Gram matrices
    K_X = b.matmul(X_c, b.transpose(X_c))
    K_Y = b.matmul(Y_c, b.transpose(Y_c))
    b.eval(K_X, K_Y)

    # CKA = trace(K_X @ K_Y) / sqrt(trace(K_X @ K_X) * trace(K_Y @ K_Y))
    n = int(b.shape(X)[0])

    # Centering matrix
    H = b.eye(n) - (1.0 / n) * b.ones((n, n))
    b.eval(H)

    # Centered Gram matrices
    K_X_c = b.matmul(H, b.matmul(K_X, H))
    K_Y_c = b.matmul(H, b.matmul(K_Y, H))
    b.eval(K_X_c, K_Y_c)

    # HSIC
    hsic_xy = b.sum(K_X_c * K_Y_c)
    hsic_xx = b.sum(K_X_c * K_X_c)
    hsic_yy = b.sum(K_Y_c * K_Y_c)
    b.eval(hsic_xy, hsic_xx, hsic_yy)

    hsic_xy_val = float(b.to_scalar(hsic_xy))
    hsic_xx_val = float(b.to_scalar(hsic_xx))
    hsic_yy_val = float(b.to_scalar(hsic_yy))

    denom = (hsic_xx_val * hsic_yy_val) ** 0.5
    if denom < 1e-10:
        return 0.0

    return hsic_xy_val / denom


def analyze_layer(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
    target_variance: float = 0.99,
) -> LayerAnalysis:
    """Analyze a single layer."""
    # Collect activations
    H_in, H_out, Delta = collect_layer_io_activations(
        model, tokenizer, config, layer_idx, backend
    )

    # Compute effective dimensions
    input_dim, input_var = compute_effective_dim(H_in, backend, target_variance)
    output_dim, output_var = compute_effective_dim(H_out, backend, target_variance)
    delta_dim, delta_var = compute_effective_dim(Delta, backend, target_variance)

    # Compute CKA between input and output manifolds
    cka = compute_cka(H_in, H_out, backend)

    return LayerAnalysis(
        layer_idx=layer_idx,
        input_dim=input_dim,
        output_dim=output_dim,
        delta_dim=delta_dim,
        cka_in_out=cka,
        input_var=input_var,
        output_var=output_var,
        delta_var=delta_var,
    )


def analyze_delta_components(
    model: Any,
    tokenizer: Any,
    config: dict,
    layer_idx: int,
    backend: "Backend",
    target_variance: float = 0.99,
) -> dict:
    """Analyze the attention and MLP components of delta.

    Returns dict with dimensionality info for each component.
    """
    import mlx.core as mx

    Attn_delta, Mlp_delta, Total_delta = collect_layer_component_deltas(
        model, tokenizer, config, layer_idx, backend
    )

    attn_dim, attn_var = compute_effective_dim(Attn_delta, backend, target_variance)
    mlp_dim, mlp_var = compute_effective_dim(Mlp_delta, backend, target_variance)
    total_dim, total_var = compute_effective_dim(Total_delta, backend, target_variance)

    # Also compute norms to see relative contribution
    attn_norms = [float(mx.linalg.norm(Attn_delta[i]).item()) for i in range(Attn_delta.shape[0])]
    mlp_norms = [float(mx.linalg.norm(Mlp_delta[i]).item()) for i in range(Mlp_delta.shape[0])]
    total_norms = [float(mx.linalg.norm(Total_delta[i]).item()) for i in range(Total_delta.shape[0])]

    mean_attn_norm = sum(attn_norms) / len(attn_norms)
    mean_mlp_norm = sum(mlp_norms) / len(mlp_norms)
    mean_total_norm = sum(total_norms) / len(total_norms)

    # Find dominant dimension in each
    def find_dominant_dim(X):
        """Find the dimension with most energy."""
        b = backend
        X_arr = b.array(X)
        b.eval(X_arr)
        # Sum squared values per dimension
        energy = b.sum(X_arr * X_arr, axis=0)
        b.eval(energy)
        energy_list = energy.tolist()
        max_dim = max(range(len(energy_list)), key=lambda i: energy_list[i])
        max_energy = energy_list[max_dim]
        total_energy = sum(energy_list)
        return max_dim, max_energy / total_energy if total_energy > 0 else 0.0

    attn_dom_dim, attn_dom_frac = find_dominant_dim(Attn_delta)
    mlp_dom_dim, mlp_dom_frac = find_dominant_dim(Mlp_delta)
    total_dom_dim, total_dom_frac = find_dominant_dim(Total_delta)

    return {
        "attn": {
            "dim": attn_dim,
            "var": attn_var,
            "mean_norm": mean_attn_norm,
            "dominant_dim": attn_dom_dim,
            "dominant_frac": attn_dom_frac,
        },
        "mlp": {
            "dim": mlp_dim,
            "var": mlp_var,
            "mean_norm": mean_mlp_norm,
            "dominant_dim": mlp_dom_dim,
            "dominant_frac": mlp_dom_frac,
        },
        "total": {
            "dim": total_dim,
            "var": total_var,
            "mean_norm": mean_total_norm,
            "dominant_dim": total_dom_dim,
            "dominant_frac": total_dom_frac,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Manifold rotation analysis"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--layer", type=int, default=None, help="Single layer to analyze")
    parser.add_argument("--all-layers", action="store_true", help="Analyze all layers")
    parser.add_argument("--decompose", action="store_true", help="Decompose delta into attn/mlp")
    parser.add_argument("--target-variance", type=float, default=0.99,
                        help="Target variance for dimension estimation (default: 0.99)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.layer is None and not args.all_layers:
        parser.error("Must specify --layer or --all-layers")

    # Initialize
    backend = initialize_backend()
    model, tokenizer, config = load_model(args.model)
    num_layers = config.get("num_hidden_layers", 0)
    hidden_dim = config.get("hidden_size", 0)

    # Determine layers to analyze
    if args.all_layers:
        layers = list(range(num_layers))
    else:
        layers = [args.layer]

    # Run delta decomposition if requested
    if args.decompose:
        logger.info("\n=== DELTA DECOMPOSITION ANALYSIS ===")
        logger.info("Target variance: %.1f%%", args.target_variance * 100)
        logger.info("")

        for layer_idx in layers:
            try:
                logger.info("Layer %d:", layer_idx)
                comp = analyze_delta_components(
                    model, tokenizer, config, layer_idx, backend, args.target_variance
                )

                logger.info("  ATTN: dim=%d (%.1f%% var), norm=%.4f, dom_dim=%d (%.1f%%)",
                           comp["attn"]["dim"], comp["attn"]["var"] * 100,
                           comp["attn"]["mean_norm"], comp["attn"]["dominant_dim"],
                           comp["attn"]["dominant_frac"] * 100)
                logger.info("  MLP:  dim=%d (%.1f%% var), norm=%.4f, dom_dim=%d (%.1f%%)",
                           comp["mlp"]["dim"], comp["mlp"]["var"] * 100,
                           comp["mlp"]["mean_norm"], comp["mlp"]["dominant_dim"],
                           comp["mlp"]["dominant_frac"] * 100)
                logger.info("  TOTAL: dim=%d (%.1f%% var), norm=%.4f, dom_dim=%d (%.1f%%)",
                           comp["total"]["dim"], comp["total"]["var"] * 100,
                           comp["total"]["mean_norm"], comp["total"]["dominant_dim"],
                           comp["total"]["dominant_frac"] * 100)

                # Which component dominates?
                if comp["attn"]["mean_norm"] > comp["mlp"]["mean_norm"]:
                    ratio = comp["attn"]["mean_norm"] / comp["mlp"]["mean_norm"]
                    logger.info("  => ATTN dominates (%.1fx larger norm)", ratio)
                else:
                    ratio = comp["mlp"]["mean_norm"] / comp["attn"]["mean_norm"]
                    logger.info("  => MLP dominates (%.1fx larger norm)", ratio)

                logger.info("")
            except Exception as e:
                logger.warning("Layer %d decomposition failed: %s", layer_idx, e)

        return

    # Run analysis
    logger.info("\n=== MANIFOLD ROTATION ANALYSIS ===")
    logger.info("Target variance: %.1f%%", args.target_variance * 100)
    logger.info("Hidden dim: %d", hidden_dim)
    logger.info("")

    # Table header
    logger.info(
        "Layer | In_dim | Out_dim | Delta_dim | CKA(in,out) | Notes"
    )
    logger.info(
        "------|--------|---------|-----------|-------------|------"
    )

    results = []
    for layer_idx in layers:
        try:
            analysis = analyze_layer(
                model, tokenizer, config, layer_idx, backend, args.target_variance
            )
            results.append(analysis)

            # Determine notes
            notes = []
            if analysis.delta_dim < analysis.output_dim:
                notes.append(f"delta {analysis.output_dim - analysis.delta_dim}D smaller!")
            if analysis.cka_in_out < 0.95:
                notes.append("rotates")
            if analysis.output_dim <= 5:
                notes.append("highway")

            logger.info(
                "  %2d  |   %3d  |   %3d   |    %3d    |    %.4f   | %s",
                layer_idx,
                analysis.input_dim,
                analysis.output_dim,
                analysis.delta_dim,
                analysis.cka_in_out,
                ", ".join(notes) if notes else "",
            )

        except Exception as e:
            logger.warning("Layer %d failed: %s", layer_idx, e)

    # Summary
    logger.info("")
    logger.info("=== SUMMARY ===")

    if results:
        # Check delta hypothesis
        delta_smaller = sum(1 for r in results if r.delta_dim < r.output_dim)
        logger.info(
            "Delta hypothesis: %d/%d layers have delta_dim < output_dim",
            delta_smaller, len(results)
        )

        # Check rotation hypothesis
        rotating = sum(1 for r in results if r.cka_in_out < 0.95)
        logger.info(
            "Rotation hypothesis: %d/%d layers have CKA < 0.95",
            rotating, len(results)
        )

        # Highway layers
        highway = [r for r in results if r.output_dim <= 5]
        if highway:
            logger.info(
                "Highway layers (output_dim <= 5): %s",
                [r.layer_idx for r in highway]
            )
            for r in highway:
                logger.info(
                    "  Layer %d: in=%dD, out=%dD, delta=%dD, CKA=%.4f",
                    r.layer_idx, r.input_dim, r.output_dim, r.delta_dim, r.cka_in_out
                )

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "target_variance": args.target_variance,
            "hidden_dim": hidden_dim,
            "layers": [
                {
                    "layer_idx": r.layer_idx,
                    "input_dim": r.input_dim,
                    "output_dim": r.output_dim,
                    "delta_dim": r.delta_dim,
                    "cka_in_out": r.cka_in_out,
                    "input_var": r.input_var,
                    "output_var": r.output_var,
                    "delta_var": r.delta_var,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("\nSaved results to %s", args.output)


if __name__ == "__main__":
    main()
