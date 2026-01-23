#!/usr/bin/env python3
# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# Bottleneck Perturbation Analysis
"""
Bottleneck Perturbation Analysis

THE QUESTION:
If we modify the 1D bottleneck value, what happens to the output?

PERTURBATIONS:
1. Scale by 0.5, 1.5, 2.0 (amplitude)
2. Invert sign (flip)
3. Set to zero (ablation)
4. Add noise (robustness)

If the 1D is causally important, perturbations should change outputs dramatically.
If the 1D is redundant, perturbations should have little effect.

METHOD:
Run forward pass with hooks that modify activations at bottleneck layers.
Compare output logits/tokens before and after perturbation.

Usage:
    python bottleneck_perturbation.py --model /path/to/model
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


TEST_PROMPTS = [
    "The capital of France is",
    "Dogs are known for being",
    "The color of the sky is",
    "Love is a feeling of",
    "Mathematics is the study of",
]


@dataclass
class PerturbationResult:
    """Results from a single perturbation experiment."""
    prompt: str
    perturbation: str
    layer_idx: int

    # Original output
    original_top_token: str
    original_top_prob: float

    # Perturbed output
    perturbed_top_token: str
    perturbed_top_prob: float

    # Divergence
    kl_divergence: float
    top_token_changed: bool


def get_logits_with_hook(
    model: Any,
    tokenizer: Any,
    prompt: str,
    hook_layer: int | None = None,
    hook_fn: Callable | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run forward pass with optional hook at specified layer.

    Returns:
        (logits, tokens) for the final position
    """
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    if hook_layer is None or hook_fn is None:
        # No hook - just run the model directly
        logits = model(input_ids)
        mx.eval(logits)
        logits_np = np.array(logits[0, -1, :].astype(mx.float32))
        return logits_np, tokens

    # With hook - need to manually run layers
    # Get the internal model
    inner_model = model.model if hasattr(model, 'model') else model

    h = inner_model.embed_tokens(input_ids)
    mx.eval(h)

    for idx, layer in enumerate(inner_model.layers):
        result = layer(h)
        h = result[0] if isinstance(result, tuple) else result
        mx.eval(h)

        # Apply hook after specified layer
        if idx == hook_layer:
            h_np = np.array(h.astype(mx.float32))
            h_modified = hook_fn(h_np)
            h = mx.array(h_modified).astype(h.dtype)
            mx.eval(h)

    # Final norm (some models have embedding_norm instead of norm)
    if hasattr(inner_model, 'norm'):
        h = inner_model.norm(h)
        mx.eval(h)
    elif hasattr(inner_model, 'embedding_norm'):
        h = inner_model.embedding_norm(h)
        mx.eval(h)

    # LM head - check different possible locations
    if hasattr(model, 'lm_head'):
        logits = model.lm_head(h)
    elif hasattr(inner_model, 'lm_head'):
        logits = inner_model.lm_head(h)
    elif hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        # Tied embeddings - use embed_tokens as the linear projection
        logits = inner_model.embed_tokens.as_linear(h)
    else:
        # Fallback - won't have hook effect
        logits = model(input_ids)

    mx.eval(logits)
    logits_np = np.array(logits[0, -1, :].astype(mx.float32))

    return logits_np, tokens


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D(p || q)."""
    # Softmax
    p = np.exp(p - np.max(p))
    p = p / p.sum()
    q = np.exp(q - np.max(q))
    q = q / q.sum()

    # KL with smoothing
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log(p / q)))


def scale_perturbation(scale: float) -> Callable:
    """Create a hook that scales the delta by a factor."""
    def hook(h: np.ndarray) -> np.ndarray:
        # Scale the last position's activation
        h[:, -1, :] *= scale
        return h
    return hook


def invert_perturbation() -> Callable:
    """Create a hook that inverts the sign of activations."""
    def hook(h: np.ndarray) -> np.ndarray:
        h[:, -1, :] *= -1
        return h
    return hook


def zero_perturbation() -> Callable:
    """Create a hook that zeros out activations."""
    def hook(h: np.ndarray) -> np.ndarray:
        h[:, -1, :] = 0
        return h
    return hook


def noise_perturbation(std: float = 0.1) -> Callable:
    """Create a hook that adds Gaussian noise."""
    def hook(h: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*h[:, -1, :].shape) * std * np.std(h[:, -1, :])
        h[:, -1, :] += noise
        return h
    return hook


def project_to_1d_and_scale(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
    layer_idx: int,
    scale: float,
) -> Callable:
    """Create a hook that projects to 1D subspace and scales."""
    import mlx.core as mx

    # First, compute the 1D basis from concept deltas
    deltas = []
    for word in concepts[:10]:  # Use subset for speed
        tokens = tokenizer.encode(word)
        input_ids = mx.array([tokens])
        mx.eval(input_ids)

        h = model.model.embed_tokens(input_ids)
        mx.eval(h)

        for idx, layer in enumerate(model.model.layers):
            if idx < layer_idx:
                result = layer(h)
                h = result[0] if isinstance(result, tuple) else result
                mx.eval(h)
            elif idx == layer_idx:
                h_in = np.array(h[0, -1, :].astype(mx.float32))
                result = layer(h)
                h_out = result[0] if isinstance(result, tuple) else result
                mx.eval(h_out)
                h_out_np = np.array(h_out[0, -1, :].astype(mx.float32))
                deltas.append(h_out_np - h_in)
                break

    deltas = np.stack(deltas)
    deltas = np.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)

    # Find 1D basis (first PC)
    mean = deltas.mean(axis=0)
    centered = deltas - mean
    cov = (centered.T @ centered) / len(deltas)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    pc1 = eigenvectors[:, idx[0]]

    def hook(h: np.ndarray) -> np.ndarray:
        # Project onto 1D and scale
        h_last = h[:, -1, :]
        projection = (h_last @ pc1)[:, np.newaxis] * pc1
        # Scale the 1D component
        h[:, -1, :] = h_last - projection + scale * projection
        return h

    return hook


def run_perturbation_experiment(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    perturbation_name: str,
    hook_fn: Callable,
) -> PerturbationResult:
    """Run a single perturbation experiment."""
    # Original forward pass
    orig_logits, _ = get_logits_with_hook(model, tokenizer, prompt)

    # Perturbed forward pass
    pert_logits, _ = get_logits_with_hook(
        model, tokenizer, prompt,
        hook_layer=layer_idx,
        hook_fn=hook_fn,
    )

    # Analyze
    orig_probs = np.exp(orig_logits - np.max(orig_logits))
    orig_probs = orig_probs / orig_probs.sum()
    pert_probs = np.exp(pert_logits - np.max(pert_logits))
    pert_probs = pert_probs / pert_probs.sum()

    orig_top_idx = np.argmax(orig_probs)
    pert_top_idx = np.argmax(pert_probs)

    orig_top_token = tokenizer.decode([int(orig_top_idx)])
    pert_top_token = tokenizer.decode([int(pert_top_idx)])

    kl = compute_kl_divergence(orig_logits, pert_logits)

    return PerturbationResult(
        prompt=prompt,
        perturbation=perturbation_name,
        layer_idx=layer_idx,
        original_top_token=orig_top_token,
        original_top_prob=float(orig_probs[orig_top_idx]),
        perturbed_top_token=pert_top_token,
        perturbed_top_prob=float(pert_probs[pert_top_idx]),
        kl_divergence=kl,
        top_token_changed=(orig_top_idx != pert_top_idx),
    )


def main():
    parser = argparse.ArgumentParser(description="Bottleneck perturbation analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    logger.info("Loading model from %s", args.model)
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    n_layers = len(model.model.layers)

    print(f"\n{'='*80}")
    print("BOTTLENECK PERTURBATION ANALYSIS")
    print("="*80)
    print(f"Model: {n_layers} layers")
    print(f"Testing {len(TEST_PROMPTS)} prompts")

    # Define perturbations
    perturbations = [
        ("scale_0.5", scale_perturbation(0.5)),
        ("scale_2.0", scale_perturbation(2.0)),
        ("invert", invert_perturbation()),
        ("zero", zero_perturbation()),
        ("noise_0.1", noise_perturbation(0.1)),
    ]

    # Test on bottleneck layers (7 and 14 for LFM2-350M)
    bottleneck_layers = [7, 14] if n_layers == 16 else [n_layers // 2]

    all_results = []

    for layer_idx in bottleneck_layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer_idx} PERTURBATIONS")
        print("="*80)

        print(f"\n{'Prompt':<35} | {'Perturbation':<12} | {'Orig Token':<12} | "
              f"{'Pert Token':<12} | {'KL Div':>8} | Changed")
        print("-" * 100)

        for prompt in TEST_PROMPTS:
            for pert_name, hook_fn in perturbations:
                result = run_perturbation_experiment(
                    model, tokenizer, prompt, layer_idx, pert_name, hook_fn
                )
                all_results.append(result)

                changed_marker = "YES" if result.top_token_changed else "no"
                print(f"{prompt[:35]:<35} | {pert_name:<12} | "
                      f"{result.original_top_token[:12]:<12} | "
                      f"{result.perturbed_top_token[:12]:<12} | "
                      f"{result.kl_divergence:>8.2f} | {changed_marker}")

    # Summary
    print(f"\n{'='*80}")
    print("PERTURBATION IMPACT SUMMARY")
    print("="*80)

    for layer_idx in bottleneck_layers:
        layer_results = [r for r in all_results if r.layer_idx == layer_idx]

        print(f"\nLayer {layer_idx}:")
        for pert_name, _ in perturbations:
            pert_results = [r for r in layer_results if r.perturbation == pert_name]
            n_changed = sum(1 for r in pert_results if r.top_token_changed)
            avg_kl = np.mean([r.kl_divergence for r in pert_results])
            print(f"  {pert_name:<12}: {n_changed}/{len(pert_results)} tokens changed, avg KL={avg_kl:.2f}")

    # The insight
    print(f"\n{'='*80}")
    print("CAUSAL IMPORTANCE")
    print("="*80)

    total_changed = sum(1 for r in all_results if r.top_token_changed)
    total = len(all_results)

    if total_changed > total * 0.5:
        print(f"""
The bottleneck IS causally important!

{total_changed}/{total} ({100*total_changed/total:.0f}%) of perturbations changed the output token.

This means the 1D value is not redundant - the model DEPENDS on it.
Modifying the bottleneck changes what the model outputs.

IMPLICATION: The 1D bottleneck is a critical control point.
Steering this value could steer model outputs.
""")
    else:
        print(f"""
The bottleneck has LIMITED causal importance.

Only {total_changed}/{total} ({100*total_changed/total:.0f}%) of perturbations changed the output token.

This suggests some redundancy - the model can recover from perturbations.
The bottleneck may be a "checkpoint" rather than a "control point".
""")


if __name__ == "__main__":
    main()
