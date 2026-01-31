#!/usr/bin/env python3
"""Analyze hidden state properties through layers.

Instead of attention entropy (hard to extract), analyze:
1. Hidden state entropy (uncertainty in representation)
2. Effective dimensionality (intrinsic dimension estimate)
3. Sparsity (fraction of near-zero activations)

These proxy measures can reveal similar information about
whether the model is expanding/exploring or compressing/focusing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TASK_PROBES = {
    "retrieval": "What is the capital of France?",
    "reasoning": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "creative": "Write the first line of a story about a dragon.",
    "code": "Write a Python function that returns the sum of two numbers.",
}


def analyze_hidden_state(hidden) -> dict:
    """Compute various metrics on hidden state."""
    import mlx.core as mx

    # Flatten to (batch*seq, hidden_dim)
    h = hidden.reshape(-1, hidden.shape[-1])
    # Cast to float32 for numpy compatibility
    h = h.astype(mx.float32)
    mx.eval(h)

    # Convert to numpy for analysis
    h_np = np.array(h)

    # 1. Norm (already have this)
    norm = float(np.linalg.norm(h_np))

    # 2. Mean activation magnitude
    mean_abs = float(np.mean(np.abs(h_np)))

    # 3. Sparsity (fraction of activations < threshold)
    threshold = 0.01 * mean_abs  # 1% of mean
    sparsity = float(np.mean(np.abs(h_np) < threshold))

    # 4. Kurtosis (peakedness - high = concentrated, low = spread out)
    centered = h_np - np.mean(h_np)
    std = np.std(h_np) + 1e-10
    kurtosis = float(np.mean((centered / std) ** 4) - 3)  # Excess kurtosis

    # 5. Effective dimension via participation ratio
    # PR = (sum of eigenvalues)^2 / sum(eigenvalues^2)
    # Higher = more dimensions active
    try:
        cov = h_np.T @ h_np / h_np.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
        sum_eig = np.sum(eigenvalues)
        sum_eig_sq = np.sum(eigenvalues ** 2)
        if sum_eig_sq > 1e-10:
            participation_ratio = (sum_eig ** 2) / sum_eig_sq
        else:
            participation_ratio = 0.0
    except:
        participation_ratio = 0.0

    # 6. Top-k eigenvalue concentration
    try:
        sorted_eig = np.sort(eigenvalues)[::-1]
        top_10_frac = float(np.sum(sorted_eig[:10]) / (np.sum(sorted_eig) + 1e-10))
    except:
        top_10_frac = 0.0

    return {
        "norm": norm,
        "mean_abs": mean_abs,
        "sparsity": sparsity,
        "kurtosis": kurtosis,
        "participation_ratio": participation_ratio,
        "top_10_concentration": top_10_frac,
    }


def trace_hidden_metrics(model, tokenizer, prompt: str) -> list[dict]:
    """Trace hidden state metrics through all layers."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    base = getattr(model, "model", model)

    # Embedding
    hidden = base.embed_tokens(input_ids)
    mx.eval(hidden)

    metrics = [analyze_hidden_state(hidden)]

    # Each layer
    for layer in base.layers:
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        metrics.append(analyze_hidden_state(hidden))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    args = parser.parse_args()

    from mlx_lm import load

    print("=" * 70)
    print("HIDDEN STATE ANALYSIS")
    print("=" * 70)
    print(f"Model: {Path(args.model).name}")

    model, tokenizer = load(args.model)
    n_layers = len(getattr(model, "model", model).layers)
    print(f"Layers: {n_layers}")
    print("=" * 70)

    all_results = {}

    for task_type, prompt in TASK_PROBES.items():
        print(f"\n{task_type.upper()}")
        print("-" * 60)

        metrics = trace_hidden_metrics(model, tokenizer, prompt)
        all_results[task_type] = metrics

        # Print trajectory
        print(f"{'Layer':>5} {'Norm':>10} {'Sparsity':>10} {'Kurtosis':>10} {'EffDim':>10} {'Top10%':>10}")
        print("-" * 60)

        for i, m in enumerate(metrics):
            layer_name = f"Emb" if i == 0 else f"L{i:02d}"
            print(f"{layer_name:>5} {m['norm']:>10.1f} {m['sparsity']:>10.3f} {m['kurtosis']:>10.2f} {m['participation_ratio']:>10.1f} {m['top_10_concentration']:>10.3f}")

    # Cross-task comparison
    print("\n" + "=" * 70)
    print("CROSS-TASK ANALYSIS")
    print("=" * 70)

    # Compare final layer metrics
    print("\nFinal layer comparison:")
    print(f"{'Task':>12} {'Norm':>10} {'Sparsity':>10} {'EffDim':>10}")
    for task, metrics in all_results.items():
        m = metrics[-1]
        print(f"{task:>12} {m['norm']:>10.1f} {m['sparsity']:>10.3f} {m['participation_ratio']:>10.1f}")

    # Sparsity trend
    print("\nSparsity trend (early → late):")
    for task, metrics in all_results.items():
        early_sparsity = np.mean([m['sparsity'] for m in metrics[:len(metrics)//3]])
        late_sparsity = np.mean([m['sparsity'] for m in metrics[-len(metrics)//3:]])
        trend = "↑ SPARSER" if late_sparsity > early_sparsity else "↓ DENSER"
        print(f"  {task:>12}: {early_sparsity:.3f} → {late_sparsity:.3f} {trend}")

    # Effective dimension trend
    print("\nEffective dimension trend (early → late):")
    for task, metrics in all_results.items():
        early_dim = np.mean([m['participation_ratio'] for m in metrics[:len(metrics)//3]])
        late_dim = np.mean([m['participation_ratio'] for m in metrics[-len(metrics)//3:]])
        trend = "↑ EXPANDING" if late_dim > early_dim else "↓ COMPRESSING"
        print(f"  {task:>12}: {early_dim:.1f} → {late_dim:.1f} {trend}")


if __name__ == "__main__":
    main()
