#!/usr/bin/env python3
"""
Attention Eigenvalue Analysis

Goal: Understand the spectral properties of attention matrices in trained transformers.

Questions:
1. What is the eigenvalue distribution of attention matrices?
2. How does it compare to random/untrained attention?
3. Is there a universal pattern across architectures?
4. Does eigenvalue concentration explain rank-1 Jacobians?

If attention matrices are nearly rank-1, then the Jacobian (which depends on attention)
would also be rank-1. This could be the missing link.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class AttentionSpectrum:
    """Eigenvalue spectrum for a single attention matrix."""
    layer_idx: int
    head_idx: int
    eigenvalues: List[float]  # Sorted descending
    effective_rank: float
    spectral_gap: float  # λ1/λ2
    entropy: float  # Attention entropy
    top_k_concentration: float  # Sum of top 3 eigenvalues / total


@dataclass
class LayerSpectrum:
    """Aggregated spectrum for all heads in a layer."""
    layer_idx: int
    mean_effective_rank: float
    mean_spectral_gap: float
    mean_entropy: float
    head_spectra: List[AttentionSpectrum]


@dataclass
class ModelSpectrum:
    """Full spectral analysis for a model."""
    model_name: str
    n_layers: int
    n_heads: int
    layer_spectra: List[LayerSpectrum]

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        all_ranks = [h.effective_rank for l in self.layer_spectra for h in l.head_spectra]
        all_gaps = [h.spectral_gap for l in self.layer_spectra for h in l.head_spectra]
        all_entropy = [h.entropy for l in self.layer_spectra for h in l.head_spectra]

        return {
            "model": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "effective_rank": {
                "mean": sum(all_ranks) / len(all_ranks),
                "min": min(all_ranks),
                "max": max(all_ranks),
            },
            "spectral_gap": {
                "mean": sum(all_gaps) / len(all_gaps) if all_gaps else 0,
                "min": min(all_gaps) if all_gaps else 0,
                "max": max(all_gaps) if all_gaps else 0,
            },
            "entropy": {
                "mean": sum(all_entropy) / len(all_entropy),
                "min": min(all_entropy),
                "max": max(all_entropy),
            },
        }


def effective_rank(eigenvalues: mx.array) -> float:
    """Compute Shannon effective rank from eigenvalues."""
    # Normalize to probability distribution (eigenvalues should be non-negative for attention)
    e = mx.abs(eigenvalues)
    e_sum = mx.sum(e)
    if float(e_sum) < 1e-10:
        return 0.0
    p = e / e_sum

    # Shannon entropy
    log_p = mx.where(p > 1e-10, mx.log(p), mx.zeros_like(p))
    entropy = -mx.sum(p * log_p)

    return float(mx.exp(entropy))


def attention_entropy(attn_weights: mx.array) -> float:
    """Compute entropy of attention distribution."""
    # attn_weights: [seq_len, seq_len], each row sums to 1
    # Compute mean entropy across positions
    # Cast to float32 for numerical stability
    p = attn_weights.astype(mx.float32) + 1e-10  # Avoid log(0)
    log_p = mx.log(p)
    row_entropy = -mx.sum(p * log_p, axis=-1)
    return float(mx.mean(row_entropy))


def analyze_attention_matrix(attn: mx.array, layer_idx: int, head_idx: int) -> AttentionSpectrum:
    """Analyze eigenvalue spectrum of a single attention matrix.

    Args:
        attn: Attention weights [seq_len, seq_len]
        layer_idx: Layer index
        head_idx: Head index
    """
    # Compute eigenvalues (attention matrix is not symmetric, use SVD instead)
    # For attention, singular values are more meaningful than eigenvalues
    # since A is not guaranteed symmetric
    # Cast to float32 for SVD (bf16 not supported)
    attn_f32 = attn.astype(mx.float32)
    U, S, Vt = mx.linalg.svd(attn_f32, stream=mx.cpu)
    mx.eval(S)

    # Sort descending
    S_sorted = mx.sort(S)[::-1]
    eigenvalues = [float(s) for s in S_sorted]

    # Effective rank
    eff_rank = effective_rank(S)

    # Spectral gap
    if len(eigenvalues) >= 2 and eigenvalues[1] > 1e-10:
        gap = eigenvalues[0] / eigenvalues[1]
    else:
        gap = float('inf')

    # Attention entropy
    entropy = attention_entropy(attn)

    # Top-k concentration
    total = sum(eigenvalues)
    top_k = sum(eigenvalues[:3]) if len(eigenvalues) >= 3 else total
    concentration = top_k / total if total > 1e-10 else 1.0

    return AttentionSpectrum(
        layer_idx=layer_idx,
        head_idx=head_idx,
        eigenvalues=eigenvalues[:10],  # Keep top 10
        effective_rank=eff_rank,
        spectral_gap=gap,
        entropy=entropy,
        top_k_concentration=concentration,
    )


def extract_attention_weights(model, tokenizer, prompt: str) -> Dict[int, mx.array]:
    """Extract attention weights from all layers for a prompt.

    Returns:
        Dict mapping layer_idx -> attention tensor [n_heads, seq_len, seq_len]
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # We need to hook into the model to get attention weights
    # This is architecture-specific, so we'll try multiple approaches

    attention_weights = {}

    # Try to find attention modules and hook them
    def find_attention_modules(module, prefix=""):
        """Find all attention modules in the model."""
        attns = []
        for name, child in module.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            if "attn" in name.lower() or "attention" in name.lower():
                attns.append((full_name, child))
        return attns

    # For MLX models, we need to do a forward pass and capture attention
    # This is tricky because MLX doesn't have hooks like PyTorch

    # Alternative approach: compute attention manually from Q, K
    # This works for standard transformer architectures

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        print("Warning: Could not find layers in model")
        return {}

    # Get embeddings
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        h = model.model.embed_tokens(input_ids)
    elif hasattr(model, "embed_tokens"):
        h = model.embed_tokens(input_ids)
    else:
        print("Warning: Could not find embedding layer")
        return {}

    # Process each layer and extract attention
    for layer_idx, layer in enumerate(layers):
        try:
            # Get attention module
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(layer, "attention"):
                attn_module = layer.attention
            else:
                continue

            # Apply layer norm if present
            if hasattr(layer, "input_layernorm"):
                h_normed = layer.input_layernorm(h)
            elif hasattr(layer, "ln_1"):
                h_normed = layer.ln_1(h)
            else:
                h_normed = h

            # Compute Q, K, V
            B, T, C = h_normed.shape

            if hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(h_normed)
                k = attn_module.k_proj(h_normed)
            elif hasattr(attn_module, "Wqkv"):
                # Fused QKV projection
                qkv = attn_module.Wqkv(h_normed)
                q, k, v = mx.split(qkv, 3, axis=-1)
            else:
                continue

            # Get head dimensions - handle GQA where n_kv_heads != n_heads
            n_heads = getattr(attn_module, "n_heads",
                            getattr(attn_module, "num_heads",
                            getattr(attn_module, "num_attention_heads", 8)))
            n_kv_heads = getattr(attn_module, "n_kv_heads",
                               getattr(attn_module, "num_key_value_heads", n_heads))

            # Infer head_dim from Q projection (q_proj outputs n_heads * head_dim)
            q_dim = q.shape[-1]  # n_heads * head_dim
            k_dim = k.shape[-1]  # n_kv_heads * head_dim

            head_dim = q_dim // n_heads
            kv_head_dim = k_dim // n_kv_heads  # Should equal head_dim

            # Reshape for multi-head attention
            # Q: [B, T, n_heads * head_dim] -> [B, n_heads, T, head_dim]
            q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
            # K: [B, T, n_kv_heads * head_dim] -> [B, n_kv_heads, T, head_dim]
            k = k.reshape(B, T, n_kv_heads, kv_head_dim).transpose(0, 2, 1, 3)

            # For GQA, repeat K heads to match Q heads
            if n_kv_heads != n_heads:
                n_rep = n_heads // n_kv_heads
                # k: [B, n_kv_heads, T, head_dim] -> [B, n_heads, T, head_dim]
                k = mx.repeat(k, n_rep, axis=1)

            # Compute attention scores
            scale = 1.0 / mx.sqrt(mx.array(head_dim, dtype=q.dtype))
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            # Apply softmax
            attn_weights_layer = mx.softmax(scores, axis=-1)
            mx.eval(attn_weights_layer)

            # Store: [n_heads, seq_len, seq_len]
            attention_weights[layer_idx] = attn_weights_layer[0]  # Remove batch dim

            # Continue forward pass for next layer (simplified)
            # This isn't perfect but gives us attention for analysis
            if hasattr(layer, "__call__"):
                h = layer(h)

        except Exception as e:
            # Print more diagnostic info
            if hasattr(layer, "self_attn"):
                attn_mod = layer.self_attn
                print(f"Layer {layer_idx} failed: {e}")
                print(f"  attn_module attrs: {[a for a in dir(attn_mod) if not a.startswith('_')][:10]}")
                if hasattr(attn_mod, "q_proj"):
                    print(f"  q_proj weight shape: {attn_mod.q_proj.weight.shape}")
            continue

    return attention_weights


def analyze_model(model_path: str, prompt: str = "The quick brown fox jumps over the lazy dog.") -> ModelSpectrum:
    """Analyze attention eigenvalue spectrum for a model."""
    from mlx_lm import load

    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    # Determine model name
    model_name = Path(model_path).name

    # Extract attention weights
    print(f"Extracting attention weights for: '{prompt}'")
    attention_weights = extract_attention_weights(model, tokenizer, prompt)

    if not attention_weights:
        raise ValueError("No attention weights extracted")

    print(f"Got attention from {len(attention_weights)} layers")

    # Analyze each layer
    layer_spectra = []
    n_heads = 0

    for layer_idx in sorted(attention_weights.keys()):
        attn = attention_weights[layer_idx]
        n_heads = attn.shape[0]

        head_spectra = []
        for head_idx in range(n_heads):
            head_attn = attn[head_idx]
            spectrum = analyze_attention_matrix(head_attn, layer_idx, head_idx)
            head_spectra.append(spectrum)

        # Aggregate layer stats
        layer_spectrum = LayerSpectrum(
            layer_idx=layer_idx,
            mean_effective_rank=sum(h.effective_rank for h in head_spectra) / len(head_spectra),
            mean_spectral_gap=sum(h.spectral_gap for h in head_spectra if h.spectral_gap < float('inf')) / max(1, sum(1 for h in head_spectra if h.spectral_gap < float('inf'))),
            mean_entropy=sum(h.entropy for h in head_spectra) / len(head_spectra),
            head_spectra=head_spectra,
        )
        layer_spectra.append(layer_spectrum)

        print(f"  Layer {layer_idx}: eff_rank={layer_spectrum.mean_effective_rank:.2f}, "
              f"entropy={layer_spectrum.mean_entropy:.2f}, "
              f"gap={layer_spectrum.mean_spectral_gap:.2f}")

    return ModelSpectrum(
        model_name=model_name,
        n_layers=len(layer_spectra),
        n_heads=n_heads,
        layer_spectra=layer_spectra,
    )


def compare_to_random(seq_len: int = 10, n_samples: int = 100) -> Dict[str, float]:
    """Compute expected spectrum for random attention matrices."""
    print(f"\nComputing random baseline (seq_len={seq_len}, n_samples={n_samples})...")

    ranks = []
    gaps = []
    entropies = []

    for _ in range(n_samples):
        # Random attention: uniform over sequence
        # This simulates untrained attention (before softmax sharpening)
        scores = mx.random.normal((seq_len, seq_len))
        attn = mx.softmax(scores, axis=-1)
        mx.eval(attn)

        spectrum = analyze_attention_matrix(attn, 0, 0)
        ranks.append(spectrum.effective_rank)
        if spectrum.spectral_gap < float('inf'):
            gaps.append(spectrum.spectral_gap)
        entropies.append(spectrum.entropy)

    return {
        "effective_rank": {
            "mean": sum(ranks) / len(ranks),
            "min": min(ranks),
            "max": max(ranks),
        },
        "spectral_gap": {
            "mean": sum(gaps) / len(gaps) if gaps else 0,
        },
        "entropy": {
            "mean": sum(entropies) / len(entropies),
        },
    }


def main():
    import sys

    print("="*60)
    print("  ATTENTION EIGENVALUE ANALYSIS")
    print("="*60)

    # Test on available models
    models_to_test = [
        "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen2.5-3B-Instruct-bf16",
        "/Volumes/CodeCypher/models/mlx-community/Qwen3-8B-bf16",
        "/Volumes/CodeCypher/models/mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
    ]

    # Filter to existing models
    models_to_test = [m for m in models_to_test if Path(m).exists()]

    if not models_to_test:
        print("No models found. Please provide a model path.")
        sys.exit(1)

    # Random baseline
    random_baseline = compare_to_random(seq_len=10)
    print(f"\nRandom baseline:")
    print(f"  Effective rank: {random_baseline['effective_rank']['mean']:.2f}")
    print(f"  Entropy: {random_baseline['entropy']['mean']:.2f}")

    # Analyze each model
    results = {"random_baseline": random_baseline, "models": {}}

    for model_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"  Analyzing: {Path(model_path).name}")
        print("="*60)

        try:
            spectrum = analyze_model(model_path)
            summary = spectrum.summary()
            results["models"][spectrum.model_name] = summary

            print(f"\nSummary for {spectrum.model_name}:")
            print(f"  Effective rank: {summary['effective_rank']['mean']:.2f} "
                  f"(range: {summary['effective_rank']['min']:.2f}-{summary['effective_rank']['max']:.2f})")
            print(f"  Entropy: {summary['entropy']['mean']:.2f}")
            print(f"  Spectral gap: {summary['spectral_gap']['mean']:.2f}")

        except Exception as e:
            print(f"Failed to analyze {model_path}: {e}")
            import traceback
            traceback.print_exc()

    # Compare to random
    print(f"\n{'='*60}")
    print("  COMPARISON: TRAINED vs RANDOM")
    print("="*60)

    random_rank = random_baseline['effective_rank']['mean']
    random_entropy = random_baseline['entropy']['mean']

    for model_name, summary in results["models"].items():
        trained_rank = summary['effective_rank']['mean']
        trained_entropy = summary['entropy']['mean']

        print(f"\n{model_name}:")
        print(f"  Rank: {trained_rank:.2f} vs random {random_rank:.2f} "
              f"({trained_rank/random_rank:.1%})")
        print(f"  Entropy: {trained_entropy:.2f} vs random {random_entropy:.2f} "
              f"({trained_entropy/random_entropy:.1%})")

        if trained_rank < random_rank * 0.5:
            print(f"  → Attention is significantly sharper than random (rank reduced by >{50}%)")

    # Save results
    output_path = Path("data/attention_eigenvalue_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
