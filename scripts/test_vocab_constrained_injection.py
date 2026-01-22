#!/usr/bin/env python3
"""
Test Vocabulary-Constrained Multimodal Injection

This script tests whether vocabulary-constrained projection improves
token retrieval compared to raw affine transformation.

The hypothesis:
- Affine bridge achieves 0.61 test cosine (good angle alignment)
- But token retrieval fails because high cosine != same vocabulary neighborhood
- Vocabulary-constrained projection forces output onto vocabulary manifold
- This should improve nearest-token retrieval accuracy

Usage:
    poetry run python scripts/test_vocab_constrained_injection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from safetensors import safe_open

from modelcypher.core.domain._backend import get_default_backend, set_default_backend
from modelcypher.backends import get_backend
from modelcypher.core.domain.geometry.affine_bridge import (
    AffineBridge,
    VocabConstrainedProjection,
    HybridBridge,
)

# Initialize backend before any domain code
set_default_backend(get_backend("mlx"))


# Paths
OFFRAMPS_DIR = Path("/path/to/experiments/multi-modal-compression-2026-01-09/offramps")
LFM2_MODEL_PATH = "/path/to/models/mlx-community/LFM2-350M-MLX-bf16"


def load_affine_weights():
    """Load pre-trained affine bridge weights."""
    affine_path = OFFRAMPS_DIR / "affine_bridge.safetensors"

    with safe_open(str(affine_path), framework="numpy") as f:
        W_np = f.get_tensor("W")
        b_np = f.get_tensor("b")

    print(f"Loaded affine weights: W={W_np.shape}, b={b_np.shape}")
    return W_np, b_np


def load_vocabulary_embeddings():
    """Load LFM2 vocabulary embeddings."""
    from mlx_lm import load

    print(f"Loading LFM2 from {LFM2_MODEL_PATH}...")
    model, tokenizer = load(LFM2_MODEL_PATH)

    # Get vocabulary embeddings
    vocab_embeds = model.model.embed_tokens.weight
    mx.eval(vocab_embeds)

    print(f"Vocabulary shape: {vocab_embeds.shape}")  # (vocab_size, 1024)
    return vocab_embeds, tokenizer, model


def create_test_embeddings(n_samples: int = 5):
    """Create synthetic test embeddings in LFM2 space (1024D).

    In real usage, these would come from CLIP → vision offramp → affine bridge.
    Here we use random embeddings to test the mechanism.
    """
    backend = get_default_backend()

    # Create random embeddings normalized to similar magnitude as vocab
    embeddings = backend.random_normal((n_samples, 1024))
    norms = backend.sqrt(backend.sum(embeddings * embeddings, axis=1, keepdims=True))
    embeddings = embeddings / norms * 5.0  # Similar magnitude to vocab
    backend.eval(embeddings)

    return embeddings


def find_nearest_tokens_raw(embedding, vocab_embeds, tokenizer, k=5):
    """Find nearest tokens using raw cosine similarity."""
    # Normalize
    embed_norm = embedding / (mx.sqrt(mx.sum(embedding ** 2, axis=-1, keepdims=True)) + 1e-8)
    vocab_norm = vocab_embeds / (mx.sqrt(mx.sum(vocab_embeds ** 2, axis=-1, keepdims=True)) + 1e-8)
    mx.eval(embed_norm, vocab_norm)

    # Compute similarities
    similarities = mx.matmul(embed_norm, vocab_norm.T)  # (1, vocab_size)
    mx.eval(similarities)

    # Get top-k
    sim_np = similarities.tolist()[0]
    top_indices = sorted(range(len(sim_np)), key=lambda i: sim_np[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        token = tokenizer.decode([idx])
        results.append((token, sim_np[idx]))

    return results


def test_vocab_constrained_vs_raw():
    """Compare vocabulary-constrained projection vs raw affine output."""
    print("=" * 70)
    print("VOCABULARY-CONSTRAINED PROJECTION TEST")
    print("=" * 70)

    backend = get_default_backend()

    # Load components
    print("\n[1/4] Loading affine weights...")
    W_np, b_np = load_affine_weights()

    print("\n[2/4] Loading vocabulary embeddings...")
    vocab_embeds, tokenizer, model = load_vocabulary_embeddings()

    # Convert to backend arrays
    W = backend.array(W_np)
    b = backend.array(b_np)
    vocab = backend.array(vocab_embeds)
    backend.eval(W, b, vocab)

    print("\n[3/4] Setting up HybridBridge...")
    hybrid = HybridBridge(backend)
    hybrid.load_affine_weights(W, b)
    hybrid.set_vocabulary(vocab)

    # Also set up standalone affine and vocab-constrained for comparison
    affine = AffineBridge(backend)
    affine.load_weights(W, b)

    vocab_proj = VocabConstrainedProjection(backend)
    vocab_proj.set_vocabulary(vocab)

    print("\n[4/4] Testing with sample embeddings...")

    # Create test embeddings (in the space that affine bridge expects)
    # Since affine was trained on 1024D→1024D, we use 1024D inputs
    test_embeddings = create_test_embeddings(n_samples=3)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for i in range(test_embeddings.shape[0]):
        embed = backend.reshape(test_embeddings[i], (1, -1))
        backend.eval(embed)

        print(f"\n--- Sample {i+1} ---")

        # Method 1: Raw affine output → find nearest tokens
        affine_out = affine.transform(embed)
        backend.eval(affine_out)
        affine_embed_mx = mx.array(backend.tolist(affine_out))

        raw_nearest = find_nearest_tokens_raw(affine_embed_mx, vocab_embeds, tokenizer, k=5)
        print("\nMethod 1: Raw Affine Output")
        print("  Nearest tokens:", [(t, f"{s:.4f}") for t, s in raw_nearest])

        # Method 2: Vocab-constrained projection (no affine)
        vocab_result = vocab_proj.project(embed, temperature=1.0)
        print("\nMethod 2: Vocab-Constrained Only")
        print("  Nearest token ID:", vocab_result.nearest_token_ids[0])
        print("  Nearest token:", tokenizer.decode([vocab_result.nearest_token_ids[0]]))

        # Method 3: HybridBridge (affine + vocab-constrained)
        hybrid_result = hybrid.transform(embed, temperature=1.0)
        print("\nMethod 3: Hybrid (Affine + Vocab-Constrained)")
        print("  Nearest token ID:", hybrid_result.nearest_token_ids[0])
        print("  Nearest token:", tokenizer.decode([hybrid_result.nearest_token_ids[0]]))

        # Compare attention sharpness
        attn = hybrid_result.attention_weights[0]
        max_attn = max(attn)
        print(f"  Max attention weight: {max_attn:.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Key insight:
- Raw affine finds tokens by cosine similarity AFTER transformation
- Vocab-constrained FORCES output to BE a token embedding
- HybridBridge combines direction alignment (affine) with token neighborhood (vocab)

If vocab-constrained tokens are more semantically coherent than raw affine
nearest neighbors, the hypothesis is supported.
""")


if __name__ == "__main__":
    test_vocab_constrained_vs_raw()
