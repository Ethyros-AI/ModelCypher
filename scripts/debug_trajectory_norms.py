#!/usr/bin/env python3
"""Debug trajectory norms to understand the geometry."""

import sys
from pathlib import Path
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def debug_norms():
    from mlx_lm import load

    model_path = "/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16"
    print(f"Loading: {model_path}")
    model, tokenizer = load(model_path)

    prompt = "Question: What is 2 + 2?\n\nAnswer:"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(tokens)}")

    # Track norms through layers
    hidden = model.model.embed_tokens(input_ids)
    mx.eval(hidden)

    print(f"\nLayer-by-layer norms:")
    print("-" * 40)

    norms = []
    # Embedding
    emb_norm = float(mx.sqrt(mx.sum(hidden * hidden)))
    norms.append(emb_norm)
    print(f"Embedding: {emb_norm:.4f}")

    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, mask=None, cache=None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        mx.eval(hidden)
        layer_norm = float(mx.sqrt(mx.sum(hidden * hidden)))
        norms.append(layer_norm)
        print(f"Layer {i:2d}: {layer_norm:.4f}")

    # Analysis
    print("\n" + "=" * 40)
    print("TRAJECTORY ANALYSIS")
    print("=" * 40)

    norms_arr = np.array(norms)
    peak_idx = np.argmax(norms_arr)
    peak_val = norms_arr[peak_idx]
    final_val = norms_arr[-1]
    ratio = peak_val / final_val

    print(f"Peak layer: {peak_idx} (of {len(norms)})")
    print(f"Peak norm: {peak_val:.4f}")
    print(f"Final norm: {final_val:.4f}")
    print(f"Compression ratio: {ratio:.4f}")
    print(f"φ (golden ratio): {(1 + np.sqrt(5))/2:.4f}")
    print(f"Distance from φ: {abs(ratio - (1 + np.sqrt(5))/2):.4f}")

    # Is there actual expansion-compression?
    if peak_idx > 0 and peak_idx < len(norms) - 1:
        print(f"\n✓ Peak is in the middle - expansion then compression")
    elif peak_idx == 0:
        print(f"\n✗ Peak at embedding - monotonic compression")
    else:
        print(f"\n✗ Peak at final - monotonic expansion")

    return norms


if __name__ == "__main__":
    debug_norms()
