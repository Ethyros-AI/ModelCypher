#!/usr/bin/env python3
"""
Fast Pythagorean Probe - Direct MLX extraction.

Skip the LocalInferenceEngine overhead and use mlx_lm directly.
Focus on the key test: Is 5 geometrically constrained by its
Pythagorean relationship to 3 and 4?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).parents[3] / "ModelCypher" / "src"))

SCRIPT_DIR = Path(__file__).parent.parent


def extract_embedding(model, tokenizer, text: str, layer: int = -1) -> np.ndarray:
    """Extract embedding from specified layer."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    inner = getattr(model, 'model', model)
    x = inner.embed_tokens(input_ids) if hasattr(inner, 'embed_tokens') else inner.wte(input_ids)

    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    target_layer = layer if layer >= 0 else len(layers) + layer

    for i, layer_module in enumerate(layers):
        x = layer_module(x)
        if isinstance(x, tuple):
            x = x[0]
        if i == target_layer:
            break

    result = x[0, -1, :]
    mx.eval(result)
    return np.array(result.astype(mx.float32), dtype=np.float32)


def get_number_embedding(model, tokenizer, n: int) -> np.ndarray:
    """Get embedding for a number."""
    return extract_embedding(model, tokenizer, f"The number {n}.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model_name = Path(args.model).name
    output_path = args.output or str(SCRIPT_DIR / "results" / f"fast_{model_name}.json")

    print("=" * 60)
    print("FAST PYTHAGOREAN PROBE")
    print(f"Model: {model_name}")
    print("=" * 60)

    print("\nLoading model...")
    model, tokenizer = load(args.model)

    # Extract embeddings for key numbers
    numbers = [3, 4, 5, 6, 9, 16, 25]
    embeddings = {}

    print("Extracting embeddings...")
    for n in numbers:
        embeddings[n] = get_number_embedding(model, tokenizer, n)
        print(f"  {n}: dim={embeddings[n].shape[0]}")

    # Test 1: Position test (Euclidean)
    print("\n--- Position Test (Euclidean) ---")
    centroid_34 = (embeddings[3] + embeddings[4]) / 2
    dist_5 = float(np.linalg.norm(embeddings[5] - centroid_34))
    dist_6 = float(np.linalg.norm(embeddings[6] - centroid_34))

    print(f"  5 → (3,4): {dist_5:.4f}")
    print(f"  6 → (3,4): {dist_6:.4f}")
    print(f"  5 {'CLOSER' if dist_5 < dist_6 else 'FARTHER'} than 6")

    # Test 2: Cosine similarity to (3,4) centroid direction
    print("\n--- Direction Test ---")
    dir_to_5 = embeddings[5] - centroid_34
    dir_to_6 = embeddings[6] - centroid_34
    dir_34 = embeddings[4] - embeddings[3]

    cos_5 = float(np.dot(dir_to_5, dir_34) / (np.linalg.norm(dir_to_5) * np.linalg.norm(dir_34) + 1e-8))
    cos_6 = float(np.dot(dir_to_6, dir_34) / (np.linalg.norm(dir_to_6) * np.linalg.norm(dir_34) + 1e-8))

    print(f"  5's alignment with 3→4: {cos_5:.4f}")
    print(f"  6's alignment with 3→4: {cos_6:.4f}")

    # Test 3: Squared numbers triangle
    print("\n--- Squared Triangle (9,16,25) ---")
    d_9_16 = float(np.linalg.norm(embeddings[9] - embeddings[16]))
    d_16_25 = float(np.linalg.norm(embeddings[16] - embeddings[25]))
    d_9_25 = float(np.linalg.norm(embeddings[9] - embeddings[25]))

    print(f"  9↔16: {d_9_16:.4f}")
    print(f"  16↔25: {d_16_25:.4f}")
    print(f"  9↔25: {d_9_25:.4f}")

    # Test 4: Squaring direction consistency
    print("\n--- Squaring Direction ---")
    dir_3_9 = embeddings[9] - embeddings[3]
    dir_4_16 = embeddings[16] - embeddings[4]
    dir_5_25 = embeddings[25] - embeddings[5]

    # Normalize
    dir_3_9 = dir_3_9 / (np.linalg.norm(dir_3_9) + 1e-8)
    dir_4_16 = dir_4_16 / (np.linalg.norm(dir_4_16) + 1e-8)
    dir_5_25 = dir_5_25 / (np.linalg.norm(dir_5_25) + 1e-8)

    sq_cos_1 = float(np.dot(dir_3_9, dir_4_16))
    sq_cos_2 = float(np.dot(dir_4_16, dir_5_25))
    sq_cos_3 = float(np.dot(dir_3_9, dir_5_25))

    print(f"  3→9 vs 4→16: {sq_cos_1:.4f}")
    print(f"  4→16 vs 5→25: {sq_cos_2:.4f}")
    print(f"  3→9 vs 5→25: {sq_cos_3:.4f}")

    # Results
    results = {
        "model": model_name,
        "position_test": {
            "dist_5_to_34": dist_5,
            "dist_6_to_34": dist_6,
            "5_closer": dist_5 < dist_6,
        },
        "direction_test": {
            "cos_5": cos_5,
            "cos_6": cos_6,
        },
        "squared_triangle": {
            "d_9_16": d_9_16,
            "d_16_25": d_16_25,
            "d_9_25": d_9_25,
        },
        "squaring_direction": {
            "3_9_vs_4_16": sq_cos_1,
            "4_16_vs_5_25": sq_cos_2,
            "3_9_vs_5_25": sq_cos_3,
            "avg_consistency": (sq_cos_1 + sq_cos_2 + sq_cos_3) / 3,
        },
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Position test (5 closer than 6): {'PASS' if dist_5 < dist_6 else 'FAIL'}")
    print(f"Squaring consistency: {results['squaring_direction']['avg_consistency']:.4f}")

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
