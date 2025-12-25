#!/usr/bin/env python3
"""
Arithmetic Direction Consistency Test

If mathematical operations are encoded geometrically, then:
1. "Add 3" should be a consistent direction: (5→8), (10→13), (20→23)
2. "Double" should be a consistent transformation
3. "Square" should be a consistent operation

This tests whether arithmetic is literally encoded as latent space navigation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).parents[3] / "ModelCypher" / "src"))

SCRIPT_DIR = Path(__file__).parent.parent


def load_model(model_path: str):
    from mlx_lm import load
    return load(model_path)


def extract_embedding(model, tokenizer, text: str) -> np.ndarray:
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    if hasattr(model, 'model'):
        inner = model.model
    else:
        inner = model

    if hasattr(inner, 'embed_tokens'):
        x = inner.embed_tokens(input_ids)
    else:
        x = inner.wte(input_ids)

    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    for layer in layers:
        x = layer(x)
        if isinstance(x, tuple):
            x = x[0]

    result = x[0, -1, :]
    mx.eval(result)
    result = result.astype(mx.float32)
    mx.eval(result)
    return np.array(result, dtype=np.float32)


def get_number_embedding(model, tokenizer, n: int) -> np.ndarray:
    text = f"The number {n}."
    return extract_embedding(model, tokenizer, text)


def direction_consistency(directions: list[np.ndarray]) -> float:
    """Compute average pairwise cosine similarity between directions."""
    if len(directions) < 2:
        return 1.0

    similarities = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            d1, d2 = directions[i], directions[j]
            sim = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)
            similarities.append(sim)

    return float(np.mean(similarities))


def test_addition_direction(model, tokenizer, addend: int, starts: list[int]) -> dict:
    """Test if adding a fixed number is a consistent direction."""
    directions = []

    for start in starts:
        end = start + addend
        emb_start = get_number_embedding(model, tokenizer, start)
        emb_end = get_number_embedding(model, tokenizer, end)
        direction = emb_end - emb_start
        directions.append(direction)

    consistency = direction_consistency(directions)

    # Compute average direction
    avg_direction = np.mean(directions, axis=0)
    avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)

    # Test: can the average direction predict unseen additions?
    test_start = max(starts) + 10
    test_end = test_start + addend
    emb_test_start = get_number_embedding(model, tokenizer, test_start)
    emb_test_end = get_number_embedding(model, tokenizer, test_end)

    # Predict end by adding scaled direction
    scale = np.linalg.norm(directions[0])  # Use first direction's magnitude
    predicted_end = emb_test_start + avg_direction * scale

    # Similarity between prediction and actual
    pred_sim = float(np.dot(predicted_end, emb_test_end) /
                    (np.linalg.norm(predicted_end) * np.linalg.norm(emb_test_end) + 1e-8))

    return {
        "addend": addend,
        "starts": starts,
        "consistency": consistency,
        "prediction_similarity": pred_sim,
        "test_case": f"{test_start}+{addend}={test_end}",
    }


def test_doubling(model, tokenizer, numbers: list[int]) -> dict:
    """Test if doubling is a consistent transformation."""
    directions = []
    scales = []

    for n in numbers:
        emb_n = get_number_embedding(model, tokenizer, n)
        emb_2n = get_number_embedding(model, tokenizer, 2 * n)
        direction = emb_2n - emb_n
        directions.append(direction / (np.linalg.norm(direction) + 1e-8))
        scales.append(np.linalg.norm(emb_2n) / (np.linalg.norm(emb_n) + 1e-8))

    dir_consistency = direction_consistency(directions)
    scale_consistency = 1.0 - np.std(scales) / (np.mean(scales) + 1e-8)

    return {
        "numbers_tested": numbers,
        "direction_consistency": float(dir_consistency),
        "scale_consistency": float(scale_consistency),
        "average_scale": float(np.mean(scales)),
    }


def test_squaring(model, tokenizer, numbers: list[int]) -> dict:
    """Test if squaring follows a consistent pattern."""
    directions = []

    for n in numbers:
        emb_n = get_number_embedding(model, tokenizer, n)
        emb_n2 = get_number_embedding(model, tokenizer, n * n)
        direction = emb_n2 - emb_n
        directions.append(direction / (np.linalg.norm(direction) + 1e-8))

    consistency = direction_consistency(directions)

    # Is there a consistent "squaring manifold"?
    # Check if squared numbers form a subspace
    squared_embs = [get_number_embedding(model, tokenizer, n*n) for n in numbers]
    squared_matrix = np.stack(squared_embs)

    # PCA to find effective dimensionality
    centered = squared_matrix - squared_matrix.mean(axis=0)
    _, s, _ = np.linalg.svd(centered)

    # Variance explained by first 3 components
    var_explained = np.cumsum(s**2) / np.sum(s**2)
    dim_3_coverage = float(var_explained[2]) if len(var_explained) > 2 else float(var_explained[-1])

    return {
        "numbers_tested": numbers,
        "direction_consistency": float(consistency),
        "squared_numbers": [n*n for n in numbers],
        "3d_variance_coverage": dim_3_coverage,
    }


def test_number_line_hypothesis(model, tokenizer, numbers: list[int]) -> dict:
    """Test if numbers form a roughly linear manifold."""
    embeddings = [get_number_embedding(model, tokenizer, n) for n in numbers]
    emb_matrix = np.stack(embeddings)

    # Fit a line through the embeddings (PCA first component)
    centered = emb_matrix - emb_matrix.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered)

    # Project onto first component
    projections = centered @ Vt[0]

    # Does projection order match number order?
    proj_order = np.argsort(projections)
    number_order = np.argsort(numbers)

    # Spearman correlation between projection order and number order
    from scipy.stats import spearmanr
    corr, p_value = spearmanr(projections, numbers)

    # Variance explained by first component (linearity measure)
    linearity = float(s[0]**2 / np.sum(s**2))

    return {
        "numbers": numbers,
        "projection_number_correlation": float(corr),
        "correlation_p_value": float(p_value),
        "linearity_score": linearity,
        "is_linear": linearity > 0.5 and abs(corr) > 0.8,
    }


def convert_for_json(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(v) for v in obj]
    return obj


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()

    model_name = Path(args.model).name
    output_path = args.output or str(SCRIPT_DIR / "results" / f"arithmetic_{model_name}.json")

    print("=" * 70)
    print("ARITHMETIC DIRECTION CONSISTENCY TEST")
    print(f"Model: {model_name}")
    print("=" * 70)

    model, tokenizer = load_model(args.model)

    results = {"model": model_name}

    # Test 1: Addition directions
    print("\n--- Testing Addition Directions ---")
    for addend in [1, 2, 3, 5, 10]:
        if addend <= 5:
            starts = [3, 7, 12, 20, 30]
        else:
            starts = [5, 15, 25, 35]
        result = test_addition_direction(model, tokenizer, addend, starts)
        results[f"add_{addend}"] = result
        print(f"  +{addend}: consistency={result['consistency']:.4f}, prediction={result['prediction_similarity']:.4f}")

    # Test 2: Doubling
    print("\n--- Testing Doubling ---")
    double_result = test_doubling(model, tokenizer, [2, 3, 5, 7, 10, 12, 15])
    results["doubling"] = double_result
    print(f"  Direction consistency: {double_result['direction_consistency']:.4f}")
    print(f"  Scale consistency: {double_result['scale_consistency']:.4f}")

    # Test 3: Squaring
    print("\n--- Testing Squaring ---")
    square_result = test_squaring(model, tokenizer, [2, 3, 4, 5, 6, 7])
    results["squaring"] = square_result
    print(f"  Direction consistency: {square_result['direction_consistency']:.4f}")
    print(f"  3D variance coverage: {square_result['3d_variance_coverage']:.4f}")

    # Test 4: Number line
    print("\n--- Testing Number Line Hypothesis ---")
    line_result = test_number_line_hypothesis(model, tokenizer, list(range(1, 21)))
    results["number_line"] = line_result
    print(f"  Projection-number correlation: {line_result['projection_number_correlation']:.4f}")
    print(f"  Linearity score: {line_result['linearity_score']:.4f}")
    print(f"  Is linear: {line_result['is_linear']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    add_consistency = np.mean([results[f"add_{i}"]["consistency"] for i in [1, 2, 3, 5, 10]])
    print(f"Average addition consistency: {add_consistency:.4f}")
    print(f"Doubling consistency: {double_result['direction_consistency']:.4f}")
    print(f"Squaring consistency: {square_result['direction_consistency']:.4f}")
    print(f"Number line linearity: {line_result['linearity_score']:.4f}")

    # Verdict
    if add_consistency > 0.3 and line_result["is_linear"]:
        print("\n>>> ARITHMETIC GEOMETRY DETECTED: Operations manifest as consistent directions!")
    elif line_result["is_linear"]:
        print("\n>>> PARTIAL: Numbers are linear, but operations not fully consistent")
    else:
        print("\n>>> LIMITED: Arithmetic geometry is weak or absent")

    results["summary"] = {
        "avg_addition_consistency": float(add_consistency),
        "doubling_consistency": double_result["direction_consistency"],
        "squaring_consistency": square_result["direction_consistency"],
        "number_line_linearity": line_result["linearity_score"],
    }

    # Save
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
