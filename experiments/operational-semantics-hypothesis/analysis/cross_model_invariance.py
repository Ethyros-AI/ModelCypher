#!/usr/bin/env python3
"""
Cross-Model Invariance Test

If the Pythagorean structure is a REAL feature of conceptual space,
then after Procrustes alignment, the structure should appear at the
SAME relative position across different models.

This tests the "invariant but twisted" hypothesis - the structure
exists in all models but needs rotation to align.
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


def get_number_embeddings(model, tokenizer, numbers: list[int]) -> np.ndarray:
    """Get embeddings for a list of numbers."""
    embeddings = []
    for n in numbers:
        text = f"The number {n}."
        emb = extract_embedding(model, tokenizer, text)
        embeddings.append(emb)
    return np.stack(embeddings)


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Align source to target using Procrustes.
    Returns aligned source and alignment error.
    """
    # Center both
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Find optimal rotation
    R, scale = orthogonal_procrustes(source_centered, target_centered)

    # Apply rotation
    aligned = source_centered @ R

    # Compute alignment error
    error = np.linalg.norm(aligned - target_centered) / np.linalg.norm(target_centered)

    return aligned, float(error)


def compute_pythagorean_structure(embeddings: dict[int, np.ndarray]) -> dict:
    """
    Compute the geometric signature of Pythagorean relationships.
    This signature should be invariant across models after alignment.
    """
    # Get key embeddings
    e3, e4, e5, e6 = embeddings[3], embeddings[4], embeddings[5], embeddings[6]
    e9, e16, e25 = embeddings[9], embeddings[16], embeddings[25]

    # Signature 1: Relative position of 5 to (3,4) vs 6 to (3,4)
    centroid_34 = (e3 + e4) / 2
    dist_5_to_34 = np.linalg.norm(e5 - centroid_34)
    dist_6_to_34 = np.linalg.norm(e6 - centroid_34)
    ratio_5_6 = dist_5_to_34 / (dist_6_to_34 + 1e-8)

    # Signature 2: The "squared" triangle (3², 4², 5²) = (9, 16, 25)
    # Does this form a consistent shape?
    d_9_16 = np.linalg.norm(e9 - e16)
    d_16_25 = np.linalg.norm(e16 - e25)
    d_9_25 = np.linalg.norm(e9 - e25)
    triangle_ratios = (d_9_16 / d_9_25, d_16_25 / d_9_25, 1.0)

    # Signature 3: Direction from 3 to 9 (squaring) vs 4 to 16
    dir_3_9 = (e9 - e3) / (np.linalg.norm(e9 - e3) + 1e-8)
    dir_4_16 = (e16 - e4) / (np.linalg.norm(e16 - e4) + 1e-8)
    squaring_consistency = float(np.dot(dir_3_9, dir_4_16))

    # Signature 4: The Pythagorean "manifold position"
    # Project 5 onto plane spanned by 3 and 4
    basis = np.stack([e3, e4]).T
    Q, _ = np.linalg.qr(basis)
    proj_5 = Q @ (Q.T @ e5)
    residual_5 = np.linalg.norm(e5 - proj_5)
    proj_coords = Q.T @ e5  # 2D coordinates on the (3,4) plane

    return {
        "ratio_5_6_to_centroid": ratio_5_6,
        "squared_triangle_ratios": list(triangle_ratios),
        "squaring_direction_consistency": squaring_consistency,
        "projection_residual_5": float(residual_5),
        "projection_coords_5": proj_coords.tolist(),
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
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "cross_model.json"))
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-MODEL INVARIANCE TEST")
    print("Testing if Pythagorean structure is invariant across models")
    print("=" * 70)

    # Numbers we need
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 24, 25]

    all_embeddings = {}
    all_structures = {}

    for model_path in args.models:
        name = Path(model_path).name
        print(f"\n--- Loading {name} ---")

        model, tokenizer = load_model(model_path)

        # Get embeddings
        emb_matrix = get_number_embeddings(model, tokenizer, numbers)
        embeddings = {n: emb_matrix[i] for i, n in enumerate(numbers)}
        all_embeddings[name] = embeddings

        # Compute structure
        structure = compute_pythagorean_structure(embeddings)
        all_structures[name] = structure
        print(f"  5/6 ratio: {structure['ratio_5_6_to_centroid']:.4f}")
        print(f"  Squaring consistency: {structure['squaring_direction_consistency']:.4f}")

        # Free memory
        del model, tokenizer
        import gc
        gc.collect()

    # Compare structures across models
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON")
    print("=" * 70)

    model_names = list(all_structures.keys())
    n_models = len(model_names)

    # Compare all pairs
    invariance_scores = []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            name_i, name_j = model_names[i], model_names[j]
            struct_i, struct_j = all_structures[name_i], all_structures[name_j]

            # Compare signatures
            ratio_diff = abs(struct_i["ratio_5_6_to_centroid"] - struct_j["ratio_5_6_to_centroid"])
            sq_diff = abs(struct_i["squaring_direction_consistency"] - struct_j["squaring_direction_consistency"])

            # Triangle shape similarity
            tri_i = np.array(struct_i["squared_triangle_ratios"])
            tri_j = np.array(struct_j["squared_triangle_ratios"])
            tri_sim = float(np.dot(tri_i, tri_j) / (np.linalg.norm(tri_i) * np.linalg.norm(tri_j)))

            print(f"\n{name_i} vs {name_j}:")
            print(f"  5/6 ratio difference: {ratio_diff:.4f}")
            print(f"  Squaring consistency difference: {sq_diff:.4f}")
            print(f"  Triangle shape similarity: {tri_sim:.4f}")

            # Procrustes alignment
            emb_i = np.stack([all_embeddings[name_i][n] for n in numbers])
            emb_j = np.stack([all_embeddings[name_j][n] for n in numbers])

            # Handle dimension mismatch by projecting to common space
            min_dim = min(emb_i.shape[1], emb_j.shape[1])
            emb_i_proj = emb_i[:, :min_dim]
            emb_j_proj = emb_j[:, :min_dim]

            aligned_i, align_error = procrustes_align(emb_i_proj, emb_j_proj)
            print(f"  Procrustes alignment error: {align_error:.4f}")

            # After alignment, check if 5 is in similar position relative to (3,4)
            idx_3, idx_4, idx_5 = numbers.index(3), numbers.index(4), numbers.index(5)
            centroid_34_aligned = (aligned_i[idx_3] + aligned_i[idx_4]) / 2
            centroid_34_target = (emb_j_proj[idx_3] + emb_j_proj[idx_4]) / 2

            pos_5_aligned = aligned_i[idx_5] - centroid_34_aligned
            pos_5_target = emb_j_proj[idx_5] - centroid_34_target

            pos_5_sim = float(np.dot(pos_5_aligned, pos_5_target) /
                            (np.linalg.norm(pos_5_aligned) * np.linalg.norm(pos_5_target) + 1e-8))
            print(f"  Position of 5 similarity after alignment: {pos_5_sim:.4f}")

            invariance_scores.append({
                "model_a": name_i,
                "model_b": name_j,
                "ratio_diff": float(ratio_diff),
                "triangle_similarity": tri_sim,
                "procrustes_error": align_error,
                "position_5_similarity": pos_5_sim,
            })

    # Summary
    print("\n" + "=" * 70)
    print("INVARIANCE SUMMARY")
    print("=" * 70)

    avg_pos_sim = np.mean([s["position_5_similarity"] for s in invariance_scores])
    avg_tri_sim = np.mean([s["triangle_similarity"] for s in invariance_scores])
    avg_proc_err = np.mean([s["procrustes_error"] for s in invariance_scores])

    print(f"\nAverage position-5 similarity after alignment: {avg_pos_sim:.4f}")
    print(f"Average triangle shape similarity: {avg_tri_sim:.4f}")
    print(f"Average Procrustes error: {avg_proc_err:.4f}")

    if avg_pos_sim > 0.5 and avg_tri_sim > 0.9:
        print("\n>>> INVARIANCE SUPPORTED: Pythagorean structure is consistent across models!")
    else:
        print("\n>>> INVARIANCE NOT CLEAR: Structure varies significantly between models")

    # Save results
    results = {
        "structures": all_structures,
        "comparisons": invariance_scores,
        "summary": {
            "avg_position_similarity": float(avg_pos_sim),
            "avg_triangle_similarity": float(avg_tri_sim),
            "avg_procrustes_error": float(avg_proc_err),
        }
    }

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
