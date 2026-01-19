#!/usr/bin/env python3
"""Cross-Model Subspace Alignment Analysis.

Key question: If SmolLM compresses to ~150 dims and LFM2 to ~200 dims at middle layers,
what's the overlap? Is there a shared "universal" subspace?

Metrics:
1. Principal angles between subspaces (Grassmann distance)
2. Shared variance explained
3. Subspace CKA (alignment of effective subspaces, not full spaces)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For tests.fixtures

import logging
import json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_models():
    """Load both models."""
    from mlx_lm import load
    import mlx.core as mx

    source_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/HuggingFaceTB--SmolLM-135M")
    target_path = str(Path.home() / "ModelCypher/tests/fixtures/.models/mlx-community--LFM2-350M-MLX-bf16")

    logger.info("Loading SmolLM-135M...")
    source_model, source_tokenizer = load(source_path)
    mx.eval(source_model.parameters())

    logger.info("Loading LFM2-350M...")
    target_model, target_tokenizer = load(target_path)
    mx.eval(target_model.parameters())

    return (source_model, source_tokenizer), (target_model, target_tokenizer)


def get_layer_activation(model, tokenizer, text: str, layer_idx: int):
    """Get mean-pooled activation at specific layer."""
    import mlx.core as mx

    inner = model.model if hasattr(model, "model") else model

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    mx.eval(input_ids)

    if hasattr(inner, "embed_tokens"):
        h = inner.embed_tokens(input_ids)
    elif hasattr(inner, "wte"):
        h = inner.wte(input_ids)
    else:
        return None

    for idx, layer in enumerate(inner.layers):
        if idx > layer_idx:
            break
        result = layer(h)
        if isinstance(result, tuple):
            h = result[0]
        else:
            h = result

    # Mean pool
    pooled = mx.mean(h, axis=(0, 1))
    mx.eval(pooled)
    return pooled


def get_activations(model, tokenizer, probes: list[str], layer_idx: int):
    """Get activations for probes at a specific layer."""
    import mlx.core as mx

    activations = []

    for probe in probes:
        act = get_layer_activation(model, tokenizer, probe, layer_idx)
        if act is not None:
            activations.append(act)

    if not activations:
        return None

    stacked = mx.stack(activations, axis=0)
    # Convert to float32 before numpy (bfloat16 doesn't convert directly)
    stacked = stacked.astype(mx.float32)
    mx.eval(stacked)
    return np.array(stacked)


def compute_effective_subspace(activations: np.ndarray, threshold_ratio: float = 3.45e-4):
    """Compute the effective subspace via SVD.

    Returns:
        U: Left singular vectors (activation space basis)
        S: Singular values
        effective_rank: Number of significant dimensions
        U_effective: Basis for the effective subspace
    """
    # Center the activations
    centered = activations - activations.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Compute effective rank using relative threshold
    max_sv = S[0] if len(S) > 0 else 1.0
    threshold = max_sv * threshold_ratio
    effective_rank = int(np.sum(S > threshold))

    # Get the effective subspace basis (in feature space)
    # Vt rows are right singular vectors - these span the effective subspace in feature space
    V_effective = Vt[:effective_rank, :].T  # Shape: (hidden_dim, effective_rank)

    return U, S, effective_rank, V_effective


def principal_angles(V1: np.ndarray, V2: np.ndarray):
    """Compute principal angles between two subspaces.

    V1: (d, k1) - basis for subspace 1
    V2: (d, k2) - basis for subspace 2

    Returns angles in radians (sorted ascending).
    """
    # Orthonormalize both bases (they should already be, but ensure)
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)

    # Compute SVD of Q1.T @ Q2
    # Singular values are cos(angles)
    M = Q1.T @ Q2
    _, S, _ = np.linalg.svd(M)

    # Clip to [-1, 1] for numerical stability
    S = np.clip(S, -1.0, 1.0)

    # Angles
    angles = np.arccos(S)
    return angles


def subspace_overlap(V1: np.ndarray, V2: np.ndarray):
    """Compute overlap metrics between two subspaces.

    Returns:
        mean_cos: Mean cosine of principal angles (1 = identical, 0 = orthogonal)
        shared_dim: Approximate shared dimensionality
        grassmann_dist: Grassmann distance (sum of squared angles)
    """
    angles = principal_angles(V1, V2)

    # Mean cosine (how aligned are they on average)
    mean_cos = np.mean(np.cos(angles))

    # "Shared" dimensions: angles close to 0 (cos > 0.9 means angle < 25°)
    shared_dim = int(np.sum(np.cos(angles) > 0.9))

    # Grassmann distance
    grassmann_dist = np.sqrt(np.sum(angles ** 2))

    return {
        "mean_cos": float(mean_cos),
        "shared_dim": shared_dim,
        "total_angles": len(angles),
        "grassmann_dist": float(grassmann_dist),
        "min_angle_deg": float(np.degrees(angles.min())) if len(angles) > 0 else 0,
        "max_angle_deg": float(np.degrees(angles.max())) if len(angles) > 0 else 0,
        "median_angle_deg": float(np.degrees(np.median(angles))) if len(angles) > 0 else 0,
    }


def subspace_cka(V1: np.ndarray, V2: np.ndarray):
    """Compute CKA between two subspaces.

    Project random points onto each subspace and compute CKA of the projections.
    """
    d = V1.shape[0]
    n_samples = 1000

    # Random points
    X = np.random.randn(n_samples, d)

    # Project onto each subspace
    P1 = V1 @ V1.T  # Projection matrix for subspace 1
    P2 = V2 @ V2.T  # Projection matrix for subspace 2

    X1 = X @ P1
    X2 = X @ P2

    # Compute CKA
    def centering_matrix(n):
        return np.eye(n) - np.ones((n, n)) / n

    H = centering_matrix(n_samples)

    K1 = X1 @ X1.T
    K2 = X2 @ X2.T

    K1_centered = H @ K1 @ H
    K2_centered = H @ K2 @ H

    hsic = np.trace(K1_centered @ K2_centered)
    norm1 = np.sqrt(np.trace(K1_centered @ K1_centered))
    norm2 = np.sqrt(np.trace(K2_centered @ K2_centered))

    cka = hsic / (norm1 * norm2 + 1e-10)
    return float(cka)


def main():
    from tests.fixtures.models import get_atlas_probes

    # Load probes (get_atlas_probes returns list of strings directly)
    probe_texts = get_atlas_probes(n_samples=1000)

    logger.info(f"Using {len(probe_texts)} probes")

    # Load models
    (source_model, source_tok), (target_model, target_tok) = load_models()

    # Get layer counts
    source_inner = source_model.model if hasattr(source_model, "model") else source_model
    target_inner = target_model.model if hasattr(target_model, "model") else target_model

    source_layers = len(source_inner.layers)
    target_layers = len(target_inner.layers)

    logger.info(f"SmolLM: {source_layers} layers, LFM2: {target_layers} layers")

    # Test at multiple depths
    depths = [0.25, 0.50, 0.75]

    results = {
        "source_model": "SmolLM-135M",
        "target_model": "LFM2-350M",
        "n_probes": len(probe_texts),
        "depths": [],
    }

    for depth in depths:
        source_layer = int(depth * (source_layers - 1))
        target_layer = int(depth * (target_layers - 1))

        logger.info("=" * 60)
        logger.info(f"DEPTH {depth:.0%}: SmolLM layer {source_layer}, LFM2 layer {target_layer}")
        logger.info("=" * 60)

        # Get activations
        logger.info("Collecting SmolLM activations...")
        source_acts = get_activations(source_model, source_tok, probe_texts, source_layer)

        logger.info("Collecting LFM2 activations...")
        target_acts = get_activations(target_model, target_tok, probe_texts, target_layer)

        if source_acts is None or target_acts is None:
            logger.error("Failed to get activations")
            continue

        logger.info(f"Source shape: {source_acts.shape}, Target shape: {target_acts.shape}")

        # Compute effective subspaces
        logger.info("Computing effective subspaces...")
        _, S_source, source_eff_rank, V_source = compute_effective_subspace(source_acts)
        _, S_target, target_eff_rank, V_target = compute_effective_subspace(target_acts)

        logger.info(f"SmolLM effective rank: {source_eff_rank}/{source_acts.shape[1]}")
        logger.info(f"LFM2 effective rank: {target_eff_rank}/{target_acts.shape[1]}")

        # Compute overlap metrics
        logger.info("Computing subspace overlap...")

        # We need to compare in a common space
        # Option 1: Compare via the Gram matrices (dimension-agnostic)
        # Option 2: Use the smaller effective subspace

        # Let's compute overlap using the Gram matrices
        # G_source = source_acts @ source_acts.T  (n x n)
        # G_target = target_acts @ target_acts.T  (n x n)
        # These are in the same space (probe space)!

        G_source = source_acts @ source_acts.T
        G_target = target_acts @ target_acts.T

        # Compute CKA between Gram matrices
        def centering_matrix(n):
            return np.eye(n) - np.ones((n, n)) / n

        n = G_source.shape[0]
        H = centering_matrix(n)

        K1_centered = H @ G_source @ H
        K2_centered = H @ G_target @ H

        hsic = np.trace(K1_centered @ K2_centered)
        norm1 = np.sqrt(np.trace(K1_centered @ K1_centered))
        norm2 = np.sqrt(np.trace(K2_centered @ K2_centered))

        gram_cka = hsic / (norm1 * norm2 + 1e-10)

        logger.info(f"Raw Gram CKA (before alignment): {gram_cka:.4f}")

        # Now compute effective subspace overlap
        # Project Gram matrices onto their effective subspaces
        U_source, S_source_full, _ = np.linalg.svd(G_source, full_matrices=False)
        U_target, S_target_full, _ = np.linalg.svd(G_target, full_matrices=False)

        # Effective ranks in Gram space
        source_gram_rank = int(np.sum(S_source_full > S_source_full[0] * 3.45e-4))
        target_gram_rank = int(np.sum(S_target_full > S_target_full[0] * 3.45e-4))

        logger.info(f"SmolLM Gram effective rank: {source_gram_rank}/{n}")
        logger.info(f"LFM2 Gram effective rank: {target_gram_rank}/{n}")

        # Compute overlap between effective Gram subspaces
        U_source_eff = U_source[:, :source_gram_rank]
        U_target_eff = U_target[:, :target_gram_rank]

        overlap = subspace_overlap(U_source_eff, U_target_eff)

        logger.info(f"Subspace overlap metrics:")
        logger.info(f"  Mean cosine: {overlap['mean_cos']:.4f}")
        logger.info(f"  Shared dimensions (angle < 25°): {overlap['shared_dim']}")
        logger.info(f"  Grassmann distance: {overlap['grassmann_dist']:.4f}")
        logger.info(f"  Angle range: {overlap['min_angle_deg']:.1f}° - {overlap['max_angle_deg']:.1f}°")
        logger.info(f"  Median angle: {overlap['median_angle_deg']:.1f}°")

        # Compute CKA between effective subspaces only
        G_source_eff = U_source_eff @ np.diag(S_source_full[:source_gram_rank]) @ U_source_eff.T
        G_target_eff = U_target_eff @ np.diag(S_target_full[:target_gram_rank]) @ U_target_eff.T

        K1_eff_centered = H @ G_source_eff @ H
        K2_eff_centered = H @ G_target_eff @ H

        hsic_eff = np.trace(K1_eff_centered @ K2_eff_centered)
        norm1_eff = np.sqrt(np.trace(K1_eff_centered @ K1_eff_centered))
        norm2_eff = np.sqrt(np.trace(K2_eff_centered @ K2_eff_centered))

        effective_cka = hsic_eff / (norm1_eff * norm2_eff + 1e-10)

        logger.info(f"Effective subspace CKA: {effective_cka:.4f}")

        depth_result = {
            "depth": depth,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "source_hidden_dim": int(source_acts.shape[1]),
            "target_hidden_dim": int(target_acts.shape[1]),
            "source_effective_rank": source_eff_rank,
            "target_effective_rank": target_eff_rank,
            "source_gram_rank": source_gram_rank,
            "target_gram_rank": target_gram_rank,
            "raw_gram_cka": float(gram_cka),
            "effective_cka": float(effective_cka),
            "subspace_overlap": overlap,
        }

        results["depths"].append(depth_result)

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY: Cross-Model Subspace Alignment")
    logger.info("=" * 60)

    for d in results["depths"]:
        logger.info(f"\nDepth {d['depth']:.0%}:")
        logger.info(f"  Effective ranks: SmolLM={d['source_effective_rank']}, LFM2={d['target_effective_rank']}")
        logger.info(f"  Gram ranks: SmolLM={d['source_gram_rank']}, LFM2={d['target_gram_rank']}")
        logger.info(f"  Raw CKA: {d['raw_gram_cka']:.4f}")
        logger.info(f"  Effective CKA: {d['effective_cka']:.4f}")
        logger.info(f"  Shared dims: {d['subspace_overlap']['shared_dim']}")
        logger.info(f"  Mean cos(angle): {d['subspace_overlap']['mean_cos']:.4f}")

    # Save results
    output_path = Path(__file__).parent / "cross_model_subspace_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
