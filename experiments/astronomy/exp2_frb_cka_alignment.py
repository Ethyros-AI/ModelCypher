#!/usr/bin/env python3
"""Experiment 2: Cross-Modal CKA Alignment.

Tests whether FRB feature geometry aligns with known information-encoding
modalities (random embeddings as baseline, with option to extend to real
embeddings from CLIP/Whisper/LLM in future).

Key hypothesis test:
    - If FRBs share geometric structure with information systems,
      aligned CKA should approach 1.0
    - If FRBs are uncorrelated noise, aligned CKA ~ 0.3 (random baseline)

Usage:
    poetry run python experiments/astronomy/exp2_frb_cka_alignment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.core.domain._backend import get_default_backend, initialize_default_backend
from modelcypher.core.domain.geometry.gram_aligner import find_alignment
from modelcypher.core.domain.geometry.cka import compute_cka

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features, get_feature_dimension


def generate_reference_embeddings(
    n_samples: int,
    dim: int,
    embedding_type: str,
    backend,
    seed: int = 42,
) -> np.ndarray:
    """Generate reference embeddings for comparison.

    For initial experiments, we use synthetic embeddings with known structure.
    Future work: replace with actual CLIP/Whisper/LLM embeddings.

    Args:
        n_samples: Number of embedding vectors
        dim: Embedding dimension
        embedding_type: Type of embedding to generate:
            - "random": IID Gaussian (baseline, should have CKA ~ 0)
            - "structured": Low-rank structure (simulates semantic manifold)
            - "correlated": Partially correlated with input
        backend: Backend instance
        seed: Random seed for reproducibility

    Returns:
        [n_samples, dim] array of reference embeddings
    """
    rng = np.random.default_rng(seed)

    if embedding_type == "random":
        # Pure random - CKA should be ~0
        embeddings = rng.standard_normal((n_samples, dim)).astype(np.float32)

    elif embedding_type == "structured":
        # Low-rank structure - simulates semantic manifold
        # Create embeddings that lie on a ~10D manifold
        intrinsic_dim = 10
        latent = rng.standard_normal((n_samples, intrinsic_dim)).astype(np.float32)
        projection = rng.standard_normal((intrinsic_dim, dim)).astype(np.float32)
        embeddings = latent @ projection
        # Add small noise
        embeddings += 0.1 * rng.standard_normal((n_samples, dim)).astype(np.float32)

    elif embedding_type == "correlated":
        # Partially correlated - should have intermediate CKA
        base = rng.standard_normal((n_samples, dim)).astype(np.float32)
        noise = rng.standard_normal((n_samples, dim)).astype(np.float32)
        # 50% signal, 50% noise
        embeddings = 0.5 * base + 0.5 * noise

    else:
        msg = f"Unknown embedding type: {embedding_type}"
        raise ValueError(msg)

    return backend.array(embeddings)


def compute_alignment_metrics(
    source: np.ndarray,
    target: np.ndarray,
    backend,
    label: str = "",
) -> dict:
    """Compute raw and aligned CKA between two activation sets.

    Args:
        source: [n, d_source] source activations
        target: [n, d_target] target activations
        backend: Backend instance
        label: Label for logging

    Returns:
        Dict with raw_cka, aligned_cka, and alignment diagnostics
    """
    print(f"  Computing CKA for {label}...")

    # Raw CKA (before alignment)
    try:
        raw_cka_result = compute_cka(source, target, backend=backend)
        raw_cka = raw_cka_result.cka
        print(f"    Raw CKA: {raw_cka:.4f}")
    except Exception as e:
        print(f"    ERROR computing raw CKA: {e}")
        raw_cka = None

    # Find alignment and compute aligned CKA
    try:
        alignment_result = find_alignment(source, target, backend=backend)

        # Apply alignment transform
        aligned_source = backend.matmul(source, alignment_result.feature_transform)

        # Compute aligned CKA
        aligned_cka_result = compute_cka(aligned_source, target, backend=backend)
        aligned_cka = aligned_cka_result.cka
        print(f"    Aligned CKA: {aligned_cka:.4f}")

        # Get alignment diagnostics
        gram_condition = alignment_result.gram_condition_number
        alignment_error = alignment_result.alignment_error

        print(f"    Gram condition number: {gram_condition:.2f}")
        print(f"    Alignment error: {alignment_error:.6f}")

    except Exception as e:
        print(f"    ERROR computing alignment: {e}")
        aligned_cka = None
        gram_condition = None
        alignment_error = None

    return {
        "label": label,
        "raw_cka": raw_cka,
        "aligned_cka": aligned_cka,
        "gram_condition_number": gram_condition,
        "alignment_error": alignment_error,
        "n_samples": source.shape[0],
        "source_dim": source.shape[1],
        "target_dim": target.shape[1],
    }


def run_experiment() -> dict:
    """Run CKA alignment experiment on FRB features."""
    # Setup
    data_dir = Path(__file__).parent / "data" / "raw"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Initialize backend
    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 2: Cross-Modal CKA Alignment")
    print("=" * 60)
    print()

    # Find and load FRB files
    frb_files = sorted(data_dir.glob("FRB*_waterfall.h5"))

    if not frb_files:
        print(f"ERROR: No FRB files found in {data_dir}")
        print("Run the download script first:")
        print("  poetry run python experiments/astronomy/data/download_chime_frb.py --limit 20")
        return {"error": "No data files found"}

    print(f"Found {len(frb_files)} FRB files")
    print()

    # Load FRBs
    print("Loading FRB waterfalls...")
    waterfalls = load_frb_batch([str(f) for f in frb_files])
    n_frbs = len(waterfalls)
    print(f"  Successfully loaded: {n_frbs} FRBs")

    if n_frbs < 5:
        print("ERROR: Need at least 5 FRBs for meaningful CKA alignment")
        return {"error": "Insufficient data"}

    # Extract features
    print()
    print("Extracting FRB features...")
    frb_features = batch_extract_features(waterfalls, backend)
    feature_dim = get_feature_dimension()
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Feature matrix shape: [{n_frbs}, {feature_dim}]")

    # Generate reference embeddings
    print()
    print("Generating reference embeddings...")

    reference_types = [
        ("random", "Random baseline (expected CKA ~ 0)"),
        ("structured", "Structured (simulates semantic manifold)"),
    ]

    alignment_results = []

    for ref_type, description in reference_types:
        print()
        print(f"--- {description} ---")

        # Generate reference embeddings with same sample count
        ref_embeddings = generate_reference_embeddings(
            n_samples=n_frbs,
            dim=feature_dim,  # Match FRB feature dimension
            embedding_type=ref_type,
            backend=backend,
            seed=42,
        )

        # Compute alignment metrics
        metrics = compute_alignment_metrics(
            source=frb_features,
            target=ref_embeddings,
            backend=backend,
            label=ref_type,
        )
        metrics["description"] = description

        alignment_results.append(metrics)

    # Self-alignment test (sanity check - should be CKA = 1.0)
    print()
    print("--- Self-alignment (sanity check) ---")
    self_metrics = compute_alignment_metrics(
        source=frb_features,
        target=frb_features,
        backend=backend,
        label="self",
    )
    self_metrics["description"] = "FRB features aligned to themselves (should be 1.0)"
    alignment_results.append(self_metrics)

    # Summary
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Reference Type':<20} {'Raw CKA':>10} {'Aligned CKA':>12}")
    print("-" * 44)

    for result in alignment_results:
        raw = result.get("raw_cka")
        aligned = result.get("aligned_cka")
        raw_str = f"{raw:.4f}" if raw is not None else "ERROR"
        aligned_str = f"{aligned:.4f}" if aligned is not None else "ERROR"
        print(f"{result['label']:<20} {raw_str:>10} {aligned_str:>12}")

    print()
    print("INTERPRETATION:")
    print("  - If random baseline has CKA ~ 0: FRBs don't match random noise")
    print("  - If structured has CKA > 0.5: FRBs may share manifold structure")
    print("  - If self has CKA = 1.0: Sanity check passed")
    print()
    print("NEXT STEPS:")
    print("  - Replace synthetic embeddings with real CLIP/Whisper/LLM embeddings")
    print("  - Compare FRB geometry to actual information-encoding systems")

    # Build results dict
    results = {
        "experiment": "exp2_frb_cka_alignment",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_frbs": n_frbs,
            "feature_dimension": feature_dim,
            "frb_files": [str(f) for f in frb_files],
        },
        "alignment_results": alignment_results,
        "frb_metadata": [
            {
                "tns_name": w.metadata.tns_name,
                "dm": w.metadata.dm,
                "snr": w.metadata.snr,
            }
            for w in waterfalls
        ],
    }

    # Save results
    output_path = results_dir / "exp2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
