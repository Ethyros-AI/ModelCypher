#!/usr/bin/env python3
"""Experiment 2: Cross-Modal CKA Alignment.

Tests whether FRB feature geometry aligns with known information-encoding
modalities (CLIP vision encoder, Whisper audio decoder, and synthetic baselines).

Key hypothesis test:
    - If FRBs share geometric structure with information systems,
      aligned CKA should approach 1.0
    - If FRBs are uncorrelated noise, aligned CKA ~ 0.3 (random baseline)

Usage:
    poetry run python experiments/astronomy/exp2_frb_cka_alignment.py
    poetry run python experiments/astronomy/exp2_frb_cka_alignment.py --real-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import find_alignment
from modelcypher.core.domain.geometry.cka import compute_cka

from shared.data_loader import load_frb_batch
from shared.feature_extraction import batch_extract_features, get_feature_dimension
from shared.reference_embeddings import (
    extract_clip_embeddings,
    extract_whisper_embeddings,
    generate_synthetic_embeddings,
)


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


def run_experiment(use_real_modalities: bool = True, synthetic_only: bool = False) -> dict:
    """Run CKA alignment experiment on FRB features.

    Args:
        use_real_modalities: If True, extract embeddings from CLIP/Whisper
        synthetic_only: If True, only run synthetic baselines (faster)
    """
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

    alignment_results = []

    # --- Synthetic baselines ---
    if not use_real_modalities or synthetic_only:
        print()
        print("=" * 40)
        print("SYNTHETIC BASELINES")
        print("=" * 40)

        synthetic_types = [
            ("random", "Random baseline (expected CKA ~ 0)"),
            ("structured", "Structured (simulates semantic manifold)"),
        ]

        for ref_type, description in synthetic_types:
            print()
            print(f"--- {description} ---")

            ref_result = generate_synthetic_embeddings(
                n_samples=n_frbs,
                dim=feature_dim,
                embedding_type=ref_type,
                backend=backend,
                seed=42,
            )

            metrics = compute_alignment_metrics(
                source=frb_features,
                target=ref_result.embeddings,
                backend=backend,
                label=f"synthetic_{ref_type}",
            )
            metrics["description"] = description
            metrics["modality"] = "synthetic"
            metrics["model_name"] = ref_result.model_name

            alignment_results.append(metrics)

    # --- Real modality embeddings ---
    if use_real_modalities and not synthetic_only:
        print()
        print("=" * 40)
        print("REAL MODALITY EMBEDDINGS")
        print("=" * 40)

        # CLIP (vision encoder)
        print()
        print("--- CLIP Vision Encoder ---")
        print("  Extracting CLIP text embeddings...")
        try:
            clip_result = extract_clip_embeddings(n_frbs, backend)
            print(f"  CLIP dimension: {clip_result.hidden_dim}")

            clip_metrics = compute_alignment_metrics(
                source=frb_features,
                target=clip_result.embeddings,
                backend=backend,
                label="clip",
            )
            clip_metrics["description"] = "CLIP vision encoder (text embeddings)"
            clip_metrics["modality"] = "vision"
            clip_metrics["model_name"] = clip_result.model_name
            clip_metrics["target_dim"] = clip_result.hidden_dim

            alignment_results.append(clip_metrics)
        except Exception as e:
            print(f"  ERROR extracting CLIP embeddings: {e}")
            alignment_results.append({
                "label": "clip",
                "description": "CLIP vision encoder (FAILED)",
                "error": str(e),
            })

        # Whisper (audio encoder)
        print()
        print("--- Whisper Audio Encoder ---")
        print("  Extracting Whisper decoder embeddings...")
        try:
            whisper_result = extract_whisper_embeddings(n_frbs, backend)
            print(f"  Whisper dimension: {whisper_result.hidden_dim}")

            whisper_metrics = compute_alignment_metrics(
                source=frb_features,
                target=whisper_result.embeddings,
                backend=backend,
                label="whisper",
            )
            whisper_metrics["description"] = "Whisper audio decoder"
            whisper_metrics["modality"] = "audio"
            whisper_metrics["model_name"] = whisper_result.model_name
            whisper_metrics["target_dim"] = whisper_result.hidden_dim

            alignment_results.append(whisper_metrics)
        except Exception as e:
            print(f"  ERROR extracting Whisper embeddings: {e}")
            alignment_results.append({
                "label": "whisper",
                "description": "Whisper audio decoder (FAILED)",
                "error": str(e),
            })

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
        if "error" in result:
            print(f"{result['label']:<20} {'ERROR':>10} {'ERROR':>12}")
            continue
        raw = result.get("raw_cka")
        aligned = result.get("aligned_cka")
        raw_str = f"{raw:.4f}" if raw is not None else "ERROR"
        aligned_str = f"{aligned:.4f}" if aligned is not None else "ERROR"
        print(f"{result['label']:<20} {raw_str:>10} {aligned_str:>12}")

    print()
    print("INTERPRETATION:")
    print("  - Raw CKA measures similarity in original coordinate systems")
    print("  - Aligned CKA measures similarity after Procrustes rotation")
    print("  - If aligned CKA > 0.7: Significant geometric structure shared")
    print("  - If aligned CKA ~ 0.3: No shared structure (random baseline)")

    # Build results dict
    results = {
        "experiment": "exp2_frb_cka_alignment",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_frbs": n_frbs,
            "feature_dimension": feature_dim,
            "use_real_modalities": use_real_modalities,
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


def main():
    parser = argparse.ArgumentParser(
        description="Cross-modal CKA alignment experiment for FRB features"
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Only use real modality embeddings (CLIP, Whisper)",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Only use synthetic baselines (faster, no model downloads)",
    )
    args = parser.parse_args()

    if args.real_only and args.synthetic_only:
        print("ERROR: Cannot specify both --real-only and --synthetic-only")
        sys.exit(1)

    run_experiment(
        use_real_modalities=not args.synthetic_only,
        synthetic_only=args.synthetic_only,
    )


if __name__ == "__main__":
    main()
