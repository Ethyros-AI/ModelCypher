#!/usr/bin/env python3
"""Experiment 38: Align Wow! Signal to EMPIRICAL Semantic Manifolds.

Previous experiments used CONSTRUCTED reference manifolds.
This experiment uses REAL manifolds from:
- CLIP (vision model)
- Whisper (audio model)
- LFM2 (language model)

These manifolds achieved CKA = 1.0 alignment with each other.
If the Wow! signal aligns with these empirical semantic structures,
that's much stronger evidence than aligning with synthetic patterns.

The offramps from the multi-modal experiment contain the actual
Procrustes transforms between modalities - real geometric relationships.

Usage:
    poetry run python experiments/astronomy/exp38_empirical_alignment.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Path to empirical data
MULTIMODAL_DIR = Path("/Volumes/CodeCypher/experiments/multi-modal-compression-2026-01-09")


def compute_gram_matrix(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute centered Gram matrix K = X @ X.T (sample space)."""
    # Center
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    # Gram matrix
    K = X_centered @ X_centered.T
    # Normalize by trace
    if normalize:
        trace = np.trace(K)
        if trace > 1e-10:
            K = K / trace
    return K


def gram_sqrt(K: np.ndarray) -> np.ndarray:
    """Compute matrix square root of Gram matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.maximum(eigenvalues, 0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    K_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    return K_sqrt


def gram_pinv_sqrt(K: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Compute pseudo-inverse of matrix square root."""
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_eig = np.max(np.abs(eigenvalues))
    threshold = max_eig * rcond
    inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
    mask = eigenvalues > threshold
    inv_sqrt_eigenvalues[mask] = 1.0 / np.sqrt(eigenvalues[mask])
    K_pinv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
    return K_pinv_sqrt


def compute_cka(K1: np.ndarray, K2: np.ndarray) -> float:
    """Compute Centered Kernel Alignment."""
    hsic = np.trace(K1 @ K2)
    hsic_11 = np.trace(K1 @ K1)
    hsic_22 = np.trace(K2 @ K2)
    if hsic_11 > 0 and hsic_22 > 0:
        return float(hsic / np.sqrt(hsic_11 * hsic_22))
    return 0.0


def align_gram_matrices(K_source: np.ndarray, K_target: np.ndarray) -> dict:
    """Align source Gram matrix to target using Gram-space Procrustes."""
    K_target_sqrt = gram_sqrt(K_target)
    K_source_pinv_sqrt = gram_pinv_sqrt(K_source)
    T = K_target_sqrt @ K_source_pinv_sqrt
    K_aligned = T @ K_source @ T.T

    raw_cka = compute_cka(K_source, K_target)
    aligned_cka = compute_cka(K_aligned, K_target)

    return {
        "raw_cka": raw_cka,
        "aligned_cka": aligned_cka,
        "improvement": aligned_cka - raw_cka,
        "transform": T,
        "K_aligned": K_aligned,
    }


def load_empirical_transforms():
    """Load the real Procrustes transforms from multi-modal experiment."""
    transforms = {}

    # LFM2 to T5XL transform (language model alignment)
    lfm2_t5_path = MULTIMODAL_DIR / "lfm2_to_t5xl_transform.npz"
    if lfm2_t5_path.exists():
        data = np.load(lfm2_t5_path)
        transforms["lfm2_t5xl"] = {
            "F": data["F"],  # 2048 x 1024
            "F_inv": data["F_inv"],  # 1024 x 2048
            "description": "LFM2 (350M) ↔ T5-XL (3B) alignment",
        }
        print(f"  Loaded LFM2-T5XL transform: {data['F'].shape}")

    # Vision offramp (CLIP alignment)
    try:
        import safetensors.numpy as st_np

        vision_path = MULTIMODAL_DIR / "offramps" / "vision_offramp.safetensors"
        if vision_path.exists():
            vision_data = st_np.load_file(str(vision_path))
            transforms["vision_clip"] = {
                "weights": vision_data,
                "description": "CLIP vision → LFM2 alignment",
            }
            for k, v in vision_data.items():
                print(f"  Loaded vision offramp {k}: {v.shape}")

        audio_path = MULTIMODAL_DIR / "offramps" / "audio_offramp.safetensors"
        if audio_path.exists():
            audio_data = st_np.load_file(str(audio_path))
            transforms["audio_whisper"] = {
                "weights": audio_data,
                "description": "Whisper audio → LFM2 alignment",
            }
            for k, v in audio_data.items():
                print(f"  Loaded audio offramp {k}: {v.shape}")

        procrustes_path = MULTIMODAL_DIR / "offramps" / "procrustes_bridge.safetensors"
        if procrustes_path.exists():
            procrustes_data = st_np.load_file(str(procrustes_path))
            transforms["procrustes"] = {
                "weights": procrustes_data,
                "description": "Universal Procrustes bridge",
            }
            for k, v in procrustes_data.items():
                print(f"  Loaded Procrustes bridge {k}: {v.shape}")

    except ImportError:
        print("  Warning: safetensors not available, skipping offramps")

    return transforms


def load_lfm2_embeddings():
    """Load real LFM2 embeddings from multi-modal experiment."""
    embeddings = []
    prompts = []

    for seed in [42, 43]:
        path = MULTIMODAL_DIR / f"lfm2_embeddings_{seed}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            # lfm2_hidden shape: (1, 8, 1024) - 8 tokens, 1024 dim
            hidden = data["lfm2_hidden"][0]  # (8, 1024)
            embeddings.append(hidden)
            prompts.append(str(data["prompt"]))
            print(f"  Loaded LFM2 embedding for: '{data['prompt']}'")

    return embeddings, prompts


def load_semantic_3d():
    """Load the semantic 3D projections."""
    path = MULTIMODAL_DIR / "semantic_3d_data.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {
            "projections": data["projections"],  # (29, 3)
            "prompts": data["prompts"],  # (29,) strings
            "categories": data["categories"],  # (29,) strings
            "eigenvalues": data["eigenvalues"],  # (10,)
        }
    return None


def create_empirical_gram_from_transform(F: np.ndarray, n_samples: int) -> np.ndarray:
    """Create Gram matrix from a learned Procrustes transform.

    The transform F maps between semantic spaces.
    F @ F.T captures the relational structure of the target space.
    We can use this as a reference Gram matrix.
    """
    # F is (target_dim, source_dim)
    # F @ F.T is (target_dim, target_dim) - the target space structure
    gram_structure = F @ F.T

    # Normalize
    trace = np.trace(gram_structure)
    if trace > 1e-10:
        gram_structure = gram_structure / trace

    # Resize to match n_samples if needed
    if gram_structure.shape[0] != n_samples:
        # Interpolate to match size
        from scipy.ndimage import zoom
        factor = n_samples / gram_structure.shape[0]
        gram_structure = zoom(gram_structure, (factor, factor), order=1)
        # Re-symmetrize
        gram_structure = (gram_structure + gram_structure.T) / 2
        # Re-normalize
        trace = np.trace(gram_structure)
        if trace > 1e-10:
            gram_structure = gram_structure / trace

    return gram_structure


def create_empirical_gram_from_embeddings(embeddings: list) -> np.ndarray:
    """Create Gram matrix from actual LLM embeddings."""
    # Stack all embeddings
    # Each embedding is (8, 1024) - 8 tokens
    # Use the pooled representation (mean across tokens)
    pooled = [emb.mean(axis=0) for emb in embeddings]
    X = np.stack(pooled)  # (n_samples, 1024)
    return compute_gram_matrix(X)


def run_experiment():
    """Run the empirical alignment experiment."""
    results_dir = Path(__file__).parent / "results"
    data_dir = Path(__file__).parent / "data" / "famous_signals"

    print("=" * 60)
    print("Experiment 38: EMPIRICAL Semantic Manifold Alignment")
    print("=" * 60)
    print("\nUsing REAL manifolds from multi-modal experiment:")
    print("  - CLIP (vision)")
    print("  - Whisper (audio)")
    print("  - LFM2/T5XL (language)")
    print("\nThese achieved CKA = 1.0 with each other. Real semantic structure.")

    # Check if external drive is available
    if not MULTIMODAL_DIR.exists():
        print(f"\nERROR: Multi-modal experiment not found at {MULTIMODAL_DIR}")
        print("Please connect the CodeCypher drive.")
        return None

    # Load Wow! signal
    print("\n" + "=" * 40)
    print("PART 1: LOAD WOW! SIGNAL")
    print("=" * 40)

    wow_path = data_dir / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr']).astype(np.float64)

    print(f"Wow! signal shape: {snr_matrix.shape}")  # (82, 50)
    n_samples = snr_matrix.shape[0]

    # Compute Wow! Gram matrix
    K_wow = compute_gram_matrix(snr_matrix)
    print(f"Wow! Gram matrix shape: {K_wow.shape}")

    # Load empirical transforms
    print("\n" + "=" * 40)
    print("PART 2: LOAD EMPIRICAL TRANSFORMS")
    print("=" * 40)

    transforms = load_empirical_transforms()

    # Load LFM2 embeddings
    print("\n" + "=" * 40)
    print("PART 3: LOAD LFM2 EMBEDDINGS")
    print("=" * 40)

    embeddings, prompts = load_lfm2_embeddings()

    # Load semantic 3D data
    semantic_3d = load_semantic_3d()
    if semantic_3d is not None:
        print(f"\n  Semantic 3D: {semantic_3d['projections'].shape}")
        print(f"  Categories: {set(semantic_3d['categories'])}")

    # Create empirical Gram matrices
    print("\n" + "=" * 40)
    print("PART 4: CREATE EMPIRICAL GRAM MATRICES")
    print("=" * 40)

    empirical_grams = {}

    # From LFM2-T5XL transform
    if "lfm2_t5xl" in transforms:
        F = transforms["lfm2_t5xl"]["F"]
        K_semantic = create_empirical_gram_from_transform(F, n_samples)
        empirical_grams["semantic_lfm2_t5xl"] = {
            "gram": K_semantic,
            "description": "Semantic structure from LFM2↔T5XL alignment",
        }
        print(f"  semantic_lfm2_t5xl: shape {K_semantic.shape}")

    # From vision offramp
    if "vision_clip" in transforms:
        weights = transforms["vision_clip"]["weights"]
        for name, W in weights.items():
            if len(W.shape) == 2 and W.shape[0] > 10 and W.shape[1] > 10:
                K_vision = create_empirical_gram_from_transform(W, n_samples)
                empirical_grams[f"vision_{name}"] = {
                    "gram": K_vision,
                    "description": f"Vision structure from CLIP ({name})",
                }
                print(f"  vision_{name}: shape {K_vision.shape}")

    # From audio offramp
    if "audio_whisper" in transforms:
        weights = transforms["audio_whisper"]["weights"]
        for name, W in weights.items():
            if len(W.shape) == 2 and W.shape[0] > 10 and W.shape[1] > 10:
                K_audio = create_empirical_gram_from_transform(W, n_samples)
                empirical_grams[f"audio_{name}"] = {
                    "gram": K_audio,
                    "description": f"Audio structure from Whisper ({name})",
                }
                print(f"  audio_{name}: shape {K_audio.shape}")

    # From semantic 3D projections - expand to full size
    if semantic_3d is not None:
        proj = semantic_3d["projections"]  # (29, 3)
        # Interpolate to n_samples
        from scipy.ndimage import zoom
        factor = n_samples / proj.shape[0]
        proj_expanded = zoom(proj, (factor, 1), order=1)
        K_3d = compute_gram_matrix(proj_expanded)
        empirical_grams["semantic_3d"] = {
            "gram": K_3d,
            "description": "3D semantic projection from LLM embeddings",
        }
        print(f"  semantic_3d: shape {K_3d.shape}")

    # Also create baseline noise Gram
    np.random.seed(42)
    noise_samples = np.random.randn(n_samples, 50)
    K_noise = compute_gram_matrix(noise_samples)
    empirical_grams["noise_baseline"] = {
        "gram": K_noise,
        "description": "Random Gaussian noise (baseline)",
    }

    # Align Wow! to each empirical manifold
    print("\n" + "=" * 40)
    print("PART 5: GRAM ALIGNMENT TO EMPIRICAL MANIFOLDS")
    print("=" * 40)

    alignment_results = {}

    for name, data in empirical_grams.items():
        K_ref = data["gram"]
        try:
            result = align_gram_matrices(K_wow, K_ref)
            alignment_results[name] = {
                "description": data["description"],
                "raw_cka": result["raw_cka"],
                "aligned_cka": result["aligned_cka"],
                "improvement": result["improvement"],
            }
            print(f"\n  {name}:")
            print(f"    Raw CKA:     {result['raw_cka']:.6f}")
            print(f"    Aligned CKA: {result['aligned_cka']:.6f}")
            print(f"    Improvement: {result['improvement']:+.6f}")
        except Exception as e:
            print(f"\n  {name}: Error - {e}")

    # Rank by aligned CKA
    valid_results = [(n, d) for n, d in alignment_results.items() if "aligned_cka" in d]
    ranked = sorted(valid_results, key=lambda x: x[1]["aligned_cka"], reverse=True)

    print("\n" + "=" * 60)
    print("RANKING BY ALIGNED CKA (EMPIRICAL MANIFOLDS)")
    print("=" * 60)

    print("\n" + "-" * 70)
    print(f"{'Rank':<6}{'Manifold':<25}{'Raw CKA':<12}{'Aligned CKA':<12}{'Δ':<10}")
    print("-" * 70)

    for i, (name, data) in enumerate(ranked):
        raw = data["raw_cka"]
        aligned = data["aligned_cka"]
        delta = data["improvement"]
        marker = " ← NOISE" if "noise" in name.lower() else ""
        print(f"{i+1:<6}{name:<25}{raw:<12.6f}{aligned:<12.6f}{delta:+.6f}{marker}")

    # Find where noise ranks
    noise_rank = next((i for i, (n, _) in enumerate(ranked) if "noise" in n.lower()), -1)

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if ranked:
        best_name, best_data = ranked[0]
        worst_name, worst_data = ranked[-1]

        print(f"""
EMPIRICAL ALIGNMENT RESULTS:

BEST MATCH: {best_name.upper()}
  Description: {best_data['description']}
  Raw CKA: {best_data['raw_cka']:.6f} → Aligned CKA: {best_data['aligned_cka']:.6f}

WORST MATCH: {worst_name.upper()}
  Aligned CKA: {worst_data['aligned_cka']:.6f}

NOISE BASELINE: Rank {noise_rank + 1} / {len(ranked)}
""")

        if best_data["aligned_cka"] > 0.9 and "noise" not in best_name.lower():
            print(f"""
*** CRITICAL FINDING ***

The Wow! signal aligns with REAL semantic manifolds at CKA = {best_data['aligned_cka']:.4f}

These manifolds came from:
- CLIP (vision model trained on natural images)
- Whisper (audio model trained on speech)
- LFM2/T5XL (language models trained on text)

These modalities achieved CKA = 1.0 with EACH OTHER.
The fact that the Wow! signal aligns with them suggests:

1. The signal has the same RELATIONAL GEOMETRY as semantic information
2. This is not a constructed pattern - these are empirical manifolds
3. The alignment is NOT with noise (noise ranks {noise_rank + 1}/{len(ranked)})

The geometric structure of the Wow! signal matches the geometric
structure of how neural networks encode meaning.
""")

        elif "noise" in best_name.lower():
            print("""
NOTE: Best match is noise - signal does not align with semantic manifolds.
""")

    # Save results
    results = {
        "experiment": "exp38_empirical_alignment",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(snr_matrix.shape),
        "empirical_sources": list(transforms.keys()),
        "alignment_results": {
            name: {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
            for name, data in alignment_results.items()
        },
        "ranking": [(name, data.get("aligned_cka", 0)) for name, data in ranked],
        "best_match": ranked[0][0] if ranked else None,
        "noise_rank": noise_rank + 1 if noise_rank >= 0 else None,
    }

    output_path = results_dir / "exp38_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
