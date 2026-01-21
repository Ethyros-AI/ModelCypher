#!/usr/bin/env python3
"""Experiment 31: Where Does the Wow! Signal Fit in Semantic Space?

We have a 60-dimensional encoding of the Wow! signal from the high-res scan.
The theory says: if information has invariant geometric structure, different
encodings of the same information should align via GramAligner.

This experiment:
1. Load the Wow! signal encoding
2. Generate reference embeddings from known semantic systems
3. Use CKA to measure geometric similarity
4. Use GramAligner to find the optimal rotation
5. See WHERE in semantic space the signal maps

The key question: Does the Wow! signal's geometry match ANY known information
encoding? If CKA approaches 1.0 after alignment, the structures are identical.

Usage:
    poetry run python experiments/astronomy/exp31_semantic_embedding.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import readsav
from scipy.linalg import svd, lstsq

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Centered Kernel Alignment between two representations.

    CKA measures geometric similarity independent of dimension.
    CKA = 1.0 means identical relational structure.
    """
    # Center the matrices
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)

    # Gram matrices (sample space)
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # HSIC normalization
    hsic_xy = np.trace(K_X @ K_Y)
    hsic_xx = np.trace(K_X @ K_X)
    hsic_yy = np.trace(K_Y @ K_Y)

    if hsic_xx > 0 and hsic_yy > 0:
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    else:
        cka = 0.0

    return float(cka)


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple:
    """Find optimal rotation from source to target space.

    Returns the alignment matrix F such that source @ F ≈ target.
    """
    # Normalize
    source_norm = source / (np.linalg.norm(source, axis=1, keepdims=True) + 1e-10)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-10)

    # Closed-form: F = pinv(source) @ target
    F, residuals, rank, s = lstsq(source_norm, target_norm, lapack_driver='gelsd')

    # Compute aligned CKA
    aligned = source_norm @ F
    cka_after = compute_cka(aligned, target_norm)

    return F, cka_after


def generate_semantic_references(n_samples: int, seed: int = 42) -> dict:
    """Generate reference embeddings that simulate different semantic structures.

    Since we don't have actual CLIP/LLM embeddings loaded, we'll create
    synthetic embeddings that match known properties of semantic spaces:

    1. Language-like: Low-D manifold (ID ~10-15) with clustering
    2. Vision-like: Medium-D manifold (ID ~20-30) with spatial structure
    3. Audio-like: Temporal structure with harmonic patterns
    4. Random: High-D noise (baseline)
    5. Structured noise: Correlated noise (control)
    """
    np.random.seed(seed)

    references = {}

    # 1. Language-like embedding (simulates LLM hidden states)
    # Low intrinsic dimension, semantic clusters
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, 64) * 2
    cluster_labels = np.random.randint(0, n_clusters, n_samples)
    language_embed = cluster_centers[cluster_labels] + np.random.randn(n_samples, 64) * 0.3
    references["language"] = {
        "embedding": language_embed,
        "description": "Language-like: clustered, low-ID (~10)",
        "expected_id": 10,
    }

    # 2. Vision-like embedding (simulates CLIP visual features)
    # Spatial grid structure
    grid_size = int(np.sqrt(n_samples))
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    spatial = np.stack([xx.ravel(), yy.ravel()], axis=1)[:n_samples]
    # Expand to higher dim with smooth functions
    vision_features = []
    for k in range(32):
        freq = (k + 1) * 0.5
        vision_features.append(np.sin(spatial[:, 0] * freq) * np.cos(spatial[:, 1] * freq))
    vision_embed = np.stack(vision_features, axis=1) + np.random.randn(n_samples, 32) * 0.1
    references["vision"] = {
        "embedding": vision_embed,
        "description": "Vision-like: spatial grid, medium-ID (~20)",
        "expected_id": 20,
    }

    # 3. Audio-like embedding (simulates Whisper features)
    # Temporal + harmonic structure
    t = np.linspace(0, 4*np.pi, n_samples)
    audio_features = []
    for harmonic in range(1, 33):
        freq = harmonic * 0.7
        phase = np.random.rand() * 2 * np.pi
        audio_features.append(np.sin(t * freq + phase) * (1/harmonic))
    audio_embed = np.stack(audio_features, axis=1) + np.random.randn(n_samples, 32) * 0.05
    references["audio"] = {
        "embedding": audio_embed,
        "description": "Audio-like: temporal harmonics, structured",
        "expected_id": 15,
    }

    # 4. Mathematical patterns (primes, constants)
    # Encodes mathematical structure
    math_embed = np.zeros((n_samples, 32))
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    for i in range(n_samples):
        for j, p in enumerate(primes[:16]):
            math_embed[i, j] = np.sin(i * np.pi / p)
            math_embed[i, j + 16] = np.cos(i * np.pi / p)
    references["mathematical"] = {
        "embedding": math_embed,
        "description": "Mathematical: prime-based periodicity",
        "expected_id": 8,
    }

    # 5. Random noise (baseline)
    random_embed = np.random.randn(n_samples, 64)
    references["random"] = {
        "embedding": random_embed,
        "description": "Random: iid Gaussian noise",
        "expected_id": 64,
    }

    # 6. Information-bearing (compressed structure)
    # Low rank + modulation (like an actual message)
    n_dims = 8
    basis = np.random.randn(n_dims, 48)  # 8D subspace
    coords = np.random.randn(n_samples, n_dims) * np.array([10, 5, 3, 2, 1, 0.5, 0.3, 0.2])
    info_embed = coords @ basis + np.random.randn(n_samples, 48) * 0.1
    references["information"] = {
        "embedding": info_embed,
        "description": "Information: low-rank subspace (~8D)",
        "expected_id": 8,
    }

    return references


def load_wow_encoding() -> np.ndarray:
    """Load the Wow! signal encoding from exp30."""
    encoding_path = Path(__file__).parent / "results" / "wow_encoding.npy"

    if not encoding_path.exists():
        print(f"Encoding not found at {encoding_path}")
        print("Run exp30_image_vectorization.py first.")
        return None

    encoding = np.load(encoding_path)
    return encoding


def create_wow_sample_matrix(encoding: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Create a sample matrix from the Wow! encoding for CKA comparison.

    The encoding is a single 60-D vector. To compute CKA, we need multiple
    samples. We'll create variations by:
    1. Adding small noise (local neighborhood)
    2. Using different projections
    3. Scaling along different dimensions
    """
    np.random.seed(42)
    d = len(encoding)

    # Method: Create samples as noisy variations of the encoding
    # This represents "similar signals" in the same geometric neighborhood
    samples = np.zeros((n_samples, d))

    for i in range(n_samples):
        # Base encoding
        sample = encoding.copy()

        # Add structured variation (not just noise)
        # Scale different components
        scale = 0.5 + np.random.rand() * 1.0
        sample = sample * scale

        # Add small noise
        noise_level = 0.1 * np.std(encoding)
        sample = sample + np.random.randn(d) * noise_level

        # Small rotation in subspace
        if i > 0:
            angle = np.random.randn() * 0.1
            j, k = np.random.choice(d, 2, replace=False)
            c, s = np.cos(angle), np.sin(angle)
            sample[j], sample[k] = c*sample[j] - s*sample[k], s*sample[j] + c*sample[k]

        samples[i] = sample

    return samples


def analyze_alignment(wow_samples: np.ndarray, references: dict) -> dict:
    """Analyze alignment between Wow! signal and each reference."""
    results = {}

    # Need to match sample counts for CKA
    n_wow = wow_samples.shape[0]

    for name, ref_data in references.items():
        ref_embed = ref_data["embedding"]
        n_ref = ref_embed.shape[0]

        # Use min samples
        n = min(n_wow, n_ref)
        wow_sub = wow_samples[:n]
        ref_sub = ref_embed[:n]

        # Raw CKA (before alignment)
        raw_cka = compute_cka(wow_sub, ref_sub)

        # Aligned CKA (after Procrustes)
        F, aligned_cka = procrustes_align(wow_sub, ref_sub)

        # Improvement
        improvement = aligned_cka - raw_cka

        results[name] = {
            "description": ref_data["description"],
            "raw_cka": float(raw_cka),
            "aligned_cka": float(aligned_cka),
            "improvement": float(improvement),
            "reference_dim": int(ref_embed.shape[1]),
            "n_samples_compared": int(n),
        }

        print(f"\n  {name}:")
        print(f"    Raw CKA: {raw_cka:.4f}")
        print(f"    Aligned CKA: {aligned_cka:.4f}")
        print(f"    Improvement: {improvement:+.4f}")

    return results


def run_experiment():
    """Run the semantic embedding experiment."""
    results_dir = Path(__file__).parent / "results"

    print("=" * 60)
    print("Experiment 31: Semantic Space Embedding")
    print("=" * 60)
    print("\nQuestion: Where does the Wow! signal fit in semantic space?")
    print("Method: CKA alignment to reference embeddings")

    # Load Wow! encoding
    print("\n" + "=" * 40)
    print("PART 1: LOAD WOW! ENCODING")
    print("=" * 40)

    encoding = load_wow_encoding()
    if encoding is None:
        return None

    print(f"\nWow! encoding loaded:")
    print(f"  Shape: {encoding.shape}")
    print(f"  Range: [{encoding.min():.3f}, {encoding.max():.3f}]")

    # Create sample matrix for CKA
    n_samples = 100
    wow_samples = create_wow_sample_matrix(encoding, n_samples)
    print(f"  Sample matrix: {wow_samples.shape}")

    # Generate reference embeddings
    print("\n" + "=" * 40)
    print("PART 2: GENERATE REFERENCE EMBEDDINGS")
    print("=" * 40)

    references = generate_semantic_references(n_samples)

    for name, ref_data in references.items():
        print(f"\n  {name}: {ref_data['embedding'].shape}")
        print(f"    {ref_data['description']}")

    # Analyze alignment
    print("\n" + "=" * 40)
    print("PART 3: CKA ALIGNMENT ANALYSIS")
    print("=" * 40)

    alignment_results = analyze_alignment(wow_samples, references)

    # Find best matches
    print("\n" + "=" * 40)
    print("PART 4: RANKING")
    print("=" * 40)

    # Sort by aligned CKA
    ranked = sorted(alignment_results.items(), key=lambda x: x[1]["aligned_cka"], reverse=True)

    print("\nReference systems ranked by aligned CKA (best match first):\n")
    for i, (name, data) in enumerate(ranked):
        print(f"  {i+1}. {name}: CKA = {data['aligned_cka']:.4f}")
        print(f"     ({data['description']})")

    best_match = ranked[0]
    worst_match = ranked[-1]

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print(f"""
THE WOW! SIGNAL'S SEMANTIC NEIGHBORHOOD:

BEST MATCH: {best_match[0]} (CKA = {best_match[1]['aligned_cka']:.4f})
  {best_match[1]['description']}

WORST MATCH: {worst_match[0]} (CKA = {worst_match[1]['aligned_cka']:.4f})
  {worst_match[1]['description']}

WHAT THIS MEANS:

The Wow! signal's geometric structure most closely resembles:
→ {best_match[0].upper()}

Key observations:
""")

    # Analysis based on results
    info_cka = alignment_results.get("information", {}).get("aligned_cka", 0)
    math_cka = alignment_results.get("mathematical", {}).get("aligned_cka", 0)
    random_cka = alignment_results.get("random", {}).get("aligned_cka", 0)

    if info_cka > random_cka + 0.1:
        print("  1. ✓ Aligns BETTER with information-bearing structures than random noise")
        print(f"     (Information CKA={info_cka:.3f} vs Random CKA={random_cka:.3f})")
    else:
        print("  1. ✗ Does NOT align distinctly better with information than noise")

    if math_cka > random_cka + 0.1:
        print("  2. ✓ Shows affinity for MATHEMATICAL patterns")
        print(f"     (Mathematical CKA={math_cka:.3f})")
    else:
        print("  2. ○ No strong mathematical pattern affinity detected")

    audio_cka = alignment_results.get("audio", {}).get("aligned_cka", 0)
    if audio_cka > 0.5:
        print(f"  3. ✓ Strong alignment with TEMPORAL/HARMONIC structure")
        print(f"     (Audio CKA={audio_cka:.3f})")

    # The key finding
    print(f"""
THE GEOMETRIC HYPOTHESIS:

If the Wow! signal encodes information, its geometry should match
other information-encoding systems better than random noise.

RESULT: Information CKA = {info_cka:.4f}
        Random CKA = {random_cka:.4f}
        Difference = {info_cka - random_cka:+.4f}
""")

    if info_cka > random_cka + 0.15:
        print("""
CONCLUSION: The Wow! signal's geometry is CONSISTENT with
information-bearing structure. Its relational geometry aligns
with compressed, low-rank encodings - the hallmark of messages.
""")
    elif info_cka > random_cka + 0.05:
        print("""
CONCLUSION: The Wow! signal shows MARGINAL alignment with
information structure. The geometry is somewhat organized
but not conclusively different from noise.
""")
    else:
        print("""
CONCLUSION: The Wow! signal's geometry does NOT align well
with information-bearing systems. The structure may be natural
rather than encoded.
""")

    # Save results
    results = {
        "experiment": "exp31_semantic_embedding",
        "timestamp": datetime.now().isoformat(),
        "wow_encoding": {
            "shape": list(encoding.shape),
            "n_samples": n_samples,
        },
        "alignment_results": alignment_results,
        "ranking": [(name, data["aligned_cka"]) for name, data in ranked],
        "best_match": {
            "name": best_match[0],
            "cka": float(best_match[1]["aligned_cka"]),
        },
        "key_comparisons": {
            "information_cka": float(info_cka),
            "random_cka": float(random_cka),
            "mathematical_cka": float(math_cka),
            "difference_info_vs_random": float(info_cka - random_cka),
        },
    }

    output_path = results_dir / "exp31_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
