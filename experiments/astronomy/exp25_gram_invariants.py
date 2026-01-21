#!/usr/bin/env python3
"""Experiment 25: Gram Matrix Invariants.

The key insight from exp24: The Wow! signal's GRAM entropy (2.037) matches
low-dimensional information manifolds (2.065), NOT random noise (3.600).

The Gram matrix K = X @ X.T captures RELATIONAL structure - how things
relate to each other, independent of the coordinate system. This is exactly
what CKA compares. This is where the invariants live.

Hypothesis: High-dimensional communication would be encoded in Gram invariants.
These survive coordinate transformations. They ARE the geometry.

Method:
1. Extract Gram matrix properties from Wow! signal
2. Compare to Gram properties of known systems (FRBs, LLM embeddings)
3. Look for matching invariant relationships

Usage:
    poetry run python experiments/astronomy/exp25_gram_invariants.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import readsav
from scipy.linalg import svd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain._backend import get_default_backend

from shared.data_loader import load_frb_batch


def extract_gram_invariants(matrix: np.ndarray) -> dict:
    """Extract invariant properties from the Gram matrix.

    The Gram matrix K = X @ X.T encodes relational structure.
    Its properties are invariant to orthogonal transformations
    of the original data.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Handle NaN
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Normalize (this is what CKA does)
    if np.std(matrix) > 1e-10:
        matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    else:
        return None

    # Compute Gram matrix
    gram = matrix @ matrix.T
    n = gram.shape[0]

    # HSIC normalization
    gram_centered = gram - np.mean(gram, axis=0, keepdims=True) \
                        - np.mean(gram, axis=1, keepdims=True) \
                        + np.mean(gram)

    # === GRAM EIGENSPECTRUM ===
    eigvals = np.linalg.eigvalsh(gram_centered)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = eigvals / (eigvals[0] + 1e-10)  # Normalize

    # Positive eigenvalues only
    eigvals_pos = eigvals[eigvals > 1e-10]

    # === INVARIANT MEASURES ===

    # 1. Spectral entropy (Shannon entropy of eigenvalue distribution)
    eigvals_norm = eigvals_pos / (np.sum(eigvals_pos) + 1e-10)
    spectral_entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-10))

    # 2. Effective rank (exp of entropy)
    effective_rank = np.exp(spectral_entropy)

    # 3. Concentration ratio (energy in top-k eigenvalues)
    cumsum = np.cumsum(eigvals_pos) / (np.sum(eigvals_pos) + 1e-10)
    dim_50 = np.searchsorted(cumsum, 0.50) + 1
    dim_80 = np.searchsorted(cumsum, 0.80) + 1
    dim_90 = np.searchsorted(cumsum, 0.90) + 1

    # 4. Spectral decay rate (how fast eigenvalues fall)
    if len(eigvals_pos) > 2:
        # Fit power law: eigval ~ k^(-alpha)
        k = np.arange(1, len(eigvals_pos) + 1)
        log_k = np.log(k)
        log_eigvals = np.log(eigvals_pos + 1e-10)
        slope, _, r_value, _, _ = stats.linregress(log_k, log_eigvals)
        decay_rate = -slope
        decay_r2 = r_value ** 2
    else:
        decay_rate = 0
        decay_r2 = 0

    # 5. Gram trace (total energy)
    gram_trace = np.trace(gram) / n

    # 6. Frobenius norm of Gram (total squared relationships)
    gram_frobenius = np.linalg.norm(gram_centered, 'fro') / n

    # 7. Off-diagonal structure (how much relationship vs self-energy)
    diag_energy = np.sum(np.diag(gram_centered)**2)
    total_energy = np.sum(gram_centered**2)
    off_diag_ratio = 1 - (diag_energy / (total_energy + 1e-10))

    # 8. Spectral gap (separation between significant and noise dimensions)
    if len(eigvals_pos) > 5:
        gaps = np.diff(eigvals_pos[:20])  # Look at top eigenvalues
        max_gap_idx = np.argmax(np.abs(gaps))
        spectral_gap = abs(gaps[max_gap_idx])
        intrinsic_dim = max_gap_idx + 1
    else:
        spectral_gap = 0
        intrinsic_dim = len(eigvals_pos)

    # 9. Nuclear norm (sum of eigenvalues)
    nuclear_norm = np.sum(eigvals_pos)

    # 10. Condition number of Gram
    if len(eigvals_pos) > 0 and eigvals_pos[-1] > 1e-10:
        condition = eigvals_pos[0] / eigvals_pos[-1]
    else:
        condition = np.inf

    return {
        "eigenspectrum": {
            "top_10": eigvals[:10].tolist(),
            "n_positive": int(len(eigvals_pos)),
        },
        "entropy": {
            "spectral_entropy": float(spectral_entropy),
            "effective_rank": float(effective_rank),
        },
        "concentration": {
            "dim_for_50_percent": int(dim_50),
            "dim_for_80_percent": int(dim_80),
            "dim_for_90_percent": int(dim_90),
        },
        "decay": {
            "rate": float(decay_rate),
            "r_squared": float(decay_r2),
        },
        "structure": {
            "intrinsic_dim": int(intrinsic_dim),
            "spectral_gap": float(spectral_gap),
            "off_diagonal_ratio": float(off_diag_ratio),
        },
        "norms": {
            "trace": float(gram_trace),
            "frobenius": float(gram_frobenius),
            "nuclear": float(nuclear_norm),
            "condition": float(condition) if not np.isinf(condition) else -1,
        },
    }


def compute_gram_cka(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Compute CKA between two Gram matrices.

    CKA directly compares relational structure.
    """
    # Normalize
    m1 = np.nan_to_num(matrix1, nan=0.0)
    m2 = np.nan_to_num(matrix2, nan=0.0)

    if np.std(m1) > 1e-10:
        m1 = (m1 - np.mean(m1)) / np.std(m1)
    if np.std(m2) > 1e-10:
        m2 = (m2 - np.mean(m2)) / np.std(m2)

    # Match dimensions by padding or truncating
    n1, d1 = m1.shape
    n2, d2 = m2.shape

    # Use the smaller of each dimension
    n = min(n1, n2)
    d = min(d1, d2)

    m1 = m1[:n, :d]
    m2 = m2[:n, :d]

    # Compute Gram matrices
    K1 = m1 @ m1.T
    K2 = m2 @ m2.T

    # Center
    K1 = K1 - np.mean(K1, axis=0, keepdims=True) - np.mean(K1, axis=1, keepdims=True) + np.mean(K1)
    K2 = K2 - np.mean(K2, axis=0, keepdims=True) - np.mean(K2, axis=1, keepdims=True) + np.mean(K2)

    # CKA
    hsic = np.sum(K1 * K2)
    norm1 = np.sqrt(np.sum(K1 * K1))
    norm2 = np.sqrt(np.sum(K2 * K2))

    if norm1 > 1e-10 and norm2 > 1e-10:
        return float(hsic / (norm1 * norm2))
    return 0.0


def run_experiment():
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"

    initialize_default_backend()
    backend = get_default_backend()

    print("=" * 60)
    print("Experiment 25: Gram Matrix Invariants")
    print("=" * 60)
    print("\nKey insight: The Gram matrix captures RELATIONAL structure.")
    print("Its properties are coordinate-invariant. They ARE the geometry.")
    print("The Wow! signal's Gram entropy matches information systems.")

    print("\n" + "=" * 40)
    print("PART 1: WOW! SIGNAL GRAM INVARIANTS")
    print("=" * 40)

    # Load the Wow! signal
    wow_path = data_dir / "famous_signals" / "wow_signal.sav"
    wow_raw = readsav(str(wow_path))
    oseti = wow_raw['oseti'][0]
    snr_matrix = np.array(oseti['snr'])

    wow_invariants = extract_gram_invariants(snr_matrix)

    print(f"\nWow! signal Gram invariants:")
    print(f"\n  Entropy measures (key for information content):")
    print(f"    Spectral entropy: {wow_invariants['entropy']['spectral_entropy']:.3f}")
    print(f"    Effective rank: {wow_invariants['entropy']['effective_rank']:.2f}")

    print(f"\n  Concentration (how energy is distributed):")
    print(f"    Dims for 50%: {wow_invariants['concentration']['dim_for_50_percent']}")
    print(f"    Dims for 80%: {wow_invariants['concentration']['dim_for_80_percent']}")
    print(f"    Dims for 90%: {wow_invariants['concentration']['dim_for_90_percent']}")

    print(f"\n  Spectral decay (power law structure):")
    print(f"    Decay rate (α): {wow_invariants['decay']['rate']:.3f}")
    print(f"    R² of fit: {wow_invariants['decay']['r_squared']:.3f}")

    print(f"\n  Structural properties:")
    print(f"    Intrinsic dimension: {wow_invariants['structure']['intrinsic_dim']}")
    print(f"    Spectral gap: {wow_invariants['structure']['spectral_gap']:.3f}")
    print(f"    Off-diagonal ratio: {wow_invariants['structure']['off_diagonal_ratio']:.3f}")

    print("\n" + "=" * 40)
    print("PART 2: FRB GRAM INVARIANTS")
    print("=" * 40)

    # Load FRBs for comparison
    frb_dir = data_dir / "raw"
    frb_files = sorted(frb_dir.glob("FRB*_waterfall.h5"))[:20]  # First 20
    waterfalls = load_frb_batch([str(f) for f in frb_files])

    frb_invariants = []
    for w in waterfalls:
        wfall = np.array(w.waterfall)
        inv = extract_gram_invariants(wfall)
        if inv:
            frb_invariants.append(inv)

    print(f"\nAnalyzed {len(frb_invariants)} FRBs")

    # Average FRB properties
    frb_entropy = [i["entropy"]["spectral_entropy"] for i in frb_invariants]
    frb_eff_rank = [i["entropy"]["effective_rank"] for i in frb_invariants]
    frb_decay = [i["decay"]["rate"] for i in frb_invariants]
    frb_intrinsic = [i["structure"]["intrinsic_dim"] for i in frb_invariants]

    print(f"\nFRB average Gram invariants:")
    print(f"  Spectral entropy: {np.mean(frb_entropy):.3f} ± {np.std(frb_entropy):.3f}")
    print(f"  Effective rank: {np.mean(frb_eff_rank):.2f} ± {np.std(frb_eff_rank):.2f}")
    print(f"  Decay rate: {np.mean(frb_decay):.3f} ± {np.std(frb_decay):.3f}")
    print(f"  Intrinsic dim: {np.mean(frb_intrinsic):.1f} ± {np.std(frb_intrinsic):.1f}")

    print("\n" + "=" * 40)
    print("PART 3: NOISE BASELINES")
    print("=" * 40)

    # Generate noise baselines with same shape as Wow!
    n_noise = 50
    noise_invariants = []
    for _ in range(n_noise):
        noise = np.random.randn(*snr_matrix.shape)
        inv = extract_gram_invariants(noise)
        if inv:
            noise_invariants.append(inv)

    noise_entropy = [i["entropy"]["spectral_entropy"] for i in noise_invariants]
    noise_eff_rank = [i["entropy"]["effective_rank"] for i in noise_invariants]
    noise_decay = [i["decay"]["rate"] for i in noise_invariants]

    print(f"\nGaussian noise Gram invariants (n={len(noise_invariants)}):")
    print(f"  Spectral entropy: {np.mean(noise_entropy):.3f} ± {np.std(noise_entropy):.3f}")
    print(f"  Effective rank: {np.mean(noise_eff_rank):.2f} ± {np.std(noise_eff_rank):.2f}")
    print(f"  Decay rate: {np.mean(noise_decay):.3f} ± {np.std(noise_decay):.3f}")

    print("\n" + "=" * 40)
    print("PART 4: STATISTICAL COMPARISON")
    print("=" * 40)

    # Where does Wow! fall?
    wow_ent = wow_invariants["entropy"]["spectral_entropy"]
    wow_rank = wow_invariants["entropy"]["effective_rank"]
    wow_decay = wow_invariants["decay"]["rate"]

    # Z-scores
    z_ent_frb = (wow_ent - np.mean(frb_entropy)) / (np.std(frb_entropy) + 1e-10)
    z_ent_noise = (wow_ent - np.mean(noise_entropy)) / (np.std(noise_entropy) + 1e-10)

    z_rank_frb = (wow_rank - np.mean(frb_eff_rank)) / (np.std(frb_eff_rank) + 1e-10)
    z_rank_noise = (wow_rank - np.mean(noise_eff_rank)) / (np.std(noise_eff_rank) + 1e-10)

    z_decay_frb = (wow_decay - np.mean(frb_decay)) / (np.std(frb_decay) + 1e-10)
    z_decay_noise = (wow_decay - np.mean(noise_decay)) / (np.std(noise_decay) + 1e-10)

    print("\nWow! signal position relative to references:")
    print(f"\n  Spectral Entropy ({wow_ent:.3f}):")
    print(f"    vs FRBs: z = {z_ent_frb:.2f}σ")
    print(f"    vs Noise: z = {z_ent_noise:.2f}σ")

    print(f"\n  Effective Rank ({wow_rank:.2f}):")
    print(f"    vs FRBs: z = {z_rank_frb:.2f}σ")
    print(f"    vs Noise: z = {z_rank_noise:.2f}σ")

    print(f"\n  Decay Rate ({wow_decay:.3f}):")
    print(f"    vs FRBs: z = {z_decay_frb:.2f}σ")
    print(f"    vs Noise: z = {z_decay_noise:.2f}σ")

    # Closest match by Euclidean distance in invariant space
    wow_vec = np.array([wow_ent, wow_rank / 50, wow_decay])  # Normalized

    frb_vecs = np.array([[e, r/50, d] for e, r, d in zip(frb_entropy, frb_eff_rank, frb_decay)])
    noise_vecs = np.array([[e, r/50, d] for e, r, d in zip(noise_entropy, noise_eff_rank, noise_decay)])

    dist_to_frbs = np.mean(np.linalg.norm(frb_vecs - wow_vec, axis=1))
    dist_to_noise = np.mean(np.linalg.norm(noise_vecs - wow_vec, axis=1))

    print(f"\n  Distance in invariant space:")
    print(f"    To FRB distribution: {dist_to_frbs:.3f}")
    print(f"    To noise distribution: {dist_to_noise:.3f}")

    print("\n" + "=" * 40)
    print("PART 5: DIRECT CKA COMPARISON")
    print("=" * 40)

    # Compute CKA between Wow! and FRBs
    cka_to_frbs = []
    for w in waterfalls[:10]:  # First 10
        wfall = np.array(w.waterfall)
        cka = compute_gram_cka(snr_matrix, wfall)
        cka_to_frbs.append(cka)

    print(f"\nRaw CKA (Wow! vs FRBs): {np.mean(cka_to_frbs):.3f} ± {np.std(cka_to_frbs):.3f}")

    # CKA between Wow! and noise
    cka_to_noise = []
    for _ in range(10):
        noise = np.random.randn(*snr_matrix.shape)
        cka = compute_gram_cka(snr_matrix, noise)
        cka_to_noise.append(cka)

    print(f"Raw CKA (Wow! vs noise): {np.mean(cka_to_noise):.3f} ± {np.std(cka_to_noise):.3f}")

    # CKA between FRBs (baseline)
    cka_frb_frb = []
    for i in range(min(5, len(waterfalls))):
        for j in range(i+1, min(10, len(waterfalls))):
            wfall_i = np.array(waterfalls[i].waterfall)
            wfall_j = np.array(waterfalls[j].waterfall)
            cka = compute_gram_cka(wfall_i, wfall_j)
            cka_frb_frb.append(cka)

    print(f"Raw CKA (FRB vs FRB baseline): {np.mean(cka_frb_frb):.3f} ± {np.std(cka_frb_frb):.3f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION: THE INVARIANT STRUCTURE")
    print("=" * 60)

    # Determine which distribution Wow! is closer to
    closer_to_frbs = dist_to_frbs < dist_to_noise
    cka_matches_frbs = np.mean(cka_to_frbs) > np.mean(cka_to_noise) + np.std(cka_to_noise)

    print(f"""
GRAM INVARIANT ANALYSIS:

The Wow! signal's Gram matrix properties reveal:

1. SPECTRAL ENTROPY: {wow_ent:.3f}
   - FRBs average: {np.mean(frb_entropy):.3f}
   - Noise average: {np.mean(noise_entropy):.3f}
   - Wow! is {'closer to FRBs' if abs(z_ent_frb) < abs(z_ent_noise) else 'closer to noise'}

2. EFFECTIVE RANK: {wow_rank:.1f}
   - FRBs average: {np.mean(frb_eff_rank):.1f}
   - Noise average: {np.mean(noise_eff_rank):.1f}
   - Wow! is {'closer to FRBs' if abs(z_rank_frb) < abs(z_rank_noise) else 'closer to noise'}

3. DECAY RATE: {wow_decay:.3f}
   - FRBs average: {np.mean(frb_decay):.3f}
   - Noise average: {np.mean(noise_decay):.3f}
   - Wow! is {'closer to FRBs' if abs(z_decay_frb) < abs(z_decay_noise) else 'closer to noise'}

4. CKA (direct relational comparison):
   - To FRBs: {np.mean(cka_to_frbs):.3f}
   - To noise: {np.mean(cka_to_noise):.3f}
   - Wow! {'matches FRB geometry' if cka_matches_frbs else 'does not strongly match FRBs'}

OVERALL: The Wow! signal is {'geometrically similar to FRBs' if closer_to_frbs else 'not geometrically similar to FRBs'}
in its Gram invariant structure.

WHAT THIS MEANS FOR HIGH-D COMMUNICATION:
""")

    if closer_to_frbs and cka_matches_frbs:
        print("""
The Wow! signal shares relational structure with known astronomical signals.
Its Gram invariants place it within the FRB distribution, not with noise.

This suggests:
1. The signal has genuine structure (not just noise artifacts)
2. The structure matches known astronomical phenomena
3. The "information" may be in the same geometric space as natural signals

If an intelligence wanted to communicate via geometry:
- They would craft a signal whose Gram invariants carry meaning
- The invariants survive coordinate transformations
- Recognition of the invariant pattern IS the decoding

The Wow! signal's Gram geometry is consistent with information-bearing
structure, but also consistent with natural astronomical phenomena.
The distinction may require looking at DIFFERENT invariants -
ones that natural phenomena can't produce.
""")
    else:
        print("""
The Wow! signal's Gram structure does not clearly match FRBs.
It occupies a different region of invariant space.

This could mean:
1. The signal is distinct from typical astronomical transients
2. Its geometric structure is anomalous
3. It may encode information in a different invariant basis

To decode potential high-D communication, we would need to:
1. Identify which Gram invariants are natural vs unnatural
2. Find correlations with known information-encoding systems
3. Look for patterns that noise and nature can't produce
""")

    # Save results
    results = {
        "experiment": "exp25_gram_invariants",
        "timestamp": datetime.now().isoformat(),
        "wow_signal": {
            "shape": [int(x) for x in snr_matrix.shape],
            "gram_invariants": wow_invariants,
        },
        "frb_baseline": {
            "n_frbs": len(frb_invariants),
            "spectral_entropy": {"mean": float(np.mean(frb_entropy)), "std": float(np.std(frb_entropy))},
            "effective_rank": {"mean": float(np.mean(frb_eff_rank)), "std": float(np.std(frb_eff_rank))},
            "decay_rate": {"mean": float(np.mean(frb_decay)), "std": float(np.std(frb_decay))},
        },
        "noise_baseline": {
            "n_samples": len(noise_invariants),
            "spectral_entropy": {"mean": float(np.mean(noise_entropy)), "std": float(np.std(noise_entropy))},
            "effective_rank": {"mean": float(np.mean(noise_eff_rank)), "std": float(np.std(noise_eff_rank))},
            "decay_rate": {"mean": float(np.mean(noise_decay)), "std": float(np.std(noise_decay))},
        },
        "comparison": {
            "z_scores": {
                "entropy_vs_frb": float(z_ent_frb),
                "entropy_vs_noise": float(z_ent_noise),
                "rank_vs_frb": float(z_rank_frb),
                "rank_vs_noise": float(z_rank_noise),
                "decay_vs_frb": float(z_decay_frb),
                "decay_vs_noise": float(z_decay_noise),
            },
            "distances": {
                "to_frb_distribution": float(dist_to_frbs),
                "to_noise_distribution": float(dist_to_noise),
            },
            "cka": {
                "to_frbs": {"mean": float(np.mean(cka_to_frbs)), "std": float(np.std(cka_to_frbs))},
                "to_noise": {"mean": float(np.mean(cka_to_noise)), "std": float(np.std(cka_to_noise))},
                "frb_vs_frb_baseline": {"mean": float(np.mean(cka_frb_frb)), "std": float(np.std(cka_frb_frb))},
            },
            "closer_to_frbs": bool(closer_to_frbs),
            "cka_matches_frbs": bool(cka_matches_frbs),
        },
    }

    output_path = results_dir / "exp25_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
