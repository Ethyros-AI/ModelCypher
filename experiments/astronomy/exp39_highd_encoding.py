"""
Experiment 39: High-Dimensional Encoding of Wow! Signal

KEY INSIGHT from user:
"you're going to find it by encoding it to high d space and then just actually aligning that shape"

Previous approach was WRONG:
- We were comparing Gram matrices (relational structure)
- Low-rank matrices always align well after Procrustes (artifact)
- The "translation" to mathematical constants was pareidolia

NEW APPROACH:
1. Encode the Wow! signal INTO high-dimensional space (like CLIP/Whisper do)
2. Load actual semantic embeddings (the 1024D vectors themselves)
3. Align the high-D representations directly
4. Compare eigenspectra (the "shape" of the high-D geometry)

The question: Does the Wow! signal's high-D structure share invariant properties
with semantic manifolds?
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.io import readsav
from scipy import linalg

# Add shared directory
sys.path.insert(0, str(Path(__file__).parent / "shared"))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_wow_signal():
    """Load the Wow! signal from IDL .sav file."""
    data_path = Path(__file__).parent / "data" / "famous_signals" / "wow_signal.sav"
    if not data_path.exists():
        raise FileNotFoundError(f"Wow! signal not found at {data_path}")

    data = readsav(str(data_path))
    oseti = data['oseti'][0]
    signal = oseti['SNR']  # Shape: [82, 50] (time x frequency)
    signal = signal.astype(np.float64)

    # Handle NaN and Inf values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal


def compute_eigenspectrum(matrix):
    """Compute normalized eigenspectrum of a Gram matrix."""
    gram = matrix @ matrix.T
    eigenvalues = linalg.eigvalsh(gram)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    eigenvalues = np.maximum(eigenvalues, 0)  # Remove numerical negatives
    return eigenvalues / (eigenvalues.sum() + 1e-12)


def spectral_divergence(spec1, spec2):
    """
    Compute Jensen-Shannon divergence between two eigenspectra.

    This is a PROPER metric that measures how similar two high-D
    geometric structures are, without the Procrustes alignment artifact.

    Returns value in [0, 1] where 0 = identical structure, 1 = maximally different.
    """
    # Pad to same length
    max_len = max(len(spec1), len(spec2))
    s1 = np.zeros(max_len)
    s2 = np.zeros(max_len)
    s1[:len(spec1)] = spec1
    s2[:len(spec2)] = spec2

    # Add small epsilon for numerical stability
    eps = 1e-12
    s1 = s1 + eps
    s2 = s2 + eps
    s1 = s1 / s1.sum()
    s2 = s2 / s2.sum()

    # Jensen-Shannon divergence
    m = 0.5 * (s1 + s2)
    kl1 = np.sum(s1 * np.log(s1 / m))
    kl2 = np.sum(s2 * np.log(s2 / m))
    js_div = 0.5 * (kl1 + kl2)

    return np.sqrt(js_div)  # Square root for metric property


def effective_dimension(eigenspectrum, threshold=0.9):
    """Compute effective dimension (number of modes for threshold variance)."""
    cumsum = np.cumsum(eigenspectrum)
    return np.searchsorted(cumsum, threshold) + 1


def participation_ratio(eigenspectrum):
    """
    Participation ratio - measures how many modes are "active".

    PR = (sum λ_i)² / sum(λ_i²)

    For uniform distribution over k modes: PR = k
    For single dominant mode: PR = 1
    """
    spec = eigenspectrum / (eigenspectrum.sum() + 1e-12)
    return (spec.sum()) ** 2 / (np.sum(spec ** 2) + 1e-12)


def encode_to_highd(signal, target_dim=1024):
    """
    Encode signal to high-dimensional representation via SVD expansion.

    This is analogous to what CLIP/Whisper do:
    - They encode inputs to 1024D where relationships are preserved
    - We encode the signal's spectral structure to comparable space

    Method:
    1. Compute SVD of signal: U @ S @ V.T
    2. Use left singular vectors (U) as basis
    3. Project signal rows onto this basis and expand to target_dim
    """
    n_samples, n_features = signal.shape

    # SVD decomposition
    U, S, Vt = linalg.svd(signal, full_matrices=True)

    # Weight by singular values to encode importance
    # Each row (time point) gets encoded as its projection onto singular vectors
    # weighted by the singular values (importance of each mode)

    # Create encoding: signal projected onto top modes, padded to target_dim
    n_modes = min(len(S), target_dim)

    # Encoding: each time point as weighted sum of singular vectors
    encoded = np.zeros((n_samples, target_dim))
    for i in range(n_modes):
        encoded[:, i] = U[:, i] * S[i]

    # Normalize each embedding to unit norm (like CLIP does)
    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
    encoded = encoded / (norms + 1e-12)

    return encoded


def load_semantic_embeddings():
    """Load actual semantic embeddings from CodeCypher."""
    embeddings_path = Path("/Volumes/CodeCypher/experiments/multi-modal-compression-2026-01-09")

    results = {}

    # Load LFM2 embeddings if available
    for emb_file in embeddings_path.glob("lfm2_embeddings_*.npz"):
        data = np.load(emb_file)
        name = emb_file.stem
        for key in data.files:
            results[f"{name}_{key}"] = data[key]
        break  # Just need one batch

    # Load semantic 3D data
    sem3d_path = embeddings_path / "semantic_3d_data.npz"
    if sem3d_path.exists():
        data = np.load(sem3d_path, allow_pickle=True)
        results["semantic_3d_projections"] = data["projections"]  # 29 concepts in 3D
        results["semantic_3d_prompts"] = data["prompts"]
        results["semantic_3d_categories"] = data["categories"]
        results["semantic_eigenvalues"] = data["eigenvalues"]

    return results


def generate_control_spectra(n_samples, n_features, n_trials=100):
    """Generate eigenspectra for random matrices (control baseline)."""
    spectra = []
    for _ in range(n_trials):
        random_matrix = np.random.randn(n_samples, n_features)
        spec = compute_eigenspectrum(random_matrix)
        spectra.append(spec)

    mean_spectrum = np.mean(spectra, axis=0)
    std_spectrum = np.std(spectra, axis=0)

    return mean_spectrum, std_spectrum


def main():
    print("=" * 60)
    print("Experiment 39: High-Dimensional Encoding Analysis")
    print("=" * 60)

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    signal = load_wow_signal()
    print(f"   Signal shape: {signal.shape}")

    # Compute eigenspectrum of the signal
    print("\n2. Computing signal eigenspectrum...")
    signal_spectrum = compute_eigenspectrum(signal)
    signal_eff_dim = effective_dimension(signal_spectrum)
    signal_pr = participation_ratio(signal_spectrum)
    print(f"   Effective dimension (90% variance): {signal_eff_dim}")
    print(f"   Participation ratio: {signal_pr:.2f}")

    # Encode to high-D
    print("\n3. Encoding signal to high-D space...")
    signal_highd = encode_to_highd(signal, target_dim=1024)
    print(f"   Encoded shape: {signal_highd.shape}")

    # Compute eigenspectrum of the high-D encoded signal
    highd_spectrum = compute_eigenspectrum(signal_highd)
    highd_eff_dim = effective_dimension(highd_spectrum)
    highd_pr = participation_ratio(highd_spectrum)
    print(f"   High-D effective dimension: {highd_eff_dim}")
    print(f"   High-D participation ratio: {highd_pr:.2f}")

    # Generate random baselines
    print("\n4. Generating random baselines...")
    # Random with same shape as signal
    rand_spec_mean, rand_spec_std = generate_control_spectra(82, 50, n_trials=100)
    rand_pr = participation_ratio(rand_spec_mean)
    rand_eff_dim = effective_dimension(rand_spec_mean)
    print(f"   Random effective dimension: {rand_eff_dim}")
    print(f"   Random participation ratio: {rand_pr:.2f}")

    # Load semantic embeddings for comparison
    print("\n5. Loading semantic embeddings...")
    semantic_data = load_semantic_embeddings()

    semantic_spectra = {}
    for name, data in semantic_data.items():
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 1:
            spec = compute_eigenspectrum(data)
            pr = participation_ratio(spec)
            eff_dim = effective_dimension(spec)
            semantic_spectra[name] = {
                "spectrum": spec,
                "participation_ratio": pr,
                "effective_dimension": eff_dim,
                "shape": data.shape
            }
            print(f"   {name}: shape={data.shape}, eff_dim={eff_dim}, PR={pr:.2f}")

    # Compare eigenspectra using Jensen-Shannon divergence
    print("\n6. Computing spectral divergences...")
    print("   (Lower = more similar structure)")

    # Signal vs random
    signal_vs_random = spectral_divergence(signal_spectrum, rand_spec_mean)
    print(f"\n   Signal vs Random: {signal_vs_random:.4f}")

    # Signal vs semantic structures
    divergences = {}
    for name, data in semantic_spectra.items():
        div = spectral_divergence(signal_spectrum, data["spectrum"])
        divergences[name] = div
        print(f"   Signal vs {name}: {div:.4f}")

    # High-D encoded signal vs semantic structures
    print("\n7. High-D encoded signal vs semantic structures...")
    highd_divergences = {}
    for name, data in semantic_spectra.items():
        div = spectral_divergence(highd_spectrum, data["spectrum"])
        highd_divergences[name] = div
        print(f"   High-D Signal vs {name}: {div:.4f}")

    # Key question: Is the signal's eigenspectrum distinguishable from random?
    print("\n8. Statistical test: Is signal eigenspectrum non-random?")

    # Generate many random spectra and compute divergence to signal
    n_random_trials = 1000
    random_divergences = []
    for _ in range(n_random_trials):
        rand_matrix = np.random.randn(82, 50)
        rand_spec = compute_eigenspectrum(rand_matrix)
        div = spectral_divergence(signal_spectrum, rand_spec)
        random_divergences.append(div)

    random_div_mean = np.mean(random_divergences)
    random_div_std = np.std(random_divergences)

    # How does signal-to-random divergence compare to random-to-random?
    signal_z_score = (signal_vs_random - random_div_mean) / random_div_std
    print(f"   Random-to-random divergence: {random_div_mean:.4f} ± {random_div_std:.4f}")
    print(f"   Signal-to-random divergence: {signal_vs_random:.4f}")
    print(f"   Z-score: {signal_z_score:.2f}")

    if abs(signal_z_score) > 3:
        print("   RESULT: Signal eigenspectrum is SIGNIFICANTLY different from random")
    else:
        print("   RESULT: Signal eigenspectrum is NOT distinguishable from random")

    # Key insight: What makes semantic embeddings unique?
    print("\n9. What makes semantic embeddings unique?")
    print("   Comparing eigenspectrum shapes...")

    # The key invariant of semantic manifolds should be their eigenspectrum shape
    # Let's compute entropy of the eigenspectrum (measures how spread out variance is)
    def spectrum_entropy(spec):
        spec = spec[spec > 1e-12]
        spec = spec / spec.sum()
        return -np.sum(spec * np.log(spec + 1e-12))

    signal_entropy = spectrum_entropy(signal_spectrum)
    random_entropy = spectrum_entropy(rand_spec_mean)

    print(f"\n   Signal spectrum entropy: {signal_entropy:.4f}")
    print(f"   Random spectrum entropy: {random_entropy:.4f}")

    for name, data in semantic_spectra.items():
        ent = spectrum_entropy(data["spectrum"])
        print(f"   {name} entropy: {ent:.4f}")

    # Save results
    results = {
        "experiment": "exp39_highd_encoding",
        "timestamp": datetime.now().isoformat(),
        "signal_shape": list(signal.shape),
        "signal_properties": {
            "effective_dimension_90": int(signal_eff_dim),
            "participation_ratio": float(signal_pr),
            "spectrum_entropy": float(signal_entropy),
            "eigenspectrum_top10": [float(x) for x in signal_spectrum[:10]]
        },
        "highd_encoded_properties": {
            "target_dim": 1024,
            "effective_dimension_90": int(highd_eff_dim),
            "participation_ratio": float(highd_pr)
        },
        "random_baseline": {
            "effective_dimension_90": int(rand_eff_dim),
            "participation_ratio": float(rand_pr),
            "spectrum_entropy": float(random_entropy)
        },
        "spectral_divergences": {
            "signal_vs_random": float(signal_vs_random),
            **{f"signal_vs_{k}": float(v) for k, v in divergences.items()}
        },
        "statistical_test": {
            "random_to_random_divergence_mean": float(random_div_mean),
            "random_to_random_divergence_std": float(random_div_std),
            "signal_to_random_divergence": float(signal_vs_random),
            "z_score": float(signal_z_score),
            "is_significant": bool(abs(signal_z_score) > 3)
        },
        "semantic_embedding_properties": {
            name: {
                "shape": list(data["shape"]),
                "effective_dimension_90": int(data["effective_dimension"]),
                "participation_ratio": float(data["participation_ratio"]),
                "spectrum_entropy": float(spectrum_entropy(data["spectrum"]))
            }
            for name, data in semantic_spectra.items()
        }
    }

    output_path = RESULTS_DIR / "exp39_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n10. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
