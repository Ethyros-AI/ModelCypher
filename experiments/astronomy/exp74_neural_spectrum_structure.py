"""
Experiment 74: Neural Spectrum Structure Analysis

Inspired by the Wow! signal analysis: does neural network activation
spectra have similar "protected mode" structure with characteristic ratios?

Questions:
1. What is the isolation ratio (signal/noise separation) in LLM activations?
2. Do consecutive signal eigenvalues have characteristic ratios (like φ)?
3. Is there self-similar structure across layers?

This experiment collects real activations from a small model and analyzes
the singular value spectrum in the same way we analyzed the Wow! signal.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fixtures.models import ensure_model, collect_real_activations, get_atlas_probes
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.backends import initialize_default_backend
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    separate_signal_noise,
    compute_signal_rank_from_singular_values,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618


def analyze_spectrum_structure(activations, layer_name: str):
    """Analyze singular value spectrum structure like Wow! analysis."""
    # SVD
    U, S, Vt = linalg.svd(activations, full_matrices=False)
    S = S / S[0]  # Normalize to max=1

    n_samples, n_features = activations.shape

    # 1. RMT signal/noise separation
    backend = get_default_backend()
    rmt_result = separate_signal_noise(backend.array(activations), backend=backend)

    signal_rank = rmt_result.signal_rank
    noise_rank = rmt_result.noise_rank

    # 2. Isolation ratio (Wow! style)
    # Edge gap = S[signal_rank-1] - S[signal_rank] (last signal - first noise)
    # Bulk gap = mean gap in noise region
    if signal_rank > 0 and signal_rank < len(S) - 1:
        edge_gap = S[signal_rank - 1] - S[signal_rank]
        noise_S = S[signal_rank:]
        if len(noise_S) > 1:
            bulk_gaps = noise_S[:-1] - noise_S[1:]
            bulk_gap = np.mean(bulk_gaps) if len(bulk_gaps) > 0 else 1e-10
            isolation_ratio = edge_gap / max(bulk_gap, 1e-10)
        else:
            isolation_ratio = float('inf')
    else:
        edge_gap = 0
        bulk_gap = 0
        isolation_ratio = 0

    # 3. Signal mode ratios (looking for φ-like structure)
    signal_ratios = []
    if signal_rank > 1:
        signal_S = S[:signal_rank]
        for i in range(len(signal_S) - 1):
            ratio = signal_S[i] / max(signal_S[i + 1], 1e-10)
            signal_ratios.append(ratio)

    # 4. Check for φ-proximity
    phi_proximity = []
    for r in signal_ratios:
        # Distance from φ, φ², φ³
        distances = [abs(r - PHI), abs(r - PHI**2), abs(r - PHI**3)]
        closest_phi_power = np.argmin(distances) + 1
        closest_distance = min(distances)
        phi_proximity.append({
            "ratio": r,
            "closest_phi_power": closest_phi_power,
            "distance_from_phi": closest_distance,
            "relative_error": closest_distance / (PHI ** closest_phi_power)
        })

    return {
        "layer": layer_name,
        "shape": [n_samples, n_features],
        "signal_rank": signal_rank,
        "noise_rank": noise_rank,
        "isolation_ratio": float(isolation_ratio),
        "edge_gap": float(edge_gap),
        "bulk_gap": float(bulk_gap),
        "signal_variance_fraction": rmt_result.signal_variance_fraction,
        "signal_ratios": [float(r) for r in signal_ratios],
        "phi_proximity": phi_proximity,
        "top_10_singular_values": [float(s) for s in S[:10]],
    }


def main():
    print("=" * 60)
    print("Experiment 74: Neural Spectrum Structure Analysis")
    print("=" * 60)

    # Initialize backend
    initialize_default_backend()

    # Use small model for speed
    model_path = ensure_model()  # SmolLM-135M
    print(f"\nModel: {model_path}")

    # Get probes
    probes = get_atlas_probes(n_samples=50)
    print(f"Probes: {len(probes)}")

    # Collect activations
    backend = get_default_backend()
    activations_by_layer = collect_real_activations(
        model_path, probes, backend, layer_indices=[0, 2, 4, 6, 8]
    )

    print(f"\nCollected activations for {len(activations_by_layer)} layers")

    # Analyze each layer
    results = []
    for layer_idx, activations in activations_by_layer.items():
        print(f"\n--- Layer {layer_idx} ---")
        act_np = np.array(activations)

        analysis = analyze_spectrum_structure(act_np, f"layer_{layer_idx}")
        results.append(analysis)

        print(f"  Shape: {analysis['shape']}")
        print(f"  Signal rank: {analysis['signal_rank']}, Noise rank: {analysis['noise_rank']}")
        print(f"  Isolation ratio: {analysis['isolation_ratio']:.2f}x")
        print(f"  Signal variance: {analysis['signal_variance_fraction']*100:.1f}%")

        if analysis['signal_ratios']:
            print(f"  Signal mode ratios: {[f'{r:.3f}' for r in analysis['signal_ratios'][:5]]}")
            for pp in analysis['phi_proximity'][:3]:
                if pp['relative_error'] < 0.1:  # Within 10% of φ^n
                    print(f"    -> {pp['ratio']:.3f} ≈ φ^{pp['closest_phi_power']} (error: {pp['relative_error']*100:.1f}%)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    isolation_ratios = [r['isolation_ratio'] for r in results if r['isolation_ratio'] > 0]
    if isolation_ratios:
        print(f"Isolation ratios: min={min(isolation_ratios):.1f}, max={max(isolation_ratios):.1f}, mean={np.mean(isolation_ratios):.1f}")

    # Check for φ-proximity across all layers
    all_ratios = []
    for r in results:
        all_ratios.extend(r['signal_ratios'])

    if all_ratios:
        phi_hits = sum(1 for r in all_ratios if any(
            abs(r - PHI**k) / PHI**k < 0.1 for k in [1, 2, 3]
        ))
        print(f"φ-proximity: {phi_hits}/{len(all_ratios)} ratios within 10% of φ^n")

    # Comparison to Wow! signal
    print("\n--- Comparison to Wow! Signal ---")
    print(f"Wow! isolation ratio: 49.5x")
    if isolation_ratios:
        print(f"Neural mean isolation: {np.mean(isolation_ratios):.1f}x")
    print(f"Wow! φ-ratio: S0/S1 = 1.56 ≈ φ (3% error)")

    # Save
    with open(RESULTS_DIR / "exp74_results.json", "w") as f:
        json.dump({
            "experiment": "exp74_neural_spectrum_structure",
            "timestamp": datetime.now().isoformat(),
            "model": str(model_path),
            "n_probes": len(probes),
            "layers": results,
            "summary": {
                "isolation_ratios": isolation_ratios,
                "all_signal_ratios": all_ratios,
                "phi": PHI,
            }
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'exp74_results.json'}")


if __name__ == "__main__":
    main()
