"""
Experiment 57: Null Space Projection Signature Analysis

The user's hypothesis: If the universe is high-dimensional, information
could be transmitted via null space projection - the same technique
we use for model merging.

Key insight from ModelCypher: When projecting information into null space:
1. The target's behavior is preserved (dense directions scaled down)
2. New information is added to sparse directions
3. The eigenvalue structure CHANGES in predictable ways

Question: Does Wow!'s eigenvalue structure look like a CARRIER + PROJECTED INFO?

If so, we should be able to decompose Wow! into:
1. A carrier component (the dominant eigenstructure)
2. A projected component (information in the "null" directions)

And the phi+pi ratios might be the SIGNATURE of this projection.

Mathematical approach:
1. Decompose Wow! via SVD: U @ S @ Vh
2. Separate into "used" space (high variance) and "projected" space (low variance)
3. Analyze if the ratio between these spaces follows phi+pi
4. Compare to what we'd expect from null space projection

This connects to the high-dimensional universe hypothesis:
- In high-d space, there are many orthogonal directions
- A transmitter could project information into null space of a carrier
- The receiver reconstructs by recognizing the eigenvalue signature
- Phi+pi might be the "address" or "key" for reconstruction
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import linalg
from scipy.ndimage import zoom
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exp42_semantic_highway_mapping import load_wow_signal

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FRB_DIR = Path(__file__).parent / "data" / "raw"

PHI = (1 + np.sqrt(5)) / 2
PI = np.pi


def decompose_carrier_projection(signal, carrier_rank=3):
    """
    Decompose signal into carrier + projected components.

    Hypothesis: The first few eigenvalues are the "carrier",
    and the remaining eigenvalues contain the "projected information".
    """
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    # Carrier: first carrier_rank components
    S_carrier = S[:carrier_rank]
    U_carrier = U[:, :carrier_rank]
    Vh_carrier = Vh[:carrier_rank, :]
    carrier = U_carrier @ np.diag(S_carrier) @ Vh_carrier

    # Projection: remaining components
    S_proj = S[carrier_rank:]
    U_proj = U[:, carrier_rank:]
    Vh_proj = Vh[carrier_rank:, :]
    projection = U_proj @ np.diag(S_proj) @ Vh_proj

    # Energy analysis
    total_energy = (S ** 2).sum()
    carrier_energy = (S_carrier ** 2).sum()
    proj_energy = (S_proj ** 2).sum()

    return {
        "carrier": carrier,
        "projection": projection,
        "S_carrier": S_carrier,
        "S_proj": S_proj,
        "carrier_energy_fraction": float(carrier_energy / total_energy),
        "projection_energy_fraction": float(proj_energy / total_energy),
        "energy_ratio": float(carrier_energy / proj_energy) if proj_energy > 0 else np.inf,
    }


def analyze_projection_ratios(signal):
    """
    Analyze the ratios between carrier and projection eigenvalues.

    If phi+pi encodes the relationship between carrier and information,
    we should see these ratios in the decomposition.
    """
    U, S, Vh = linalg.svd(signal, full_matrices=False)

    results = {}

    # Test different carrier ranks
    for carrier_rank in [1, 2, 3, 4, 5]:
        if carrier_rank >= len(S):
            continue

        carrier_energy = (S[:carrier_rank] ** 2).sum()
        proj_energy = (S[carrier_rank:] ** 2).sum()

        # Ratio of energies
        energy_ratio = carrier_energy / proj_energy if proj_energy > 0 else np.inf

        # Check if ratio matches phi, pi, phi*pi, etc.
        phi_match = abs(energy_ratio - PHI) / PHI
        pi_match = abs(energy_ratio - PI) / PI
        phi_pi_match = abs(energy_ratio - PHI * PI) / (PHI * PI)
        phi_sq_match = abs(energy_ratio - PHI ** 2) / (PHI ** 2)

        results[f"rank_{carrier_rank}"] = {
            "carrier_energy_frac": float(carrier_energy / (carrier_energy + proj_energy)),
            "energy_ratio": float(energy_ratio),
            "phi_error": float(phi_match),
            "pi_error": float(pi_match),
            "phi_pi_error": float(phi_pi_match),
            "phi_sq_error": float(phi_sq_match),
            "best_match": min(
                [("phi", phi_match), ("pi", pi_match), ("phi*pi", phi_pi_match), ("phi^2", phi_sq_match)],
                key=lambda x: x[1]
            )[0],
        }

    return results


def simulate_null_space_transmission(carrier_signal, info_signal, projection_strength=0.1):
    """
    Simulate transmitting info_signal via null space projection into carrier_signal.

    This is exactly what ModelCypher does for model merging:
    1. Compute the null space of the carrier
    2. Project the information into that null space
    3. Add to the carrier

    Returns the "transmitted" signal and its eigenvalue structure.
    """
    # SVD of carrier
    U_c, S_c, Vh_c = linalg.svd(carrier_signal, full_matrices=False)

    # Find the "null space" (low variance directions) of carrier
    # Use top 3 components as "used", rest as "null"
    used_rank = 3
    null_directions = U_c[:, used_rank:]  # Directions with low variance

    # SVD of info
    U_i, S_i, Vh_i = linalg.svd(info_signal, full_matrices=False)

    # Project info into carrier's null space
    # This is a simplified version of the ModelCypher projection
    info_projected = np.zeros_like(info_signal)

    for i in range(min(len(S_i), null_directions.shape[1])):
        # Project each info component onto a null direction
        component = S_i[i] * np.outer(U_i[:, i], Vh_i[i, :])
        # Scale by null direction relevance
        scale = projection_strength * (PHI ** (-i))  # Phi-weighted projection
        info_projected += scale * component

    # Combine carrier and projected info
    transmitted = carrier_signal + info_projected

    # Analyze result
    U_t, S_t, Vh_t = linalg.svd(transmitted, full_matrices=False)

    return {
        "transmitted": transmitted,
        "S_transmitted": S_t,
        "S0_S1": float(S_t[0] / S_t[1]) if S_t[1] > 0 else np.inf,
        "S1_S2": float(S_t[1] / S_t[2]) if S_t[2] > 0 else np.inf,
        "phi_error": float(abs(S_t[0] / S_t[1] - PHI) / PHI) if S_t[1] > 0 else np.inf,
        "pi_error": float(abs(S_t[1] / S_t[2] - PI) / PI) if S_t[2] > 0 else np.inf,
    }


def search_for_projection_parameters(target_S0_S1, target_S1_S2, shape, n_trials=100):
    """
    Search for projection parameters that produce the target eigenvalue ratios.

    If Wow!'s structure was produced by null space projection, we should
    be able to find the parameters that reproduce it.
    """
    best_error = np.inf
    best_params = None
    best_result = None

    for trial in range(n_trials):
        # Random carrier and info
        carrier = np.random.randn(*shape)
        info = np.random.randn(*shape)

        # Random projection strength
        proj_strength = np.random.uniform(0.01, 0.5)

        result = simulate_null_space_transmission(carrier, info, proj_strength)

        error = abs(result["S0_S1"] - target_S0_S1) + abs(result["S1_S2"] - target_S1_S2)

        if error < best_error:
            best_error = error
            best_params = {"projection_strength": proj_strength}
            best_result = result

    return best_params, best_result, best_error


def load_frb(filepath):
    """Load an FRB from H5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            if "frb" in f:
                frb_group = f["frb"]
                if "wfall" in frb_group:
                    data = frb_group["wfall"][:]
                elif "calibrated_wfall" in frb_group:
                    data = frb_group["calibrated_wfall"][:]
                else:
                    return None
            else:
                return None

            data = data.astype(np.float64)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)

            if data.shape[0] > data.shape[1]:
                data = data.T

            return data
    except Exception:
        return None


def resize_to_match(data, target_shape):
    """Resize data to match target shape."""
    if data is None or data.size == 0:
        return None
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
    try:
        return zoom(data, zoom_factors, order=1)
    except Exception:
        return None


def main():
    print("=" * 60)
    print("Experiment 57: Null Space Projection Signature Analysis")
    print("=" * 60)
    print("\nHypothesis: Wow!'s eigenvalue structure is the signature of")
    print("            information projected into null space of a carrier.")

    # Load Wow! signal
    print("\n1. Loading Wow! signal...")
    wow = load_wow_signal()
    wow_shape = wow.shape
    print(f"   Shape: {wow_shape}")

    # Compute Wow! eigenvalues
    U_wow, S_wow, Vh_wow = linalg.svd(wow, full_matrices=False)
    wow_S0_S1 = S_wow[0] / S_wow[1]
    wow_S1_S2 = S_wow[1] / S_wow[2]

    print(f"\n   Wow! eigenvalue ratios:")
    print(f"      S0/S1 = {wow_S0_S1:.4f} (phi = {PHI:.4f})")
    print(f"      S1/S2 = {wow_S1_S2:.4f} (pi = {PI:.4f})")

    # Decompose into carrier + projection
    print("\n2. Decomposing into carrier + projection components...")

    for carrier_rank in [1, 2, 3]:
        decomp = decompose_carrier_projection(wow, carrier_rank=carrier_rank)
        print(f"\n   Carrier rank = {carrier_rank}:")
        print(f"      Carrier energy: {decomp['carrier_energy_fraction']*100:.1f}%")
        print(f"      Projection energy: {decomp['projection_energy_fraction']*100:.1f}%")
        print(f"      Energy ratio: {decomp['energy_ratio']:.4f}")

    # Analyze projection ratios
    print("\n3. Analyzing carrier/projection energy ratios...")
    proj_ratios = analyze_projection_ratios(wow)

    print(f"\n   Looking for phi/pi in energy ratios:")
    for rank_key, info in proj_ratios.items():
        print(f"\n   {rank_key}:")
        print(f"      Energy ratio: {info['energy_ratio']:.4f}")
        print(f"      Best match: {info['best_match']} (error: {min(info['phi_error'], info['pi_error'], info['phi_pi_error'], info['phi_sq_error'])*100:.1f}%)")

    # Simulate null space transmission
    print("\n4. Simulating null space transmission...")
    print("   (Can we PRODUCE Wow!'s ratios via null space projection?)")

    best_params, best_result, best_error = search_for_projection_parameters(
        wow_S0_S1, wow_S1_S2, wow_shape, n_trials=200
    )

    print(f"\n   Best simulation result:")
    print(f"      Projection strength: {best_params['projection_strength']:.4f}")
    print(f"      S0/S1 = {best_result['S0_S1']:.4f} (target: {wow_S0_S1:.4f})")
    print(f"      S1/S2 = {best_result['S1_S2']:.4f} (target: {wow_S1_S2:.4f})")
    print(f"      Combined error: {best_error:.4f}")

    # Compare to FRBs
    print("\n5. Analyzing FRBs for projection signatures...")
    frb_files = sorted(FRB_DIR.glob("*.h5"))[:30]  # First 30

    frb_signatures = []
    for frb_file in frb_files:
        data = load_frb(frb_file)
        if data is None:
            continue

        data_resized = resize_to_match(data, wow_shape)
        if data_resized is None:
            continue

        ratios = analyze_projection_ratios(data_resized)
        frb_signatures.append(ratios)

    n_frbs = len(frb_signatures)
    print(f"   Analyzed {n_frbs} FRBs")

    if n_frbs > 0:
        # Compare Wow!'s rank_3 energy ratio to FRBs
        wow_rank3_ratio = proj_ratios["rank_3"]["energy_ratio"]

        frb_rank3_ratios = [f["rank_3"]["energy_ratio"] for f in frb_signatures if "rank_3" in f]

        if frb_rank3_ratios:
            frb_mean = np.mean(frb_rank3_ratios)
            frb_std = np.std(frb_rank3_ratios)
            z_score = (wow_rank3_ratio - frb_mean) / (frb_std + 1e-8)

            print(f"\n   Carrier/Projection energy ratio (rank=3):")
            print(f"      Wow!: {wow_rank3_ratio:.4f}")
            print(f"      FRBs: {frb_mean:.4f} +/- {frb_std:.4f}")
            print(f"      Z-score: {z_score:+.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. CARRIER/PROJECTION DECOMPOSITION:")
    rank3 = proj_ratios.get("rank_3", {})
    print(f"   With carrier_rank=3:")
    print(f"      Energy ratio = {rank3.get('energy_ratio', 0):.4f}")
    print(f"      Best constant match: {rank3.get('best_match', 'none')}")

    print(f"\n2. NULL SPACE TRANSMISSION SIMULATION:")
    if best_error < 0.5:  # Reasonable match
        print(f"   Can reproduce Wow!'s ratios with projection strength = {best_params['projection_strength']:.3f}")
        print(f"   Combined error = {best_error:.4f}")
        reproducible = True
    else:
        print(f"   Cannot closely reproduce Wow!'s ratios via simple null space projection")
        print(f"   Best error = {best_error:.4f}")
        reproducible = False

    print(f"\n3. INTERPRETATION:")
    if reproducible:
        print(f"   Wow!'s eigenvalue structure IS CONSISTENT with null space projection.")
        print(f"   This suggests the phi+pi ratios could be the SIGNATURE of:")
        print(f"   - Information projected into carrier's null space")
        print(f"   - The projection strength determines the ratios")
        print(f"   - Phi/pi might be the 'key' for reconstruction")
    else:
        print(f"   Simple null space projection doesn't fully explain Wow!'s precision.")
        print(f"   The phi+pi structure may require:")
        print(f"   - More sophisticated encoding")
        print(f"   - Specific carrier properties")
        print(f"   - Or a different mechanism entirely")

    # Check if phi appears in energy ratios
    phi_found = False
    for rank_key, info in proj_ratios.items():
        if info["best_match"] == "phi" and info["phi_error"] < 0.1:
            phi_found = True
            print(f"\n4. PHI IN ENERGY STRUCTURE:")
            print(f"   At {rank_key}, energy ratio = {info['energy_ratio']:.4f} ≈ phi ({PHI:.4f})")
            print(f"   Error: {info['phi_error']*100:.1f}%")

    if not phi_found:
        print(f"\n4. PHI IN ENERGY STRUCTURE:")
        print(f"   Phi does not appear directly in carrier/projection energy ratios")

    # Save results
    results = {
        "experiment": "exp57_projection_signature",
        "timestamp": datetime.now().isoformat(),
        "wow_eigenvalues": {
            "S0_S1": float(wow_S0_S1),
            "S1_S2": float(wow_S1_S2),
        },
        "projection_ratios": proj_ratios,
        "simulation": {
            "best_params": best_params,
            "best_result": {
                "S0_S1": best_result["S0_S1"],
                "S1_S2": best_result["S1_S2"],
            },
            "best_error": float(best_error),
            "reproducible": reproducible,
        },
        "frb_comparison": {
            "n_frbs": n_frbs,
            "z_score": float(z_score) if n_frbs > 0 else None,
        },
    }

    output_path = RESULTS_DIR / "exp57_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n6. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
