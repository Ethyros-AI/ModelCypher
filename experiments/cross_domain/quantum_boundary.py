#!/usr/bin/env python3
"""
Experiment 2.4: Quantum Systems - The Boundary Between Information and Physics

HYPOTHESIS:
Quantum mechanics sits at the boundary between information and physics.
- Superposition (unmeasured) might show π/e (pure information)
- Collapsed states (measured) might show φ/√3 (physical reality)
- This would explain the measurement problem!

METHODOLOGY:
1. Generate density matrices for quantum states
2. Compare superposition states vs classical mixtures
3. Analyze entanglement structure
4. Test if measurement transition shows π/e → φ/√3

PREDICTION:
If true, the measurement problem becomes:
  "Measurement is the transition from informational (π/e) to physical (φ/√3) regime"
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import logm, sqrtm

# Constants
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)

CONSTANTS = {
    "pi/e": PI / E,
    "e/pi": E / PI,
    "phi": PHI,
    "1/phi": 1 / PHI,
    "sqrt2": SQRT2,
    "1/sqrt2": 1 / SQRT2,
    "sqrt3": SQRT3,
    "e": E,
    "pi": PI,
}

MATCH_THRESHOLD = 0.05


def count_constant_matches(S: np.ndarray, bidirectional: bool = True) -> Dict[str, int]:
    """Count matches for each constant in singular value ratios."""
    matches = {name: 0 for name in CONSTANTS}

    for i in range(min(len(S) - 1, 20)):
        for j in range(i + 1, min(len(S), i + 6)):
            if S[j] > 1e-10:
                ratio1 = S[i] / S[j]
                ratio2 = S[j] / S[i] if bidirectional else None

                for const_name, const_val in CONSTANTS.items():
                    error1 = abs(ratio1 - const_val) / const_val
                    if error1 < MATCH_THRESHOLD:
                        matches[const_name] += 1

                    if bidirectional and ratio2 is not None:
                        error2 = abs(ratio2 - const_val) / const_val
                        if error2 < MATCH_THRESHOLD:
                            matches[const_name] += 1

    return matches


# ============================================================================
# QUANTUM STATE GENERATION
# ============================================================================

def create_pure_state(n_qubits: int, state_type: str = "random") -> np.ndarray:
    """Create a pure quantum state vector.

    Pure states represent complete quantum information (superposition).
    """
    dim = 2 ** n_qubits

    if state_type == "random":
        # Random pure state (Haar distributed)
        real_part = np.random.randn(dim)
        imag_part = np.random.randn(dim)
        state = real_part + 1j * imag_part
        state = state / np.linalg.norm(state)

    elif state_type == "ghz":
        # GHZ state: |00...0⟩ + |11...1⟩ (maximally entangled)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1 / SQRT2
        state[-1] = 1 / SQRT2

    elif state_type == "w":
        # W state: |100...0⟩ + |010...0⟩ + ... (different entanglement)
        state = np.zeros(dim, dtype=complex)
        for i in range(n_qubits):
            idx = 2 ** (n_qubits - 1 - i)
            state[idx] = 1 / math.sqrt(n_qubits)

    elif state_type == "superposition":
        # Equal superposition of all basis states
        state = np.ones(dim, dtype=complex) / math.sqrt(dim)

    else:
        raise ValueError(f"Unknown state type: {state_type}")

    return state


def create_density_matrix(state: np.ndarray) -> np.ndarray:
    """Create density matrix ρ = |ψ⟩⟨ψ| from pure state."""
    return np.outer(state, np.conj(state))


def create_mixed_state(n_qubits: int, purity: float = 0.5) -> np.ndarray:
    """Create a mixed state density matrix.

    Mixed states represent classical uncertainty + quantum coherence.
    purity = 1: pure state
    purity = 1/dim: maximally mixed (no quantum information)
    """
    dim = 2 ** n_qubits

    # Start with maximally mixed state
    rho = np.eye(dim, dtype=complex) / dim

    if purity > 1/dim:
        # Add quantum coherence by mixing with a pure state
        pure_state = create_pure_state(n_qubits, "random")
        pure_rho = create_density_matrix(pure_state)

        # Interpolate to achieve target purity
        # Purity = Tr(ρ²), for maximally mixed = 1/dim, for pure = 1
        # Linear interpolation: ρ = (1-α)ρ_mixed + α*ρ_pure
        # At α=0: purity = 1/dim, at α=1: purity = 1
        alpha = (purity - 1/dim) / (1 - 1/dim)
        alpha = max(0, min(1, alpha))

        rho = (1 - alpha) * rho + alpha * pure_rho

    return rho


def create_classical_mixture(n_qubits: int) -> np.ndarray:
    """Create a classical mixture (diagonal density matrix).

    Classical mixtures have NO quantum coherence - they represent
    states that have "collapsed" into definite outcomes.
    """
    dim = 2 ** n_qubits

    # Random classical probabilities
    probs = np.random.dirichlet(np.ones(dim))

    # Diagonal density matrix (no off-diagonal coherence)
    rho = np.diag(probs.astype(complex))

    return rho


def partial_trace(rho: np.ndarray, n_qubits: int, keep_qubits: List[int]) -> np.ndarray:
    """Compute partial trace of density matrix.

    Used to examine subsystems and measure entanglement.
    """
    dim = 2 ** n_qubits

    # Reshape into tensor form
    shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(shape)

    # Trace over qubits not in keep_qubits
    trace_qubits = [i for i in range(n_qubits) if i not in keep_qubits]

    result = rho_tensor
    for q in sorted(trace_qubits, reverse=True):
        # Trace over qubit q: sum over indices q and q+n_qubits
        result = np.trace(result, axis1=q, axis2=q+len(keep_qubits)+len(trace_qubits))
        # Note: This is simplified - full implementation needs careful index tracking

    # For simplicity, return reduced density matrix directly
    n_keep = len(keep_qubits)
    dim_keep = 2 ** n_keep

    # Alternative: direct computation
    reduced = np.zeros((dim_keep, dim_keep), dtype=complex)

    for i in range(dim_keep):
        for j in range(dim_keep):
            for k in range(dim // dim_keep):
                # Map indices back to full system
                full_i = i * (dim // dim_keep) + k
                full_j = j * (dim // dim_keep) + k
                reduced[i, j] += rho[full_i, full_j]

    return reduced


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S = -Tr(ρ log ρ).

    Measures quantum information content.
    Pure states: S = 0
    Maximally mixed: S = log(dim)
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Avoid log(0)

    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def purity(rho: np.ndarray) -> float:
    """Compute purity Tr(ρ²).

    Pure states: purity = 1
    Maximally mixed: purity = 1/dim
    """
    return float(np.real(np.trace(rho @ rho)))


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_density_matrix(rho: np.ndarray, name: str) -> Dict:
    """Analyze a density matrix for constant matches."""

    dim = rho.shape[0]

    # SVD of density matrix
    U, S, Vh = np.linalg.svd(rho)

    # Count matches
    matches = count_constant_matches(S.real, bidirectional=True)
    total = sum(matches.values())

    # Compute fractions
    pi_e_total = matches["pi/e"] + matches["e/pi"]
    phi_sqrt3_total = matches["phi"] + matches["1/phi"] + matches["sqrt3"]

    pi_e_frac = pi_e_total / total if total > 0 else 0
    phi_sqrt3_frac = phi_sqrt3_total / total if total > 0 else 0

    # Quantum properties
    state_purity = purity(rho)
    entropy = von_neumann_entropy(rho)

    return {
        "name": name,
        "dimension": dim,
        "matches": matches,
        "total_matches": total,
        "pi_e_matches": pi_e_total,
        "phi_sqrt3_matches": phi_sqrt3_total,
        "pi_e_fraction": float(pi_e_frac),
        "phi_sqrt3_fraction": float(phi_sqrt3_frac),
        "purity": float(state_purity),
        "entropy": float(entropy),
        "max_entropy": float(np.log2(dim)),
        "top_singular_values": list(S.real[:10]),
    }


def run_quantum_comparison(n_samples: int = 50, n_qubits: int = 4) -> Dict:
    """Compare pure states, mixed states, and classical mixtures."""

    print(f"\nAnalyzing {n_samples} samples of each state type...")
    print(f"System size: {n_qubits} qubits ({2**n_qubits} dimensional Hilbert space)")

    results = {
        "pure_superposition": [],
        "mixed_state": [],
        "classical_mixture": [],
        "ghz_entangled": [],
        "w_entangled": [],
    }

    # Pure superposition states (maximum quantum information)
    print("\n1. Pure superposition states (max quantum info)...")
    for _ in range(n_samples):
        state = create_pure_state(n_qubits, "superposition")
        rho = create_density_matrix(state)
        analysis = analyze_density_matrix(rho, "pure_superposition")
        results["pure_superposition"].append(analysis)

    # Random mixed states (partial quantum information)
    print("2. Mixed states (partial quantum info)...")
    for _ in range(n_samples):
        rho = create_mixed_state(n_qubits, purity=0.5)
        analysis = analyze_density_matrix(rho, "mixed_state")
        results["mixed_state"].append(analysis)

    # Classical mixtures (no quantum coherence - "collapsed")
    print("3. Classical mixtures (no quantum coherence)...")
    for _ in range(n_samples):
        rho = create_classical_mixture(n_qubits)
        analysis = analyze_density_matrix(rho, "classical_mixture")
        results["classical_mixture"].append(analysis)

    # GHZ entangled states
    print("4. GHZ entangled states (max entanglement)...")
    for _ in range(n_samples):
        state = create_pure_state(n_qubits, "ghz")
        rho = create_density_matrix(state)
        analysis = analyze_density_matrix(rho, "ghz_entangled")
        results["ghz_entangled"].append(analysis)

    # W entangled states
    print("5. W entangled states (different entanglement type)...")
    for _ in range(n_samples):
        state = create_pure_state(n_qubits, "w")
        rho = create_density_matrix(state)
        analysis = analyze_density_matrix(rho, "w_entangled")
        results["w_entangled"].append(analysis)

    return results


def compute_statistics(results: Dict) -> Dict:
    """Compute aggregate statistics for each state type."""

    stats_results = {}

    for state_type, analyses in results.items():
        if not analyses:
            continue

        pi_e_fracs = [a["pi_e_fraction"] for a in analyses]
        phi_sqrt3_fracs = [a["phi_sqrt3_fraction"] for a in analyses]
        purities = [a["purity"] for a in analyses]
        entropies = [a["entropy"] for a in analyses]
        totals = [a["total_matches"] for a in analyses]

        stats_results[state_type] = {
            "n_samples": len(analyses),
            "pi_e_mean": float(np.mean(pi_e_fracs)),
            "pi_e_std": float(np.std(pi_e_fracs)),
            "phi_sqrt3_mean": float(np.mean(phi_sqrt3_fracs)),
            "phi_sqrt3_std": float(np.std(phi_sqrt3_fracs)),
            "purity_mean": float(np.mean(purities)),
            "entropy_mean": float(np.mean(entropies)),
            "total_matches_mean": float(np.mean(totals)),
        }

    return stats_results


def main():
    """Run quantum boundary experiment."""

    print("=" * 70)
    print("EXPERIMENT 2.4: QUANTUM SYSTEMS - THE BOUNDARY")
    print("=" * 70)
    print("\nHypothesis: Quantum mechanics sits at the information/geometry boundary")
    print("  - Superposition (unmeasured) → π/e (information)")
    print("  - Classical mixture (collapsed) → φ/√3 (geometry)")

    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "2.4_quantum_boundary",
        "hypothesis": "Measurement transitions from π/e (info) to φ/√3 (geometry)",
    }

    # Run comparison
    raw_results = run_quantum_comparison(n_samples=100, n_qubits=4)

    # Compute statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    stats_results = compute_statistics(raw_results)
    results["statistics"] = stats_results

    print("\n{:<25} {:>10} {:>10} {:>10} {:>10}".format(
        "State Type", "π/e %", "φ/√3 %", "Purity", "Entropy"
    ))
    print("-" * 70)

    for state_type, s in stats_results.items():
        print("{:<25} {:>10.1f} {:>10.1f} {:>10.3f} {:>10.2f}".format(
            state_type,
            s["pi_e_mean"] * 100,
            s["phi_sqrt3_mean"] * 100,
            s["purity_mean"],
            s["entropy_mean"],
        ))

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Compare pure vs classical
    pure_pi_e = [a["pi_e_fraction"] for a in raw_results["pure_superposition"]]
    classical_pi_e = [a["pi_e_fraction"] for a in raw_results["classical_mixture"]]

    if len(pure_pi_e) >= 2 and len(classical_pi_e) >= 2:
        t_stat, p_value = stats.ttest_ind(pure_pi_e, classical_pi_e)
        print(f"\nPure vs Classical (π/e fraction):")
        print(f"  Pure mean: {np.mean(pure_pi_e)*100:.1f}%")
        print(f"  Classical mean: {np.mean(classical_pi_e)*100:.1f}%")
        print(f"  T-test: t={t_stat:.2f}, p={p_value:.4f}")

        results["pure_vs_classical_test"] = {
            "pure_mean": float(np.mean(pure_pi_e)),
            "classical_mean": float(np.mean(classical_pi_e)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    pure_phi = [a["phi_sqrt3_fraction"] for a in raw_results["pure_superposition"]]
    classical_phi = [a["phi_sqrt3_fraction"] for a in raw_results["classical_mixture"]]

    if len(pure_phi) >= 2 and len(classical_phi) >= 2:
        t_stat, p_value = stats.ttest_ind(pure_phi, classical_phi)
        print(f"\nPure vs Classical (φ/√3 fraction):")
        print(f"  Pure mean: {np.mean(pure_phi)*100:.1f}%")
        print(f"  Classical mean: {np.mean(classical_phi)*100:.1f}%")
        print(f"  T-test: t={t_stat:.2f}, p={p_value:.4f}")

        results["pure_vs_classical_phi_test"] = {
            "pure_mean": float(np.mean(pure_phi)),
            "classical_mean": float(np.mean(classical_phi)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    pure_pi_e_mean = stats_results.get("pure_superposition", {}).get("pi_e_mean", 0)
    classical_pi_e_mean = stats_results.get("classical_mixture", {}).get("pi_e_mean", 0)
    pure_phi_mean = stats_results.get("pure_superposition", {}).get("phi_sqrt3_mean", 0)
    classical_phi_mean = stats_results.get("classical_mixture", {}).get("phi_sqrt3_mean", 0)

    hypothesis_supported = (
        pure_pi_e_mean > classical_pi_e_mean and
        classical_phi_mean > pure_phi_mean
    )

    if hypothesis_supported:
        print("\n✓ HYPOTHESIS SUPPORTED:")
        print(f"  Pure superposition shows MORE π/e ({pure_pi_e_mean*100:.1f}% vs {classical_pi_e_mean*100:.1f}%)")
        print(f"  Classical mixture shows MORE φ/√3 ({classical_phi_mean*100:.1f}% vs {pure_phi_mean*100:.1f}%)")
        print("\n  → Measurement may transition from informational to physical regime!")
        verdict = "SUPPORTED"
    else:
        print("\n✗ HYPOTHESIS NOT CLEARLY SUPPORTED:")
        print(f"  Pure π/e: {pure_pi_e_mean*100:.1f}%")
        print(f"  Classical π/e: {classical_pi_e_mean*100:.1f}%")
        print(f"  Pure φ/√3: {pure_phi_mean*100:.1f}%")
        print(f"  Classical φ/√3: {classical_phi_mean*100:.1f}%")
        verdict = "NOT_SUPPORTED"

    results["verdict"] = {
        "hypothesis_supported": hypothesis_supported,
        "pure_pi_e": float(pure_pi_e_mean),
        "classical_pi_e": float(classical_pi_e_mean),
        "pure_phi_sqrt3": float(pure_phi_mean),
        "classical_phi_sqrt3": float(classical_phi_mean),
        "interpretation": verdict,
    }

    # Additional analysis: Entanglement
    print("\n" + "-" * 40)
    print("Entanglement Analysis:")

    ghz_stats = stats_results.get("ghz_entangled", {})
    w_stats = stats_results.get("w_entangled", {})

    print(f"\nGHZ state (max entanglement):")
    print(f"  π/e: {ghz_stats.get('pi_e_mean', 0)*100:.1f}%")
    print(f"  φ/√3: {ghz_stats.get('phi_sqrt3_mean', 0)*100:.1f}%")

    print(f"\nW state (different entanglement):")
    print(f"  π/e: {w_stats.get('pi_e_mean', 0)*100:.1f}%")
    print(f"  φ/√3: {w_stats.get('phi_sqrt3_mean', 0)*100:.1f}%")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"quantum_boundary_{timestamp}.json"

    # Don't save all raw results (too large)
    results["raw_sample_counts"] = {k: len(v) for k, v in raw_results.items()}

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
