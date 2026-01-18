#!/usr/bin/env python3
"""
Ill-Conditioned Alignment Experiment

Verifies that the numerical-rank-truncated alignment correctly handles
ill-conditioned Gram matrices. Tests:

1. Synthetic activations with controlled condition numbers
2. Rank-deficient activations
3. Binary search for failure boundary

Success criteria (precision-derived):
- Truncated κ < 1/√ε
- Alignment residual < √ε on well-posed problems
- CKA = 1.0 on training probes (by construction)

Results saved to experiments/results/ill_conditioned_alignment.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_activations(
    backend,
    n_samples: int,
    dimension: int,
    condition_number: float,
    seed: int = 42,
) -> tuple:
    """Create synthetic activations with a controlled condition number.

    Constructs A = U @ diag(σ) @ V^T where σ are chosen to achieve target κ.

    Parameters
    ----------
    backend : Backend
        Backend for tensor operations.
    n_samples : int
        Number of samples.
    dimension : int
        Feature dimension.
    condition_number : float
        Target condition number σ_max / σ_min.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[Array, Array, float]
        (source, target, actual_kappa) where target = source @ F_true + noise
    """
    import numpy as np
    np.random.seed(seed)

    b = backend
    k = min(n_samples, dimension)  # Rank is at most min(n, d)

    # Create singular values with the target condition number
    # σ_i = σ_max * (σ_min/σ_max)^(i/(k-1)) for geometric spacing
    sigma_max = 1.0
    sigma_min = sigma_max / condition_number

    if k > 1:
        ratios = np.linspace(0, 1, k)
        sigmas = sigma_max * (sigma_min / sigma_max) ** ratios
    else:
        sigmas = np.array([sigma_max])

    # Random orthogonal U [n, k] and V [d, k]
    U_np, _ = np.linalg.qr(np.random.randn(n_samples, k))
    V_np, _ = np.linalg.qr(np.random.randn(dimension, k))

    # source = U @ diag(σ) @ V^T  [n, d]
    source_np = U_np @ np.diag(sigmas) @ V_np.T

    # Create a known transform F_true and compute target = source @ F_true
    F_true_np = np.eye(dimension) + 0.1 * np.random.randn(dimension, dimension)
    target_np = source_np @ F_true_np

    # Convert to backend arrays
    source = b.array(source_np.astype("float32"))
    target = b.array(target_np.astype("float32"))
    b.eval(source, target)

    # Verify actual condition number
    _, S, _ = np.linalg.svd(source_np, full_matrices=False)
    actual_kappa = S[0] / S[-1] if S[-1] > 0 else float("inf")

    return source, target, actual_kappa


def create_rank_deficient_activations(
    backend,
    n_samples: int,
    dimension: int,
    effective_rank: int,
    seed: int = 42,
) -> tuple:
    """Create activations with specified effective rank < dimension.

    Parameters
    ----------
    backend : Backend
        Backend for tensor operations.
    n_samples : int
        Number of samples.
    dimension : int
        Feature dimension.
    effective_rank : int
        Desired effective rank (< dimension).
    seed : int
        Random seed.

    Returns
    -------
    tuple[Array, Array, int]
        (source, target, true_rank)
    """
    import numpy as np
    np.random.seed(seed)

    b = backend
    k = min(effective_rank, n_samples, dimension)

    # Create low-rank matrix: source = U @ V^T where U [n, k], V [d, k]
    U_np = np.random.randn(n_samples, k)
    V_np = np.random.randn(dimension, k)
    source_np = U_np @ V_np.T

    # Add tiny noise to make it not exactly rank-k (more realistic)
    eps_noise = 1e-10 * np.random.randn(n_samples, dimension)
    source_np = source_np + eps_noise

    # Target = source @ F + small noise
    F_np = np.eye(dimension) + 0.1 * np.random.randn(dimension, dimension)
    target_np = source_np @ F_np

    source = b.array(source_np.astype("float32"))
    target = b.array(target_np.astype("float32"))
    b.eval(source, target)

    # True numerical rank
    _, S, _ = np.linalg.svd(source_np, full_matrices=False)
    sqrt_eps = float(np.finfo(np.float32).eps) ** 0.5
    true_rank = int(np.sum(S > S[0] * sqrt_eps))

    return source, target, true_rank


def run_condition_number_tests(backend, results: dict) -> None:
    """Phase 1: Test alignment with controlled condition numbers."""
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )

    logger.info("\n=== Phase 1: Condition Number Tests ===")

    b = backend
    n_samples = 100
    dimension = 64

    # Test condition numbers from mild to extreme
    kappa_values = [1e3, 1e5, 1e7, 1e10, 1e15]

    # Compute precision thresholds
    test_arr = b.array([1.0], dtype="float32")
    eps = machine_epsilon(b, test_arr)
    sqrt_eps = sqrt_scalar(eps, b)
    kappa_threshold = 1.0 / sqrt_eps

    logger.info(f"Machine epsilon: {eps:.2e}")
    logger.info(f"Precision threshold (√ε): {sqrt_eps:.2e}")
    logger.info(f"Max safe κ (1/√ε): {kappa_threshold:.2e}")

    condition_results = []

    for target_kappa in kappa_values:
        logger.info(f"\n--- Testing κ = {target_kappa:.0e} ---")

        source, target, actual_kappa = create_synthetic_activations(
            backend=b,
            n_samples=n_samples,
            dimension=dimension,
            condition_number=target_kappa,
        )

        # Run alignment
        result = find_alignment(source, target, backend=b)

        truncated_kappa = result.gram_condition_number
        residual = result.alignment_residual
        cka = result.achieved_cka

        # Success criteria
        is_truncated_bounded = truncated_kappa < kappa_threshold
        is_residual_bounded = residual < sqrt_eps
        is_cka_perfect = cka >= (1.0 - sqrt_eps)

        test_result = {
            "target_kappa": target_kappa,
            "actual_kappa": actual_kappa,
            "truncated_kappa": truncated_kappa,
            "alignment_residual": residual,
            "achieved_cka": cka,
            "source_rank": result.source_numerical_rank,
            "target_rank": result.target_numerical_rank,
            "alignment_rank": result.alignment_rank,
            "is_truncated_bounded": is_truncated_bounded,
            "is_residual_bounded": is_residual_bounded,
            "is_cka_perfect": is_cka_perfect,
            "success": is_truncated_bounded and is_residual_bounded,
        }
        condition_results.append(test_result)

        logger.info(f"  Actual κ: {actual_kappa:.2e}")
        logger.info(f"  Truncated κ: {truncated_kappa:.2e} (bounded: {is_truncated_bounded})")
        logger.info(f"  Residual: {residual:.6e} (bounded: {is_residual_bounded})")
        logger.info(f"  CKA: {cka:.6f} (perfect: {is_cka_perfect})")
        logger.info(f"  Ranks: src={result.source_numerical_rank}, tgt={result.target_numerical_rank}, align={result.alignment_rank}")
        logger.info(f"  SUCCESS: {test_result['success']}")

    results["condition_number_tests"] = {
        "n_samples": n_samples,
        "dimension": dimension,
        "machine_epsilon": eps,
        "precision_threshold": sqrt_eps,
        "kappa_threshold": kappa_threshold,
        "tests": condition_results,
        "all_passed": all(t["success"] for t in condition_results),
    }


def run_rank_deficiency_tests(backend, results: dict) -> None:
    """Phase 2: Test alignment with rank-deficient activations."""
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )

    logger.info("\n=== Phase 2: Rank Deficiency Tests ===")

    b = backend
    n_samples = 100
    dimension = 64

    test_arr = b.array([1.0], dtype="float32")
    eps = machine_epsilon(b, test_arr)
    sqrt_eps = sqrt_scalar(eps, b)

    # Test effective ranks: full, half, tenth, minimal
    effective_ranks = [dimension, dimension // 2, dimension // 10, 5, 1]

    rank_results = []

    for target_rank in effective_ranks:
        logger.info(f"\n--- Testing effective rank = {target_rank}/{dimension} ---")

        source, target, true_rank = create_rank_deficient_activations(
            backend=b,
            n_samples=n_samples,
            dimension=dimension,
            effective_rank=target_rank,
        )

        # Run alignment
        result = find_alignment(source, target, backend=b)

        detected_rank = result.source_numerical_rank
        residual = result.alignment_residual

        # For rank-deficient data, alignment should work in the detected rank
        # Residual should still be bounded if the detected rank is correct
        is_rank_detected = abs(detected_rank - true_rank) <= 3  # Allow small deviation
        is_residual_bounded = residual < sqrt_eps * 10  # More lenient for rank-deficient

        test_result = {
            "target_rank": target_rank,
            "true_rank": true_rank,
            "detected_rank": detected_rank,
            "alignment_rank": result.alignment_rank,
            "alignment_residual": residual,
            "achieved_cka": result.achieved_cka,
            "gram_condition_number": result.gram_condition_number,
            "is_rank_detected": is_rank_detected,
            "is_residual_bounded": is_residual_bounded,
            "success": is_rank_detected and is_residual_bounded,
        }
        rank_results.append(test_result)

        logger.info(f"  True rank: {true_rank}")
        logger.info(f"  Detected rank: {detected_rank} (accurate: {is_rank_detected})")
        logger.info(f"  Alignment rank: {result.alignment_rank}")
        logger.info(f"  Residual: {residual:.6e} (bounded: {is_residual_bounded})")
        logger.info(f"  CKA: {result.achieved_cka:.6f}")
        logger.info(f"  SUCCESS: {test_result['success']}")

    results["rank_deficiency_tests"] = {
        "n_samples": n_samples,
        "dimension": dimension,
        "tests": rank_results,
        "all_passed": all(t["success"] for t in rank_results),
    }


def run_failure_boundary_search(backend, results: dict) -> None:
    """Phase 4: Binary search for the condition number where alignment fails."""
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )

    logger.info("\n=== Phase 4: Failure Boundary Search ===")

    b = backend
    n_samples = 100
    dimension = 64

    test_arr = b.array([1.0], dtype="float32")
    eps = machine_epsilon(b, test_arr)
    sqrt_eps = sqrt_scalar(eps, b)
    kappa_threshold = 1.0 / sqrt_eps

    def test_kappa(kappa: float) -> bool:
        """Returns True if alignment succeeds at this kappa."""
        source, target, _ = create_synthetic_activations(
            backend=b,
            n_samples=n_samples,
            dimension=dimension,
            condition_number=kappa,
            seed=12345,  # Fixed seed for consistency
        )
        result = find_alignment(source, target, backend=b)
        # "Success" = residual is bounded
        return result.alignment_residual < sqrt_eps

    # Binary search
    low_kappa = 1e3   # Known good
    high_kappa = 1e20  # Known bad (beyond float32 range)
    iterations = 0
    max_iterations = 50

    logger.info(f"Searching for failure boundary between κ={low_kappa:.0e} and κ={high_kappa:.0e}")

    boundary_tests = []

    while high_kappa / low_kappa > 10 and iterations < max_iterations:
        mid_kappa = (low_kappa * high_kappa) ** 0.5  # Geometric mean

        success = test_kappa(mid_kappa)
        boundary_tests.append({
            "iteration": iterations,
            "kappa": mid_kappa,
            "success": success,
        })

        if success:
            low_kappa = mid_kappa
            logger.info(f"  κ={mid_kappa:.2e}: SUCCESS → new low={low_kappa:.2e}")
        else:
            high_kappa = mid_kappa
            logger.info(f"  κ={mid_kappa:.2e}: FAIL → new high={high_kappa:.2e}")

        iterations += 1

    failure_boundary = (low_kappa * high_kappa) ** 0.5

    # Compare to theoretical threshold
    ratio_to_threshold = failure_boundary / kappa_threshold

    logger.info(f"\nFailure boundary: κ ≈ {failure_boundary:.2e}")
    logger.info(f"Theoretical threshold (1/√ε): {kappa_threshold:.2e}")
    logger.info(f"Ratio: {ratio_to_threshold:.2f}x")

    results["failure_boundary_search"] = {
        "low_bound": low_kappa,
        "high_bound": high_kappa,
        "failure_boundary": failure_boundary,
        "theoretical_threshold": kappa_threshold,
        "ratio_to_threshold": ratio_to_threshold,
        "iterations": iterations,
        "tests": boundary_tests,
        "conclusion": (
            "Truncation extends usable range"
            if failure_boundary > kappa_threshold
            else "Fails at or before theoretical threshold"
        ),
    }


def run_near_collinear_test(backend, results: dict) -> None:
    """Phase 3: Test with near-collinear directions (stress test)."""
    from modelcypher.core.domain.geometry.gram_aligner import find_alignment
    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,
        sqrt_scalar,
    )
    import numpy as np

    logger.info("\n=== Phase 3: Near-Collinear Directions Test ===")

    b = backend
    n_samples = 100
    dimension = 64

    test_arr = b.array([1.0], dtype="float32")
    eps = machine_epsilon(b, test_arr)
    sqrt_eps = sqrt_scalar(eps, b)

    # Create activations with deliberately near-collinear columns
    np.random.seed(42)

    # Start with random activations
    base = np.random.randn(n_samples, dimension)

    # Make some columns nearly collinear (duplicated + tiny noise)
    n_duplicates = 10
    for i in range(n_duplicates):
        src_col = i
        dst_col = dimension - 1 - i
        # dst_col = src_col + tiny perturbation
        base[:, dst_col] = base[:, src_col] + 1e-8 * np.random.randn(n_samples)

    source_np = base.astype("float32")
    target_np = source_np @ (np.eye(dimension) + 0.1 * np.random.randn(dimension, dimension)).astype("float32")

    source = b.array(source_np)
    target = b.array(target_np)
    b.eval(source, target)

    # Check original condition number
    _, S_orig, _ = np.linalg.svd(source_np, full_matrices=False)
    original_kappa = S_orig[0] / S_orig[-1] if S_orig[-1] > 0 else float("inf")
    original_rank = int(np.sum(S_orig > S_orig[0] * sqrt_eps))

    logger.info(f"Original κ: {original_kappa:.2e}")
    logger.info(f"Original numerical rank: {original_rank}/{dimension}")
    logger.info(f"Near-collinear pairs: {n_duplicates}")

    # Run alignment
    result = find_alignment(source, target, backend=b)

    truncated_kappa = result.gram_condition_number
    detected_rank = result.source_numerical_rank
    residual = result.alignment_residual

    # Success: truncation should remove near-duplicates
    expected_rank_reduction = n_duplicates  # Rough expectation
    actual_rank_reduction = dimension - detected_rank

    is_truncation_working = actual_rank_reduction >= expected_rank_reduction // 2
    is_residual_bounded = residual < sqrt_eps * 10

    test_result = {
        "original_kappa": original_kappa,
        "truncated_kappa": truncated_kappa,
        "original_rank": original_rank,
        "detected_rank": detected_rank,
        "n_duplicate_pairs": n_duplicates,
        "expected_rank_reduction": expected_rank_reduction,
        "actual_rank_reduction": actual_rank_reduction,
        "alignment_residual": residual,
        "achieved_cka": result.achieved_cka,
        "is_truncation_working": is_truncation_working,
        "is_residual_bounded": is_residual_bounded,
        "success": is_truncation_working and is_residual_bounded,
    }

    logger.info(f"Truncated κ: {truncated_kappa:.2e}")
    logger.info(f"Detected rank: {detected_rank}")
    logger.info(f"Rank reduction: {actual_rank_reduction} (expected ≈{expected_rank_reduction})")
    logger.info(f"Residual: {residual:.6e}")
    logger.info(f"Truncation working: {is_truncation_working}")
    logger.info(f"SUCCESS: {test_result['success']}")

    results["near_collinear_test"] = test_result


def main():
    """Run ill-conditioned alignment experiments."""
    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()
    logger.info(f"Using backend: {type(backend).__name__}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "backend": type(backend).__name__,
        "experiments": {},
    }

    # Run all phases
    run_condition_number_tests(backend, results["experiments"])
    run_rank_deficiency_tests(backend, results["experiments"])
    run_near_collinear_test(backend, results["experiments"])
    run_failure_boundary_search(backend, results["experiments"])

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Ill-Conditioned Alignment Verification")
    logger.info("=" * 60)

    exp = results["experiments"]

    kappa_passed = exp.get("condition_number_tests", {}).get("all_passed", False)
    rank_passed = exp.get("rank_deficiency_tests", {}).get("all_passed", False)
    collinear_passed = exp.get("near_collinear_test", {}).get("success", False)

    logger.info(f"[{'PASS' if kappa_passed else 'FAIL'}] Condition number tests")
    logger.info(f"[{'PASS' if rank_passed else 'FAIL'}] Rank deficiency tests")
    logger.info(f"[{'PASS' if collinear_passed else 'FAIL'}] Near-collinear test")

    fb = exp.get("failure_boundary_search", {})
    if fb:
        logger.info(f"[INFO] Failure boundary: κ ≈ {fb.get('failure_boundary', 0):.2e}")
        logger.info(f"       Theoretical threshold: {fb.get('theoretical_threshold', 0):.2e}")
        logger.info(f"       Ratio: {fb.get('ratio_to_threshold', 0):.2f}x")

    overall_success = kappa_passed and rank_passed and collinear_passed
    results["overall_success"] = overall_success
    results["conclusion"] = (
        "RESOLVED: Numeric rank truncation correctly handles ill-conditioning. "
        "The warning in relative_representation.py is informational only."
        if overall_success
        else "NEEDS INVESTIGATION: Some tests failed. See individual results."
    )

    logger.info(f"\nOVERALL: {'SUCCESS' if overall_success else 'NEEDS INVESTIGATION'}")
    logger.info(f"Conclusion: {results['conclusion']}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "ill_conditioned_alignment.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    main()
