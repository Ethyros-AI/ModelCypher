# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

"""
Prime Number Geometry Analysis.

Explores the hypothesis that prime number distribution has hidden geometric
structure visible through high-dimensional analysis techniques.

Mathematical Motivation:
    1. The zeros of the Riemann zeta function behave like eigenvalues of
       random Hermitian matrices (Montgomery's pair correlation conjecture).

    2. Prime distribution is encoded in the spectrum of an unknown operator.
       If we can find the right embedding, the eigenvalue statistics should
       reveal this structure.

    3. Concept relationships in neural networks are invariant across models.
       Primes provide a "pure signal" - number-theoretic structure with no
       training noise - to test if our geometric tools can detect invariants.

Approach:
    - Embed prime gaps/positions into high-dimensional space via time-delay
    - Use multiple embedding strategies: time-delay, residue classes, digit patterns
    - Compute Gram matrices (relational structure independent of coordinates)
    - Analyze eigenvalue distributions
    - Compare to multiple baselines: exponential, uniform, Poisson, Cramér model
    - Use intrinsic dimension, topological fingerprinting, and curvature
    - Apply statistical testing with bootstrap CIs and effect sizes

Hypotheses (H1-H8):
    H1: Spectral Concentration - participation_ratio(primes) < participation_ratio(random)
    H2: Lower Spectral Entropy - spectral_entropy(primes) < spectral_entropy(random)
    H3: Distinct Intrinsic Dimension - |ID(primes) - ID(random)| > 1.0
    H4: Topological Distinctiveness - betti_diff > 0 OR bottleneck/scale > 0.1
    H5: Curvature Signature - mean_ricci differs significantly
    H6: Cross-Representation Coherence - CKA(prime embeds) > CKA(random embeds)
    H7: Scale Invariance - Effect sizes stable/increase with n
    H8: Perturbation Robustness - Primes more stable under noise

References:
    - Montgomery (1973): Pair correlation of zeros of the zeta function
    - Berry & Keating (1999): The Riemann zeros and eigenvalue asymptotics
    - Facco et al. (2017): TwoNN intrinsic dimension estimation
    - Cramér (1936): Prime number theorem, probabilistic model
"""

from __future__ import annotations

from .prime_geometry_analysis import (
    analyze_prime_geometry,
    format_comprehensive_result,
    format_result,
    run_comprehensive_analysis,
    run_perturbation_study,
    run_scale_sweep,
)
from .prime_geometry_baselines import (
    generate_baseline,
    generate_cramer_model,
    generate_poisson_gaps,
    generate_random_gaps,
    generate_uniform_gaps,
    shuffled_gaps,
)
from .prime_geometry_embeddings import (
    binary_digit_embedding,
    digit_embedding,
    generate_primes,
    residue_embedding,
    time_delay_embedding,
)
from .prime_geometry_spectral import (
    analyze_eigenvalues,
    compare_distributions,
    compute_gram_matrix,
)
from .prime_geometry_stats import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    permutation_test,
    run_hypothesis_test,
)
from .prime_geometry_types import (
    BaselineType,
    ComprehensiveResult,
    ConfidenceInterval,
    EffectSize,
    EigenvalueDistribution,
    EmbeddingType,
    HypothesisTest,
    PerturbationResult,
    PrimeGeometryResult,
    PrimeSequence,
    ScaleSweepResult,
    SpectralComparison,
)

__all__ = [
    "BaselineType",
    "ComprehensiveResult",
    "ConfidenceInterval",
    "EffectSize",
    "EigenvalueDistribution",
    "EmbeddingType",
    "HypothesisTest",
    "PerturbationResult",
    "PrimeGeometryResult",
    "PrimeSequence",
    "ScaleSweepResult",
    "SpectralComparison",
    "analyze_eigenvalues",
    "analyze_prime_geometry",
    "binary_digit_embedding",
    "bootstrap_confidence_interval",
    "compare_distributions",
    "compute_cohens_d",
    "compute_gram_matrix",
    "digit_embedding",
    "format_comprehensive_result",
    "format_result",
    "generate_baseline",
    "generate_cramer_model",
    "generate_poisson_gaps",
    "generate_primes",
    "generate_random_gaps",
    "generate_uniform_gaps",
    "permutation_test",
    "residue_embedding",
    "run_comprehensive_analysis",
    "run_hypothesis_test",
    "run_perturbation_study",
    "run_scale_sweep",
    "shuffled_gaps",
    "time_delay_embedding",
]
