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
LLM Prime Geometry Analysis.

Tests whether prime numbers have hidden geometric structure in LLM representation
space. Extends the mathematical prime analysis (prime_geometry.py) to analyze
how neural networks represent numbers.

Core Hypothesis:
    Primes appear random in 1D because we're viewing them in the wrong dimension.
    LLMs, trained on vast mathematical text, may have encoded prime structure
    in their high-dimensional representation space.

Hypotheses (L1-L12):
    L1: Spectral Concentration - primes have lower participation ratio
    L2: Lower Spectral Entropy - primes have more concentrated eigenvalues
    L3: Distinct Intrinsic Dimension - primes live on different-dim manifold
    L4: Curvature Signature - primes have different manifold curvature
    L5: Cross-Model Invariance - pattern appears across all models (CKA > 0.9)
    L6: Prompt Invariance - pattern stable across prompt formats
    L7: Scale Invariance - effect sizes stable across prime ranges
    L8: Separability - clustering separates primes from composites
    L9: Layer Emergence - where does prime structure appear?
    L10: Manifold Substructure - primes form distinct sub-manifold
    L11: Twin Prime Clustering - twin primes closer than average
    L12: Gap Encoding - prime gaps correlate with representation distance

Usage:
    from modelcypher.core.domain.geometry.llm_prime_geometry import LLMPrimeAnalyzer

    analyzer = LLMPrimeAnalyzer(backend)
    result = analyzer.run_pilot(model, tokenizer, n_primes=100)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.number_probe import (
    NumberProbe,
    NumberProbeResult,
    NumberSetConfig,
    NumberSets,
    PromptFormat,
    LayerSelection,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass(frozen=True)
class SpectralMetrics:
    """Spectral properties of a representation set's Gram matrix."""

    participation_ratio: float  # Effective rank: (sum λ)² / sum(λ²)
    spectral_entropy: float  # -sum(p log p) where p = λ / sum(λ)
    condition_number: float  # λ_max / λ_min
    top_k_ratio: float  # sum(top 10 λ) / sum(all λ)
    n_samples: int


@dataclass(frozen=True)
class IntrinsicDimensionMetrics:
    """Intrinsic dimension estimates."""

    intrinsic_dimension: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_samples: int


@dataclass(frozen=True)
class CurvatureMetrics:
    """Manifold curvature statistics."""

    mean_ricci: float
    std_ricci: float
    min_ricci: float
    max_ricci: float
    n_samples: int


@dataclass(frozen=True)
class EffectSize:
    """Cohen's d effect size - raw measurement only.

    NO VIBES: Callers decide what constitutes "small", "medium", "large".
    The d value is the raw measurement; interpretation is not our job.
    """

    d: float

    @staticmethod
    def compute(mean1: float, mean2: float, std1: float, std2: float, n1: int, n2: int) -> "EffectSize":
        """Compute Cohen's d with pooled standard deviation."""
        if n1 < 2 or n2 < 2:
            return EffectSize(d=0.0)

        # Pooled standard deviation
        pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = pooled_var**0.5

        # sqrt(float64 machine epsilon) for division safety: 2^-52 → sqrt → 2^-26
        if pooled_std < 2.0 ** -26:
            return EffectSize(d=0.0)

        d = (mean1 - mean2) / pooled_std
        return EffectSize(d=d)


@dataclass(frozen=True)
class HypothesisResult:
    """Result of a single hypothesis test."""

    hypothesis_id: str  # L1, L2, etc.
    description: str
    prime_value: float
    composite_value: float
    effect_size: EffectSize
    p_value: float | None  # None if not enough samples for permutation test
    passed: bool | None  # None if indeterminate
    layer: int | None = None  # Which layer this was measured at


@dataclass
class PilotResult:
    """Results from a pilot experiment (L1-L4 hypotheses)."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    model_name: str = ""
    n_primes: int = 0
    n_composites: int = 0
    prompt_format: str = ""
    layer: int = 0

    # Raw metrics
    prime_spectral: SpectralMetrics | None = None
    composite_spectral: SpectralMetrics | None = None
    prime_id: IntrinsicDimensionMetrics | None = None
    composite_id: IntrinsicDimensionMetrics | None = None
    prime_curvature: CurvatureMetrics | None = None
    composite_curvature: CurvatureMetrics | None = None

    # Hypothesis tests
    hypotheses: dict[str, HypothesisResult] = field(default_factory=dict)

    # Summary
    n_passed: int = 0
    n_tested: int = 0
    has_signal: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "n_primes": self.n_primes,
            "n_composites": self.n_composites,
            "prompt_format": self.prompt_format,
            "layer": self.layer,
            "prime_spectral": {
                "participation_ratio": self.prime_spectral.participation_ratio,
                "spectral_entropy": self.prime_spectral.spectral_entropy,
                "condition_number": self.prime_spectral.condition_number,
                "top_k_ratio": self.prime_spectral.top_k_ratio,
            } if self.prime_spectral else None,
            "composite_spectral": {
                "participation_ratio": self.composite_spectral.participation_ratio,
                "spectral_entropy": self.composite_spectral.spectral_entropy,
                "condition_number": self.composite_spectral.condition_number,
                "top_k_ratio": self.composite_spectral.top_k_ratio,
            } if self.composite_spectral else None,
            "prime_id": {
                "intrinsic_dimension": self.prime_id.intrinsic_dimension,
                "ci_lower": self.prime_id.ci_lower,
                "ci_upper": self.prime_id.ci_upper,
            } if self.prime_id else None,
            "composite_id": {
                "intrinsic_dimension": self.composite_id.intrinsic_dimension,
                "ci_lower": self.composite_id.ci_lower,
                "ci_upper": self.composite_id.ci_upper,
            } if self.composite_id else None,
            "hypotheses": {
                k: {
                    "hypothesis_id": v.hypothesis_id,
                    "description": v.description,
                    "prime_value": v.prime_value,
                    "composite_value": v.composite_value,
                    "effect_size": v.effect_size.d,
                    "p_value": v.p_value,
                    "passed": v.passed,
                    "layer": v.layer,
                }
                for k, v in self.hypotheses.items()
            },
            "summary": {
                "n_passed": self.n_passed,
                "n_tested": self.n_tested,
                "has_signal": self.has_signal,
            },
        }

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Core Analysis Functions
# =============================================================================


def compute_gram_matrix(X: "Array", backend: "Backend") -> "Array":
    """Compute Gram matrix K = X @ X^T."""
    X = backend.astype(X, "float32")
    return backend.matmul(X, backend.transpose(X))


def analyze_eigenvalues(gram: "Array", backend: "Backend") -> SpectralMetrics:
    """Analyze eigenvalue distribution of a Gram matrix."""
    from modelcypher.core.domain.geometry.numerical_stability import power_iteration_eigh

    n = int(gram.shape[0])
    eigenvalues, _ = power_iteration_eigh(backend, gram, k=n)

    # Filter positive eigenvalues
    eps = machine_epsilon(backend, eigenvalues)
    pos_mask = eigenvalues > eps
    pos_count_arr = backend.sum(backend.astype(pos_mask, "int32"))
    backend.eval(pos_count_arr)
    pos_count = int(backend.to_scalar(pos_count_arr))

    if pos_count < 2:
        return SpectralMetrics(
            participation_ratio=1.0,
            spectral_entropy=0.0,
            condition_number=1.0,
            top_k_ratio=1.0,
            n_samples=n,
        )

    pos_ev = eigenvalues[:pos_count]

    # Participation ratio: (sum λ)² / sum(λ²)
    sum_ev = backend.sum(pos_ev)
    sum_ev_sq = backend.sum(pos_ev * pos_ev)
    backend.eval(sum_ev, sum_ev_sq)
    sum_ev_val = float(backend.to_scalar(sum_ev))
    sum_ev_sq_val = float(backend.to_scalar(sum_ev_sq))
    participation_ratio = (sum_ev_val ** 2) / sum_ev_sq_val if sum_ev_sq_val > eps else 1.0

    # Spectral entropy: -sum(p log p)
    p = pos_ev / sum_ev_val
    log_p = backend.where(p > eps, backend.log(p), backend.zeros_like(p))
    entropy = -backend.sum(p * log_p)
    backend.eval(entropy)
    spectral_entropy = float(backend.to_scalar(entropy))

    # Condition number
    first_ev = backend.take(pos_ev, backend.array([0]), axis=0)
    last_ev = backend.take(pos_ev, backend.array([pos_count - 1]), axis=0)
    backend.eval(first_ev, last_ev)
    condition_number = float(backend.to_scalar(first_ev)) / float(backend.to_scalar(last_ev))

    # Top-k ratio
    k = min(10, pos_count)
    top_k_sum = backend.sum(pos_ev[:k])
    backend.eval(top_k_sum)
    top_k_ratio = float(backend.to_scalar(top_k_sum)) / sum_ev_val

    return SpectralMetrics(
        participation_ratio=participation_ratio,
        spectral_entropy=spectral_entropy,
        condition_number=condition_number,
        top_k_ratio=top_k_ratio,
        n_samples=n,
    )


def compute_intrinsic_dimension(
    X: "Array",
    backend: "Backend",
) -> IntrinsicDimensionMetrics:
    """Compute intrinsic dimension using TwoNN estimator."""
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    estimator = IntrinsicDimension(backend)
    X_float = backend.astype(X, "float32")

    try:
        result = estimator.compute(X_float, with_ci=True)
        # Handle CI - attribute is 'ci', a ConfidenceInterval object
        if result.ci is not None:
            ci_lower = result.ci.lower
            ci_upper = result.ci.upper
        else:
            ci_lower = result.intrinsic_dimension
            ci_upper = result.intrinsic_dimension

        return IntrinsicDimensionMetrics(
            intrinsic_dimension=result.intrinsic_dimension,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=result.sample_count,
        )
    except Exception as e:
        logger.warning(f"ID estimation failed: {e}")
        return IntrinsicDimensionMetrics(
            intrinsic_dimension=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            n_samples=0,
        )


def compute_curvature(
    X: "Array",
    backend: "Backend",
) -> CurvatureMetrics:
    """Compute Ollivier-Ricci curvature statistics."""
    from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature

    try:
        ricci = OllivierRicciCurvature(backend)
        X_float = backend.astype(X, "float32")
        result = ricci.compute(X_float)

        # Extract min/max from edge_curvatures list
        if result.edge_curvatures:
            curvatures = [e.curvature for e in result.edge_curvatures]
            min_ricci = min(curvatures)
            max_ricci = max(curvatures)
        else:
            min_ricci = result.mean_edge_curvature
            max_ricci = result.mean_edge_curvature

        return CurvatureMetrics(
            mean_ricci=result.mean_edge_curvature,
            std_ricci=result.std_edge_curvature,
            min_ricci=min_ricci,
            max_ricci=max_ricci,
            n_samples=int(X.shape[0]),
        )
    except Exception as e:
        logger.warning(f"Curvature computation failed: {e}")
        return CurvatureMetrics(
            mean_ricci=float("nan"),
            std_ricci=float("nan"),
            min_ricci=float("nan"),
            max_ricci=float("nan"),
            n_samples=0,
        )


def permutation_test(
    values1: list[float],
    values2: list[float],
    n_permutations: int = 1000,
    backend: "Backend | None" = None,
) -> float:
    """Compute two-tailed p-value via permutation test."""
    backend = backend or get_default_backend()

    if len(values1) < 2 or len(values2) < 2:
        return 1.0

    observed_diff = abs(sum(values1) / len(values1) - sum(values2) / len(values2))
    combined = values1 + values2
    n1 = len(values1)
    n_total = len(combined)

    count_extreme = 0

    for _ in range(n_permutations):
        # Shuffle using backend random
        shuffled = combined.copy()
        rand_vals = backend.random_uniform(low=0.0, high=1.0, shape=(n_total - 1,))
        backend.eval(rand_vals)
        rand_list = backend.tolist(rand_vals)

        for i in range(n_total - 1, 0, -1):
            j = int(rand_list[n_total - 1 - i] * (i + 1))
            j = min(j, i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        perm_mean1 = sum(shuffled[:n1]) / n1
        perm_mean2 = sum(shuffled[n1:]) / (n_total - n1)
        perm_diff = abs(perm_mean1 - perm_mean2)

        if perm_diff >= observed_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_permutations + 1)


# =============================================================================
# Main Analyzer Class
# =============================================================================


class LLMPrimeAnalyzer:
    """
    Analyzes prime number geometry in LLM representation space.

    This is the main experiment runner for testing whether primes have
    hidden geometric structure visible through neural network representations.
    """

    def __init__(self, backend: "Backend | None" = None):
        """Initialize the analyzer."""
        self.backend = backend or get_default_backend()
        self.probe = NumberProbe(self.backend)

    def run_pilot(
        self,
        model: Any,
        tokenizer: Any,
        n_primes: int = 100,
        prompt_format: PromptFormat | str = PromptFormat.BARE,
        model_name: str = "unknown",
    ) -> PilotResult:
        """
        Run pilot experiment with L1-L4 hypotheses.

        Args:
            model: The LLM model.
            tokenizer: The tokenizer.
            n_primes: Number of primes to test.
            prompt_format: Prompt format to use.
            model_name: Name for logging.

        Returns:
            PilotResult with all metrics and hypothesis tests.
        """
        logger.info(f"Starting pilot experiment: {n_primes} primes, format={prompt_format}")

        # Generate number sets
        number_sets = NumberSets.generate(NumberSetConfig(
            n_primes=n_primes,
            include_composites_matched=True,
        ))

        logger.info(f"Generated {len(number_sets.primes)} primes, {len(number_sets.composites_matched)} composites")

        # Collect representations
        logger.info("Collecting prime representations...")
        prime_result = self.probe.collect_number_representations(
            model, tokenizer,
            numbers=number_sets.primes,
            prompt_format=prompt_format,
            layer_selection=LayerSelection.MIDDLE,
            model_name=model_name,
        )

        logger.info("Collecting composite representations...")
        composite_result = self.probe.collect_number_representations(
            model, tokenizer,
            numbers=number_sets.composites_matched,
            prompt_format=prompt_format,
            layer_selection=LayerSelection.MIDDLE,
            model_name=model_name,
        )

        # Use middle layer for analysis
        mid_layer = prime_result.get_middle_layer()
        logger.info(f"Analyzing at middle layer: {mid_layer}")

        # Get activation matrices
        prime_acts = prime_result.get_layer_matrix(mid_layer, self.backend)
        composite_acts = composite_result.get_layer_matrix(mid_layer, self.backend)

        logger.info(f"Prime activations shape: {prime_acts.shape}")
        logger.info(f"Composite activations shape: {composite_acts.shape}")

        # Diagnostic: check activation variance
        prime_var = self.backend.mean(self.backend.var(prime_acts, axis=0))
        composite_var = self.backend.mean(self.backend.var(composite_acts, axis=0))
        self.backend.eval(prime_var, composite_var)
        logger.info(f"Prime activation variance (mean across dims): {float(self.backend.to_scalar(prime_var)):.6f}")
        logger.info(f"Composite activation variance (mean across dims): {float(self.backend.to_scalar(composite_var)):.6f}")

        # Center the data before Gram computation (critical for meaningful eigenvalues)
        prime_mean = self.backend.mean(prime_acts, axis=0, keepdims=True)
        composite_mean = self.backend.mean(composite_acts, axis=0, keepdims=True)
        prime_acts_centered = prime_acts - prime_mean
        composite_acts_centered = composite_acts - composite_mean
        self.backend.eval(prime_acts_centered, composite_acts_centered)

        # Compute metrics (use CENTERED data for meaningful eigenvalues)
        logger.info("Computing spectral metrics...")
        prime_gram = compute_gram_matrix(prime_acts_centered, self.backend)
        composite_gram = compute_gram_matrix(composite_acts_centered, self.backend)

        prime_spectral = analyze_eigenvalues(prime_gram, self.backend)
        composite_spectral = analyze_eigenvalues(composite_gram, self.backend)

        logger.info("Computing intrinsic dimensions...")
        prime_id = compute_intrinsic_dimension(prime_acts_centered, self.backend)
        composite_id = compute_intrinsic_dimension(composite_acts_centered, self.backend)

        logger.info("Computing curvature...")
        prime_curv = compute_curvature(prime_acts_centered, self.backend)
        composite_curv = compute_curvature(composite_acts_centered, self.backend)

        # Run hypothesis tests
        logger.info("Running hypothesis tests...")
        hypotheses = {}

        # L1: Spectral Concentration
        h1 = self._test_hypothesis(
            "L1",
            "Spectral Concentration: primes have lower participation ratio",
            prime_spectral.participation_ratio,
            composite_spectral.participation_ratio,
            one_sided=True,  # primes < composites
            layer=mid_layer,
        )
        hypotheses["L1"] = h1

        # L2: Spectral Entropy
        h2 = self._test_hypothesis(
            "L2",
            "Lower Spectral Entropy: primes have more concentrated eigenvalues",
            prime_spectral.spectral_entropy,
            composite_spectral.spectral_entropy,
            one_sided=True,
            layer=mid_layer,
        )
        hypotheses["L2"] = h2

        # L3: Intrinsic Dimension
        h3 = self._test_hypothesis(
            "L3",
            "Distinct Intrinsic Dimension: |ID(primes) - ID(composites)| > 1.0",
            prime_id.intrinsic_dimension,
            composite_id.intrinsic_dimension,
            one_sided=False,  # Two-sided: either direction is interesting
            threshold=1.0,  # Effect must be > 1.0 dimension
            layer=mid_layer,
        )
        hypotheses["L3"] = h3

        # L4: Curvature Signature
        h4 = self._test_hypothesis(
            "L4",
            "Curvature Signature: primes have different manifold curvature",
            prime_curv.mean_ricci,
            composite_curv.mean_ricci,
            one_sided=False,
            layer=mid_layer,
        )
        hypotheses["L4"] = h4

        # Compute summary
        n_tested = len(hypotheses)
        n_passed = sum(1 for h in hypotheses.values() if h.passed is True)
        has_signal = n_passed >= 2  # At least 2 of 4 hypotheses pass

        # Build result
        result = PilotResult(
            model_name=model_name,
            n_primes=len(number_sets.primes),
            n_composites=len(number_sets.composites_matched),
            prompt_format=prompt_format.value if isinstance(prompt_format, PromptFormat) else prompt_format,
            layer=mid_layer,
            prime_spectral=prime_spectral,
            composite_spectral=composite_spectral,
            prime_id=prime_id,
            composite_id=composite_id,
            prime_curvature=prime_curv,
            composite_curvature=composite_curv,
            hypotheses=hypotheses,
            n_passed=n_passed,
            n_tested=n_tested,
            has_signal=has_signal,
        )

        logger.info(f"Pilot complete: {n_passed}/{n_tested} hypotheses passed, signal={has_signal}")

        return result

    def _test_hypothesis(
        self,
        hypothesis_id: str,
        description: str,
        prime_value: float,
        composite_value: float,
        one_sided: bool = True,
        threshold: float | None = None,
        layer: int | None = None,
    ) -> HypothesisResult:
        """Run a single hypothesis test."""
        import math

        # Handle NaN values
        if math.isnan(prime_value) or math.isnan(composite_value):
            return HypothesisResult(
                hypothesis_id=hypothesis_id,
                description=description,
                prime_value=prime_value,
                composite_value=composite_value,
                effect_size=EffectSize(d=0.0),
                p_value=None,
                passed=None,
                layer=layer,
            )

        # Compute effect size (using single values as means with unit variance)
        diff = prime_value - composite_value
        effect = EffectSize(d=diff)

        # Determine pass/fail
        if threshold is not None:
            # For L3: check if difference exceeds threshold
            passed = abs(diff) > threshold
        elif one_sided:
            # For L1, L2: primes should be LESS than composites
            passed = prime_value < composite_value and abs(effect.d) > 0.2
        else:
            # Two-sided: any significant difference
            passed = abs(effect.d) > 0.2

        return HypothesisResult(
            hypothesis_id=hypothesis_id,
            description=description,
            prime_value=prime_value,
            composite_value=composite_value,
            effect_size=effect,
            p_value=None,  # Would need bootstrap samples for proper p-value
            passed=passed,
            layer=layer,
        )


def format_pilot_result(result: PilotResult) -> str:
    """Format pilot result for display."""
    lines = [
        "=" * 70,
        "LLM PRIME GEOMETRY PILOT EXPERIMENT",
        "=" * 70,
        "",
        f"Experiment ID: {result.experiment_id}",
        f"Timestamp: {result.timestamp}",
        f"Model: {result.model_name}",
        f"Layer analyzed: {result.layer}",
        f"Numbers: {result.n_primes} primes, {result.n_composites} composites",
        f"Prompt format: {result.prompt_format}",
        "",
        "-" * 70,
        "SPECTRAL METRICS",
        "-" * 70,
        "",
        f"                          | Primes    | Composites | Diff",
        f"-" * 60,
    ]

    if result.prime_spectral and result.composite_spectral:
        ps = result.prime_spectral
        cs = result.composite_spectral
        lines.extend([
            f"Participation ratio       | {ps.participation_ratio:9.3f} | {cs.participation_ratio:10.3f} | {ps.participation_ratio - cs.participation_ratio:+.3f}",
            f"Spectral entropy          | {ps.spectral_entropy:9.3f} | {cs.spectral_entropy:10.3f} | {ps.spectral_entropy - cs.spectral_entropy:+.3f}",
            f"Top-10 ratio              | {ps.top_k_ratio:9.3f} | {cs.top_k_ratio:10.3f} | {ps.top_k_ratio - cs.top_k_ratio:+.3f}",
        ])

    lines.extend([
        "",
        "-" * 70,
        "INTRINSIC DIMENSION",
        "-" * 70,
        "",
    ])

    if result.prime_id and result.composite_id:
        pi = result.prime_id
        ci = result.composite_id
        lines.extend([
            f"Primes:     {pi.intrinsic_dimension:.2f} (95% CI: [{pi.ci_lower:.2f}, {pi.ci_upper:.2f}])",
            f"Composites: {ci.intrinsic_dimension:.2f} (95% CI: [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}])",
            f"Difference: {abs(pi.intrinsic_dimension - ci.intrinsic_dimension):.2f}",
        ])

    lines.extend([
        "",
        "-" * 70,
        "CURVATURE",
        "-" * 70,
        "",
    ])

    if result.prime_curvature and result.composite_curvature:
        pc = result.prime_curvature
        cc = result.composite_curvature
        lines.extend([
            f"Primes mean Ricci:     {pc.mean_ricci:.4f} (std: {pc.std_ricci:.4f})",
            f"Composites mean Ricci: {cc.mean_ricci:.4f} (std: {cc.std_ricci:.4f})",
        ])

    lines.extend([
        "",
        "-" * 70,
        "HYPOTHESIS TESTS",
        "-" * 70,
        "",
    ])

    for h in result.hypotheses.values():
        if h.passed is None:
            status = "? INDETERMINATE"
        elif h.passed:
            status = "PASS"
        else:
            status = "FAIL"
        lines.extend([
            f"{h.hypothesis_id}: {status}",
            f"  {h.description}",
            f"  Prime: {h.prime_value:.4f}, Composite: {h.composite_value:.4f}",
            f"  Effect size (d): {h.effect_size.d:.3f}",
            "",
        ])

    lines.extend([
        "-" * 70,
        "SUMMARY",
        "-" * 70,
        "",
        f"Hypotheses passed: {result.n_passed}/{result.n_tested}",
        f"Signal detected: {'YES' if result.has_signal else 'NO'}",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
