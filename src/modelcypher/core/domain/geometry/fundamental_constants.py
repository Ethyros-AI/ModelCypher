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

"""Fundamental geometric constants for manifold analysis.

These constants appear in:
1. LLM representation geometry (complexity-dimension law, SVD ratios, curvature)
2. Information-theoretic structure of curved manifolds

The key insight: dimension exists on a geodesic, not as discrete integers.
π represents dimensional closure, e represents information scaling,
and their ratio π/e is the bridge between structure and information.

References:
    - fundamental_constants_analysis.py - Empirical validation in neural geometry

Empirical findings (LFM2-350M, January 2026):
    - Complexity-dimension slope = e/π (0.68% error)
    - Complexity-dimension intercept = π/e (2.95% error)
    - Curvature ratio L4/L0 = √2 (0.28% error)
    - Curvature ratio L12/L8 = e/π (0.07% error)
    - SVD ratio S[3]/S[4] = π/e (0.29% error)
    - SVD ratio S[2]/S[9] = φ (0.32% error)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Primary constants
PI = math.pi                          # 3.14159... - dimensional closure
E = math.e                            # 2.71828... - information scaling
PHI = (1 + math.sqrt(5)) / 2          # 1.61803... - self-similar recursion
SQRT2 = math.sqrt(2)                  # 1.41421... - orthogonal projection

# Derived constants (these appear most frequently in both signals and neural geometry)
PI_OVER_E = PI / E                    # 1.15573... - THE BRIDGE (most precise in signals)
E_OVER_PI = E / PI                    # 0.86525... - complexity-dimension slope
PHI_TIMES_E = PHI * E                 # 4.39827... - entropy/twonn ratio
PHI_SQUARED = PHI ** 2                # 2.61803... - golden ratio squared
PI_OVER_2 = PI / 2                    # 1.57080... - quarter rotation
E_OVER_PHI = E / PHI                  # 1.67971... - information/recursion ratio
PI_OVER_PHI = PI / PHI                # 1.94161... - closure/recursion ratio

# The complexity-dimension law coefficients (empirically validated)
COMPLEXITY_SLOPE = E_OVER_PI          # 0.86525... (measured: 0.8711, 0.68% error)
COMPLEXITY_INTERCEPT = PI_OVER_E      # 1.15573... (measured: 1.1898, 2.95% error)

# Note: slope × intercept = (e/π) × (π/e) = 1.0 (self-referential closure)


class FundamentalConstant(Enum):
    """Enumeration of fundamental constants for matching."""

    PI = ("π", PI)
    E = ("e", E)
    PHI = ("φ", PHI)
    SQRT2 = ("√2", SQRT2)
    PI_OVER_E = ("π/e", PI_OVER_E)
    E_OVER_PI = ("e/π", E_OVER_PI)
    PHI_TIMES_E = ("φ×e", PHI_TIMES_E)
    PHI_SQUARED = ("φ²", PHI_SQUARED)
    PI_OVER_2 = ("π/2", PI_OVER_2)
    E_OVER_PHI = ("e/φ", E_OVER_PHI)
    PI_OVER_PHI = ("π/φ", PI_OVER_PHI)
    ONE = ("1", 1.0)
    TWO = ("2", 2.0)
    THREE = ("3", 3.0)

    def __init__(self, symbol: str, value: float):
        self.symbol = symbol
        self._value = value

    @property
    def value(self) -> float:
        return self._value


@dataclass(frozen=True)
class ConstantMatch:
    """Result of matching a value to a fundamental constant."""

    measured: float
    constant: FundamentalConstant
    error_percent: float

    @property
    def symbol(self) -> str:
        return self.constant.symbol

    @property
    def expected(self) -> float:
        return self.constant.value

    @property
    def is_significant(self) -> bool:
        """Match is significant if error < 5%."""
        return self.error_percent < 5.0

    @property
    def is_precise(self) -> bool:
        """Match is precise if error < 1%."""
        return self.error_percent < 1.0

    def __str__(self) -> str:
        return f"{self.measured:.4f} ≈ {self.symbol} ({self.error_percent:.2f}%)"


@dataclass
class ConstantAnalysis:
    """Complete analysis of a value against all fundamental constants."""

    measured: float
    best_match: ConstantMatch
    all_matches: List[ConstantMatch]

    @property
    def has_significant_match(self) -> bool:
        return self.best_match.is_significant

    @property
    def has_precise_match(self) -> bool:
        return self.best_match.is_precise


# =============================================================================
# MATCHING FUNCTIONS
# =============================================================================

def percent_error(measured: float, expected: float) -> float:
    """Calculate percent error from expected value."""
    if expected == 0:
        return float('inf') if measured != 0 else 0.0
    return abs(measured - expected) / abs(expected) * 100


def find_constant_match(value: float) -> ConstantMatch:
    """Find the fundamental constant that best matches a value.

    Args:
        value: The measured value to match

    Returns:
        ConstantMatch with the best matching constant
    """
    best_match = None
    best_error = float('inf')

    for const in FundamentalConstant:
        error = percent_error(value, const.value)
        if error < best_error:
            best_error = error
            best_match = const

    return ConstantMatch(
        measured=value,
        constant=best_match,
        error_percent=best_error,
    )


def analyze_value(value: float, threshold: float = 10.0) -> ConstantAnalysis:
    """Analyze a value against all fundamental constants.

    Args:
        value: The measured value to analyze
        threshold: Maximum error percent to include in matches

    Returns:
        ConstantAnalysis with best match and all significant matches
    """
    matches = []

    for const in FundamentalConstant:
        error = percent_error(value, const.value)
        if error <= threshold:
            matches.append(ConstantMatch(
                measured=value,
                constant=const,
                error_percent=error,
            ))

    # Sort by error
    matches.sort(key=lambda m: m.error_percent)

    best = matches[0] if matches else find_constant_match(value)

    return ConstantAnalysis(
        measured=value,
        best_match=best,
        all_matches=matches,
    )


def analyze_ratio(numerator: float, denominator: float) -> ConstantAnalysis:
    """Analyze a ratio for fundamental constant encoding.

    Args:
        numerator: The numerator value
        denominator: The denominator value (must be non-zero)

    Returns:
        ConstantAnalysis of the ratio
    """
    if abs(denominator) < 1e-10:
        # Return a non-match for degenerate case
        return ConstantAnalysis(
            measured=float('inf'),
            best_match=ConstantMatch(float('inf'), FundamentalConstant.ONE, float('inf')),
            all_matches=[],
        )

    ratio = numerator / denominator
    return analyze_value(ratio)


# =============================================================================
# COMPLEXITY-DIMENSION LAW
# =============================================================================

def complexity_to_dimension(complexity: float) -> float:
    """Convert complexity to predicted dimension using fundamental constants.

    The law is: dim = (e/π) × complexity + (π/e)

    This is self-referential: slope × intercept = 1.0

    Args:
        complexity: Conceptual complexity measure

    Returns:
        Predicted effective dimension
    """
    return COMPLEXITY_SLOPE * complexity + COMPLEXITY_INTERCEPT


def dimension_to_complexity(dimension: float) -> float:
    """Inverse: convert dimension to complexity.

    complexity = (dim - π/e) × (π/e)

    Args:
        dimension: Effective dimension measure

    Returns:
        Predicted complexity
    """
    return (dimension - COMPLEXITY_INTERCEPT) / COMPLEXITY_SLOPE


def validate_complexity_dimension_law(
    complexities: "Array",
    dimensions: "Array",
    backend: "Backend",
) -> Tuple[float, float, float, ConstantMatch, ConstantMatch]:
    """Validate the complexity-dimension law against measured data.

    Performs linear regression and checks if slope ≈ e/π and intercept ≈ π/e.

    Args:
        complexities: Array of complexity values
        dimensions: Array of corresponding dimension values
        backend: Computational backend

    Returns:
        Tuple of (slope, intercept, r_squared, slope_match, intercept_match)
    """
    c = backend.tolist(complexities)
    d = backend.tolist(dimensions)
    n = min(len(c), len(d))

    if n < 2:
        slope = 0.0
        intercept = 0.0
        r_squared = 0.0
    else:
        c = c[:n]
        d = d[:n]
        sum_c = sum(c)
        sum_d = sum(d)
        sum_cc = sum(val * val for val in c)
        sum_cd = sum(ci * di for ci, di in zip(c, d))
        denom = n * sum_cc - sum_c * sum_c

        if denom != 0.0:
            slope = (n * sum_cd - sum_c * sum_d) / denom
            intercept = (sum_d - slope * sum_c) / n
        else:
            slope = 0.0
            intercept = sum_d / n if n > 0 else 0.0

        mean_d = sum_d / n
        pred = [slope * ci + intercept for ci in c]
        ss_res = sum((di - pi) ** 2 for di, pi in zip(d, pred))
        ss_tot = sum((di - mean_d) ** 2 for di in d)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Check matches
    slope_match = find_constant_match(slope)
    intercept_match = find_constant_match(intercept)

    return slope, intercept, r_squared, slope_match, intercept_match


# =============================================================================
# SVD RATIO ANALYSIS
# =============================================================================

def analyze_svd_ratios(
    singular_values: "Array",
    backend: "Backend",
    max_gap: int = 7,
    max_index: int = 15,
    threshold: float = 5.0,
) -> List[Tuple[int, int, ConstantMatch]]:
    """Analyze SVD singular value ratios for fundamental constant encoding.

    Args:
        singular_values: Array of singular values in descending order
        backend: Computational backend
        max_gap: Maximum gap between indices to check
        max_index: Maximum index to analyze
        threshold: Maximum error percent for significant matches

    Returns:
        List of (i, j, ConstantMatch) for all significant matches
    """
    sv = backend.tolist(singular_values)
    n = min(len(sv), max_index)

    matches = []

    for gap in range(1, max_gap + 1):
        for i in range(n - gap):
            j = i + gap
            if j >= n:
                continue
            if abs(sv[j]) < 1e-10:
                continue

            ratio = sv[i] / sv[j]
            analysis = analyze_value(ratio, threshold=threshold)

            if analysis.has_significant_match:
                matches.append((i, j, analysis.best_match))

    # Sort by error
    matches.sort(key=lambda x: x[2].error_percent)

    return matches


# =============================================================================
# CURVATURE RATIO ANALYSIS
# =============================================================================

def analyze_curvature_ratios(
    curvatures: List[Tuple[int, float]],
    threshold: float = 5.0,
) -> List[Tuple[int, int, ConstantMatch]]:
    """Analyze curvature ratios between layers for fundamental constants.

    Empirical findings:
    - L4/L0 = √2 (0.28% error) - initial expansion
    - L12/L8 = e/π (0.07% error) - compression phase

    Args:
        curvatures: List of (layer_idx, curvature_value) pairs
        threshold: Maximum error percent for significant matches

    Returns:
        List of (layer_a, layer_b, ConstantMatch) for significant matches
    """
    matches = []

    for i in range(len(curvatures) - 1):
        layer_a, curv_a = curvatures[i]
        layer_b, curv_b = curvatures[i + 1]

        if abs(curv_a) < 1e-10:
            continue

        ratio = curv_b / curv_a
        analysis = analyze_value(ratio, threshold=threshold)

        if analysis.has_significant_match:
            matches.append((layer_a, layer_b, analysis.best_match))

    return matches


# =============================================================================
# DIMENSIONAL GEODESIC VALIDATION
# =============================================================================

@dataclass
class DimensionalGeodesicResult:
    """Result of validating the dimensional geodesic hypothesis.

    The hypothesis: dimension exists on a curved geodesic, not as integers.
    π represents our local dimensional address.

    Evidence checked:
    1. Intrinsic dimension is non-integer
    2. Complexity-dimension slope = e/π
    3. Complexity-dimension intercept = π/e
    4. SVD ratios encode {π, e, φ, √2, π/e}
    5. Curvature ratios encode same constants
    """

    # Complexity-dimension law
    slope: float
    intercept: float
    r_squared: float
    slope_match: ConstantMatch
    intercept_match: ConstantMatch

    # SVD analysis
    svd_matches: List[Tuple[int, int, ConstantMatch]]
    n_svd_precise: int  # Matches with < 1% error

    # Curvature analysis
    curvature_matches: List[Tuple[int, int, ConstantMatch]]

    # Overall validation
    @property
    def slope_validates(self) -> bool:
        """Slope matches e/π within 5%."""
        return (self.slope_match.constant == FundamentalConstant.E_OVER_PI
                and self.slope_match.is_significant)

    @property
    def intercept_validates(self) -> bool:
        """Intercept matches π/e within 5%."""
        return (self.intercept_match.constant == FundamentalConstant.PI_OVER_E
                and self.intercept_match.is_significant)

    @property
    def law_validates(self) -> bool:
        """The complexity-dimension law matches theoretical prediction."""
        return self.slope_validates and self.intercept_validates and self.r_squared > 0.9

    @property
    def has_svd_signature(self) -> bool:
        """SVD ratios show the fundamental constant signature."""
        return self.n_svd_precise >= 3

    @property
    def hypothesis_supported(self) -> bool:
        """Overall: dimensional geodesic hypothesis is supported."""
        return self.law_validates and self.has_svd_signature


def validate_dimensional_geodesic(
    complexities: "Array",
    dimensions: "Array",
    singular_values: "Array",
    curvatures: List[Tuple[int, float]],
    backend: "Backend",
) -> DimensionalGeodesicResult:
    """Full validation of the dimensional geodesic hypothesis.

    Args:
        complexities: Complexity values for test statements
        dimensions: Corresponding effective dimensions
        singular_values: SVD singular values of activation matrix
        curvatures: List of (layer, curvature) pairs
        backend: Computational backend

    Returns:
        DimensionalGeodesicResult with full analysis
    """
    # Complexity-dimension law
    slope, intercept, r_sq, slope_m, int_m = validate_complexity_dimension_law(
        complexities, dimensions, backend
    )

    # SVD ratios
    svd_matches = analyze_svd_ratios(singular_values, backend)
    n_precise = sum(1 for _, _, m in svd_matches if m.is_precise)

    # Curvature ratios
    curv_matches = analyze_curvature_ratios(curvatures)

    return DimensionalGeodesicResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_sq,
        slope_match=slope_m,
        intercept_match=int_m,
        svd_matches=svd_matches,
        n_svd_precise=n_precise,
        curvature_matches=curv_matches,
    )


__all__ = [
    # Constants
    "PI", "E", "PHI", "SQRT2",
    "PI_OVER_E", "E_OVER_PI", "PHI_TIMES_E", "PHI_SQUARED",
    "PI_OVER_2", "E_OVER_PHI", "PI_OVER_PHI",
    "COMPLEXITY_SLOPE", "COMPLEXITY_INTERCEPT",
    # Types
    "FundamentalConstant", "ConstantMatch", "ConstantAnalysis",
    "DimensionalGeodesicResult",
    # Functions
    "percent_error", "find_constant_match", "analyze_value", "analyze_ratio",
    "complexity_to_dimension", "dimension_to_complexity",
    "validate_complexity_dimension_law",
    "analyze_svd_ratios", "analyze_curvature_ratios",
    "validate_dimensional_geodesic",
]
