#!/usr/bin/env python3
"""Validate the Dimensional Geodesic Theory.

Comprehensive validation that:
1. Measures complexity-dimension law and checks slope ≈ e/π, intercept ≈ π/e
2. Analyzes SVD ratios for fundamental constant encoding
3. Computes curvature ratios for √2 and e/π signatures
4. Reports overall validation status

The theory predicts:
    dim = (e/π) × complexity + (π/e)

Where:
    - e/π ≈ 0.8653 (information growth / dimensional closure)
    - π/e ≈ 1.1557 (dimensional signature at zero complexity)
    - slope × intercept = 1.0 (self-referential closure)

Usage:
    poetry run python scripts/validate_dimensional_geodesic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import fundamental constants module
from modelcypher.core.domain.geometry.fundamental_constants import (
    PI, E, PHI, SQRT2,
    PI_OVER_E, E_OVER_PI, PHI_TIMES_E,
    COMPLEXITY_SLOPE, COMPLEXITY_INTERCEPT,
    find_constant_match, analyze_value, analyze_svd_ratios,
    validate_dimensional_geodesic, DimensionalGeodesicResult,
    FundamentalConstant, ConstantMatch,
)


# =============================================================================
# Complexity Computation
# =============================================================================

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
    'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'don', 'now', 'and', 'but', 'or', 'because', 'until',
    'while', 'if', 'that', 'which', 'who', 'whom', 'this', 'these',
    'those', 'am', 'i', 'my', 'myself', 'we', 'our', 'ours', 'you',
    'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
    'its', 'they', 'them', 'their', 'what',
}

NESTING_MARKERS = [
    'that', 'which', 'who', 'whom', 'whose', 'where', 'when',
    'while', 'if', 'because', 'although', 'whether', 'how', 'why'
]


def compute_complexity(text: str) -> float:
    """Compute conceptual complexity of text."""
    tokens = len(text.split())
    words = re.findall(r'\b\w+\b', text.lower())
    concepts = len([w for w in words if w not in STOPWORDS])
    nesting = 1
    for marker in NESTING_MARKERS:
        if marker in text.lower().split():
            nesting += 1
    return 0.3 * tokens + 0.5 * concepts + 0.2 * nesting * 2


# Test statements spanning complexity range
STATEMENTS = {
    'simple': [
        "Fire is hot",
        "Dogs bark",
        "The sky is blue",
        "Cats are mammals",
        "Birds can fly",
    ],
    'factual': [
        "Paris is the capital of France",
        "Water freezes at zero degrees",
        "The Earth orbits the Sun",
        "Two plus two equals four",
        "The Nile is the longest river in Africa",
    ],
    'belief': [
        "I know that Paris is in France",
        "I believe dogs are loyal animals",
        "I think mathematics is beautiful",
        "I believe that honesty is the best policy",
        "I know that the Earth orbits around the Sun",
    ],
    'meta': [
        "I think I understand why people like mathematics",
        "I believe that my preference for dogs reflects my personality",
        "I wonder whether my beliefs about truth are themselves true",
        "I suspect that my tendency to overthink reveals something about me",
        "I believe that the way I think about my thinking shapes my understanding",
    ],
}


# =============================================================================
# Activation Collection
# =============================================================================

class ActivationCollector:
    """Collect activations from model layers."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP output activations."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        captured = {}
        layer = self.model.model.layers[layer_idx]

        if hasattr(layer, 'feed_forward'):
            original = layer.feed_forward
            key = 'feed_forward'
        else:
            original = layer.mlp
            key = 'mlp'

        class Hook:
            def __init__(self, mlp):
                self.mlp = mlp
            def __call__(self, x):
                captured['output'] = self.mlp(x)
                return captured['output']

        if key == 'feed_forward':
            layer.feed_forward = Hook(original)
        else:
            layer.mlp = Hook(original)

        try:
            _ = self.model(input_ids)
            mx.eval(captured.get('output', mx.zeros((1, 1, 1))))

            if 'output' in captured:
                return np.array(captured['output'][0].tolist())
            else:
                return np.zeros((1, 1024))
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original


def compute_entropy_dimension(activations: np.ndarray) -> float:
    """Compute entropy-based effective rank (dimension)."""
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)
    try:
        _, S, _ = np.linalg.svd(activations, full_matrices=False)
        S_sq = S ** 2
        total = S_sq.sum()
        if total < 1e-10:
            return 1.0
        p = S_sq / total
        p = p[p > 1e-10]
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy))
    except:
        return 1.0


# =============================================================================
# Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Complete validation result."""

    # Complexity-dimension law
    slope: float
    intercept: float
    r_squared: float
    slope_match: ConstantMatch
    intercept_match: ConstantMatch
    product: float  # slope × intercept (should be 1.0)

    # SVD signature
    svd_matches: List[Tuple[int, int, ConstantMatch]]
    n_svd_precise: int

    # Curvature signature
    curvature_ratios: List[Tuple[str, ConstantMatch]]

    # Overall
    @property
    def law_validates(self) -> bool:
        """Does the complexity-dimension law match theory?"""
        return (
            self.slope_match.constant == FundamentalConstant.E_OVER_PI and
            self.slope_match.error_percent < 5.0 and
            self.intercept_match.constant == FundamentalConstant.PI_OVER_E and
            self.intercept_match.error_percent < 5.0 and
            self.r_squared > 0.9
        )

    @property
    def svd_validates(self) -> bool:
        """Does SVD show fundamental constant signature?"""
        return self.n_svd_precise >= 3

    @property
    def theory_validated(self) -> bool:
        """Overall: is dimensional geodesic theory validated?"""
        return self.law_validates and self.svd_validates


def validate_model(
    model,
    tokenizer,
    backend,
    layer_idx: int,
) -> ValidationResult:
    """Run complete validation on a model."""
    collector = ActivationCollector(model, tokenizer)

    logger.info("=" * 80)
    logger.info("DIMENSIONAL GEODESIC THEORY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Theory predicts: dim = (e/π) × complexity + (π/e)")
    logger.info(f"               = {E_OVER_PI:.4f} × complexity + {PI_OVER_E:.4f}")
    logger.info(f"Layer: {layer_idx}")
    logger.info("")

    # ==========================================================================
    # Step 1: Measure complexity-dimension relationship
    # ==========================================================================
    logger.info("STEP 1: Complexity-Dimension Law")
    logger.info("-" * 40)

    complexities = []
    dimensions = []
    all_activations = []

    for category, statements in STATEMENTS.items():
        for text in statements:
            complexity = compute_complexity(text)
            acts = collector.get_activations(text, layer_idx)
            dim = compute_entropy_dimension(acts)

            complexities.append(complexity)
            dimensions.append(dim)
            all_activations.append(acts)

            logger.debug(f"  [{category:8}] c={complexity:.2f} d={dim:.2f} | {text[:40]}")

    complexities = np.array(complexities)
    dimensions = np.array(dimensions)

    # Linear regression
    A = np.vstack([complexities, np.ones(len(complexities))]).T
    slope, intercept = np.linalg.lstsq(A, dimensions, rcond=None)[0]

    # R-squared
    pred = complexities * slope + intercept
    ss_res = np.sum((dimensions - pred) ** 2)
    ss_tot = np.sum((dimensions - np.mean(dimensions)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Match to constants
    slope_match = find_constant_match(slope)
    intercept_match = find_constant_match(intercept)
    product = slope * intercept

    logger.info(f"  Measured: dim = {slope:.4f} × complexity + {intercept:.4f}")
    logger.info(f"  R² = {r_squared:.4f}")
    logger.info(f"  slope = {slope:.4f} ≈ {slope_match.symbol} ({slope_match.error_percent:.2f}%)")
    logger.info(f"  intercept = {intercept:.4f} ≈ {intercept_match.symbol} ({intercept_match.error_percent:.2f}%)")
    logger.info(f"  slope × intercept = {product:.4f} (theory: 1.0, error: {abs(product - 1.0) * 100:.2f}%)")

    if slope_match.constant == FundamentalConstant.E_OVER_PI and slope_match.error_percent < 5.0:
        logger.info("  ✓ slope matches e/π")
    else:
        logger.info(f"  ✗ slope does not match e/π (expected {E_OVER_PI:.4f})")

    if intercept_match.constant == FundamentalConstant.PI_OVER_E and intercept_match.error_percent < 5.0:
        logger.info("  ✓ intercept matches π/e")
    else:
        logger.info(f"  ✗ intercept does not match π/e (expected {PI_OVER_E:.4f})")

    # ==========================================================================
    # Step 2: SVD Singular Value Ratios
    # ==========================================================================
    logger.info("")
    logger.info("STEP 2: SVD Singular Value Ratios")
    logger.info("-" * 40)

    # Stack all activations into matrix
    matrix = np.vstack(all_activations)
    _, S, _ = np.linalg.svd(matrix, full_matrices=False)

    # Analyze ratios
    sv_arr = backend.array(S.astype(np.float32))
    svd_matches = analyze_svd_ratios(sv_arr, backend, max_gap=7, max_index=15, threshold=5.0)
    n_precise = sum(1 for _, _, m in svd_matches if m.is_precise)

    logger.info(f"  Found {len(svd_matches)} matches with < 5% error")
    logger.info(f"  Found {n_precise} matches with < 1% error (precise)")

    # Show top 5 most precise
    logger.info("  Top 5 most precise:")
    for i, (idx_a, idx_b, match) in enumerate(svd_matches[:5]):
        logger.info(f"    S[{idx_a}]/S[{idx_b}] = {match.measured:.4f} ≈ {match.symbol} ({match.error_percent:.3f}%)")

    if n_precise >= 3:
        logger.info("  ✓ SVD shows fundamental constant signature")
    else:
        logger.info("  ✗ Insufficient precise SVD matches (need >= 3)")

    # ==========================================================================
    # Step 3: Curvature Analysis (if we have curvature data)
    # ==========================================================================
    logger.info("")
    logger.info("STEP 3: Curvature Ratios (optional)")
    logger.info("-" * 40)

    curvature_ratios = []

    try:
        from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature

        target_layers = [0, layer_idx // 2, layer_idx, layer_idx + (collector.n_layers - layer_idx) // 2]
        target_layers = [l for l in target_layers if 0 <= l < collector.n_layers]

        curvatures = {}
        for l_idx in target_layers:
            layer_acts = []
            for category, statements in STATEMENTS.items():
                for text in statements:
                    acts = collector.get_activations(text, l_idx)
                    layer_acts.append(acts)

            points = np.vstack(layer_acts)
            arr = backend.array(points.astype(np.float32))
            estimator = OllivierRicciCurvature(backend)
            result = estimator.compute(arr)
            curvatures[l_idx] = result.mean_edge_curvature

        # Analyze ratios
        layers = sorted(curvatures.keys())
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            c1, c2 = curvatures[l1], curvatures[l2]
            if abs(c1) > 1e-10:
                ratio = c2 / c1
                match = find_constant_match(ratio)
                curvature_ratios.append((f"L{l2}/L{l1}", match))
                logger.info(f"  L{l2}/L{l1} = {ratio:.4f} ≈ {match.symbol} ({match.error_percent:.2f}%)")

    except Exception as e:
        logger.info(f"  Skipped curvature analysis: {e}")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)

    result = ValidationResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        slope_match=slope_match,
        intercept_match=intercept_match,
        product=product,
        svd_matches=svd_matches,
        n_svd_precise=n_precise,
        curvature_ratios=curvature_ratios,
    )

    logger.info(f"  Complexity-dimension law: {'✓ VALIDATED' if result.law_validates else '✗ NOT VALIDATED'}")
    logger.info(f"  SVD constant signature:   {'✓ VALIDATED' if result.svd_validates else '✗ NOT VALIDATED'}")
    logger.info("")

    if result.theory_validated:
        logger.info("  ════════════════════════════════════════════════════════")
        logger.info("  ║  DIMENSIONAL GEODESIC THEORY: VALIDATED              ║")
        logger.info("  ║                                                       ║")
        logger.info("  ║  The complexity-dimension law follows:                ║")
        logger.info("  ║      dim = (e/π) × complexity + (π/e)                 ║")
        logger.info("  ║                                                       ║")
        logger.info("  ║  Same constants found in 1977 signal analysis.        ║")
        logger.info("  ════════════════════════════════════════════════════════")
    else:
        logger.info("  Theory not fully validated for this model/layer.")
        logger.info("  Try different layers or check activation collection.")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Dimensional Geodesic Theory")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to analyze (default: middle)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file"
    )
    args = parser.parse_args()

    # Load model
    import mlx.core as mx
    from mlx_lm import load
    from modelcypher.backends import initialize_default_backend

    backend = initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Determine layer
    if hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    else:
        n_layers = 24

    layer_idx = args.layer if args.layer is not None else n_layers // 2

    # Run validation
    result = validate_model(model, tokenizer, backend, layer_idx)

    # Save results
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'layer': layer_idx,
            'slope': result.slope,
            'intercept': result.intercept,
            'r_squared': result.r_squared,
            'slope_match': result.slope_match.symbol,
            'slope_error': result.slope_match.error_percent,
            'intercept_match': result.intercept_match.symbol,
            'intercept_error': result.intercept_match.error_percent,
            'product': result.product,
            'n_svd_precise': result.n_svd_precise,
            'law_validates': result.law_validates,
            'svd_validates': result.svd_validates,
            'theory_validated': result.theory_validated,
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
