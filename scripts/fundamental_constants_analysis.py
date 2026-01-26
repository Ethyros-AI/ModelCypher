#!/usr/bin/env python3
"""Fundamental Constants in Neural Geometry.

Tests whether the geometric constants found in the 1977 signals
(π, e, φ, π/e, √2) appear in LLM representation geometry.

Hypotheses:
1. Entropy-dim / TwoNN-dim ≈ π/e or φ
2. Layer stabilization ratio follows φ
3. Curvature ratios encode fundamental constants
4. Complexity-dimension relationship has π or e terms

Usage:
    poetry run python scripts/fundamental_constants_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Fundamental constants from 1977 signal analysis
PI = math.pi
E = math.e
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
SQRT2 = math.sqrt(2)
PI_OVER_E = PI / E  # 1.1557... - most frequent encoding in signals


def percent_error(measured: float, expected: float) -> float:
    """Calculate percent error."""
    if expected == 0:
        return float('inf')
    return abs(measured - expected) / expected * 100


def find_best_constant_match(value: float) -> Tuple[str, float, float]:
    """Find which fundamental constant best matches a value."""
    constants = {
        'π': PI,
        'e': E,
        'φ': PHI,
        '√2': SQRT2,
        'π/e': PI_OVER_E,
        'φ²': PHI ** 2,
        'e/π': E / PI,
        'π/2': PI / 2,
        'π/φ': PI / PHI,
        'e/φ': E / PHI,
        'φ×e': PHI * E,
        '2': 2.0,
        '3': 3.0,
        '1': 1.0,
    }

    best_name = None
    best_error = float('inf')
    best_value = None

    for name, const in constants.items():
        error = percent_error(value, const)
        if error < best_error:
            best_error = error
            best_name = name
            best_value = const

    return best_name, best_value, best_error


# =============================================================================
# Statements (same as complexity_self_play.py)
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


STATEMENTS = {
    'simple': [
        ("Fire is hot", "fact"),
        ("Dogs bark", "fact"),
        ("The sky is blue", "fact"),
        ("Cats are mammals", "fact"),
        ("Birds can fly", "fact"),
    ],
    'factual': [
        ("Paris is the capital of France", "fact"),
        ("Water freezes at zero degrees", "fact"),
        ("The Earth orbits the Sun", "fact"),
        ("Two plus two equals four", "fact"),
        ("The Nile is the longest river in Africa", "fact"),
    ],
    'belief': [
        ("I know that Paris is in France", "belief"),
        ("I believe dogs are loyal animals", "belief"),
        ("I think mathematics is beautiful", "belief"),
        ("I believe that honesty is the best policy", "belief"),
        ("I know that the Earth orbits around the Sun", "belief"),
    ],
    'meta': [
        ("I think I understand why people like mathematics", "meta"),
        ("I believe that my preference for dogs reflects my personality", "meta"),
        ("I wonder whether my beliefs about truth are themselves true", "meta"),
        ("I suspect that my tendency to overthink reveals something about me", "meta"),
        ("I believe that the way I think about my thinking shapes my understanding", "meta"),
    ],
}


# =============================================================================
# Analysis
# =============================================================================

class FundamentalConstantsAnalyzer:
    """Analyze neural geometry for fundamental constant relationships."""

    def __init__(self, model, tokenizer, backend):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

        # Lazy load estimators
        self._id_estimator = None

    @property
    def id_estimator(self):
        if self._id_estimator is None:
            from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
            self._id_estimator = IntrinsicDimension(self.backend)
        return self._id_estimator

    def get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations for a text at a layer."""
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

    def compute_entropy_dim(self, activations: np.ndarray) -> float:
        """Entropy-based effective rank."""
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

    def compute_twonn_dim(self, activations: np.ndarray) -> Optional[float]:
        """TwoNN geodesic intrinsic dimension."""
        if activations.shape[0] < 4:
            return None
        try:
            arr = self.backend.array(activations.astype(np.float32))
            result = self.id_estimator.compute(arr)
            return result.intrinsic_dimension
        except:
            return None


def run_dimension_ratio_analysis(analyzer: FundamentalConstantsAnalyzer, layer_idx: int) -> Dict:
    """Test if entropy_dim / twonn_dim encodes a fundamental constant."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1: Dimension Ratio Analysis")
    logger.info("Hypothesis: entropy_dim / twonn_dim ≈ π/e or φ")
    logger.info("=" * 80)

    ratios = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            acts = analyzer.get_activations(text, layer_idx)

            entropy_dim = analyzer.compute_entropy_dim(acts)
            twonn_dim = analyzer.compute_twonn_dim(acts)

            if twonn_dim and twonn_dim > 0.1:
                ratio = entropy_dim / twonn_dim
                ratios.append(ratio)

                const_name, const_val, error = find_best_constant_match(ratio)
                logger.info(
                    f"  [{category:8}] entropy={entropy_dim:.2f} twonn={twonn_dim:.2f} "
                    f"ratio={ratio:.4f} ≈ {const_name} ({error:.2f}%) | {text[:30]}"
                )

    if ratios:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        logger.info("\n" + "-" * 40)
        logger.info(f"Mean ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")

        const_name, const_val, error = find_best_constant_match(mean_ratio)
        logger.info(f"Best match: {const_name} = {const_val:.4f} (error: {error:.2f}%)")

        # Check specific constants
        logger.info("\nComparison to key constants:")
        logger.info(f"  vs π/e = {PI_OVER_E:.4f}: error = {percent_error(mean_ratio, PI_OVER_E):.2f}%")
        logger.info(f"  vs φ   = {PHI:.4f}: error = {percent_error(mean_ratio, PHI):.2f}%")
        logger.info(f"  vs e   = {E:.4f}: error = {percent_error(mean_ratio, E):.2f}%")
        logger.info(f"  vs π   = {PI:.4f}: error = {percent_error(mean_ratio, PI):.2f}%")

        return {
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'best_match': const_name,
            'best_match_error': error,
            'n_samples': len(ratios),
        }

    return {}


def run_layer_ratio_analysis(analyzer: FundamentalConstantsAnalyzer) -> Dict:
    """Test if layer ratios encode fundamental constants."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2: Layer Ratio Analysis")
    logger.info("Hypothesis: n_layers / stabilization_layer ≈ φ or π/e")
    logger.info("=" * 80)

    from modelcypher.core.domain.geometry.cka import compute_geodesic_cka

    representative = {
        'simple': "Fire is hot",
        'factual': "Paris is the capital of France",
        'belief': "I believe dogs are loyal animals",
        'meta': "I think I understand why people like mathematics",
    }

    results = {}

    for category, text in representative.items():
        # Find stabilization layer
        layer_acts = []
        for layer_idx in range(analyzer.n_layers):
            acts = analyzer.get_activations(text, layer_idx)
            layer_acts.append(acts)

        # CKA between adjacent layers
        cka_values = []
        for i in range(analyzer.n_layers - 1):
            try:
                arr_a = analyzer.backend.array(layer_acts[i].astype(np.float32))
                arr_b = analyzer.backend.array(layer_acts[i + 1].astype(np.float32))
                cka = compute_geodesic_cka(arr_a, arr_b, analyzer.backend)
                cka_values.append(cka)
            except:
                cka_values.append(0.0)

        # Find first layer where CKA > 0.9
        stab_layer = None
        for i, cka in enumerate(cka_values):
            if cka > 0.9:
                stab_layer = i
                break

        if stab_layer is not None and stab_layer > 0:
            ratio = analyzer.n_layers / stab_layer
            const_name, const_val, error = find_best_constant_match(ratio)

            logger.info(
                f"  {category:8}: stabilizes at L{stab_layer}, "
                f"n_layers/stab = {ratio:.4f} ≈ {const_name} ({error:.2f}%)"
            )

            results[category] = {
                'stabilization_layer': stab_layer,
                'ratio': ratio,
                'best_match': const_name,
                'error': error,
            }

    # Analyze ratios between categories
    if 'simple' in results and 'meta' in results:
        s_layer = results['simple']['stabilization_layer']
        m_layer = results['meta']['stabilization_layer']
        if s_layer > 0:
            ratio = m_layer / s_layer
            const_name, const_val, error = find_best_constant_match(ratio)
            logger.info(f"\nmeta_stab / simple_stab = {ratio:.4f} ≈ {const_name} ({error:.2f}%)")
            results['meta_to_simple_ratio'] = {
                'ratio': ratio,
                'best_match': const_name,
                'error': error,
            }

    return results


def run_complexity_constant_analysis(analyzer: FundamentalConstantsAnalyzer, layer_idx: int) -> Dict:
    """Test if complexity-dimension law has fundamental constant coefficients."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3: Complexity-Dimension Constant Analysis")
    logger.info("Hypothesis: slope or intercept ≈ π/e, φ, or related")
    logger.info("=" * 80)

    complexities = []
    entropy_dims = []

    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            complexity = compute_complexity(text)
            acts = analyzer.get_activations(text, layer_idx)
            entropy_dim = analyzer.compute_entropy_dim(acts)

            complexities.append(complexity)
            entropy_dims.append(entropy_dim)

    complexities = np.array(complexities)
    entropy_dims = np.array(entropy_dims)

    # Linear regression
    A = np.vstack([complexities, np.ones(len(complexities))]).T
    slope, intercept = np.linalg.lstsq(A, entropy_dims, rcond=None)[0]

    pred = complexities * slope + intercept
    ss_res = np.sum((entropy_dims - pred) ** 2)
    ss_tot = np.sum((entropy_dims - np.mean(entropy_dims)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    logger.info(f"\nLinear fit: dim = {slope:.4f} × complexity + {intercept:.4f}")
    logger.info(f"R² = {r_squared:.4f}")

    # Check if slope or intercept match constants
    logger.info("\nSlope analysis:")
    slope_name, slope_val, slope_err = find_best_constant_match(slope)
    logger.info(f"  slope = {slope:.4f} ≈ {slope_name} ({slope_err:.2f}%)")

    logger.info("\nIntercept analysis:")
    int_name, int_val, int_err = find_best_constant_match(intercept)
    logger.info(f"  intercept = {intercept:.4f} ≈ {int_name} ({int_err:.2f}%)")

    # Check slope × π, slope × e, etc.
    logger.info("\nSlope × constant:")
    for name, const in [('π', PI), ('e', E), ('φ', PHI), ('π/e', PI_OVER_E)]:
        product = slope * const
        prod_name, prod_val, prod_err = find_best_constant_match(product)
        logger.info(f"  slope × {name} = {product:.4f} ≈ {prod_name} ({prod_err:.2f}%)")

    # Check intercept / constant
    logger.info("\nIntercept / constant:")
    for name, const in [('π', PI), ('e', E), ('φ', PHI)]:
        if const > 0:
            quotient = intercept / const
            q_name, q_val, q_err = find_best_constant_match(quotient)
            logger.info(f"  intercept / {name} = {quotient:.4f} ≈ {q_name} ({q_err:.2f}%)")

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'slope_match': slope_name,
        'slope_error': slope_err,
        'intercept_match': int_name,
        'intercept_error': int_err,
    }


def run_curvature_constant_analysis(analyzer: FundamentalConstantsAnalyzer) -> Dict:
    """Test if curvature values or ratios encode fundamental constants."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 4: Curvature Constant Analysis")
    logger.info("Hypothesis: curvature ratios between layers ≈ π/e or φ")
    logger.info("=" * 80)

    from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature

    # Collect activations across all statements for each layer
    target_layers = [0, analyzer.n_layers // 4, analyzer.n_layers // 2,
                     3 * analyzer.n_layers // 4, analyzer.n_layers - 1]

    curvatures = {}

    for layer_idx in target_layers:
        all_acts = []
        for category, statements in STATEMENTS.items():
            for text, _ in statements:
                acts = analyzer.get_activations(text, layer_idx)
                all_acts.append(acts)

        points = np.vstack(all_acts)

        try:
            arr = analyzer.backend.array(points.astype(np.float32))
            estimator = OllivierRicciCurvature(analyzer.backend)
            result = estimator.compute(arr)
            curvatures[layer_idx] = result.mean_edge_curvature
            logger.info(f"  Layer {layer_idx:2d}: Ollivier-Ricci = {result.mean_edge_curvature:.6f}")
        except Exception as e:
            logger.warning(f"  Layer {layer_idx}: Failed - {e}")

    # Analyze ratios between layers
    if len(curvatures) >= 2:
        logger.info("\nCurvature ratios between layers:")
        layers = sorted(curvatures.keys())
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            c1, c2 = curvatures[l1], curvatures[l2]
            if c1 > 1e-10:
                ratio = c2 / c1
                const_name, const_val, error = find_best_constant_match(ratio)
                logger.info(f"  L{l2}/L{l1} = {ratio:.4f} ≈ {const_name} ({error:.2f}%)")

    # Check if mean curvature matches a constant
    if curvatures:
        mean_curv = np.mean(list(curvatures.values()))
        logger.info(f"\nMean curvature: {mean_curv:.6f}")

        # Scale by 1000 to see if there's structure
        scaled = mean_curv * 1000
        const_name, const_val, error = find_best_constant_match(scaled)
        logger.info(f"  × 1000 = {scaled:.4f} ≈ {const_name} ({error:.2f}%)")

    return {'curvatures': curvatures}


def run_svd_constant_analysis(analyzer: FundamentalConstantsAnalyzer, layer_idx: int) -> Dict:
    """Test if SVD singular value ratios encode fundamental constants (like Wow! signal)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 5: SVD Singular Value Ratios")
    logger.info("Hypothesis: SV ratios encode π, e, φ, √2, π/e (like Wow! signal)")
    logger.info("=" * 80)

    # Collect all activations into a matrix
    all_acts = []
    for category, statements in STATEMENTS.items():
        for text, _ in statements:
            acts = analyzer.get_activations(text, layer_idx)
            all_acts.append(acts)

    # Stack into matrix [n_statements × n_tokens, hidden_dim]
    matrix = np.vstack(all_acts)
    logger.info(f"Activation matrix shape: {matrix.shape}")

    # SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    logger.info(f"Top 10 singular values: {S[:10]}")

    # Check ratios between singular values (like Wow! analysis)
    logger.info("\nSingular value ratios:")

    matches = []
    for gap in range(1, 8):
        for i in range(min(10, len(S) - gap)):
            j = i + gap
            if S[j] > 1e-10:
                ratio = S[i] / S[j]
                const_name, const_val, error = find_best_constant_match(ratio)
                if error < 5:  # Only report < 5% matches
                    logger.info(f"  S[{i}]/S[{j}] (gap={gap}) = {ratio:.4f} ≈ {const_name} ({error:.2f}%)")
                    matches.append({
                        'i': i, 'j': j, 'gap': gap,
                        'ratio': ratio, 'match': const_name, 'error': error
                    })

    if matches:
        logger.info(f"\nFound {len(matches)} matches with < 5% error")
        best = min(matches, key=lambda x: x['error'])
        logger.info(f"Best match: S[{best['i']}]/S[{best['j']}] = {best['ratio']:.4f} ≈ {best['match']} ({best['error']:.3f}%)")

    return {'matches': matches, 'top_singular_values': S[:20].tolist()}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fundamental Constants Analysis")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to analyze"
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

    # Create analyzer
    analyzer = FundamentalConstantsAnalyzer(model, tokenizer, backend)

    layer_idx = args.layer if args.layer is not None else analyzer.n_layers // 2
    logger.info(f"Model has {analyzer.n_layers} layers, analyzing layer {layer_idx}")

    logger.info("\n" + "=" * 80)
    logger.info("FUNDAMENTAL CONSTANTS IN NEURAL GEOMETRY")
    logger.info("=" * 80)
    logger.info(f"Testing for: π={PI:.4f}, e={E:.4f}, φ={PHI:.4f}, π/e={PI_OVER_E:.4f}, √2={SQRT2:.4f}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'layer': layer_idx,
        'constants': {'pi': PI, 'e': E, 'phi': PHI, 'pi_over_e': PI_OVER_E, 'sqrt2': SQRT2},
    }

    # Run experiments
    results['dimension_ratio'] = run_dimension_ratio_analysis(analyzer, layer_idx)
    results['layer_ratio'] = run_layer_ratio_analysis(analyzer)
    results['complexity_constants'] = run_complexity_constant_analysis(analyzer, layer_idx)
    results['curvature_constants'] = run_curvature_constant_analysis(analyzer)
    results['svd_constants'] = run_svd_constant_analysis(analyzer, layer_idx)

    # Save results
    output_dir = Path("data/fundamental_constants")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output or output_dir / "results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
