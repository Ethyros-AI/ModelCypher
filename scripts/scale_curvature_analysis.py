#!/usr/bin/env python3
"""Scale Curvature Analysis.

The local curvature experiment showed flat manifolds at single-statement scale.
This script aggregates activations across MANY statements to look for curvature
at scale - the hypothesis being that curvature emerges from global structure,
not local neighborhoods.

Approach:
1. Collect activations for 100+ diverse statements
2. Build a global point cloud per layer
3. Measure manifold curvature on the aggregate
4. Track curvature evolution through layers
5. Compare curvature signature by statement category

The π dimension hypothesis: if the true manifold has irrational dimension,
we'd expect to see it in the scaling behavior of dimension estimates as
sample size increases.

Usage:
    python scale_curvature_analysis.py --model /path/to/model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import json
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


# =============================================================================
# Diverse Statement Set (for aggregate analysis)
# =============================================================================

# We need diverse statements to properly sample the manifold
DIVERSE_STATEMENTS = [
    # Simple entities
    "red", "blue", "green", "hot", "cold", "big", "small", "fast", "slow",
    "water", "fire", "earth", "air", "stone", "metal", "wood", "glass",

    # Simple facts
    "Fire is hot", "Ice is cold", "Water is wet", "The sun is bright",
    "Dogs bark", "Cats meow", "Birds fly", "Fish swim",
    "The sky is blue", "Grass is green", "Snow is white", "Night is dark",

    # Compound facts
    "Paris is the capital of France", "Tokyo is in Japan",
    "The Earth orbits the Sun", "Water freezes at zero degrees",
    "Two plus two equals four", "The moon reflects sunlight",
    "Plants need water to grow", "Humans need oxygen to breathe",

    # Relational facts
    "Dogs are mammals", "Whales are not fish", "Penguins are birds",
    "The Nile is the longest river in Africa",
    "Mount Everest is the tallest mountain",
    "The Pacific is the largest ocean",

    # Beliefs
    "I know that Paris is in France", "I believe dogs are loyal",
    "I think mathematics is beautiful", "I feel happy today",
    "I believe honesty is important", "I know the Earth is round",

    # Opinions
    "Pizza is delicious", "Summer is the best season",
    "Music brings joy", "Reading is relaxing",
    "Nature is beautiful", "Kindness matters",

    # Meta-cognitive
    "I think I understand why people like art",
    "I wonder if my beliefs are correct",
    "I believe my thoughts shape my reality",
    "I suspect that learning never ends",
    "I know that I know nothing",

    # Abstract concepts
    "Time flows forward", "Change is constant",
    "Truth is relative", "Beauty is subjective",
    "Existence precedes essence", "Meaning is constructed",

    # Counterfactuals for contrast
    "Fire is cold", "Ice is hot", "The sky is green",
    "Dogs meow", "Cats bark", "Fish fly",
    "Two plus two equals five", "Paris is in Germany",

    # Complex nested
    "The theory suggests that consciousness emerges from complexity",
    "Scientists believe that the universe is expanding",
    "Philosophers argue that free will is an illusion",
    "Research indicates that memory is reconstructive",
]


# =============================================================================
# Activation Collection
# =============================================================================

class ActivationCollector:
    """Collect activations from model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_activations(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP activations for all tokens."""
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


# =============================================================================
# Scale Analysis
# =============================================================================

@dataclass
class ScaleCurvatureResult:
    """Result of curvature analysis at scale."""
    layer_idx: int
    n_statements: int
    n_tokens: int

    # Global curvature
    mean_curvature: float
    curvature_variance: float
    curvature_sign: str

    # Dimension scaling
    dimension_at_10: float
    dimension_at_50: float
    dimension_at_100: float
    dimension_scaling_exponent: float  # How dimension grows with samples

    # Per-category breakdown (if computed)
    category_curvatures: Dict[str, float] = None

    def as_dict(self) -> dict:
        return {
            'layer_idx': self.layer_idx,
            'n_statements': self.n_statements,
            'n_tokens': self.n_tokens,
            'mean_curvature': self.mean_curvature,
            'curvature_variance': self.curvature_variance,
            'curvature_sign': self.curvature_sign,
            'dimension_at_10': self.dimension_at_10,
            'dimension_at_50': self.dimension_at_50,
            'dimension_at_100': self.dimension_at_100,
            'dimension_scaling_exponent': self.dimension_scaling_exponent,
            'category_curvatures': self.category_curvatures,
        }


class ScaleCurvatureAnalyzer:
    """Analyze curvature at scale by aggregating many activations."""

    def __init__(self, model, tokenizer, backend):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.collector = ActivationCollector(model, tokenizer)

        # Lazy-load geometry tools
        self._id_estimator = None
        self._curvature_estimator = None

    @property
    def id_estimator(self):
        if self._id_estimator is None:
            from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
            self._id_estimator = IntrinsicDimension(self.backend)
        return self._id_estimator

    @property
    def curvature_estimator(self):
        if self._curvature_estimator is None:
            from modelcypher.core.domain.geometry.ollivier_ricci import OllivierRicciCurvature
            self._curvature_estimator = OllivierRicciCurvature(self.backend)
        return self._curvature_estimator

    def collect_aggregate_activations(
        self,
        statements: List[str],
        layer_idx: int,
        mode: str = 'all_tokens'  # 'all_tokens', 'last_token', 'mean_token'
    ) -> np.ndarray:
        """Collect activations for many statements into one point cloud."""
        all_points = []

        for text in statements:
            try:
                acts = self.collector.get_activations(text, layer_idx)

                if mode == 'all_tokens':
                    all_points.append(acts)  # All tokens
                elif mode == 'last_token':
                    all_points.append(acts[-1:])  # Last token only
                elif mode == 'mean_token':
                    all_points.append(acts.mean(axis=0, keepdims=True))  # Mean pooled
            except Exception as e:
                logger.debug(f"Failed: {text[:30]}... ({e})")

        if not all_points:
            return np.zeros((1, 1024))

        return np.vstack(all_points)

    def compute_dimension_scaling(self, points: np.ndarray) -> Tuple[float, float, float, float]:
        """Measure how dimension scales with number of samples.

        If dimension is truly D, it should stabilize as samples increase.
        If it keeps growing, we might be seeing fractal/irrational dimension behavior.
        """
        n_total = points.shape[0]

        dims = []
        sample_sizes = [10, 50, 100, min(200, n_total)]

        for n in sample_sizes:
            if n > n_total:
                continue
            # Random subsample
            idx = np.random.choice(n_total, size=min(n, n_total), replace=False)
            subsample = points[idx]

            try:
                arr = self.backend.array(subsample.astype(np.float32))
                result = self.id_estimator.compute(arr)
                dims.append((n, result.intrinsic_dimension))
            except:
                dims.append((n, 1.0))

        # Fit power law: dim = a * n^b
        # log(dim) = log(a) + b * log(n)
        if len(dims) >= 2:
            log_n = np.log([d[0] for d in dims])
            log_dim = np.log([max(d[1], 0.1) for d in dims])

            # Linear regression in log space
            A = np.vstack([log_n, np.ones(len(log_n))]).T
            b, log_a = np.linalg.lstsq(A, log_dim, rcond=None)[0]
            exponent = b
        else:
            exponent = 0.0

        dim_10 = dims[0][1] if len(dims) > 0 else 0.0
        dim_50 = dims[1][1] if len(dims) > 1 else 0.0
        dim_100 = dims[2][1] if len(dims) > 2 else 0.0

        return dim_10, dim_50, dim_100, exponent

    def analyze_layer(self, statements: List[str], layer_idx: int) -> ScaleCurvatureResult:
        """Full scale analysis for a single layer."""
        logger.info(f"\nCollecting activations for layer {layer_idx}...")

        # Collect all activations
        points = self.collect_aggregate_activations(statements, layer_idx, mode='all_tokens')
        n_tokens = points.shape[0]
        logger.info(f"  Collected {n_tokens} activation vectors from {len(statements)} statements")

        # Dimension scaling analysis
        logger.info("  Computing dimension scaling...")
        dim_10, dim_50, dim_100, exponent = self.compute_dimension_scaling(points)
        logger.info(f"    dim@10={dim_10:.2f}, dim@50={dim_50:.2f}, dim@100={dim_100:.2f}")
        logger.info(f"    scaling exponent: {exponent:.3f}")

        # Ollivier-Ricci curvature (graph-based, works for point clouds)
        logger.info("  Computing Ollivier-Ricci curvature...")
        try:
            arr = self.backend.array(points.astype(np.float32))
            result = self.curvature_estimator.compute(arr)
            mean_curv = result.mean_edge_curvature
            var_curv = result.std_edge_curvature ** 2  # Convert std to variance
            # Classify sign based on mean
            if mean_curv > 0.01:
                sign = "positive"
            elif mean_curv < -0.01:
                sign = "negative"
            else:
                sign = "flat"
        except Exception as e:
            logger.warning(f"    Curvature estimation failed: {e}")
            mean_curv, var_curv, sign = 0.0, 0.0, "unknown"

        logger.info(f"    mean_curvature={mean_curv:.6f}, sign={sign}")

        return ScaleCurvatureResult(
            layer_idx=layer_idx,
            n_statements=len(statements),
            n_tokens=n_tokens,
            mean_curvature=mean_curv,
            curvature_variance=var_curv,
            curvature_sign=sign,
            dimension_at_10=dim_10,
            dimension_at_50=dim_50,
            dimension_at_100=dim_100,
            dimension_scaling_exponent=exponent,
        )

    def analyze_all_layers(self, statements: List[str]) -> List[ScaleCurvatureResult]:
        """Analyze curvature evolution through all layers."""
        n_layers = self.collector.n_layers
        target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

        results = []
        for layer_idx in target_layers:
            result = self.analyze_layer(statements, layer_idx)
            results.append(result)

        return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scale Curvature Analysis")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-350M-MLX-bf16",
        help="Path to model"
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
    analyzer = ScaleCurvatureAnalyzer(model, tokenizer, backend)

    logger.info("\n" + "=" * 80)
    logger.info("SCALE CURVATURE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Statements: {len(DIVERSE_STATEMENTS)}")
    logger.info("Hypothesis: Curvature emerges at scale, not locally")
    logger.info("Looking for: Non-integer dimension scaling (π dimension?)")

    # Run analysis
    results = analyzer.analyze_all_layers(DIVERSE_STATEMENTS)

    # Report
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info("\nDimension scaling through layers:")
    for r in results:
        logger.info(
            f"  L{r.layer_idx:2d}: dim@10={r.dimension_at_10:.2f} → dim@100={r.dimension_at_100:.2f} "
            f"(exp={r.dimension_scaling_exponent:.3f})"
        )

    logger.info("\nCurvature through layers:")
    for r in results:
        logger.info(
            f"  L{r.layer_idx:2d}: curvature={r.mean_curvature:.6f} ({r.curvature_sign})"
        )

    # Check for irrational dimension signature
    logger.info("\nIrrational dimension check:")
    for r in results:
        # If exponent is close to 0, dimension is integer (stabilizes)
        # If exponent > 0, dimension keeps growing (fractal behavior)
        if abs(r.dimension_scaling_exponent) < 0.05:
            logger.info(f"  L{r.layer_idx}: INTEGER dimension (stable)")
        elif r.dimension_scaling_exponent > 0:
            logger.info(f"  L{r.layer_idx}: GROWING dimension (possible fractal) exp={r.dimension_scaling_exponent:.3f}")
        else:
            logger.info(f"  L{r.layer_idx}: SHRINKING dimension exp={r.dimension_scaling_exponent:.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "data" / "scale_curvature"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"

    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'n_statements': len(DIVERSE_STATEMENTS),
        'results': [r.as_dict() for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
