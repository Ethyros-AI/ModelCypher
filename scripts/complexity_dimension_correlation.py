#!/usr/bin/env python3
"""Complexity-Dimension Correlation.

Testing the hypothesis that intrinsic dimension correlates with
CONCEPTUAL COMPLEXITY, not categorical type (fact/belief/opinion).

Complexity proxies:
1. Token count (simple but correlated)
2. Unique concept count (nouns + verbs)
3. Relational depth (nested clauses)

If true: dim(statement) ≈ f(complexity(statement))
regardless of whether it's a fact, belief, or opinion.

Usage:
    python complexity_dimension_correlation.py --model /path/to/model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from datetime import datetime
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Statements with Varying Complexity
# =============================================================================

# Statements sorted by expected complexity (low to high)
# Mixed categories to show complexity matters more than type
COMPLEXITY_STATEMENTS = [
    # Very low complexity (1-2 concepts)
    ("Paris", "entity", 1),
    ("water", "entity", 1),
    ("good", "adjective", 1),
    ("red apple", "entity", 2),

    # Low complexity (2-3 concepts, simple predicate)
    ("Fire is hot", "fact", 2),
    ("Dogs bark", "fact", 2),
    ("I like pizza", "opinion", 2),
    ("The sky is blue", "fact", 3),
    ("Cats are mammals", "fact", 2),

    # Medium complexity (3-4 concepts, relation)
    ("Paris is in France", "fact", 3),
    ("Two plus two equals four", "fact", 4),
    ("I think dogs are great", "opinion", 4),
    ("Water freezes at zero degrees", "fact", 4),
    ("The Earth orbits the Sun", "fact", 4),

    # Higher complexity (4-6 concepts, nested relations)
    ("Paris is the capital of France", "fact", 5),
    ("I know that Paris is in France", "belief", 5),
    ("I believe dogs are loyal animals", "belief", 5),
    ("The Nile is the longest river in Africa", "fact", 6),
    ("I think mathematics is beautiful", "opinion", 4),

    # High complexity (6-8 concepts, multiple relations)
    ("I know that the Earth orbits around the Sun", "belief", 7),
    ("The capital of France is a city called Paris", "fact", 7),
    ("I believe that honesty is the best policy", "belief", 6),
    ("Water molecules consist of two hydrogen atoms and one oxygen atom", "fact", 8),

    # Very high complexity (8+ concepts, self-reference, nested)
    ("I think I understand why people like mathematics", "meta", 8),
    ("The relationship between mass and energy is described by Einstein", "fact", 8),
    ("I believe that my preference for dogs reflects my personality", "meta", 9),
    ("Scientists have discovered that the universe is expanding rapidly", "fact", 9),
    ("I wonder whether my beliefs about truth are themselves true", "meta", 10),

    # Maximum complexity (deep nesting, self-reference)
    ("I suspect that my tendency to overthink reveals something about my nature", "meta", 11),
    ("The theory suggests that consciousness emerges from complex information processing", "fact", 10),
    ("I believe that the way I think about my thinking shapes my understanding of myself", "meta", 13),
]


# =============================================================================
# Complexity Metrics
# =============================================================================

def count_tokens(text: str) -> int:
    """Simple token count."""
    return len(text.split())


def count_concepts(text: str) -> int:
    """Estimate concept count (content words)."""
    # Simple heuristic: non-stopword words
    stopwords = {
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
    words = re.findall(r'\b\w+\b', text.lower())
    return len([w for w in words if w not in stopwords])


def estimate_nesting_depth(text: str) -> int:
    """Estimate clause nesting depth."""
    # Simple heuristic: count subordinating conjunctions and relative pronouns
    markers = ['that', 'which', 'who', 'whom', 'whose', 'where', 'when',
               'while', 'if', 'because', 'although', 'whether', 'how', 'why']
    words = text.lower().split()
    depth = 1
    for word in words:
        if word in markers:
            depth += 1
    return depth


def compute_complexity_score(text: str) -> float:
    """Combined complexity score."""
    tokens = count_tokens(text)
    concepts = count_concepts(text)
    nesting = estimate_nesting_depth(text)

    # Weighted combination
    return 0.3 * tokens + 0.5 * concepts + 0.2 * nesting * 2


# =============================================================================
# Dimension Estimator
# =============================================================================

class DimensionEstimator:
    """Estimate intrinsic dimension of representations."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_representation(self, text: str, layer_idx: int) -> np.ndarray:
        """Get MLP representation."""
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

    def estimate_dimension(self, rep: np.ndarray) -> float:
        """Estimate effective dimension."""
        if rep.ndim == 1:
            rep = rep.reshape(1, -1)

        try:
            _, S, _ = np.linalg.svd(rep, full_matrices=False)
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
# Main Analysis
# =============================================================================

@dataclass
class CorrelationResult:
    """Result for a single statement."""
    text: str
    category: str
    expected_complexity: int
    measured_complexity: float
    intrinsic_dimension: float

    def as_dict(self) -> dict:
        return {
            'text': self.text,
            'category': self.category,
            'expected_complexity': self.expected_complexity,
            'measured_complexity': self.measured_complexity,
            'intrinsic_dimension': self.intrinsic_dimension,
        }


class ComplexityDimensionAnalysis:
    """Analyze correlation between complexity and dimension."""

    def __init__(self, model, tokenizer):
        self.estimator = DimensionEstimator(model, tokenizer)
        self.results: List[CorrelationResult] = []

    def run(self, layer_idx: int | None = None) -> Dict:
        """Run analysis."""
        if layer_idx is None:
            layer_idx = self.estimator.n_layers // 2

        logger.info(f"Analyzing complexity-dimension correlation at layer {layer_idx}")
        logger.info(f"Testing {len(COMPLEXITY_STATEMENTS)} statements")

        for text, category, expected in COMPLEXITY_STATEMENTS:
            try:
                rep = self.estimator.get_representation(text, layer_idx)
                dim = self.estimator.estimate_dimension(rep)
                complexity = compute_complexity_score(text)

                result = CorrelationResult(
                    text=text,
                    category=category,
                    expected_complexity=expected,
                    measured_complexity=complexity,
                    intrinsic_dimension=dim,
                )
                self.results.append(result)

                logger.info(
                    f"  cpx={complexity:5.1f} dim={dim:5.1f} [{category:8}] {text[:50]}"
                )
            except Exception as e:
                logger.warning(f"  Failed: {text[:30]}... ({e})")

        return {'results': [r.as_dict() for r in self.results]}

    def analyze_correlation(self) -> Dict:
        """Compute correlation statistics."""
        if len(self.results) < 3:
            return {}

        complexities = [r.measured_complexity for r in self.results]
        dimensions = [r.intrinsic_dimension for r in self.results]

        # Pearson correlation
        corr_matrix = np.corrcoef(complexities, dimensions)
        correlation = corr_matrix[0, 1]

        # Linear regression
        A = np.vstack([complexities, np.ones(len(complexities))]).T
        slope, intercept = np.linalg.lstsq(A, dimensions, rcond=None)[0]

        # R-squared
        predicted = np.array(complexities) * slope + intercept
        ss_res = np.sum((np.array(dimensions) - predicted) ** 2)
        ss_tot = np.sum((np.array(dimensions) - np.mean(dimensions)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Per-category analysis
        categories = set(r.category for r in self.results)
        category_stats = {}
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                cat_dims = [r.intrinsic_dimension for r in cat_results]
                category_stats[cat] = {
                    'mean_dim': float(np.mean(cat_dims)),
                    'std_dim': float(np.std(cat_dims)),
                    'n': len(cat_results),
                }

        return {
            'correlation': float(correlation),
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept),
            'n_samples': len(self.results),
            'category_stats': category_stats,
        }

    def report(self) -> str:
        """Generate report."""
        analysis = self.analyze_correlation()

        report = [
            "=" * 80,
            "COMPLEXITY-DIMENSION CORRELATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "HYPOTHESIS: Intrinsic dimension correlates with conceptual complexity,",
            "            NOT with statement type (fact/belief/opinion).",
            "",
            "CORRELATION ANALYSIS:",
            f"  Pearson correlation: {analysis.get('correlation', 0):.3f}",
            f"  R-squared:           {analysis.get('r_squared', 0):.3f}",
            f"  Linear fit:          dim = {analysis.get('slope', 0):.2f} * complexity + {analysis.get('intercept', 0):.2f}",
            f"  Samples:             {analysis.get('n_samples', 0)}",
            "",
            "PER-CATEGORY DIMENSION (if complexity matters, categories should overlap):",
        ]

        for cat, stats in analysis.get('category_stats', {}).items():
            report.append(
                f"  {cat:12}: {stats['mean_dim']:.2f} ± {stats['std_dim']:.2f} (n={stats['n']})"
            )

        report.extend([
            "",
            "INTERPRETATION:",
        ])

        corr = analysis.get('correlation', 0)
        if corr > 0.7:
            report.append("  STRONG CORRELATION: Dimension tracks complexity, not type!")
            report.append("  → The model's geometry reflects conceptual structure.")
        elif corr > 0.5:
            report.append("  MODERATE CORRELATION: Complexity is a significant factor.")
        elif corr > 0.3:
            report.append("  WEAK CORRELATION: Some relationship exists.")
        else:
            report.append("  NO CORRELATION: Complexity doesn't predict dimension.")

        # Show scatter hints
        report.extend([
            "",
            "SAMPLE POINTS (complexity → dimension):",
        ])

        sorted_results = sorted(self.results, key=lambda r: r.measured_complexity)
        for r in sorted_results[:5]:
            report.append(f"  {r.measured_complexity:5.1f} → {r.intrinsic_dimension:5.1f}: {r.text[:40]}")
        report.append("  ...")
        for r in sorted_results[-5:]:
            report.append(f"  {r.measured_complexity:5.1f} → {r.intrinsic_dimension:5.1f}: {r.text[:40]}")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Complexity-Dimension Correlation")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
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
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Run analysis
    analysis = ComplexityDimensionAnalysis(model, tokenizer)
    results = analysis.run(args.layer)

    # Print report
    print("\n" + analysis.report())

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "complexity_dimension.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'results': results,
        'analysis': analysis.analyze_correlation(),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
