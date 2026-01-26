#!/usr/bin/env python3
"""Dimensional Hierarchy of Knowledge.

Testing the hypothesis that knowledge exists in a dimensional hierarchy:

Level 0: The territory (reality itself - unmeasurable)
Level 1: Facts about territory     → lowest intrinsic dimension
Level 2: Beliefs about facts       → +1 dimension (adds observer)
Level 3: Opinions about beliefs    → +1 dimension (adds preference)
Level 4: Meta-cognition            → +1 dimension (thought about thought)

The Ruliad Connection:
Wolfram's ruliad suggests all possible computations exist. The relationships
between concepts (apple, orange, fruit) exist as geodesics in this space
BEFORE we name them. Language samples this structure.

"Opinions are still facts" - "Person X believes Y" is a fact with coordinates
in the ruliad. The content Y might be false, but the existence of the belief
is factual.

Hypothesis:
    intrinsic_dim("Paris is the capital of France")
    < intrinsic_dim("I believe Paris is the capital of France")
    < intrinsic_dim("I think Paris is a beautiful capital")

Each meta-level adds dimensionality because it adds a perspective/observer axis.

Usage:
    python dimensional_hierarchy.py --model /path/to/model
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Hierarchical Statement Sets
# =============================================================================

# Each tuple: (base_fact, level_1_fact, level_2_belief, level_3_opinion, level_4_meta)
HIERARCHICAL_STATEMENTS = [
    # Geography
    (
        "Paris",  # L0: the thing itself (just the word)
        "Paris is the capital of France",  # L1: fact
        "I know that Paris is the capital of France",  # L2: belief about fact
        "I think Paris is a beautiful capital city",  # L3: opinion
        "I believe I feel strongly about Paris being beautiful",  # L4: meta
    ),
    (
        "Tokyo",
        "Tokyo is in Japan",
        "I know that Tokyo is in Japan",
        "I think Tokyo is the most exciting city",
        "I suspect my preference for Tokyo reflects my values",
    ),
    (
        "water",
        "Water freezes at zero degrees Celsius",
        "I understand that water freezes at zero degrees",
        "I find it fascinating that water freezes at zero",
        "I wonder why I find water's properties so interesting",
    ),
    # Math
    (
        "four",
        "Two plus two equals four",
        "I know that two plus two equals four",
        "I think mathematics is beautiful",
        "I believe my love of math shapes how I see the world",
    ),
    (
        "nine",
        "Three times three equals nine",
        "I understand that three times three is nine",
        "I find multiplication tables satisfying",
        "I wonder if my appreciation for patterns is innate",
    ),
    # Science
    (
        "Earth",
        "The Earth orbits the Sun",
        "I know that the Earth orbits the Sun",
        "I find our solar system magnificent",
        "I think my awe of space reflects something deep in me",
    ),
    (
        "gravity",
        "Gravity pulls objects toward Earth",
        "I understand that gravity pulls things down",
        "I think gravity is a fascinating force",
        "I wonder why I'm so curious about physical forces",
    ),
    # Logic
    (
        "mammals",
        "All dogs are mammals",
        "I know that all dogs are mammals",
        "I think dogs are wonderful mammals",
        "I believe my love of dogs says something about me",
    ),
    (
        "mortal",
        "All humans are mortal",
        "I understand that all humans are mortal",
        "I find mortality thought-provoking",
        "I wonder how my awareness of death shapes my choices",
    ),
    # Abstract
    (
        "time",
        "Time moves forward",
        "I perceive that time moves forward",
        "I feel time moves too quickly",
        "I think my perception of time reveals my mental state",
    ),
]

# Level labels
LEVEL_NAMES = [
    "L0_entity",      # Just the word
    "L1_fact",        # Fact about the thing
    "L2_belief",      # Belief about the fact
    "L3_opinion",     # Opinion/feeling
    "L4_meta",        # Meta-cognition
]


# =============================================================================
# Intrinsic Dimension Estimator
# =============================================================================

class IntrinsicDimensionEstimator:
    """Estimate intrinsic dimension using TwoNN method."""

    def __init__(self, backend=None):
        if backend is None:
            from modelcypher.core.domain._backend import get_default_backend
            backend = get_default_backend()
        self.backend = backend

    def estimate_twonn(self, activations: np.ndarray) -> float:
        """Estimate intrinsic dimension using TwoNN (Facco et al.).

        Uses the ratio of distances to first and second nearest neighbors.
        ID = 1 / mean(log(r2/r1))
        """
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        n_samples = activations.shape[0]
        if n_samples < 3:
            # Not enough samples, use effective rank instead
            return self._effective_rank(activations)

        # Compute pairwise distances
        dists = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                dists[i, j] = np.linalg.norm(activations[i] - activations[j])

        # For each point, find r1 (nearest) and r2 (second nearest)
        mus = []
        for i in range(n_samples):
            row = dists[i].copy()
            row[i] = np.inf  # Exclude self
            sorted_dists = np.sort(row)
            r1 = sorted_dists[0]
            r2 = sorted_dists[1] if len(sorted_dists) > 1 else r1

            if r1 > 1e-10:
                mu = r2 / r1
                if mu > 1:
                    mus.append(np.log(mu))

        if not mus:
            return self._effective_rank(activations)

        # ID = 1 / mean(log(mu))
        mean_log_mu = np.mean(mus)
        if mean_log_mu > 1e-10:
            return 1.0 / mean_log_mu
        else:
            return self._effective_rank(activations)

    def _effective_rank(self, activations: np.ndarray) -> float:
        """Fallback: effective rank from SVD."""
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

    def estimate_from_representation(self, rep: np.ndarray) -> Dict[str, float]:
        """Estimate multiple dimension metrics from a single representation."""
        if rep.ndim == 1:
            rep = rep.reshape(1, -1)

        # Effective rank (always available)
        eff_rank = self._effective_rank(rep)

        # Variance concentration
        try:
            _, S, _ = np.linalg.svd(rep, full_matrices=False)
            S_sq = S ** 2
            total = S_sq.sum()
            var_top1 = float(S_sq[0] / total) if total > 1e-10 else 1.0
            var_top3 = float(S_sq[:3].sum() / total) if total > 1e-10 and len(S_sq) >= 3 else var_top1
        except:
            var_top1 = 1.0
            var_top3 = 1.0

        # Spectral entropy
        try:
            S_norm = S / S.sum() if S.sum() > 1e-10 else S
            spectral_entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-10)))
        except:
            spectral_entropy = 0.0

        return {
            'effective_rank': eff_rank,
            'var_top1': var_top1,
            'var_top3': var_top3,
            'spectral_entropy': spectral_entropy,
        }


# =============================================================================
# Hierarchy Analyzer
# =============================================================================

@dataclass
class LevelSignature:
    """Signature for a single level of the hierarchy."""
    level: int
    level_name: str
    statement: str
    effective_rank: float
    var_top1: float
    var_top3: float
    spectral_entropy: float

    def as_dict(self) -> dict:
        return {
            'level': self.level,
            'level_name': self.level_name,
            'statement': self.statement,
            'effective_rank': self.effective_rank,
            'var_top1': self.var_top1,
            'var_top3': self.var_top3,
            'spectral_entropy': self.spectral_entropy,
        }


@dataclass
class HierarchyResult:
    """Result for a complete hierarchy (L0 through L4)."""
    base_concept: str
    levels: List[LevelSignature]

    @property
    def dimension_progression(self) -> List[float]:
        """Get effective rank progression across levels."""
        return [l.effective_rank for l in self.levels]

    @property
    def is_monotonic_increasing(self) -> bool:
        """Does dimension increase with each level?"""
        dims = self.dimension_progression
        for i in range(1, len(dims)):
            if dims[i] <= dims[i-1]:
                return False
        return True

    @property
    def total_dimension_increase(self) -> float:
        """Total dimension increase from L0 to L4."""
        dims = self.dimension_progression
        return dims[-1] - dims[0] if len(dims) >= 2 else 0.0

    def as_dict(self) -> dict:
        return {
            'base_concept': self.base_concept,
            'levels': [l.as_dict() for l in self.levels],
            'dimension_progression': self.dimension_progression,
            'is_monotonic': self.is_monotonic_increasing,
            'total_increase': self.total_dimension_increase,
        }


class HierarchyAnalyzer:
    """Analyze dimensional hierarchy of knowledge."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dim_estimator = IntrinsicDimensionEstimator()

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_representation(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get MLP representation from a specific layer."""
        import mlx.core as mx

        tokens = self.tokenizer.encode(prompt)
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
                # Get ALL token positions for richer dimension estimate
                return np.array(captured['output'][0].tolist())
            else:
                return np.zeros((1, 1024))
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def analyze_statement(
        self,
        statement: str,
        level: int,
        level_name: str,
        layer_idx: int
    ) -> LevelSignature:
        """Analyze a single statement."""
        rep = self.get_representation(statement, layer_idx)
        metrics = self.dim_estimator.estimate_from_representation(rep)

        return LevelSignature(
            level=level,
            level_name=level_name,
            statement=statement,
            effective_rank=metrics['effective_rank'],
            var_top1=metrics['var_top1'],
            var_top3=metrics['var_top3'],
            spectral_entropy=metrics['spectral_entropy'],
        )

    def analyze_hierarchy(
        self,
        hierarchy: Tuple[str, ...],
        layer_idx: int
    ) -> HierarchyResult:
        """Analyze a complete hierarchy from L0 to L4."""
        levels = []

        for i, (statement, level_name) in enumerate(zip(hierarchy, LEVEL_NAMES)):
            sig = self.analyze_statement(statement, i, level_name, layer_idx)
            levels.append(sig)

        return HierarchyResult(
            base_concept=hierarchy[0],
            levels=levels,
        )


# =============================================================================
# Main Discovery
# =============================================================================

class DimensionalHierarchyDiscovery:
    """Discover dimensional hierarchy patterns."""

    def __init__(self, model, tokenizer):
        self.analyzer = HierarchyAnalyzer(model, tokenizer)
        self.results: List[HierarchyResult] = []

    def run(self, layer_idx: int | None = None) -> Dict:
        """Run hierarchy analysis."""
        if layer_idx is None:
            layer_idx = self.analyzer.n_layers // 2

        logger.info(f"Analyzing dimensional hierarchy at layer {layer_idx}")
        logger.info(f"Testing {len(HIERARCHICAL_STATEMENTS)} concept hierarchies")

        for hierarchy in HIERARCHICAL_STATEMENTS:
            try:
                result = self.analyzer.analyze_hierarchy(hierarchy, layer_idx)
                self.results.append(result)

                # Log progression
                dims = result.dimension_progression
                mono = "↑" if result.is_monotonic_increasing else "~"
                logger.info(
                    f"  {result.base_concept:12} {mono} "
                    f"L0={dims[0]:.1f} → L1={dims[1]:.1f} → L2={dims[2]:.1f} → "
                    f"L3={dims[3]:.1f} → L4={dims[4]:.1f} "
                    f"(Δ={result.total_dimension_increase:+.1f})"
                )
            except Exception as e:
                logger.warning(f"  Failed: {hierarchy[0]} ({e})")

        return {'hierarchies': [r.as_dict() for r in self.results]}

    def analyze_results(self) -> Dict:
        """Analyze patterns across all hierarchies."""
        if not self.results:
            return {}

        # Per-level statistics
        level_dims = {name: [] for name in LEVEL_NAMES}
        for result in self.results:
            for level in result.levels:
                level_dims[level.level_name].append(level.effective_rank)

        level_stats = {}
        for name, dims in level_dims.items():
            if dims:
                level_stats[name] = {
                    'mean': float(np.mean(dims)),
                    'std': float(np.std(dims)),
                    'min': float(np.min(dims)),
                    'max': float(np.max(dims)),
                }

        # Monotonicity
        n_monotonic = sum(1 for r in self.results if r.is_monotonic_increasing)
        monotonicity_rate = n_monotonic / len(self.results) if self.results else 0

        # Average dimension increase per level
        all_progressions = [r.dimension_progression for r in self.results]
        avg_progression = np.mean(all_progressions, axis=0).tolist()

        # Level-to-level increases
        level_increases = []
        for i in range(1, len(LEVEL_NAMES)):
            prev_dims = [r.dimension_progression[i-1] for r in self.results]
            curr_dims = [r.dimension_progression[i] for r in self.results]
            avg_increase = np.mean([c - p for c, p in zip(curr_dims, prev_dims)])
            level_increases.append({
                'from': LEVEL_NAMES[i-1],
                'to': LEVEL_NAMES[i],
                'avg_increase': float(avg_increase),
            })

        return {
            'n_hierarchies': len(self.results),
            'monotonicity_rate': monotonicity_rate,
            'level_stats': level_stats,
            'avg_progression': avg_progression,
            'level_increases': level_increases,
        }

    def report(self) -> str:
        """Generate report."""
        analysis = self.analyze_results()

        report = [
            "=" * 80,
            "DIMENSIONAL HIERARCHY DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "HYPOTHESIS: Each meta-level adds dimensionality",
            "  L0 (entity) < L1 (fact) < L2 (belief) < L3 (opinion) < L4 (meta)",
            "",
            "RESULTS:",
            f"  Hierarchies analyzed: {analysis.get('n_hierarchies', 0)}",
            f"  Monotonically increasing: {100*analysis.get('monotonicity_rate', 0):.0f}%",
            "",
            "AVERAGE DIMENSION BY LEVEL:",
        ]

        for name in LEVEL_NAMES:
            stats = analysis.get('level_stats', {}).get(name, {})
            report.append(
                f"  {name:12}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}"
            )

        report.extend([
            "",
            "LEVEL-TO-LEVEL INCREASES:",
        ])

        for inc in analysis.get('level_increases', []):
            arrow = "↑" if inc['avg_increase'] > 0 else "↓" if inc['avg_increase'] < 0 else "→"
            report.append(
                f"  {inc['from']:12} → {inc['to']:12}: {arrow} {inc['avg_increase']:+.2f}"
            )

        # Interpretation
        report.extend([
            "",
            "INTERPRETATION:",
        ])

        mono_rate = analysis.get('monotonicity_rate', 0)
        if mono_rate > 0.7:
            report.append("  STRONG SUPPORT: Dimension increases with meta-level!")
            report.append("  → The model's geometry reflects the hierarchy of abstraction.")
        elif mono_rate > 0.5:
            report.append("  MODERATE SUPPORT: Tendency for dimension to increase.")
        elif mono_rate > 0.3:
            report.append("  WEAK SUPPORT: Some hierarchical structure detected.")
        else:
            report.append("  NO SUPPORT: Dimension does not follow meta-level hierarchy.")

        # Check specific hypothesis
        level_stats = analysis.get('level_stats', {})
        if level_stats:
            l1_mean = level_stats.get('L1_fact', {}).get('mean', 0)
            l2_mean = level_stats.get('L2_belief', {}).get('mean', 0)
            l3_mean = level_stats.get('L3_opinion', {}).get('mean', 0)

            if l1_mean < l2_mean < l3_mean:
                report.append("")
                report.append("  KEY FINDING: fact_dim < belief_dim < opinion_dim ✓")
                report.append("  → Beliefs about facts have more dimensions than facts themselves!")
                report.append("  → Opinions add even more dimensionality (preference axis).")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Dimensional Hierarchy Discovery")
    parser.add_argument(
        "--model",
        default="/Volumes/CodeCypher/models/mlx-community/LFM2-1.2B-bf16",
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
    initialize_default_backend()

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Run discovery
    discovery = DimensionalHierarchyDiscovery(model, tokenizer)
    results = discovery.run(args.layer)

    # Print report
    print("\n" + discovery.report())

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "dimensional_hierarchy.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'results': results,
        'analysis': discovery.analyze_results(),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
