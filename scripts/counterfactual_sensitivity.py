#!/usr/bin/env python3
"""Counterfactual Sensitivity Test.

Key insight from semantic invariance test: Models learn LINGUISTIC patterns
more than factual content. Hedging language is highly invariant because
"Maybe X" and "X might be" and "It could be X" are syntactically similar.

New hypothesis: FACTUAL knowledge should be SENSITIVE to counterfactuals.
- "2 + 2 = 4" and "2 + 2 = 5" should have DIFFERENT representations
- "Paris is the capital of France" and "London is the capital of France" DIFFERENT
- But "Pizza is the best food" and "Sushi is the best food" should be SIMILAR
  (both are opinions, the model doesn't distinguish)

Metric: Counterfactual Sensitivity
- High sensitivity = representation changes when fact is violated = model "knows"
- Low sensitivity = representation unchanged = model doesn't distinguish true/false

This is the inverse of the previous test:
- Previous: same meaning, different words -> should be similar (invariance)
- New: different meaning, similar words -> should be different (sensitivity)

For facts: HIGH sensitivity expected (model knows 2+2≠5)
For opinions: LOW sensitivity expected (model doesn't know "best" food)
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
# Counterfactual Pairs
# =============================================================================

# True statement -> False counterfactual
FACTUAL_COUNTERFACTUALS = [
    # Math - model should KNOW these are different
    ("2 + 2 = 4", "2 + 2 = 5"),
    ("3 * 3 = 9", "3 * 3 = 8"),
    ("10 - 5 = 5", "10 - 5 = 6"),

    # Geography - model should know capitals
    ("Paris is the capital of France", "London is the capital of France"),
    ("Tokyo is in Japan", "Tokyo is in China"),
    ("The Nile is in Africa", "The Nile is in Europe"),

    # Science - basic facts
    ("Water freezes at 0 degrees", "Water freezes at 100 degrees"),
    ("The Earth orbits the Sun", "The Sun orbits the Earth"),
    ("Fire is hot", "Fire is cold"),

    # Logic - model should know implications
    ("All dogs are mammals", "All dogs are reptiles"),
    ("Humans need oxygen to breathe", "Humans need carbon dioxide to breathe"),
]

# Both are opinions - model should treat similarly
OPINION_COUNTERFACTUALS = [
    # Preferences - both equally valid opinions
    ("Pizza is the best food", "Sushi is the best food"),
    ("Blue is the nicest color", "Red is the nicest color"),
    ("Summer is the best season", "Winter is the best season"),

    # Subjective judgments
    ("Python is a great language", "Java is a great language"),
    ("Dogs are better pets than cats", "Cats are better pets than dogs"),
    ("Morning is the best time to work", "Evening is the best time to work"),

    # Speculative - both equally uncertain
    ("AI will help humanity", "AI will harm humanity"),
    ("The future looks bright", "The future looks dark"),
    ("Change is good", "Change is bad"),
]


# =============================================================================
# Counterfactual Sensitivity Analyzer
# =============================================================================

@dataclass
class SensitivityResult:
    """Result of counterfactual sensitivity analysis."""
    true_statement: str
    false_statement: str
    category: str  # 'factual' or 'opinion'
    cosine_distance: float  # 1 - cosine_similarity
    euclidean_distance: float
    representation_diff_norm: float

    @property
    def sensitivity_score(self) -> float:
        """Higher = more sensitive to counterfactual = "knows" the difference."""
        # Combine metrics - normalize by expected ranges
        # Cosine distance: 0 to 2 (but usually 0 to 1)
        # Use cosine distance as primary metric
        return self.cosine_distance

    def as_dict(self) -> dict:
        return {
            'true_statement': self.true_statement,
            'false_statement': self.false_statement,
            'category': self.category,
            'cosine_distance': self.cosine_distance,
            'euclidean_distance': self.euclidean_distance,
            'sensitivity_score': self.sensitivity_score,
        }


class CounterfactualSensitivityAnalyzer:
    """Analyze sensitivity to counterfactual changes."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if hasattr(model.model, 'layers'):
            self.n_layers = len(model.model.layers)
        else:
            self.n_layers = 24

    def get_representation(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Get representation from a specific layer (last token position)."""
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
                return np.array(captured['output'][0, -1, :].tolist())
            else:
                return np.zeros(1024)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def analyze_pair(
        self,
        true_stmt: str,
        false_stmt: str,
        category: str,
        layer_idx: int,
    ) -> SensitivityResult:
        """Analyze sensitivity between true and counterfactual statements."""

        rep_true = self.get_representation(true_stmt, layer_idx)
        rep_false = self.get_representation(false_stmt, layer_idx)

        # Cosine distance (1 - similarity)
        n_true, n_false = np.linalg.norm(rep_true), np.linalg.norm(rep_false)
        if n_true > 1e-10 and n_false > 1e-10:
            cosine_sim = np.dot(rep_true, rep_false) / (n_true * n_false)
            cosine_dist = 1 - cosine_sim
        else:
            cosine_dist = 1.0

        # Euclidean distance (normalized by dimension)
        diff = rep_true - rep_false
        euclidean_dist = np.linalg.norm(diff) / np.sqrt(len(diff))

        return SensitivityResult(
            true_statement=true_stmt,
            false_statement=false_stmt,
            category=category,
            cosine_distance=float(cosine_dist),
            euclidean_distance=float(euclidean_dist),
            representation_diff_norm=float(np.linalg.norm(diff)),
        )


# =============================================================================
# Main Discovery
# =============================================================================

class CounterfactualDiscovery:
    """Discover factual knowledge through counterfactual sensitivity."""

    def __init__(self, model, tokenizer):
        self.analyzer = CounterfactualSensitivityAnalyzer(model, tokenizer)
        self.results: List[SensitivityResult] = []

    def run_discovery(self, layer_idx: int | None = None) -> Dict:
        """Run counterfactual discovery."""

        if layer_idx is None:
            layer_idx = self.analyzer.n_layers // 2

        logger.info(f"Analyzing at layer {layer_idx}")

        output = {'factual': [], 'opinion': []}

        # Analyze factual counterfactuals
        logger.info("\nAnalyzing FACTUAL counterfactuals (should show HIGH sensitivity)...")
        for true_stmt, false_stmt in FACTUAL_COUNTERFACTUALS:
            result = self.analyzer.analyze_pair(
                true_stmt, false_stmt, 'factual', layer_idx
            )
            self.results.append(result)
            output['factual'].append(result.as_dict())

            logger.info(
                f"  sens={result.sensitivity_score:.3f} | "
                f"'{true_stmt[:25]}' vs '{false_stmt[:25]}'"
            )

        # Analyze opinion counterfactuals
        logger.info("\nAnalyzing OPINION counterfactuals (should show LOW sensitivity)...")
        for true_stmt, false_stmt in OPINION_COUNTERFACTUALS:
            result = self.analyzer.analyze_pair(
                true_stmt, false_stmt, 'opinion', layer_idx
            )
            self.results.append(result)
            output['opinion'].append(result.as_dict())

            logger.info(
                f"  sens={result.sensitivity_score:.3f} | "
                f"'{true_stmt[:25]}' vs '{false_stmt[:25]}'"
            )

        return output

    def analyze_separation(self) -> Dict:
        """Analyze separation between factual and opinion sensitivity."""

        factual_sens = [
            r.sensitivity_score for r in self.results if r.category == 'factual'
        ]
        opinion_sens = [
            r.sensitivity_score for r in self.results if r.category == 'opinion'
        ]

        if not factual_sens or not opinion_sens:
            return {}

        analysis = {
            'factual_mean_sensitivity': float(np.mean(factual_sens)),
            'factual_std_sensitivity': float(np.std(factual_sens)),
            'opinion_mean_sensitivity': float(np.mean(opinion_sens)),
            'opinion_std_sensitivity': float(np.std(opinion_sens)),
        }

        # For counterfactual sensitivity, we WANT facts > opinions
        gap = analysis['factual_mean_sensitivity'] - analysis['opinion_mean_sensitivity']
        pooled_std = np.sqrt(
            (analysis['factual_std_sensitivity']**2 +
             analysis['opinion_std_sensitivity']**2) / 2
        )
        analysis['separation_gap'] = gap
        analysis['effect_size'] = gap / pooled_std if pooled_std > 0 else 0

        return analysis

    def report(self) -> str:
        """Generate report."""
        analysis = self.analyze_separation()

        report = [
            "=" * 80,
            "COUNTERFACTUAL SENSITIVITY DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total pairs analyzed: {len(self.results)}",
            "",
            "HYPOTHESIS: Facts should show HIGHER sensitivity to counterfactuals",
            "            (model should 'know' 2+2≠5 but not 'know' best food)",
            "",
            "RESULTS:",
            f"  Factual counterfactuals:",
            f"    Mean sensitivity: {analysis.get('factual_mean_sensitivity', 0):.3f}",
            f"    Std:              {analysis.get('factual_std_sensitivity', 0):.3f}",
            "",
            f"  Opinion counterfactuals:",
            f"    Mean sensitivity: {analysis.get('opinion_mean_sensitivity', 0):.3f}",
            f"    Std:              {analysis.get('opinion_std_sensitivity', 0):.3f}",
            "",
            f"  Separation gap:    {analysis.get('separation_gap', 0):.3f}",
            f"  Effect size:       {analysis.get('effect_size', 0):.2f}",
            "",
            "INTERPRETATION:",
        ]

        effect_size = analysis.get('effect_size', 0)
        if effect_size > 0.8:
            report.append("  STRONG: Model distinguishes true facts from counterfactuals!")
            report.append("  -> Model has learned factual knowledge, not just patterns.")
        elif effect_size > 0.5:
            report.append("  MODERATE: Some factual knowledge signal detected.")
        elif effect_size > 0.2:
            report.append("  WEAK: Slight factual knowledge signal.")
        elif effect_size > -0.2:
            report.append("  NONE: Model doesn't distinguish facts from opinions.")
            report.append("  -> Model may be pattern-matching, not 'knowing'.")
        else:
            report.append("  REVERSED: Model is MORE sensitive to opinion changes!")
            report.append("  -> This is unexpected and may indicate a bug or")
            report.append("     that opinions have more linguistic variation.")

        report.extend([
            "",
            "INDIVIDUAL RESULTS (sorted by sensitivity):",
        ])

        sorted_results = sorted(
            self.results, key=lambda r: r.sensitivity_score, reverse=True
        )
        for r in sorted_results:
            report.append(
                f"  {r.sensitivity_score:.3f} [{r.category:7}] "
                f"{r.true_statement[:35]} vs ..."
            )

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Sensitivity Discovery")
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
    discovery = CounterfactualDiscovery(model, tokenizer)
    results = discovery.run_discovery(args.layer)

    # Print report
    print("\n" + discovery.report())

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "counterfactual_sensitivity.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'results': results,
        'analysis': discovery.analyze_separation(),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
