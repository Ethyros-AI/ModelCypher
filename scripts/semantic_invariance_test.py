#!/usr/bin/env python3
"""Semantic Invariance Test.

The key insight: Model "confidence" is NOT the same as "factual knowledge."
A model can be very confident about completing "I think maybe..." because
hedging language is predictable, but that doesn't mean it "knows" anything.

TRUE factual knowledge should exhibit SEMANTIC INVARIANCE:
- "The capital of France is Paris" and "Paris is the capital of France"
  should have similar internal representations
- "2 + 2 = 4" and "Four is equal to two plus two" should be similar

UNCERTAIN/OPINION content lacks this invariance:
- "The best food is pizza" vs "Pizza is the best food"
  could have very different representations depending on context

This script tests semantic invariance by:
1. Creating paraphrase pairs for facts and opinions
2. Measuring representation similarity across paraphrases
3. Computing "invariance score" = mean CKA across paraphrases

Hypothesis: Facts have HIGH invariance, opinions have LOW invariance.
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
# Paraphrase Sets
# =============================================================================

FACTUAL_PARAPHRASES = [
    # Math facts - should be highly invariant
    (
        "2 + 2 = 4",
        "Two plus two equals four",
        "Four is the sum of two and two",
        "The result of adding 2 and 2 is 4",
    ),
    (
        "Pi is approximately 3.14159",
        "The value of pi is about 3.14",
        "3.14159 is an approximation of pi",
    ),
    # Geography facts
    (
        "Paris is the capital of France",
        "The capital of France is Paris",
        "France has Paris as its capital",
    ),
    (
        "Tokyo is in Japan",
        "Japan contains Tokyo",
        "Tokyo is a city in Japan",
    ),
    # Science facts
    (
        "Water freezes at 0 degrees Celsius",
        "Zero degrees Celsius is the freezing point of water",
        "At 0 degrees C, water freezes",
    ),
    (
        "The Earth orbits the Sun",
        "The Sun is orbited by the Earth",
        "Earth goes around the Sun",
    ),
    # Logical facts
    (
        "All mammals are warm-blooded",
        "Mammals are warm-blooded animals",
        "Being warm-blooded is a property of mammals",
    ),
]

OPINION_PARAPHRASES = [
    # Subjective preferences - should vary more
    (
        "Pizza is the best food",
        "The best food is pizza",
        "Pizza is better than other foods",
    ),
    (
        "Python is a great programming language",
        "Programming in Python is great",
        "A great language for programming is Python",
    ),
    (
        "Summer is the nicest season",
        "The nicest season is summer",
        "Summer is nicer than other seasons",
    ),
    # Speculative statements
    (
        "AI will transform the world",
        "The world will be transformed by AI",
        "Transforming the world is what AI will do",
    ),
    (
        "The future looks bright",
        "Looking at the future, it seems bright",
        "A bright future lies ahead",
    ),
    # Hedged statements
    (
        "Maybe the answer is yes",
        "The answer might be yes",
        "It could be that the answer is yes",
    ),
]


# =============================================================================
# Semantic Invariance Analyzer
# =============================================================================

@dataclass
class InvarianceResult:
    """Result of semantic invariance analysis."""
    paraphrase_set: Tuple[str, ...]
    category: str
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    std_similarity: float
    pairwise_similarities: List[float]

    def as_dict(self) -> dict:
        return {
            'paraphrases': list(self.paraphrase_set),
            'category': self.category,
            'mean_similarity': self.mean_similarity,
            'min_similarity': self.min_similarity,
            'max_similarity': self.max_similarity,
            'std_similarity': self.std_similarity,
            'invariance_score': self.invariance_score,
        }

    @property
    def invariance_score(self) -> float:
        """Invariance score: high mean + low variance = stable representation."""
        # Penalize high variance (inconsistent paraphrases)
        return self.mean_similarity - self.std_similarity


class SemanticInvarianceAnalyzer:
    """Analyze semantic invariance across paraphrases."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # Detect number of layers
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

        # Hook the MLP output
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
                # Take last token position
                return np.array(captured['output'][0, -1, :].tolist())
            else:
                return np.zeros(1024)
        finally:
            if key == 'feed_forward':
                layer.feed_forward = original
            else:
                layer.mlp = original

    def compute_similarity(self, rep1: np.ndarray, rep2: np.ndarray) -> float:
        """Compute cosine similarity between representations."""
        n1, n2 = np.linalg.norm(rep1), np.linalg.norm(rep2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(rep1, rep2) / (n1 * n2))

    def analyze_paraphrase_set(
        self,
        paraphrases: Tuple[str, ...],
        category: str,
        layer_idx: int,
    ) -> InvarianceResult:
        """Analyze invariance across a set of paraphrases."""

        # Get representations for all paraphrases
        reps = []
        for p in paraphrases:
            rep = self.get_representation(p, layer_idx)
            reps.append(rep)

        # Compute all pairwise similarities
        similarities = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                sim = self.compute_similarity(reps[i], reps[j])
                similarities.append(sim)

        if not similarities:
            return InvarianceResult(
                paraphrase_set=paraphrases,
                category=category,
                mean_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                std_similarity=0.0,
                pairwise_similarities=[],
            )

        return InvarianceResult(
            paraphrase_set=paraphrases,
            category=category,
            mean_similarity=float(np.mean(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            std_similarity=float(np.std(similarities)),
            pairwise_similarities=similarities,
        )


# =============================================================================
# Main Discovery
# =============================================================================

class SemanticInvarianceDiscovery:
    """Discover geometric invariance through paraphrase testing."""

    def __init__(self, model, tokenizer):
        self.analyzer = SemanticInvarianceAnalyzer(model, tokenizer)
        self.results: List[InvarianceResult] = []

    def run_discovery(self, layer_idx: int | None = None) -> Dict:
        """Run invariance discovery."""

        if layer_idx is None:
            layer_idx = self.analyzer.n_layers // 2

        logger.info(f"Analyzing at layer {layer_idx}")

        output = {'factual': [], 'opinion': []}

        # Analyze factual paraphrases
        logger.info("\nAnalyzing FACTUAL paraphrase sets...")
        for paraphrase_set in FACTUAL_PARAPHRASES:
            result = self.analyzer.analyze_paraphrase_set(
                paraphrase_set, 'factual', layer_idx
            )
            self.results.append(result)
            output['factual'].append(result.as_dict())

            logger.info(
                f"  inv={result.invariance_score:.3f} "
                f"mean={result.mean_similarity:.3f} "
                f"std={result.std_similarity:.3f} | "
                f"{paraphrase_set[0][:40]}"
            )

        # Analyze opinion paraphrases
        logger.info("\nAnalyzing OPINION paraphrase sets...")
        for paraphrase_set in OPINION_PARAPHRASES:
            result = self.analyzer.analyze_paraphrase_set(
                paraphrase_set, 'opinion', layer_idx
            )
            self.results.append(result)
            output['opinion'].append(result.as_dict())

            logger.info(
                f"  inv={result.invariance_score:.3f} "
                f"mean={result.mean_similarity:.3f} "
                f"std={result.std_similarity:.3f} | "
                f"{paraphrase_set[0][:40]}"
            )

        return output

    def analyze_separation(self) -> Dict:
        """Analyze separation between factual and opinion invariance."""

        factual_scores = [
            r.invariance_score for r in self.results if r.category == 'factual'
        ]
        opinion_scores = [
            r.invariance_score for r in self.results if r.category == 'opinion'
        ]

        if not factual_scores or not opinion_scores:
            return {}

        analysis = {
            'factual_mean': float(np.mean(factual_scores)),
            'factual_std': float(np.std(factual_scores)),
            'opinion_mean': float(np.mean(opinion_scores)),
            'opinion_std': float(np.std(opinion_scores)),
        }

        gap = analysis['factual_mean'] - analysis['opinion_mean']
        pooled_std = np.sqrt(
            (analysis['factual_std']**2 + analysis['opinion_std']**2) / 2
        )
        analysis['separation_gap'] = gap
        analysis['effect_size'] = gap / pooled_std if pooled_std > 0 else 0

        # Also check raw similarity
        factual_sim = np.mean([r.mean_similarity for r in self.results if r.category == 'factual'])
        opinion_sim = np.mean([r.mean_similarity for r in self.results if r.category == 'opinion'])
        analysis['factual_mean_similarity'] = float(factual_sim)
        analysis['opinion_mean_similarity'] = float(opinion_sim)

        return analysis

    def report(self) -> str:
        """Generate report."""
        analysis = self.analyze_separation()

        report = [
            "=" * 80,
            "SEMANTIC INVARIANCE DISCOVERY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total paraphrase sets analyzed: {len(self.results)}",
            "",
            "HYPOTHESIS: Facts should have HIGHER paraphrase invariance than opinions.",
            "",
            "RESULTS:",
            f"  Factual paraphrases:",
            f"    Mean invariance score: {analysis.get('factual_mean', 0):.3f}",
            f"    Mean similarity:       {analysis.get('factual_mean_similarity', 0):.3f}",
            f"    Std:                   {analysis.get('factual_std', 0):.3f}",
            "",
            f"  Opinion paraphrases:",
            f"    Mean invariance score: {analysis.get('opinion_mean', 0):.3f}",
            f"    Mean similarity:       {analysis.get('opinion_mean_similarity', 0):.3f}",
            f"    Std:                   {analysis.get('opinion_std', 0):.3f}",
            "",
            f"  Separation gap:    {analysis.get('separation_gap', 0):.3f}",
            f"  Effect size:       {analysis.get('effect_size', 0):.2f}",
            "",
            "INTERPRETATION:",
        ]

        effect_size = analysis.get('effect_size', 0)
        if effect_size > 0.8:
            report.append("  STRONG: Facts show significantly more invariance than opinions!")
            report.append("  -> Semantic invariance IS a valid signal for factual knowledge.")
        elif effect_size > 0.5:
            report.append("  MODERATE: Some signal, but overlap between categories.")
        elif effect_size > 0.2:
            report.append("  WEAK: Slight signal, but noisy.")
        elif effect_size > -0.2:
            report.append("  NONE: No meaningful separation detected.")
        else:
            report.append("  REVERSED: Opinions show MORE invariance than facts!")
            report.append("  -> This could indicate the model learned linguistic patterns")
            report.append("     more than factual content.")

        # Show individual results
        report.extend([
            "",
            "INDIVIDUAL RESULTS (sorted by invariance score):",
        ])

        sorted_results = sorted(self.results, key=lambda r: r.invariance_score, reverse=True)
        for r in sorted_results:
            report.append(
                f"  {r.invariance_score:.3f} [{r.category:7}] {r.paraphrase_set[0][:50]}"
            )

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Semantic Invariance Discovery")
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
    discovery = SemanticInvarianceDiscovery(model, tokenizer)
    results = discovery.run_discovery(args.layer)

    # Print report
    print("\n" + discovery.report())

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "semantic_invariance.json"

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
